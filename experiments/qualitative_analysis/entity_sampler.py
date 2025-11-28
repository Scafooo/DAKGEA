#!/usr/bin/env python3
"""Sample extractor for human evaluation of generated entities.

This module helps extract representative samples of synthetic entities
for manual human evaluation. It provides structured output formats
suitable for annotation tasks.

Features:
    - Random sampling
    - Stratified sampling (by dataset, by diversity level)
    - Export to TSV for spreadsheet annotation
    - Side-by-side comparison of original vs synthetic
    - Extract aligned pairs for consistency checking
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.data_structures import Dataset, Entity, KnowledgeGraph


class EntitySampler:
    """Extracts representative samples of synthetic entities."""

    def __init__(self, seed: int = 42):
        """Initialize sampler.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.encoder = None

    def extract_samples(
        self,
        original_dataset: Dataset,
        augmented_dataset: Dataset,
        n_samples: int = 50,
        strategy: str = "random",
        stage: str = "augmentation"
    ) -> List[Dict]:
        """Extract n samples of synthetic entities.

        Args:
            original_dataset: Original dataset
            augmented_dataset: Augmented dataset
            n_samples: Number of samples to extract
            strategy: Sampling strategy ('random', 'diverse', 'aligned')
            stage: Stage name ('reduction' or 'augmentation')

        Returns:
            List of sample dictionaries with entity data
        """
        # Get KGs based on stage
        if stage == "augmentation":
            orig_kg = original_dataset.kg1
            aug_kg = augmented_dataset.kg1
            paired_kg = augmented_dataset.kg2
        else:
            orig_kg = original_dataset.kg2
            aug_kg = augmented_dataset.kg2
            paired_kg = augmented_dataset.kg1

        # Identify synthetic entities
        orig_uris = set(orig_kg.entities.keys())
        aug_uris = set(aug_kg.entities.keys())
        synthetic_uris = list(aug_uris - orig_uris)

        if not synthetic_uris:
            return []

        # Sample based on strategy
        if strategy == "random":
            sampled_uris = self._random_sample(synthetic_uris, n_samples)
        elif strategy == "diverse":
            sampled_uris = self._diverse_sample(aug_kg, synthetic_uris, n_samples)
        elif strategy == "aligned":
            sampled_uris = self._aligned_sample(
                augmented_dataset, synthetic_uris, n_samples
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Extract entity data
        samples = []
        for uri in sampled_uris:
            sample = self._extract_entity_sample(
                uri,
                aug_kg,
                paired_kg,
                augmented_dataset,
                orig_kg
            )
            if sample:
                samples.append(sample)

        return samples

    def _random_sample(
        self,
        uris: List[str],
        n: int
    ) -> List[str]:
        """Random sampling."""
        return random.sample(uris, min(n, len(uris)))

    def _diverse_sample(
        self,
        kg: KnowledgeGraph,
        uris: List[str],
        n: int
    ) -> List[str]:
        """Sample diverse entities using embedding clustering.

        Maximizes diversity by selecting entities far apart in embedding space.
        """
        if len(uris) <= n:
            return uris

        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Create text representations
        texts = []
        valid_uris = []
        for uri in uris:
            entity = kg.entities.get(uri)
            if not entity:
                continue

            # Concatenate all attribute values
            text = " ".join(
                str(attr.value) for attr in entity.attributes
                if attr.value
            )
            if text:
                texts.append(text)
                valid_uris.append(uri)

        if not texts:
            return random.sample(uris, min(n, len(uris)))

        # Encode
        embeddings = self.encoder.encode(texts, convert_to_tensor=False, show_progress_bar=False)

        # Greedy diverse sampling
        selected_indices = []
        selected_indices.append(random.randint(0, len(embeddings) - 1))

        for _ in range(n - 1):
            if len(selected_indices) >= len(embeddings):
                break

            # Find entity farthest from selected ones
            max_min_dist = -1
            best_idx = -1

            for i in range(len(embeddings)):
                if i in selected_indices:
                    continue

                # Compute minimum distance to any selected entity
                min_dist = min(
                    1 - np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    for j in selected_indices
                )

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)

        return [valid_uris[i] for i in selected_indices]

    def _aligned_sample(
        self,
        dataset: Dataset,
        synthetic_uris: List[str],
        n: int
    ) -> List[str]:
        """Sample synthetic entities that have aligned pairs."""
        # Find synthetic entities in alignment pairs
        synthetic_set = set(synthetic_uris)
        aligned_synthetic = [
            src_uri for src_uri, tgt_uri in dataset.alignment_pairs
            if src_uri in synthetic_set
        ]

        return random.sample(aligned_synthetic, min(n, len(aligned_synthetic)))

    def _extract_entity_sample(
        self,
        uri: str,
        kg: KnowledgeGraph,
        paired_kg: KnowledgeGraph,
        dataset: Dataset,
        orig_kg: KnowledgeGraph
    ) -> Optional[Dict]:
        """Extract detailed sample for a single entity."""
        entity = kg.entities.get(uri)
        if not entity:
            return None

        sample = {
            "uri": uri,
            "attributes": [],
            "num_attributes": len(entity.attributes),
        }

        # Extract attributes
        for attr in entity.attributes:
            sample["attributes"].append({
                "predicate": str(attr.predicate),
                "value": str(attr.value) if attr.value else "",
            })

        # Find aligned pair if exists
        aligned_uri = None
        for src_uri, tgt_uri in dataset.alignment_pairs:
            if src_uri == uri:
                aligned_uri = tgt_uri
                break

        if aligned_uri:
            aligned_entity = paired_kg.entities.get(aligned_uri)
            if aligned_entity:
                sample["aligned_uri"] = aligned_uri
                sample["aligned_attributes"] = [
                    {
                        "predicate": str(attr.predicate),
                        "value": str(attr.value) if attr.value else "",
                    }
                    for attr in aligned_entity.attributes
                ]

        # Find closest original entity (for comparison)
        sample["closest_original"] = self._find_closest_original(
            entity, orig_kg
        )

        return sample

    def _find_closest_original(
        self,
        synthetic_entity: Entity,
        orig_kg: KnowledgeGraph,
        top_k: int = 1
    ) -> Optional[Dict]:
        """Find most similar original entity."""
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Text from synthetic
        synth_text = " ".join(
            str(attr.value) for attr in synthetic_entity.attributes
            if attr.value
        )
        if not synth_text:
            return None

        synth_emb = self.encoder.encode([synth_text], convert_to_tensor=False, show_progress_bar=False)[0]

        # Sample originals for efficiency
        orig_uris = list(orig_kg.entities.keys())
        if len(orig_uris) > 200:
            orig_uris = random.sample(orig_uris, 200)

        # Find closest
        best_sim = -1
        best_uri = None
        best_entity = None

        for uri in orig_uris:
            entity = orig_kg.entities.get(uri)
            if not entity:
                continue

            orig_text = " ".join(
                str(attr.value) for attr in entity.attributes
                if attr.value
            )
            if not orig_text:
                continue

            orig_emb = self.encoder.encode([orig_text], convert_to_tensor=False, show_progress_bar=False)[0]

            sim = np.dot(synth_emb, orig_emb) / (
                np.linalg.norm(synth_emb) * np.linalg.norm(orig_emb)
            )

            if sim > best_sim:
                best_sim = sim
                best_uri = uri
                best_entity = entity

        if best_entity:
            return {
                "uri": best_uri,
                "similarity": float(best_sim),
                "attributes": [
                    {
                        "predicate": str(attr.predicate),
                        "value": str(attr.value) if attr.value else "",
                    }
                    for attr in best_entity.attributes
                ]
            }

        return None

    def export_to_tsv(
        self,
        samples: List[Dict],
        output_path: Path,
        include_comparison: bool = True
    ) -> None:
        """Export samples to TSV for human annotation.

        Args:
            samples: List of sample dictionaries
            output_path: Path to save TSV
            include_comparison: Include closest original for comparison
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "sample_id",
                "synthetic_uri",
                "synthetic_attributes",
                "aligned_uri",
                "aligned_attributes",
                "realism_score",  # For annotator
                "consistency_score",  # For annotator
                "notes",  # For annotator
            ]

            if include_comparison:
                fieldnames.insert(3, "closest_original_uri")
                fieldnames.insert(4, "closest_original_attributes")
                fieldnames.insert(5, "similarity")

            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for i, sample in enumerate(samples):
                row = {
                    "sample_id": i + 1,
                    "synthetic_uri": sample["uri"],
                    "synthetic_attributes": self._format_attributes(sample["attributes"]),
                    "aligned_uri": sample.get("aligned_uri", ""),
                    "aligned_attributes": self._format_attributes(
                        sample.get("aligned_attributes", [])
                    ),
                    "realism_score": "",
                    "consistency_score": "",
                    "notes": "",
                }

                if include_comparison and sample.get("closest_original"):
                    closest = sample["closest_original"]
                    row["closest_original_uri"] = closest["uri"]
                    row["closest_original_attributes"] = self._format_attributes(
                        closest["attributes"]
                    )
                    row["similarity"] = f"{closest['similarity']:.3f}"

                writer.writerow(row)

    def _format_attributes(self, attributes: List[Dict]) -> str:
        """Format attributes as readable string."""
        if not attributes:
            return ""

        parts = []
        for attr in attributes:
            pred = attr["predicate"].split("/")[-1].split("#")[-1]  # Get local name
            value = attr["value"]
            parts.append(f"{pred}: {value}")

        return " | ".join(parts)


def sample_entities(
    original_path: Path,
    augmented_path: Path,
    output_path: Path,
    n_samples: int = 50,
    strategy: str = "random",
    stage: str = "augmentation",
    format: str = "tsv"
) -> List[Dict]:
    """Sample and export synthetic entities for evaluation.

    Args:
        original_path: Path to original dataset
        augmented_path: Path to augmented dataset
        output_path: Path to save samples
        n_samples: Number of samples
        strategy: Sampling strategy ('random', 'diverse', 'aligned')
        stage: Stage name
        format: Export format ('tsv', 'json')

    Returns:
        List of sample dictionaries
    """
    from src.core.data_io import load_dataset

    # Load datasets
    orig_dataset = load_dataset(original_path)
    aug_dataset = load_dataset(augmented_path)

    # Sample
    sampler = EntitySampler()
    samples = sampler.extract_samples(
        orig_dataset,
        aug_dataset,
        n_samples=n_samples,
        strategy=strategy,
        stage=stage
    )

    # Export
    if format == "tsv":
        sampler.export_to_tsv(samples, output_path, include_comparison=True)
    elif format == "json":
        with output_path.open("w") as f:
            json.dump(samples, f, indent=2)

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample entities for human evaluation")
    parser.add_argument("--original", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--augmented", type=str, required=True, help="Path to augmented dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save samples")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--strategy", type=str, default="random",
                       choices=["random", "diverse", "aligned"],
                       help="Sampling strategy")
    parser.add_argument("--stage", type=str, default="augmentation",
                       choices=["reduction", "augmentation"])
    parser.add_argument("--format", type=str, default="tsv",
                       choices=["tsv", "json"])

    args = parser.parse_args()

    samples = sample_entities(
        Path(args.original),
        Path(args.augmented),
        Path(args.output),
        n_samples=args.n_samples,
        strategy=args.strategy,
        stage=args.stage,
        format=args.format
    )

    print(f"\n✓ Extracted {len(samples)} samples")
    print(f"✓ Saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"1. Open {args.output} in a spreadsheet")
    print(f"2. Annotate realism_score (1-5) and consistency_score (1-5)")
    print(f"3. Add notes about quality issues")
    print(f"4. Save and use for analysis")
