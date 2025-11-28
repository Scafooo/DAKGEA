#!/usr/bin/env python3
"""Entity sampler for human evaluation of augmented entities.

Extracts representative samples of synthetic entities for manual quality assessment.
Supports different sampling strategies: random, diverse, aligned.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rdflib import Literal, URIRef
from sentence_transformers import SentenceTransformer

from src.core import Dataset, KnowledgeGraph


class EntitySampler:
    """Extract representative samples of synthetic entities."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize entity sampler.

        Args:
            embedding_model: Model for computing diversity
        """
        self.embedding_model_name = embedding_model
        self.encoder = None

    def extract_samples(
        self,
        original_dataset: Dataset,
        augmented_dataset: Dataset,
        n_samples: int = 50,
        strategy: str = "random",
        stage: str = "augmentation"
    ) -> List[Dict]:
        """Extract samples of synthetic entities.

        Args:
            original_dataset: Original dataset
            augmented_dataset: Augmented dataset
            n_samples: Number of samples to extract
            strategy: Sampling strategy ('random', 'diverse', or 'aligned')
            stage: Stage name

        Returns:
            List of sample dictionaries
        """
        # Get appropriate KGs
        if stage == "augmentation":
            orig_kg = original_dataset.knowledge_graph_source
            aug_kg = augmented_dataset.knowledge_graph_source
        else:
            orig_kg = original_dataset.knowledge_graph_target
            aug_kg = augmented_dataset.knowledge_graph_target

        # Identify synthetic entities
        orig_uris = self._get_entity_uris(orig_kg)
        aug_uris = self._get_entity_uris(aug_kg)
        synthetic_uris = list(aug_uris - orig_uris)

        if not synthetic_uris:
            return []

        # Sample based on strategy
        if strategy == "random":
            sampled_uris = self._random_sample(synthetic_uris, n_samples)
        elif strategy == "diverse":
            sampled_uris = self._diverse_sample(aug_kg, synthetic_uris, n_samples)
        elif strategy == "aligned":
            sampled_uris = self._aligned_sample(augmented_dataset, synthetic_uris, n_samples, stage)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Build sample records
        samples = []
        for uri in sampled_uris:
            sample = self._build_sample_record(
                uri, aug_kg, orig_kg, augmented_dataset, stage
            )
            samples.append(sample)

        return samples

    def _get_entity_uris(self, kg: KnowledgeGraph) -> Set[str]:
        """Extract all entity URIs from knowledge graph."""
        entities = set()
        for s, p, o in kg.triples((None, None, None)):
            if isinstance(s, URIRef):
                entities.add(str(s))
        return entities

    def _random_sample(self, uris: List[str], n: int) -> List[str]:
        """Random sampling."""
        n = min(n, len(uris))
        return np.random.choice(uris, n, replace=False).tolist()

    def _diverse_sample(self, kg: KnowledgeGraph, uris: List[str], n: int) -> List[str]:
        """Sample diverse entities using embeddings."""
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        # Get text representations
        entity_texts = {}
        for uri in uris:
            texts = []
            for s, p, o in kg.triples((URIRef(uri), None, None)):
                if isinstance(o, Literal):
                    texts.append(str(o))
            entity_texts[uri] = " ".join(texts[:10]) if texts else uri

        # Encode
        uris_list = list(entity_texts.keys())
        texts_list = [entity_texts[uri] for uri in uris_list]
        embeddings = self.encoder.encode(texts_list, convert_to_tensor=False, show_progress_bar=False)

        # Greedy diversity sampling
        selected = []
        selected_embs = []

        # Start with random
        first_idx = np.random.randint(len(uris_list))
        selected.append(uris_list[first_idx])
        selected_embs.append(embeddings[first_idx])

        # Greedily select most diverse
        while len(selected) < min(n, len(uris_list)):
            max_min_dist = -1
            best_idx = -1

            for i, emb in enumerate(embeddings):
                if uris_list[i] in selected:
                    continue

                # Compute minimum distance to selected
                min_dist = min([
                    1 - np.dot(emb, sel_emb) / (np.linalg.norm(emb) * np.linalg.norm(sel_emb))
                    for sel_emb in selected_embs
                ])

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected.append(uris_list[best_idx])
                selected_embs.append(embeddings[best_idx])

        return selected

    def _aligned_sample(
        self,
        dataset: Dataset,
        uris: List[str],
        n: int,
        stage: str
    ) -> List[str]:
        """Sample synthetic entities that are part of aligned pairs."""
        # For augmentation stage, source entities are the ones we augmented
        if stage == "augmentation":
            kg = dataset.knowledge_graph_source
        else:
            kg = dataset.knowledge_graph_target

        # Filter to only entities that have synthetic counterparts
        aligned_synthetic = []
        for uri in uris:
            # Check if this entity appears in any alignment
            for src_uri, tgt_uri in dataset.aligned_entities:
                if stage == "augmentation" and str(src_uri) == uri:
                    aligned_synthetic.append(uri)
                    break
                elif stage == "reduction" and str(tgt_uri) == uri:
                    aligned_synthetic.append(uri)
                    break

        if not aligned_synthetic:
            # Fall back to random if no aligned synthetics
            return self._random_sample(uris, n)

        return self._random_sample(aligned_synthetic, n)

    def _build_sample_record(
        self,
        synthetic_uri: str,
        aug_kg: KnowledgeGraph,
        orig_kg: KnowledgeGraph,
        dataset: Dataset,
        stage: str
    ) -> Dict:
        """Build a sample record for human evaluation."""
        # Get attributes of synthetic entity
        synth_attrs = {}
        for s, p, o in aug_kg.triples((URIRef(synthetic_uri), None, None)):
            if isinstance(o, Literal):
                pred_name = str(p).split('/')[-1].split('#')[-1]
                synth_attrs[pred_name] = str(o)

        # Find closest original entity
        closest_orig_uri = self._find_closest_original(
            synthetic_uri, aug_kg, orig_kg, self._get_entity_uris(orig_kg)
        )

        # Get attributes of closest original
        orig_attrs = {}
        if closest_orig_uri:
            for s, p, o in orig_kg.triples((URIRef(closest_orig_uri), None, None)):
                if isinstance(o, Literal):
                    pred_name = str(p).split('/')[-1].split('#')[-1]
                    orig_attrs[pred_name] = str(o)

        # Get aligned counterpart if exists
        aligned_uri = None
        if stage == "augmentation":
            for src, tgt in dataset.aligned_entities:
                if str(src) == synthetic_uri:
                    aligned_uri = str(tgt)
                    break
        else:
            for src, tgt in dataset.aligned_entities:
                if str(tgt) == synthetic_uri:
                    aligned_uri = str(src)
                    break

        return {
            "synthetic_uri": synthetic_uri,
            "synthetic_attributes": synth_attrs,
            "closest_original_uri": closest_orig_uri or "N/A",
            "closest_original_attributes": orig_attrs,
            "aligned_counterpart_uri": aligned_uri or "N/A",
            "realism_score": "",  # To be filled by human annotator
            "consistency_score": "",  # To be filled by human annotator
            "notes": "",
        }

    def _find_closest_original(
        self,
        synth_uri: str,
        aug_kg: KnowledgeGraph,
        orig_kg: KnowledgeGraph,
        orig_uris: Set[str]
    ) -> str:
        """Find most similar original entity."""
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        # Get text of synthetic
        synth_texts = []
        for s, p, o in aug_kg.triples((URIRef(synth_uri), None, None)):
            if isinstance(o, Literal):
                synth_texts.append(str(o))

        if not synth_texts:
            return None

        synth_text = " ".join(synth_texts[:10])
        synth_emb = self.encoder.encode([synth_text], convert_to_tensor=False, show_progress_bar=False)[0]

        # Compare with originals
        best_sim = -1
        best_uri = None

        # Sample originals for performance
        sampled_orig = list(orig_uris)
        if len(sampled_orig) > 200:
            sampled_orig = np.random.choice(sampled_orig, 200, replace=False).tolist()

        for orig_uri in sampled_orig:
            orig_texts = []
            for s, p, o in orig_kg.triples((URIRef(orig_uri), None, None)):
                if isinstance(o, Literal):
                    orig_texts.append(str(o))

            if not orig_texts:
                continue

            orig_text = " ".join(orig_texts[:10])
            orig_emb = self.encoder.encode([orig_text], convert_to_tensor=False, show_progress_bar=False)[0]

            sim = np.dot(synth_emb, orig_emb) / (np.linalg.norm(synth_emb) * np.linalg.norm(orig_emb))
            if sim > best_sim:
                best_sim = sim
                best_uri = orig_uri

        return best_uri

    def export_to_tsv(self, samples: List[Dict], output_path: Path) -> None:
        """Export samples to TSV for human annotation."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline='', encoding='utf-8') as f:
            # Flatten attributes for TSV export
            if not samples:
                return

            # Write header
            writer = csv.writer(f, delimiter='\t')
            header = [
                "synthetic_uri",
                "synthetic_attributes",
                "closest_original_uri",
                "closest_original_attributes",
                "aligned_counterpart_uri",
                "realism_score_1_5",
                "consistency_score_1_5",
                "notes"
            ]
            writer.writerow(header)

            # Write samples
            for sample in samples:
                synth_attrs_str = "; ".join([f"{k}: {v}" for k, v in sample["synthetic_attributes"].items()])
                orig_attrs_str = "; ".join([f"{k}: {v}" for k, v in sample["closest_original_attributes"].items()])

                row = [
                    sample["synthetic_uri"],
                    synth_attrs_str,
                    sample["closest_original_uri"],
                    orig_attrs_str,
                    sample["aligned_counterpart_uri"],
                    sample.get("realism_score", ""),
                    sample.get("consistency_score", ""),
                    sample.get("notes", "")
                ]
                writer.writerow(row)


def sample_entities(
    original_path: Path,
    augmented_path: Path,
    output_path: Path,
    n_samples: int = 50,
    strategy: str = "random",
    stage: str = "augmentation"
) -> List[Dict]:
    """Extract and export entity samples.

    Args:
        original_path: Path to original dataset
        augmented_path: Path to augmented dataset
        output_path: Path to save TSV
        n_samples: Number of samples
        strategy: Sampling strategy
        stage: Stage name

    Returns:
        List of samples
    """
    from src.core import DatasetReaderFactory

    # Load datasets
    reader = DatasetReaderFactory.create_reader("openea")
    orig_dataset = reader.read(original_path)
    aug_dataset = reader.read(augmented_path)

    # Sample
    sampler = EntitySampler()
    samples = sampler.extract_samples(
        orig_dataset, aug_dataset, n_samples=n_samples, strategy=strategy, stage=stage
    )

    # Export
    sampler.export_to_tsv(samples, output_path)

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--augmented", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "diverse", "aligned"])
    parser.add_argument("--stage", type=str, default="augmentation")

    args = parser.parse_args()

    samples = sample_entities(
        Path(args.original),
        Path(args.augmented),
        Path(args.output),
        n_samples=args.n_samples,
        strategy=args.strategy,
        stage=args.stage
    )

    print(f"Extracted {len(samples)} samples to {args.output}")
