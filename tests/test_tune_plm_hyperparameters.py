"""
Empirical hyperparameter tuning for PLM augmentation.

This script tests different generation parameter configurations and evaluates
the quality of generated attribute values using automatic metrics.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import json
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.dataset import Dataset
from src.core.dataset.reader import DatasetReaderFactory
from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM
from rdflib import Literal, URIRef
import torch

# For semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("⚠️  sentence-transformers not available, using simpler metrics")


from src.logger import get_logger

logger = get_logger(__name__, level="INFO")


class HyperparameterEvaluator:
    """Evaluates generation quality for different hyperparameter configurations."""

    def __init__(self, dataset: Dataset, sample_size: int = 30):
        self.dataset = dataset
        self.sample_size = sample_size

        # Initialize similarity model if available
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.similarity_model = None

    def collect_sample_pairs(self) -> List[Tuple[str, str, str]]:
        """Collect sample attribute value pairs from aligned entities."""
        pairs = []

        # Get aligned entity pairs
        alignment_pairs = list(self.dataset.aligned_entities)[:200]  # More entities for better coverage

        for src_entity, tgt_entity in alignment_pairs:
            # Get all literals from source
            src_literals = []
            for s, p, o in self.dataset.knowledge_graph_source.triples((src_entity, None, None)):
                if isinstance(o, Literal):
                    val = str(o).strip()
                    if 3 <= len(val) <= 200:  # Filter early
                        src_literals.append((val, str(p)))

            # Get all literals from target
            tgt_literals = []
            for s, p, o in self.dataset.knowledge_graph_target.triples((tgt_entity, None, None)):
                if isinstance(o, Literal):
                    val = str(o).strip()
                    if 3 <= len(val) <= 200:  # Filter early
                        tgt_literals.append((val, str(p)))

            # Create pairs (cartesian product limited)
            # This gives us pairs even if predicates don't match
            if src_literals and tgt_literals:
                # Sample up to 3 pairs per entity pair
                for i, (src_val, src_p) in enumerate(src_literals[:3]):
                    if i < len(tgt_literals):
                        tgt_val, tgt_p = tgt_literals[i]
                        # Use a generic predicate name since they don't match
                        predicate = "attribute"
                        pairs.append((src_val, tgt_val, predicate))

                        if len(pairs) >= self.sample_size:
                            return pairs

        return pairs

    def evaluate_configuration(
        self,
        config: Dict[str, Any],
        sample_pairs: List[Tuple[str, str, str]]
    ) -> Dict[str, float]:
        """Evaluate a specific hyperparameter configuration."""

        # Extract generation config
        gen_config = config['augmentation']['bart']['generation']

        # Initialize BART interpolator directly
        device = "cuda" if torch.cuda.is_available() else "cpu"

        bart = BartInterpolatorPLM(
            model_name="facebook/bart-base",
            device=device,
            seed=42,
            base_alpha=0.35,
            alpha_spread=0.15,
            generation_config=gen_config,
        )

        generated_outputs = []
        input_lengths = []
        output_lengths = []

        # Generate outputs for all samples
        for src_val, tgt_val, predicate in sample_pairs:
            try:
                # Use the interpolator directly
                out_src, out_tgt = bart.interpolate_pair(
                    src_val, tgt_val, predicate=predicate
                )

                if out_src and out_tgt and len(out_src) > 0 and len(out_tgt) > 0:
                    generated_outputs.append({
                        'input_src': src_val,
                        'input_tgt': tgt_val,
                        'output_src': out_src,
                        'output_tgt': out_tgt,
                        'predicate': predicate
                    })

                    input_lengths.append((len(src_val), len(tgt_val)))
                    output_lengths.append((len(out_src), len(out_tgt)))

            except Exception as e:
                logger.debug(f"Generation failed: {e}")
                continue

        if not generated_outputs:
            return {
                'semantic_similarity': 0.0,
                'diversity': 0.0,
                'coherence': 0.0,
                'length_ratio': 0.0,
                'success_rate': 0.0,
                'total_score': 0.0
            }

        # Compute metrics
        metrics = {}

        # 1. Semantic Similarity (should be moderate: 0.6-0.85)
        if self.similarity_model:
            similarities = []
            for gen in generated_outputs:
                # Compare output to input
                sim_src = util.cos_sim(
                    self.similarity_model.encode(gen['output_src']),
                    self.similarity_model.encode(gen['input_src'])
                ).item()
                sim_tgt = util.cos_sim(
                    self.similarity_model.encode(gen['output_tgt']),
                    self.similarity_model.encode(gen['input_tgt'])
                ).item()

                similarities.append((sim_src + sim_tgt) / 2)

            avg_sim = np.mean(similarities)
            # Prefer 0.65-0.85 range
            if 0.65 <= avg_sim <= 0.85:
                semantic_score = 1.0
            elif avg_sim < 0.65:
                semantic_score = avg_sim / 0.65
            else:  # > 0.85
                semantic_score = max(0.0, 1.0 - (avg_sim - 0.85) / 0.15)

            metrics['semantic_similarity'] = avg_sim
            metrics['semantic_score'] = semantic_score
        else:
            # Fallback: simple string similarity
            similarities = []
            for gen in generated_outputs:
                sim_src = self._simple_similarity(gen['output_src'], gen['input_src'])
                sim_tgt = self._simple_similarity(gen['output_tgt'], gen['input_tgt'])
                similarities.append((sim_src + sim_tgt) / 2)

            avg_sim = np.mean(similarities)
            metrics['semantic_similarity'] = avg_sim
            metrics['semantic_score'] = min(1.0, avg_sim / 0.7)

        # 2. Diversity (outputs should vary)
        if len(generated_outputs) > 1:
            all_outputs = [g['output_src'] for g in generated_outputs] + \
                         [g['output_tgt'] for g in generated_outputs]
            unique_ratio = len(set(all_outputs)) / len(all_outputs)
            metrics['diversity'] = unique_ratio
            metrics['diversity_score'] = unique_ratio
        else:
            metrics['diversity'] = 0.5
            metrics['diversity_score'] = 0.5

        # 3. Coherence (no excessive repetition)
        coherence_scores = []
        for gen in generated_outputs:
            words_src = gen['output_src'].split()
            words_tgt = gen['output_tgt'].split()

            rep_src = self._repetition_score(words_src)
            rep_tgt = self._repetition_score(words_tgt)

            coherence_scores.append((rep_src + rep_tgt) / 2)

        metrics['coherence'] = np.mean(coherence_scores) if coherence_scores else 0.0
        metrics['coherence_score'] = metrics['coherence']

        # 4. Length appropriateness
        if input_lengths and output_lengths:
            input_avg = np.mean([sum(pair) / 2 for pair in input_lengths])
            output_avg = np.mean([sum(pair) / 2 for pair in output_lengths])

            ratio = min(input_avg, output_avg) / max(input_avg, output_avg) if max(input_avg, output_avg) > 0 else 0
            metrics['length_ratio'] = ratio
            metrics['length_score'] = ratio
        else:
            metrics['length_ratio'] = 0.0
            metrics['length_score'] = 0.0

        # 5. Success rate
        metrics['success_rate'] = len(generated_outputs) / len(sample_pairs)

        # Total weighted score
        metrics['total_score'] = (
            0.35 * metrics.get('semantic_score', 0.0) +
            0.25 * metrics.get('diversity_score', 0.0) +
            0.25 * metrics.get('coherence_score', 0.0) +
            0.10 * metrics.get('length_score', 0.0) +
            0.05 * metrics['success_rate']
        )

        # Clean up
        del bart

        return metrics

    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Simple character-level similarity."""
        s1_chars = set(s1.lower())
        s2_chars = set(s2.lower())

        if not s1_chars or not s2_chars:
            return 0.0

        intersection = s1_chars & s2_chars
        union = s1_chars | s2_chars

        return len(intersection) / len(union)

    def _repetition_score(self, words: List[str]) -> float:
        """Score based on word repetition (1.0 = no repetition)."""
        if len(words) <= 1:
            return 1.0

        unique = len(set(words))
        total = len(words)

        return unique / total


def generate_configurations() -> List[Dict[str, Any]]:
    """Generate hyperparameter configurations to test."""
    configurations = []

    # Define search space (focused grid)
    temperatures = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    num_beams_options = [2, 3, 5]
    top_p_options = [0.90, 0.95]
    repetition_penalties = [1.5, 1.7]

    # Test combinations
    for temp in temperatures:
        for beams in num_beams_options:
            for top_p in top_p_options:
                for rep_pen in repetition_penalties:
                    config = {
                        "augmentation": {
                            "bart": {
                                "generation": {
                                    "max_new_tokens": 32,
                                    "do_sample": True,
                                    "top_k": 0,
                                    "top_p": top_p,
                                    "temperature": temp,
                                    "num_beams": beams,
                                    "repetition_penalty": rep_pen,
                                    "length_penalty": 1.0,
                                    "no_repeat_ngram_size": 3,
                                }
                            }
                        }
                    }
                    configurations.append(config)

    logger.info(f"Generated {len(configurations)} configurations to test")
    return configurations


def main():
    """Run hyperparameter tuning."""
    logger.info("=" * 80)
    logger.info("PLM Hyperparameter Tuning - Empirical Evaluation")
    logger.info("=" * 80)

    # Load dataset
    logger.info("\n[1/4] Loading dataset...")
    dataset_name = "D_W_15K_V2"
    dataset_path = f"data/raw/openea/{dataset_name}"
    reader = DatasetReaderFactory.create_reader("openea")
    dataset = reader.read(dataset_path)
    logger.info(f"✓ Loaded {dataset_name}")

    # Initialize evaluator
    logger.info("\n[2/4] Collecting sample pairs...")
    evaluator = HyperparameterEvaluator(dataset, sample_size=30)
    sample_pairs = evaluator.collect_sample_pairs()
    logger.info(f"✓ Collected {len(sample_pairs)} sample pairs")

    # Show examples
    logger.info("\nExample pairs:")
    for i, (src, tgt, pred) in enumerate(sample_pairs[:3]):
        pred_short = pred.split('/')[-1] if '/' in pred else pred
        logger.info(f"  {i+1}. '{src[:50]}...' <-> '{tgt[:50]}...' ({pred_short})")

    # Generate configurations
    logger.info("\n[3/4] Generating configurations...")
    configurations = generate_configurations()

    # Test configurations
    logger.info(f"\n[4/4] Testing {len(configurations)} configurations...")
    logger.info("(This will take a while...)\n")

    results = []

    for i, config in enumerate(configurations, 1):
        gen_cfg = config['augmentation']['bart']['generation']

        logger.info(f"[{i}/{len(configurations)}] Testing: temp={gen_cfg['temperature']}, "
                   f"beams={gen_cfg['num_beams']}, top_p={gen_cfg['top_p']}, "
                   f"rep={gen_cfg['repetition_penalty']}")

        try:
            metrics = evaluator.evaluate_configuration(config, sample_pairs)

            result = {
                'config': gen_cfg,
                'metrics': metrics
            }
            results.append(result)

            logger.info(f"  → Score: {metrics['total_score']:.3f} "
                       f"(sim={metrics['semantic_similarity']:.2f}, "
                       f"div={metrics['diversity']:.2f}, "
                       f"coh={metrics['coherence']:.2f}, "
                       f"succ={metrics['success_rate']:.2f})")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Sort by total score
    results.sort(key=lambda x: x['metrics']['total_score'], reverse=True)

    # Display top 10
    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("=" * 80)

    for i, result in enumerate(results[:10], 1):
        cfg = result['config']
        metrics = result['metrics']

        logger.info(f"\n#{i} - Total Score: {metrics['total_score']:.4f}")
        logger.info(f"  Configuration:")
        logger.info(f"    temperature: {cfg['temperature']}")
        logger.info(f"    num_beams: {cfg['num_beams']}")
        logger.info(f"    top_p: {cfg['top_p']}")
        logger.info(f"    repetition_penalty: {cfg['repetition_penalty']}")
        logger.info(f"  Metrics:")
        logger.info(f"    Semantic similarity: {metrics['semantic_similarity']:.4f}")
        logger.info(f"    Diversity: {metrics['diversity']:.4f}")
        logger.info(f"    Coherence: {metrics['coherence']:.4f}")
        logger.info(f"    Success rate: {metrics['success_rate']:.4f}")

    # Save results
    output_file = Path(__file__).parent / "plm_tuning_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Full results saved to: {output_file}")

    # Print recommended configuration
    if results:
        best = results[0]
        best_cfg = best['config']

        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDED CONFIGURATION FOR plm.yaml")
        logger.info("=" * 80)
        logger.info(f"""
generation:
  max_new_tokens: 32
  do_sample: true
  top_k: 0
  top_p: {best_cfg['top_p']}
  temperature: {best_cfg['temperature']}
  num_beams: {best_cfg['num_beams']}
  repetition_penalty: {best_cfg['repetition_penalty']}
  length_penalty: 1.0
  no_repeat_ngram_size: 3
""")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
