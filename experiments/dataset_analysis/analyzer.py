"""Dataset analyzer for HybEA/BERT-INT attribute_data format."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.logger import get_logger

logger = get_logger(__name__)


class DatasetAnalyzer:
    """Analyze HybEA attribute_data format datasets to verify structural invariants."""

    def __init__(self, dataset_path: Path):
        """
        Initialize analyzer with dataset path.

        Args:
            dataset_path: Path to attribute_data directory
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # File paths
        self.ent_ids_1 = self.dataset_path / "ent_ids_1"
        self.ent_ids_2 = self.dataset_path / "ent_ids_2"
        self.triples_1 = self.dataset_path / "triples_1"
        self.triples_2 = self.dataset_path / "triples_2"
        self.attr_triples1 = self.dataset_path / "attr_triples1"
        self.attr_triples2 = self.dataset_path / "attr_triples2"
        self.ref_pairs = self.dataset_path / "ref_pairs"
        self.sup_pairs = self.dataset_path / "sup_pairs"
        self.valid_pairs = self.dataset_path / "valid_pairs"
        self.rel_ids_1 = self.dataset_path / "rel_ids_1"
        self.rel_ids_2 = self.dataset_path / "rel_ids_2"

        # Data structures (loaded lazily)
        self._index_to_uri_kg1 = None
        self._index_to_uri_kg2 = None
        self._uri_to_index_kg1 = None
        self._uri_to_index_kg2 = None
        self._alignments = None
        self._entities_in_triples = None
        self._entities_in_attr = None

    def load_entity_mappings(self):
        """Load entity ID mappings for both KGs."""
        logger.info("Loading entity ID mappings...")

        self._index_to_uri_kg1 = {}
        self._uri_to_index_kg1 = {}
        with open(self.ent_ids_1) as f:
            for line in f:
                idx, uri = line.strip().split('\t')
                idx = int(idx)
                self._index_to_uri_kg1[idx] = uri
                self._uri_to_index_kg1[uri] = idx

        self._index_to_uri_kg2 = {}
        self._uri_to_index_kg2 = {}
        with open(self.ent_ids_2) as f:
            for line in f:
                idx, uri = line.strip().split('\t')
                idx = int(idx)
                self._index_to_uri_kg2[idx] = uri
                self._uri_to_index_kg2[uri] = idx

        logger.info(f"Loaded {len(self._index_to_uri_kg1)} KG1 entities, {len(self._index_to_uri_kg2)} KG2 entities")

    def load_alignments(self) -> Dict[str, Set[Tuple[int, int]]]:
        """Load all alignment pairs (ref, sup, valid)."""
        logger.info("Loading alignment pairs...")

        alignments = {
            "ref": set(),
            "sup": set(),
            "valid": set()
        }

        with open(self.ref_pairs) as f:
            for line in f:
                e1, e2 = map(int, line.strip().split('\t'))
                alignments["ref"].add((e1, e2))

        with open(self.sup_pairs) as f:
            for line in f:
                e1, e2 = map(int, line.strip().split('\t'))
                alignments["sup"].add((e1, e2))

        with open(self.valid_pairs) as f:
            for line in f:
                e1, e2 = map(int, line.strip().split('\t'))
                alignments["valid"].add((e1, e2))

        self._alignments = alignments
        logger.info(f"Loaded ref={len(alignments['ref'])}, sup={len(alignments['sup'])}, valid={len(alignments['valid'])}")
        return alignments

    def load_triples_entities(self) -> Dict[str, Set[int]]:
        """Load entities appearing in relation triples."""
        logger.info("Loading entities from triples...")

        entities_kg1 = set()
        with open(self.triples_1) as f:
            for line in f:
                s, p, o = map(int, line.strip().split('\t'))
                entities_kg1.add(s)
                entities_kg1.add(o)

        entities_kg2 = set()
        with open(self.triples_2) as f:
            for line in f:
                s, p, o = map(int, line.strip().split('\t'))
                entities_kg2.add(s)
                entities_kg2.add(o)

        self._entities_in_triples = {
            "kg1": entities_kg1,
            "kg2": entities_kg2
        }

        logger.info(f"Entities in triples: KG1={len(entities_kg1)}, KG2={len(entities_kg2)}")
        return self._entities_in_triples

    def load_attr_entities(self) -> Dict[str, Set[str]]:
        """Load entities appearing as subjects in attr_triples."""
        logger.info("Loading entities from attr_triples...")

        entities_kg1 = set()
        with open(self.attr_triples1) as f:
            for line in f:
                subject_uri = line.strip().split('\t', 1)[0]
                entities_kg1.add(subject_uri)

        entities_kg2 = set()
        with open(self.attr_triples2) as f:
            for line in f:
                subject_uri = line.strip().split('\t', 1)[0]
                entities_kg2.add(subject_uri)

        self._entities_in_attr = {
            "kg1": entities_kg1,
            "kg2": entities_kg2
        }

        logger.info(f"Entities in attr_triples: KG1={len(entities_kg1)}, KG2={len(entities_kg2)}")
        return self._entities_in_attr

    def analyze_alignment_structure(self) -> Dict:
        """Analyze alignment pairs structure."""
        logger.info("\n" + "="*70)
        logger.info("ANALYZING ALIGNMENT STRUCTURE")
        logger.info("="*70)

        if self._alignments is None:
            self.load_alignments()

        ref = self._alignments["ref"]
        sup = self._alignments["sup"]
        valid = self._alignments["valid"]

        all_pairs = ref | sup | valid
        overlap_all = ref & sup & valid
        overlap_ref_sup = ref & sup
        overlap_ref_valid = ref & valid
        overlap_sup_valid = sup & valid

        entities_kg1 = {e1 for e1, e2 in all_pairs}
        entities_kg2 = {e2 for e1, e2 in all_pairs}

        result = {
            "total_unique_pairs": len(all_pairs),
            "ref_pairs": len(ref),
            "sup_pairs": len(sup),
            "valid_pairs": len(valid),
            "overlap_all_three": len(overlap_all),
            "overlap_ref_sup": len(overlap_ref_sup),
            "overlap_ref_valid": len(overlap_ref_valid),
            "overlap_sup_valid": len(overlap_sup_valid),
            "unique_entities_kg1": len(entities_kg1),
            "unique_entities_kg2": len(entities_kg2),
            "kg1_index_range": (min(entities_kg1), max(entities_kg1)),
            "kg2_index_range": (min(entities_kg2), max(entities_kg2)),
        }

        logger.info(f"Total unique alignment pairs: {result['total_unique_pairs']}")
        logger.info(f"  ref_pairs:   {result['ref_pairs']} ({result['ref_pairs']/result['total_unique_pairs']*100:.1f}%)")
        logger.info(f"  sup_pairs:   {result['sup_pairs']} ({result['sup_pairs']/result['total_unique_pairs']*100:.1f}%)")
        logger.info(f"  valid_pairs: {result['valid_pairs']} ({result['valid_pairs']/result['total_unique_pairs']*100:.1f}%)")
        logger.info(f"\nOverlap between sets:")
        logger.info(f"  All three: {result['overlap_all_three']}")
        logger.info(f"  ref ∩ sup: {result['overlap_ref_sup']}")
        logger.info(f"  ref ∩ valid: {result['overlap_ref_valid']}")
        logger.info(f"  sup ∩ valid: {result['overlap_sup_valid']}")
        logger.info(f"\nUnique entities:")
        logger.info(f"  KG1: {result['unique_entities_kg1']} (range: {result['kg1_index_range'][0]}-{result['kg1_index_range'][1]})")
        logger.info(f"  KG2: {result['unique_entities_kg2']} (range: {result['kg2_index_range'][0]}-{result['kg2_index_range'][1]})")

        return result

    def verify_alignment_to_triples(self) -> Dict:
        """Verify that all aligned entities appear in triples."""
        logger.info("\n" + "="*70)
        logger.info("VERIFYING ALIGNMENT → TRIPLES COVERAGE")
        logger.info("="*70)

        if self._alignments is None:
            self.load_alignments()
        if self._entities_in_triples is None:
            self.load_triples_entities()

        all_pairs = self._alignments["ref"] | self._alignments["sup"] | self._alignments["valid"]
        aligned_kg1 = {e1 for e1, e2 in all_pairs}
        aligned_kg2 = {e2 for e1, e2 in all_pairs}

        triples_kg1 = self._entities_in_triples["kg1"]
        triples_kg2 = self._entities_in_triples["kg2"]

        missing_in_triples1 = aligned_kg1 - triples_kg1
        missing_in_triples2 = aligned_kg2 - triples_kg2

        not_aligned_in_triples1 = triples_kg1 - aligned_kg1
        not_aligned_in_triples2 = triples_kg2 - aligned_kg2

        result = {
            "aligned_kg1": len(aligned_kg1),
            "aligned_kg2": len(aligned_kg2),
            "in_triples_kg1": len(triples_kg1),
            "in_triples_kg2": len(triples_kg2),
            "missing_in_triples_kg1": len(missing_in_triples1),
            "missing_in_triples_kg2": len(missing_in_triples2),
            "not_aligned_in_triples_kg1": len(not_aligned_in_triples1),
            "not_aligned_in_triples_kg2": len(not_aligned_in_triples2),
        }

        logger.info(f"KG1:")
        logger.info(f"  Aligned entities: {result['aligned_kg1']}")
        logger.info(f"  In triples: {result['in_triples_kg1']}")
        logger.info(f"  Aligned but NOT in triples: {result['missing_in_triples_kg1']}")
        logger.info(f"  In triples but NOT aligned: {result['not_aligned_in_triples_kg1']}")

        logger.info(f"\nKG2:")
        logger.info(f"  Aligned entities: {result['aligned_kg2']}")
        logger.info(f"  In triples: {result['in_triples_kg2']}")
        logger.info(f"  Aligned but NOT in triples: {result['missing_in_triples_kg2']}")
        logger.info(f"  In triples but NOT aligned: {result['not_aligned_in_triples_kg2']}")

        # Check invariant
        if result['missing_in_triples_kg1'] == 0 and result['missing_in_triples_kg2'] == 0:
            logger.info("\n✅ INVARIANT VERIFIED: All aligned entities appear in triples")
        else:
            logger.warning("\n❌ INVARIANT VIOLATED: Some aligned entities missing from triples!")

        if result['not_aligned_in_triples_kg1'] == 0 and result['not_aligned_in_triples_kg2'] == 0:
            logger.info("✅ INVARIANT VERIFIED: All entities in triples are aligned")
        else:
            logger.warning("❌ INVARIANT VIOLATED: Some entities in triples are not aligned!")

        return result

    def analyze_attribute_coverage(self) -> Dict:
        """Analyze which entities have attributes."""
        logger.info("\n" + "="*70)
        logger.info("ANALYZING ATTRIBUTE COVERAGE")
        logger.info("="*70)

        if self._index_to_uri_kg1 is None:
            self.load_entity_mappings()
        if self._entities_in_triples is None:
            self.load_triples_entities()
        if self._entities_in_attr is None:
            self.load_attr_entities()

        # Convert triples entities (indices) to URIs
        triples_uris_kg1 = {self._index_to_uri_kg1[idx] for idx in self._entities_in_triples["kg1"]}
        triples_uris_kg2 = {self._index_to_uri_kg2[idx] for idx in self._entities_in_triples["kg2"]}

        attr_uris_kg1 = self._entities_in_attr["kg1"]
        attr_uris_kg2 = self._entities_in_attr["kg2"]

        # Entities in different categories
        only_triples_kg1 = triples_uris_kg1 - attr_uris_kg1
        only_triples_kg2 = triples_uris_kg2 - attr_uris_kg2
        only_attr_kg1 = attr_uris_kg1 - triples_uris_kg1
        only_attr_kg2 = attr_uris_kg2 - triples_uris_kg2
        in_both_kg1 = triples_uris_kg1 & attr_uris_kg1
        in_both_kg2 = triples_uris_kg2 & attr_uris_kg2

        result = {
            "total_entities_kg1": len(self._index_to_uri_kg1),
            "total_entities_kg2": len(self._index_to_uri_kg2),
            "only_triples_kg1": len(only_triples_kg1),
            "only_triples_kg2": len(only_triples_kg2),
            "only_attr_kg1": len(only_attr_kg1),
            "only_attr_kg2": len(only_attr_kg2),
            "in_both_kg1": len(in_both_kg1),
            "in_both_kg2": len(in_both_kg2),
        }

        logger.info(f"KG1 (total entities: {result['total_entities_kg1']}):")
        logger.info(f"  In triples only: {result['only_triples_kg1']} ({result['only_triples_kg1']/result['total_entities_kg1']*100:.1f}%)")
        logger.info(f"  In attr_triples only: {result['only_attr_kg1']} ({result['only_attr_kg1']/result['total_entities_kg1']*100:.1f}%)")
        logger.info(f"  In BOTH: {result['in_both_kg1']} ({result['in_both_kg1']/result['total_entities_kg1']*100:.1f}%)")

        logger.info(f"\nKG2 (total entities: {result['total_entities_kg2']}):")
        logger.info(f"  In triples only: {result['only_triples_kg2']} ({result['only_triples_kg2']/result['total_entities_kg2']*100:.1f}%)")
        logger.info(f"  In attr_triples only: {result['only_attr_kg2']} ({result['only_attr_kg2']/result['total_entities_kg2']*100:.1f}%)")
        logger.info(f"  In BOTH: {result['in_both_kg2']} ({result['in_both_kg2']/result['total_entities_kg2']*100:.1f}%)")

        # Verify that all entities have at least triples or attributes
        all_with_data_kg1 = triples_uris_kg1 | attr_uris_kg1
        all_with_data_kg2 = triples_uris_kg2 | attr_uris_kg2

        all_uris_kg1 = set(self._index_to_uri_kg1.values())
        all_uris_kg2 = set(self._index_to_uri_kg2.values())

        missing_kg1 = all_uris_kg1 - all_with_data_kg1
        missing_kg2 = all_uris_kg2 - all_with_data_kg2

        logger.info(f"\nEntities with NO data (neither triples nor attributes):")
        logger.info(f"  KG1: {len(missing_kg1)}")
        logger.info(f"  KG2: {len(missing_kg2)}")

        if len(missing_kg1) == 0 and len(missing_kg2) == 0:
            logger.info("✅ INVARIANT VERIFIED: All entities have at least triples or attributes")
        else:
            logger.warning("❌ INVARIANT VIOLATED: Some entities have no data!")

        return result

    def analyze_data_density(self) -> Dict:
        """Analyze data density by alignment category."""
        logger.info("\n" + "="*70)
        logger.info("ANALYZING DATA DENSITY BY ALIGNMENT CATEGORY")
        logger.info("="*70)

        if self._index_to_uri_kg1 is None:
            self.load_entity_mappings()
        if self._alignments is None:
            self.load_alignments()

        # Count attributes per entity
        attr_count_kg1 = defaultdict(int)
        with open(self.attr_triples1) as f:
            for line in f:
                subject_uri = line.strip().split('\t', 1)[0]
                attr_count_kg1[subject_uri] += 1

        attr_count_kg2 = defaultdict(int)
        with open(self.attr_triples2) as f:
            for line in f:
                subject_uri = line.strip().split('\t', 1)[0]
                attr_count_kg2[subject_uri] += 1

        # Count relations per entity (as subject)
        rel_count_kg1 = defaultdict(int)
        with open(self.triples_1) as f:
            for line in f:
                s, p, o = map(int, line.strip().split('\t'))
                rel_count_kg1[s] += 1

        rel_count_kg2 = defaultdict(int)
        with open(self.triples_2) as f:
            for line in f:
                s, p, o = map(int, line.strip().split('\t'))
                rel_count_kg2[s] += 1

        def compute_stats(pairs, name):
            attrs1 = [attr_count_kg1.get(self._index_to_uri_kg1[e1], 0) for e1, e2 in pairs]
            attrs2 = [attr_count_kg2.get(self._index_to_uri_kg2[e2], 0) for e1, e2 in pairs]
            rels1 = [rel_count_kg1.get(e1, 0) for e1, e2 in pairs]
            rels2 = [rel_count_kg2.get(e2, 0) for e1, e2 in pairs]

            stats = {
                "name": name,
                "count": len(pairs),
                "kg1_avg_attrs": sum(attrs1) / len(attrs1) if attrs1 else 0,
                "kg2_avg_attrs": sum(attrs2) / len(attrs2) if attrs2 else 0,
                "kg1_avg_rels": sum(rels1) / len(rels1) if rels1 else 0,
                "kg2_avg_rels": sum(rels2) / len(rels2) if rels2 else 0,
                "kg1_zero_attrs": sum(1 for x in attrs1 if x == 0),
                "kg2_zero_attrs": sum(1 for x in attrs2 if x == 0),
            }

            logger.info(f"\n{name} ({stats['count']} pairs):")
            logger.info(f"  KG1 avg attributes: {stats['kg1_avg_attrs']:.2f}")
            logger.info(f"  KG2 avg attributes: {stats['kg2_avg_attrs']:.2f}")
            logger.info(f"  KG1 avg relations: {stats['kg1_avg_rels']:.2f}")
            logger.info(f"  KG2 avg relations: {stats['kg2_avg_rels']:.2f}")
            logger.info(f"  KG1 entities with 0 attrs: {stats['kg1_zero_attrs']} ({stats['kg1_zero_attrs']/stats['count']*100:.1f}%)")
            logger.info(f"  KG2 entities with 0 attrs: {stats['kg2_zero_attrs']} ({stats['kg2_zero_attrs']/stats['count']*100:.1f}%)")

            return stats

        results = {
            "ref": compute_stats(self._alignments["ref"], "REF (test)"),
            "sup": compute_stats(self._alignments["sup"], "SUP (train)"),
            "valid": compute_stats(self._alignments["valid"], "VALID"),
        }

        # Overall statistics
        logger.info(f"\n{'='*70}")
        logger.info("OVERALL STATISTICS")
        logger.info(f"{'='*70}")
        logger.info(f"Total attr_triples1: {sum(attr_count_kg1.values())}")
        logger.info(f"Total attr_triples2: {sum(attr_count_kg2.values())}")
        logger.info(f"Total triples_1: {sum(rel_count_kg1.values())}")
        logger.info(f"Total triples_2: {sum(rel_count_kg2.values())}")

        return results

    def run_full_analysis(self) -> Dict:
        """Run complete dataset analysis."""
        logger.info("\n" + "="*70)
        logger.info(f"DATASET ANALYSIS: {self.dataset_path.name}")
        logger.info("="*70)

        results = {
            "dataset_path": str(self.dataset_path),
            "alignment_structure": self.analyze_alignment_structure(),
            "alignment_to_triples": self.verify_alignment_to_triples(),
            "attribute_coverage": self.analyze_attribute_coverage(),
            "data_density": self.analyze_data_density(),
        }

        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)

        return results
