#!/usr/bin/env python3
"""Convert datasets to RDF exports using project readers/writers.

Primary flow: load HybEA-layout datasets via `HybeaDatasetReader`, then persist
them with `RDFDatasetWriter`.  If the directory does not match the HybEA layout,
fall back to reading paired RDF graphs plus an alignment file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple, Type


def _import_dependencies():
    """Import reader/writer classes, extending sys.path when necessary."""
    try:
        from src.core.dataset.reader.hybea_dataset_reader import HybeaDatasetReader
        from src.core.dataset.writer.rdf_dataset_writer import RDFDatasetWriter
        from src.core.dataset import Dataset
        from src.core.knowledge_graph.reader import KnowledgeGraphReaderFactory

        return HybeaDatasetReader, RDFDatasetWriter, Dataset, KnowledgeGraphReaderFactory
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parent
        candidate_roots = [
            project_root,
            project_root.parent,
            project_root.parent / "DAKGEA",
        ]

        for candidate in candidate_roots:
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))

        from src.core.dataset.reader.hybea_dataset_reader import HybeaDatasetReader
        from src.core.dataset.writer.rdf_dataset_writer import RDFDatasetWriter
        from src.core.dataset import Dataset
        from src.core.knowledge_graph.reader import KnowledgeGraphReaderFactory

        return HybeaDatasetReader, RDFDatasetWriter, Dataset, KnowledgeGraphReaderFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a dataset directory into RDF artifacts."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the HybEA-formatted dataset directory.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory for the RDF export.",
    )
    return parser.parse_args()


def _try_load_hybea_dataset(reader_cls: Type, input_dir: Path):
    """Attempt to load the dataset using the HybEA reader; return None on failure."""
    required_files = { "ent_ids_1", "ent_ids_2", "ref_pairs" }
    has_signature = any((input_dir / name).exists() for name in required_files)
    reader = reader_cls()

    if not has_signature:
        siblings = ["attribute_data", "knowformer_data"]
        if not any((input_dir / sib).exists() for sib in siblings):
            return None

    try:
        return reader.read(str(input_dir))
    except Exception:
        return None


def _load_rdf_pair_dataset(
    dataset_cls: Type,
    kg_reader_factory,
    input_dir: Path,
) -> "Dataset":
    """Construct a dataset from paired RDF graphs plus an alignment file."""
    kg_reader = kg_reader_factory.create_reader("rdf")
    graph_candidates = sorted(
        p for p in input_dir.glob("*")
        if p.is_file()
        and p.suffix.lower() in {".ttl", ".nt", ".ntriples"}
        and "gold" not in p.name.lower()
    )

    if len(graph_candidates) < 2:
        raise FileNotFoundError(
            f"Expected at least two RDF graph files under {input_dir}"
        )

    kg_source_path, kg_target_path = graph_candidates[:2]
    kg_source = kg_reader.read(str(kg_source_path))
    kg_target = kg_reader.read(str(kg_target_path))

    alignment_path = _resolve_alignment_file(input_dir)
    aligned_pairs = _parse_alignment_pairs(alignment_path)

    return dataset_cls(kg_source, kg_target, aligned_pairs)


def _resolve_alignment_file(input_dir: Path) -> Path:
    candidates = [
        p for p in input_dir.glob("*")
        if p.is_file()
        and any(key in p.name.lower() for key in ("gold", "aligned", "alignment"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Unable to locate alignment file (gold/aligned) under {input_dir}"
        )
    return sorted(candidates)[0]


PREFIX_PATTERN = re.compile(r"@prefix\s+([A-Za-z0-9_-]+):\s*<([^>]+)>\s*\.")


def _parse_alignment_pairs(alignment_path: Path) -> Iterable[Tuple[str, str]]:
    prefix_map = {}
    pairs = []

    with alignment_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = PREFIX_PATTERN.match(line)
            if match:
                prefix_map[match.group(1)] = match.group(2)
                continue

            left, right = _split_alignment_line(line)
            pairs.append(
                (_expand_prefixed(left, prefix_map), _expand_prefixed(right, prefix_map))
            )

    if not pairs:
        raise ValueError(f"No alignment pairs parsed from {alignment_path}")
    return pairs


def _split_alignment_line(line: str) -> Tuple[str, str]:
    tokens = re.split(r"\s+", line)
    if len(tokens) < 2:
        raise ValueError(f"Malformed alignment line: {line}")
    return tokens[0], tokens[1]


def _expand_prefixed(token: str, prefix_map: dict) -> str:
    token = token.strip()
    if not token:
        raise ValueError("Encountered empty token while parsing alignment file.")
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if ":" in token:
        prefix, local = token.split(":", 1)
        base = prefix_map.get(prefix)
        if base is None:
            raise KeyError(f"Unknown prefix '{prefix}' for token '{token}'")
        return f"{base}{local}"
    return token


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    (
        HybeaDatasetReader,
        RDFDatasetWriter,
        DatasetCls,
        KGReaderFactory,
    ) = _import_dependencies()

    dataset = _try_load_hybea_dataset(HybeaDatasetReader, input_dir)
    if dataset is None:
        dataset = _load_rdf_pair_dataset(DatasetCls, KGReaderFactory, input_dir)

    writer = RDFDatasetWriter()
    writer.write(dataset, str(output_dir))

    print(f"Conversion complete.\n  source: {input_dir}\n  output: {output_dir}")


if __name__ == "__main__":
    main()
