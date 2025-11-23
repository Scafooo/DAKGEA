"""Sanity check for OpenEA attribute split ratios."""

from rdflib import URIRef

from src.core.dataset.writer.openea_dataset_writer import OpeneaDatasetWriter
from src.utils.reader import read_tsv
from src.utils.writer import write_tsv


def test_openea_attribute_split_ratios(tmp_path):
    """Ensure the writer produces a 20/70/10 split for sup/ref/valid."""
    dir_path = tmp_path

    # Build ent_ids mappings
    ent_ids_1 = [[str(i), f"http://e1/{i}"] for i in range(10)]
    ent_ids_2 = [[str(10 + i), f"http://e2/{i}"] for i in range(10)]
    write_tsv(dir_path / "ent_ids_1", ent_ids_1)
    write_tsv(dir_path / "ent_ids_2", ent_ids_2)

    aligned_entities = [
        (URIRef(f"http://e1/{i}"), URIRef(f"http://e2/{i}")) for i in range(10)
    ]

    writer = OpeneaDatasetWriter()
    writer._write_aligned_entities_attribute(aligned_entities, str(dir_path))

    sup_pairs = read_tsv(dir_path / "sup_pairs")
    ref_pairs = read_tsv(dir_path / "ref_pairs")
    valid_pairs = read_tsv(dir_path / "valid_pairs")

    assert len(sup_pairs) == 2  # 20% of 10
    assert len(ref_pairs) == 7  # 70% of 10
    assert len(valid_pairs) == 1  # 10% of 10

    assert sup_pairs == [[str(i), str(10 + i)] for i in range(2)]
    assert ref_pairs == [[str(i), str(10 + i)] for i in range(2, 9)]
    assert valid_pairs == [[str(i), str(10 + i)] for i in range(9, 10)]
