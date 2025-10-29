import tempfile
import unittest
from pathlib import Path

from src.util.reader import read_tsv, read_valid_pairs, read_vaild_pairs
from src.util.writer import write_tsv


class UtilIOTests(unittest.TestCase):
    def test_write_tsv_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "nested" / "output.tsv"
            rows = [["h1", "h2"], ["a", "b"]]

            write_tsv(destination, rows)

            self.assertTrue(destination.exists())
            contents = destination.read_text(encoding="utf-8").splitlines()
            self.assertEqual(contents, ["h1\th2", "a\tb"])

    def test_write_tsv_handles_string_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "log" / "notes.tsv"
            rows = ["header", ["value", 123]]

            write_tsv(destination, rows)

            contents = destination.read_text(encoding="utf-8").splitlines()
            self.assertEqual(contents, ["header", "value\t123"])

    def test_read_valid_pairs_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "pairs.tsv"
            source.write_text("id1\tid2\nx\ty\n", encoding="utf-8")

            expected = [["id1", "id2"], ["x", "y"]]
            self.assertEqual(read_tsv(source), expected)
            self.assertEqual(read_valid_pairs(source), expected)
            self.assertEqual(read_vaild_pairs(source), expected)


if __name__ == "__main__":
    unittest.main()
