import os
from src.logger import logger

def build_path(root, filename):
    return os.path.join(root, filename)


def write_tsv(file_path, rows):
    logger.info(f"Writing file: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, (list, tuple)):
                f.write("\t".join(map(str, row)) + "\n")
            else:
                f.write(str(row) + "\n")
