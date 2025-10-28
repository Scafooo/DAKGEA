import csv
from src.logger import logger

def read_tsv(file_path):
    logger.info(f"Reading file: {file_path}")
    with open(file_path, newline='', encoding='utf-8') as f:
        return [row for row in csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')]

read_vaild_pairs = read_tsv
read_sup_pairs = read_tsv
read_ref_pairs = read_tsv