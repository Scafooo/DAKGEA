import numpy as np
from collections import defaultdict
import logging
from ..reader.helper import convert_tokens_to_ids
from ..reader.helper import load_vocab
from src.logger import get_logger

logger = get_logger(__name__)


class KGDataReader(object):
    def __init__(self,
                 vocab_path,
                 data_path,
                 batch_size=4096,
                 is_training=True):
        self.vocab = load_vocab(vocab_path)
        self.mask_id = self.vocab["[MASK]"]
        self.batch_size = batch_size
        self.is_training = is_training
        self.seq_len = -1
        self.examples = self.read_example(data_path)

    def read_example(self, input_file):
        logger.info("Reading examples from %s", input_file)
        examples = []
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split("\t")
                # Check if last token is a MASK token (string starting with "MASK")
                if len(tokens) > 0 and isinstance(tokens[-1], str) and tokens[-1].startswith("MASK"):
                    # Convert all tokens except the last one to IDs
                    token_seq_ids = convert_tokens_to_ids(self.vocab, tokens[:-1])
                    self.seq_len = max(self.seq_len, len(token_seq_ids))
                    # Append the MASK token as a string (not converted to ID)
                    token_seq_ids.append(tokens[-1])
                else:
                    # Convert all tokens to IDs
                    token_seq_ids = convert_tokens_to_ids(self.vocab, tokens[:])
                    self.seq_len = max(self.seq_len, len(token_seq_ids))
                examples.append(token_seq_ids)
        return examples

    def data_generator(self):
        range_list = [i for i in range(len(self.examples))]
        if self.is_training:
            np.random.shuffle(range_list)
        for i in range(0, len(self.examples), self.batch_size):
            batch = []
            for j in range_list[i:i + self.batch_size]:
                batch.append(self.examples[j])
            yield batch
