from ..tools.Announce import Announce
from .._globals import seq_max_len


class Trainer:
    def __init__(self):
        pass

    def train(self):
        pass

    def data_prepare(self, eid2tids1: dict, eid2tids2: dict, entity_ids1: dict, entity_ids2: dict):
        pass

    @staticmethod
    def reduce_tokens(tids):
        while True:
            if len(tids) <= seq_max_len:
                break
            tids.pop()
        return tids
