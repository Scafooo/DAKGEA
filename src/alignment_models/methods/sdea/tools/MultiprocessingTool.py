import multiprocessing as mp
import sys
from itertools import chain
from threading import Thread

from tqdm import tqdm

from .Announce import Announce


class MultiprocessingTool:
    def __init__(self):
        self._solver = None
        self._kwargs = {}
        self._packs = None

    def packed_solver(self, solver, **kwargs):
        self._solver = solver
        self._kwargs = kwargs
        return self

    def send_packs(self, packs):
        self._packs = list(packs)
        return self

    def receive_results(self):
        results = []
        for pack in self._packs:
            try:
                result = self._solver(pack, **self._kwargs)
                results.append(result)
            except Exception:
                pass
        return results


class MPTool:
    @staticmethod
    def packed_solver(solver, **kwargs):
        class _Runner:
            def send_packs(self, packs):
                self._packs = list(packs)
                return self

            def receive_results(self_inner):
                results = []
                for pack in self_inner._packs:
                    try:
                        result = solver(pack, **kwargs)
                        results.append(result)
                    except Exception:
                        pass
                return results
        return _Runner()
