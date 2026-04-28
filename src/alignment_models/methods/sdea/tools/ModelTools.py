import os

import torch as t

from .._globals import PARALLEL
from .Announce import Announce


class ModelTools:
    def __init__(self, patient, min_max='min'):
        self.min_max = min_max
        if min_max == 'min':
            self.bst_score = float('inf')
        else:
            assert min_max == 'max', 'unknown min_max'
            self.bst_score = float('-inf')
        self.patient = patient
        self.current_patient = 0

    def early_stopping(self, model: t.nn.Module, path, score):
        save = False
        if self.min_max == 'min':
            if score < self.bst_score:
                save = True
        else:
            if score > self.bst_score:
                save = True
        if save:
            print(Announce.printMessage(), 'score:', self.bst_score, '->', score)
            self.bst_score = score
            ModelTools.save_model(model, path)
            self.current_patient = 0
            return False
        else:
            self.current_patient += 1
            print(Announce.printMessage(), 'bst score:', self.bst_score, 'current score:', score, 'patient:', self.current_patient)
            if self.current_patient >= self.patient:
                return True
            else:
                return False

    @staticmethod
    def save_model(model: t.nn.Module, output_sub_dir) -> None:
        print(Announce.printMessage(), 'save model:', output_sub_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        fold = os.path.dirname(output_sub_dir)
        if not os.path.exists(fold):
            os.makedirs(fold)
        t.save(model_to_save, output_sub_dir)

    @staticmethod
    def load_model(model_path):
        return t.load(model_path, weights_only=False)
