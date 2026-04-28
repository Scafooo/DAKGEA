import sys

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

from .Announce import Announce


class TrainingTools:
    def __init__(self, iter: DataLoader, epochs: int = 15, device: t.device = None):
        self.iter = iter
        self.sample_count = len(iter.dataset)
        self.epochs = epochs
        self.device = device or t.device('cpu')
        self.__init_metrics()

    def __init_metrics(self):
        self.loss_count = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.pred = t.LongTensor().to(self.device)
        self.results = t.FloatTensor().to(self.device)
        self.labels = t.LongTensor().to(self.device)

    def batches(self, get_bc):
        self.__init_metrics()
        with tqdm(total=self.sample_count, file=sys.stdout) as pbar:
            done = 0
            for i, batch in enumerate(self.iter):
                yield i, batch
                batch_size = get_bc(batch)
                done += batch_size * 2
                pbar.set_description('Loss: %f\tAcc: %f\tPrec: %.2f\tRec: %.2f\tF1: %.2f' % (
                    self.loss_count / max(done, 1), self.accuracy, self.precision, self.recall, self.f1_score))
                pbar.update(batch_size)

    @staticmethod
    def batch_iter(iter: DataLoader, desc=''):
        total = len(iter.dataset)
        with tqdm(total=total, file=sys.stdout) as pbar:
            done = 0
            pbar.set_description(desc)
            for i, batch in enumerate(iter):
                yield i, batch
                step_count = len(batch[0])
                done += step_count
                pbar.update(step_count)

    def update_metrics(self, loss, y_pred, labels: t.Tensor = None, batch_size=None):
        assert batch_size is not None
        self.__add_loss_count(loss, batch_size)
        _, values = t.max(y_pred, 1)
        if labels is not None:
            tp, tn, fp, fn = TrainingTools._confusion_matrix(y_pred, labels)
            self.__update_classify_metrics(tp, tn, fp, fn)
        values = values.to(self.device)
        self.pred = t.cat([self.pred, values])
        if labels is not None:
            labels = labels.to(self.device)
            self.labels = t.cat([self.labels, labels])

    def __add_loss_count(self, loss, batch_size):
        self.loss_count += loss.item() * batch_size

    def __update_classify_metrics(self, TP, TN, FP, FN):
        self.TP += TP
        self.TN += TN
        self.FP += FP
        self.FN += FN
        if self.TP + self.FP > 0:
            self.precision = 100 * self.TP / (self.TP + self.FP)
        if self.TP + self.FN > 0:
            self.recall = 100 * self.TP / (self.TP + self.FN)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        total = self.TP + self.TN + self.FP + self.FN
        if total > 0:
            self.accuracy = 100 * (self.TP + self.TN) / total

    @staticmethod
    def _confusion_matrix(output, target):
        predictions = output.max(1)[1].data
        correct = (predictions == target.data).float()
        incorrect = (1 - correct).float()
        positives = (target.data == 1).float()
        negatives = (target.data == 0).float()
        tp = t.dot(correct, positives)
        tn = t.dot(correct, negatives)
        fp = t.dot(incorrect, negatives)
        fn = t.dot(incorrect, positives)
        return tp, tn, fp, fn
