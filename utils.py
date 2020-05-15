import torch
import numpy as np
from loguru import logger
import time


class PrecisionRecallMetric:
    def __init__(self, tolerance):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.tolerance = tolerance
        self.eps = 1e-5

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        return precision, recall, f1

    def get_final_metrics(self):
        return self.get_metrics(self.precision_counter, self.recall_counter, self.pred_counter, self.gt_counter)

    def zero(self):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0

    def update(self, batch_y, batch_yhat):
        precision_counter = 0
        recall_counter = 0
        pred_counter = 0
        gt_counter = 0

        for (y, yhat) in zip(batch_y, batch_yhat):
            y, yhat = np.array(y), np.array(yhat)
            y, yhat = y[1:-1], yhat[1:-1]
            for yhat_i in yhat:
                min_dist = np.abs(y - yhat_i).min()
                precision_counter += (min_dist <= self.tolerance)
            for y_i in y:
                min_dist = np.abs(yhat - y_i).min()
                recall_counter += (min_dist <= self.tolerance)
            pred_counter += len(yhat)
            gt_counter += len(y)

        self.precision_counter += precision_counter
        self.recall_counter += recall_counter
        self.pred_counter += pred_counter
        self.gt_counter += gt_counter

        return self.get_metrics(precision_counter, recall_counter, pred_counter, gt_counter)


class PrecisionRecallMetricMultiple:
    def __init__(self, levels=[0, 1, 2]):
        self.prs = {level: PrecisionRecallMetric(tolerance=level) for level in levels}

    def get_final_metrics(self):
        results = {level: pr.get_final_metrics() for level, pr in self.prs.items()}
        return results

    def zero(self):
        for level, pr in self.prs.items():
            pr.zero()

    def update(self, batch_y, batch_yhat):
        results = {}
        for level, pr in self.prs.items():
            results[level] = pr.update(batch_y, batch_yhat)

        return results


class StatsMeter:
    def __init__(self):
        self.data = []

    def update(self, item):
        if type(item) == list:
            self.data.extend(item)
        else:
            self.data.append(item)

    def get_stats(self):
        data = np.array(self.data)
        return data.mean(), data.std()

    def zero(self):
        self.data.clear()
        assert len(self.data) == 0, "StatsMeter didn't clear"


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info(self.msg % (time.time() - self.start_time))
