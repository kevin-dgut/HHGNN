import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from scipy.special import softmax

import sys
sys.path.append("/home/mult-atalas/MHHGT/hgt_three_connected")

class AverageMeter(object):


    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(preds, labels):
    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, list):
        labels = np.array(labels)

    if preds.ndim == 1:
        preds = (preds > 0.5).astype(np.float32)


    correct_prediction = np.equal(preds, labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)



def auc(preds, labels, is_logit=False):

    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, list):
        labels = np.array(labels)
    if is_logit:
        preds = softmax(preds, axis=1)[:, 1]
    else:
        preds = preds[:, 1] if preds.ndim == 2 else preds

    try:
        auc_out = roc_auc_score(labels, preds)
    except:
        auc_out = 0
    return auc_out



def prf(preds, labels):

    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, list):
        labels = np.array(labels)
    if preds.ndim == 1:
        preds = (preds > 0.5).astype(np.int32)

    p, r, f, s = precision_recall_fscore_support(labels, preds, average='binary')
    return [p, r, f]



def numeric_score(preds, labels):

    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, list):
        labels = np.array(labels)
    FP = np.float64(np.sum((preds == 1) & (labels == 0)))
    FN = np.float64(np.sum((preds == 0) & (labels == 1)))
    TP = np.float64(np.sum((preds == 1) & (labels == 1)))
    TN = np.float64(np.sum((preds == 0) & (labels == 0)))
    return FP, FN, TP, TN



def metrics(preds, labels):

    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, list):
        labels = np.array(labels)
    if preds.ndim == 1:
        preds = (preds > 0.5).astype(np.int32)

    FP, FN, TP, TN = numeric_score(preds, labels)
    sen = TP / (TP + FN + 1e-10)
    spe = TN / (TN + FP + 1e-10)
    return sen, spe

