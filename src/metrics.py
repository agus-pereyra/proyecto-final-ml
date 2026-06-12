'''
Módulo de métricas de evaluación
'''

import numpy as np
from typing import Iterable

def f1_score():
    pass

def confusion_matrix(y_true: Iterable, y_pred: Iterable, n_classes: int = 6):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def _kappa_from_cm(cm):
    n = cm.sum()
    p_o = np.trace(cm) / n

    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    p_e = np.sum(row_marg * col_marg) / n**2

    return (p_o - p_e) / (1 - p_e)

def cohen_kappa(y_true, y_pred, n_classes=6):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    return _kappa_from_cm(cm)

def cohen_kappa_per_class(y_true, y_pred, n_classes=6):
    kappas = np.zeros(n_classes)
    for k in range(n_classes):
        true_bin = (y_true == k).astype(int)
        pred_bin = (y_pred == k).astype(int)
        cm = confusion_matrix(true_bin, pred_bin, n_classes=2)
        kappas[k] = _kappa_from_cm(cm)

    return kappas
