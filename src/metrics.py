import numpy as np
from sklearn.metrics import cohen_kappa_score

def cohen_kappa_per_class(y_true, y_pred, n_classes=6):
    '''
    Cohen's Kappa por clase (one-vs-rest): para cada clase k se binariza el
    problema (¿es k o no?) y se calcula el kappa de ese problema binario.
    Permite ver en qué clases específicas difieren más dos etiquetados.
    '''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    kappas = np.zeros(n_classes)
    for k in range(n_classes):
        kappas[k] = cohen_kappa_score((y_true == k).astype(int), (y_pred == k).astype(int))
    return kappas
