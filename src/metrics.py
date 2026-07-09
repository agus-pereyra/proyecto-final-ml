import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

FIG_DIR = Path(__file__).parent.parent / 'report' / 'figures'


def _slugify(text):
    '''Convierte un título en un nombre de archivo (minúsculas, guiones, sin acentos raros).'''
    text = text.strip().lower()
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'[^a-z0-9\-]', '', text)
    return re.sub(r'-+', '-', text).strip('-')

def _resolve_class_names(y_true, y_pred, class_names):
    '''Nombres de clase por defecto: las 5 etapas si entran, si no genéricos.'''
    if class_names is not None:
        return class_names
    n = int(max(np.max(y_true), np.max(y_pred))) + 1
    return STAGE_NAMES if n <= len(STAGE_NAMES) else [f'clase {k}' for k in range(n)]


def print_metrics(y_true, y_pred, class_names=None, name=''):
    '''
    Imprime el classification_report (precision/recall/F1 por clase + macro/weighted
    + accuracy) y el Cohen's Kappa.
    In:  y_true, y_pred [N]; class_names y name (encabezado) opcionales.
    Out: nada (imprime en pantalla).
    '''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = _resolve_class_names(y_true, y_pred, class_names)
    labels = list(range(len(class_names)))
    if name:
        print(name)
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, 
                                zero_division=0, digits=4))
    print(f"Cohen's Kappa:  {cohen_kappa_score(y_true, y_pred):.4f}")
    print(f"F1 macro:       {f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.4f}")
    print(f"F1 micro:       {f1_score(y_true, y_pred, labels=labels, average='micro', zero_division=0):.4f}")
    print(f"Accuracy:       {accuracy_score(y_true, y_pred):.4f}")


def _draw_confusion(cm, class_names, title, normalize, ax=None, figsize=(6, 5),
                    fontsize_scale=1.0, show_title=True):
    '''Dibuja una matriz de confusión ya calculada. `fontsize_scale` escala TODAS las
    fuentes (números internos de cada celda incluidos); `show_title` controla el título.
    Devuelve (fig, ax).'''
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    disp.plot(ax=ax, cmap='Blues', colorbar=False,
              values_format='.2f' if normalize else 'd',
              text_kw={'fontsize': 13 * fontsize_scale})   # números dentro de cada bloque
    if show_title:
        ax.set_title(title, fontsize=12 * fontsize_scale)
    ax.set_ylabel('expert label', fontsize=12 * fontsize_scale)
    ax.set_xlabel('predicted label', fontsize=12 * fontsize_scale)
    ax.tick_params(axis='both', labelsize=11 * fontsize_scale)
    fig.tight_layout()
    return fig, ax


def plot_confusion(y_true, y_pred, class_names=None, title='Matriz de confusión',
                   normalize=False, ax=None, figsize=(6, 5), save=False):
    '''
    Grafica la matriz de confusión (5 clases por defecto; `normalize=True` la
    normaliza por fila = recall por clase).
    In:  y_true, y_pred [N]; class_names, title, normalize, ax opcionales.
         save: guarda el PNG en report/figures/ (igual que plots.py). True -> nombre
         derivado del título; str -> ese nombre de archivo (sin extensión). Ignorado si
         se pasa un `ax` externo. La versión guardada se dibuja más chica y con fuentes
         más grandes (números de cada celda incluidos), conservando el título del ax.
    Out: (fig, ax) de la figura mostrada.
    '''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = _resolve_class_names(y_true, y_pred, class_names)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels,
                          normalize='true' if normalize else None)

    if save and ax is None:
        name = save if isinstance(save, str) else _slugify(title)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_fig, _ = _draw_confusion(cm, class_names, title, normalize,
                                      figsize=(4.6, 3.9), fontsize_scale=1.2,
                                      show_title=True)
        save_fig.savefig(FIG_DIR / f'{name}.png', dpi=200, bbox_inches='tight')
        plt.close(save_fig)

    return _draw_confusion(cm, class_names, title, normalize, ax=ax, figsize=figsize)

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


def roc_pr_curves(y_true, y_score, class_names=None, title='', figsize=(13, 5)):
    '''
    Curvas ROC y Precision-Recall one-vs-rest (una por clase).
    In:  y_true [N] (0..C-1), y_score [N, C] (probabilidades); class_names/title opc.
    Out: (fig, (ax_roc, ax_pr)).
    '''
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_classes = y_score.shape[1]
    if class_names is None:
        class_names = STAGE_NAMES if n_classes == 5 else [f'clase {k}' for k in range(n_classes)]

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)

    aucs, aps = [], []
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:      # sin positivos: la clase no aparece en y_true
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')

        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        aps.append(ap)
        ax_pr.plot(rec, prec, label=f'{name} (AP={ap:.2f})')

    macro_auc = float(np.mean(aucs)) if aucs else float('nan')
    macro_ap = float(np.mean(aps)) if aps else float('nan')

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='azar')
    ax_roc.set_xlabel('Tasa de falsos positivos')
    ax_roc.set_ylabel('Tasa de verdaderos positivos')
    ax_roc.set_title(f'Curvas ROC (one-vs-rest) — AUC macro={macro_auc:.3f}')
    ax_roc.legend(loc='lower right', fontsize=9)

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Curvas Precision-Recall (one-vs-rest) — AP macro={macro_ap:.3f}')
    ax_pr.legend(loc='upper right', fontsize=9)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_roc, ax_pr)
