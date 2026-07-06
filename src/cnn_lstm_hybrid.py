'''
Híbrido CNN + BiLSTM end-to-end.

Reusa DOS archivos existentes sin modificarlos:
- src/CNN.py: sus bloques convolucionales (block1/2/3, sin el clasificador) hacen
  de encoder intra-época. Por eso NO hace falta un archivo aparte para la CNN.
- src/lstm.py: el `SleepStager` (que ya prevé un `EpochEncoder` enchufable), más
  `collate_nights`, `compute_metrics`, `collect_predictions` y `class_weights`.

Lo único nuevo acá es el pegamento: un `CNNEpochEncoder` que implementa la
interfaz `EpochEncoder` de lstm.py, y un loop de entrenamiento que alimenta señal
cruda por noche (los loaders tabulares de lstm.py no sirven para señal cruda).
'''

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from CNN import CNN
    from lstm import (EpochEncoder, SleepStager, collate_nights, compute_metrics,
                      collect_predictions, class_weights, set_seed, UNKNOWN, N_CLASSES)
    from sequence_data import NightSignalDataset, split_night_files
except ImportError:
    from src.CNN import CNN
    from src.lstm import (EpochEncoder, SleepStager, collate_nights, compute_metrics,
                          collect_predictions, class_weights, set_seed, UNKNOWN, N_CLASSES)
    from src.sequence_data import NightSignalDataset, split_night_files


class CNNEpochEncoder(EpochEncoder):
    '''
    Encoder intra-época que reusa los bloques conv de src/CNN.py (sin tocar el
    archivo ni usar su clasificador). Mapea la secuencia de épocas crudas de una
    noche a la secuencia de vectores de features:

        forward: feats [B, T, 150, 4] -> [B, T, out_dim]  (out_dim = 128)
    '''
    def __init__(self, feature_dim=128):
        super().__init__()
        cnn = CNN(num_classes=5)          # sólo nos quedamos con los bloques conv
        self.block1 = cnn.block1
        self.block2 = cnn.block2
        self.block3 = cnn.block3
        self.out_dim = feature_dim

    def forward(self, x):
        B, T, L, C = x.shape
        x = x.reshape(B * T, L, C).permute(0, 2, 1)   # [B*T, C, L] para Conv1d
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)                            # [B*T, out_dim, 1]
        return x.flatten(1).reshape(B, T, -1)         # [B, T, out_dim]


def _labels_from_files(files):
    return np.concatenate([np.load(f)['y'].astype(np.int64) for f in files])


def train_hybrid(cfg, sequences_dir='../data_extraction/sequences',
                 val_frac=0.15, test_frac=0.15, feature_dim=128):
    '''
    Entrena el híbrido end-to-end (la loss de la LSTM entrena también la CNN).
    Reusa de lstm.py: SleepStager, collate_nights, class_weights, compute_metrics
    y collect_predictions. Guarda el mejor checkpoint por Kappa de validación.

    Devuelve (model, history, test_metrics). `history` es una lista de dicts por
    epoch con train_loss + las métricas de validación (kappa, macro_f1, accuracy).
    '''
    set_seed(cfg.seed)
    device = cfg.device

    train_f, val_f, test_f = split_night_files(sequences_dir, val_frac, test_frac, cfg.seed)
    make = lambda files, sh: DataLoader(NightSignalDataset(files), batch_size=cfg.batch_size,
                                        shuffle=sh, collate_fn=collate_nights)
    loaders = {'train': make(train_f, True), 'val': make(val_f, False), 'test': make(test_f, False)}
    print(f"sujetos -> train {len(train_f)} | val {len(val_f)} | test {len(test_f)}")

    encoder = CNNEpochEncoder(feature_dim=feature_dim)
    model = SleepStager(cfg, encoder).to(device)

    weight = None
    if cfg.use_class_weights:
        weight = class_weights(pd.DataFrame({'label': _labels_from_files(train_f)}), device)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=UNKNOWN)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ckpt_dir = os.path.dirname(cfg.ckpt_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    history = []
    best_kappa = -np.inf
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for feats, labels, lengths in loaders['train']:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats, lengths)                        # [B, T, C]
            loss = criterion(logits.reshape(-1, N_CLASSES), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += loss.item() * feats.shape[0]
        train_loss = total / len(loaders['train'].dataset)

        y_val, p_val = collect_predictions(model, loaders['val'], device)
        val_m = compute_metrics(y_val, p_val)
        history.append({'epoch': epoch, 'train_loss': train_loss, **val_m})

        if val_m['kappa'] > best_kappa:
            best_kappa = val_m['kappa']
            torch.save({'model_state': model.state_dict(),
                        'epoch': epoch, 'val_kappa': best_kappa}, cfg.ckpt_path)

        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | val kappa {val_m['kappa']:.4f} "
              f"| macroF1 {val_m['macro_f1']:.4f} | acc {val_m['accuracy']:.4f}")

    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    y_test, p_test = collect_predictions(model, loaders['test'], device)
    test_m = compute_metrics(y_test, p_test)
    print(f"\nTEST (mejor ckpt, val kappa {ckpt['val_kappa']:.4f}, epoch {ckpt['epoch']}):")
    print(f"  kappa {test_m['kappa']:.4f} | macroF1 {test_m['macro_f1']:.4f} | acc {test_m['accuracy']:.4f}")
    print(f"  4-clases: kappa {test_m['kappa_4']:.4f} | macroF1 {test_m['macro_f1_4']:.4f} | acc {test_m['accuracy_4']:.4f}")
    return model, history, test_m
