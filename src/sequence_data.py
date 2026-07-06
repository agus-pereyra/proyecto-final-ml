'''
Datos por NOCHE (señal cruda) para el híbrido CNN+BiLSTM.

`build_night_sequences` reprocesa la señal alineada (reusando la alineación
"fuente de verdad" de data.py) y guarda por paciente X (N,150,4), y (N,) y
night_id (N,). `NightSignalDataset` entrega una secuencia por noche para
alimentar el `SleepStager` de lstm.py; el padding lo hace `lstm.collate_nights`.

Lo genuinamente nuevo que lstm.py no tiene: lstm.py trabaja sobre las FEATURES
tabulares (epoch_features.csv), mientras que el híbrido necesita la SEÑAL CRUDA
por noche. No modifica ningún archivo existente.
'''

import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from data import EDA
except ImportError:
    from src.data import EDA


def _epoch_tensor(hr_win, acc_win, epoch_start, epoch_end, n_steps=150):
    '''
    Tensor (n_steps, 4) [HR, mag_mean, mag_std, enmo_mean] de una época, idéntico
    al que arma process_dataset_cnn.
    '''
    bin_dur = 30 / n_steps
    grid = np.linspace(epoch_start, epoch_end, n_steps, dtype=np.float32)
    hr_fixed = np.interp(grid, hr_win['Timestamp'], hr_win['hr'])

    acc_win = acc_win.copy()
    mag = np.sqrt(acc_win['x']**2 + acc_win['y']**2 + acc_win['z']**2)
    acc_win['mag'] = mag
    acc_win['enmo'] = np.maximum(mag - 1.0, 0.0)
    acc_win['bin'] = ((acc_win['Timestamp'] - epoch_start) / bin_dur).astype(int).clip(0, n_steps - 1)

    grouped = acc_win.groupby('bin')
    mag_mean = grouped['mag'].mean().reindex(range(n_steps), method='ffill').fillna(0).values
    mag_std = grouped['mag'].std().reindex(range(n_steps), method='ffill').fillna(0).values
    enmo_mean = grouped['enmo'].mean().reindex(range(n_steps), method='ffill').fillna(0).values

    return np.column_stack([hr_fixed, mag_mean, mag_std, enmo_mean])


def build_night_sequences(output_dir='../data_extraction/sequences', n_patients=None, n_steps=150):
    '''
    Como get_cnn_dataset pero conservando el borde de noche: por paciente guarda
    X (N,150,4), y (N,) y night_id (N,) en <paciente>.npz. La LSTM no puede cruzar
    de una noche a otra, así que ese night_id es imprescindible.

    Nota: se descartan las épocas con muy pocas muestras (igual que el pipeline
    original), lo que puede romper la contigüidad dentro de una noche; para esta
    primera versión se acepta (son pocas y aisladas).
    '''
    windows = EDA.valid_windows()
    gap_res = EDA.internal_gap_resolution()

    nights_by_patient = defaultdict(list)
    for patient, night in sorted(windows):
        nights_by_patient[patient].append(night)

    patients = sorted(nights_by_patient)
    if n_patients:
        patients = patients[:n_patients]

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for patient in tqdm(patients, desc='Procesando pacientes'):
        X_list, y_list, nid_list = [], [], []

        for night in nights_by_patient[patient]:
            valid_start, valid_end = windows[(patient, night)]

            res = gap_res.get((patient, night))
            if res is not None:
                if res['action'] == 'discard':
                    continue
                if res['action'] == 'trim_tail' and res['new_valid_end'] is not None:
                    valid_end = res['new_valid_end']

            hr, motion, _dreem, expert, aligned_start = EDA.load_night_clean(
                patient, night, valid_start, valid_end)

            for i, label in enumerate(expert):
                if label > 4:
                    continue
                epoch_start = aligned_start + i * 30
                epoch_end = epoch_start + 30
                hr_win = hr[(hr['Timestamp'] >= epoch_start) & (hr['Timestamp'] < epoch_end)]
                acc_win = motion[(motion['Timestamp'] >= epoch_start) & (motion['Timestamp'] < epoch_end)]
                if len(hr_win) < 2 or len(acc_win) < 10:
                    continue
                X_list.append(_epoch_tensor(hr_win, acc_win, epoch_start, epoch_end, n_steps))
                y_list.append(int(label))
                nid_list.append(night)

        if X_list:
            path = os.path.join(output_dir, f"Bidslab{patient:02d}.npz")
            np.savez_compressed(path,
                                X=np.array(X_list, dtype=np.float32),
                                y=np.array(y_list, dtype=np.int8),
                                night_id=np.array(nid_list, dtype=np.int16))
            print(f"-> Bidslab{patient:02d}: {len(X_list)} épocas, {len(set(nid_list))} noches.")
        else:
            print(f"-> Bidslab{patient:02d}: sin épocas válidas.")

    print(f"--- Listo en: {output_dir} ---")


class NightSignalDataset(Dataset):
    '''
    Una noche por item: (feats [T,150,4] float32, labels [T] int64). Lee los .npz
    de build_night_sequences y agrupa por night_id (las filas ya vienen en orden
    de época). Compatible con `lstm.collate_nights`, que padea feats con 0.0 y
    labels con UNKNOWN.
    '''
    def __init__(self, files):
        self.nights = []
        for f in files:
            d = np.load(f)
            X, y, nid = d['X'], d['y'].astype(np.int64), d['night_id']
            for n in np.unique(nid):
                m = nid == n
                self.nights.append((X[m], y[m]))

    def __len__(self):
        return len(self.nights)

    def __getitem__(self, i):
        X, y = self.nights[i]
        return torch.from_numpy(X.copy()), torch.from_numpy(y.copy())


def split_night_files(processed_dir='../data_extraction/sequences',
                      val_frac=0.15, test_frac=0.15, seed=36631):
    '''
    Divide los .npz por paciente (un archivo = un sujeto) en train/val/test, con
    el mismo criterio que `split_subjects` de lstm.py (fracciones + shuffle con
    seed). Devuelve tres listas de paths.
    '''
    files = sorted(Path(processed_dir).glob('*.npz'))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)

    n = len(files)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    pick = lambda ids: [files[i] for i in ids]
    return pick(train_idx), pick(val_idx), pick(test_idx)
