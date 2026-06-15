'''
Módulo de carga, procesamiento y división de dataset
'''

import pandas as pd
import scipy.io as sio
from pathlib import Path
import numpy as np
import torch

DATA_DIR = Path(__file__).parent.parent / 'data' if '__file__' in dir() else Path('../data')
DATA_PATH = next((p for p in DATA_DIR.iterdir() if p.is_dir()), None) # primer directorio (debería haber solo uno)

if DATA_PATH is None:
    raise FileNotFoundError(
        f'No se encontró el dataset en {DATA_DIR}. '
        'Descargar y extraer en esa carpeta (ver README.md).'
    )

PATIENCE_NUMBERS = [0,1,2,6,7,8,9,10,11,13,14,15,16,17,18,19,20,22,30,31,32,34,35,36,
                    38,39,40,41,42,43,44,45,47,49,50,51,52,53,55,56,60,62,63,64,65,66,68]

STAGES_LABELS = {
    0 : 'Wake',
    1 : 'N1',
    2 : 'N2',
    3 : 'N3',
    4 : 'REM',
    5 : 'Unkown'
}

class EDA:
    '''
    Clase de métodos estáticos para análisis y procesamiento de los datos
    '''
    @staticmethod
    def load_night(patient: int, night: int):
        path = DATA_PATH / f'Bidslab{patient:02d}' / f'{night}' 
        hr = pd.read_csv(path / 'hr.csv', header=None, names=['Timestamp', 'hr']) 
        motion = pd.read_csv(path / 'motion.csv')

        # pasaje timestamps de unix a segundos
        hr['datetime'] = pd.to_datetime(hr['Timestamp'], unit='s')
        motion['datetime'] = pd.to_datetime(motion['Timestamp'], unit='s')

        mat = sio.loadmat(path / 'labels.mat')

        dreem_labels = mat['dreem_label'].flatten()
        expert_labels = mat['expert_label'].flatten()
        rec_start = mat['recStart'][0]
        
        return hr, motion, dreem_labels, expert_labels, rec_start

    @staticmethod
    def class_distribution():
        expert_counts = np.zeros(6, dtype=int)
        dreem_counts = np.zeros(6, dtype=int)

        for patient in PATIENCE_NUMBERS:
            patient_dir = DATA_PATH / f'Bidslab{patient:02d}'
            for night_dir in patient_dir.iterdir():
                if not night_dir.is_dir():
                    continue

                mat = sio.loadmat(night_dir / 'labels.mat')
                expert_labels = mat['expert_label'].flatten()
                dreem_labels = mat['dreem_label'].flatten()

                expert_counts += np.bincount(expert_labels, minlength=6)
                dreem_counts += np.bincount(dreem_labels, minlength=6)

        return {'expert': expert_counts, 'dreem': dreem_counts}

    @staticmethod
    def all_labels():
        expert_labels = []
        dreem_labels = []

        for patient in PATIENCE_NUMBERS:
            patient_dir = DATA_PATH / f'Bidslab{patient:02d}'
            for night_dir in patient_dir.iterdir():
                if not night_dir.is_dir():
                    continue

                mat = sio.loadmat(night_dir / 'labels.mat')
                expert_labels.append(mat['expert_label'].flatten())
                dreem_labels.append(mat['dreem_label'].flatten())

        return np.concatenate(expert_labels), np.concatenate(dreem_labels)

class DataSet(torch.utils.data.Dataset):
    '''
    Clase para el manejo de splits/batches para el entrenamiento de redes
    '''
    def __init__(self, manifest):
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        path, epoch_idx, label = self.manifest[idx]
        epoch = np.load(path, mmap_mode='r')[epoch_idx]
        return torch.tensor(epoch, dtype=torch.float32), label
