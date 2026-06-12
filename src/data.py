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
