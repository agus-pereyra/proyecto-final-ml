'''
Módulo de carga, procesamiento y división de dataset
'''

import json
import pandas as pd
import scipy.io as sio
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).parent.parent / 'data' if '__file__' in dir() else Path('../data')
DATA_PATH = next((p for p in DATA_DIR.iterdir() if p.is_dir()), None) # primer directorio (debería haber solo uno)
ANALYSIS_DIR = Path(__file__).parent.parent / 'analysis' if '__file__' in dir() else Path('../analysis')

if DATA_PATH is None:
    raise FileNotFoundError(
        f'No se encontró el dataset en {DATA_DIR}. '
        'Descargar y extraer en esa carpeta (ver README.md).'
    )

PATIENCE_NUMBERS = [0,1,2,6,7,8,9,10,11,13,14,15,16,17,18,19,20,22,30,31,32,34,35,36,
                    38,39,40,41,42,43,44,45,47,49,50,51,52,53,55,56,60,62,63,64,65,66,68]

# Criterio de calidad de noches (ver EDA.quality_report / EDA.problematic_nights)
GAP_THRESHOLD_S = 60.0           # gaps de ihr mayores a esto se detectan como discontinuidades
ACC_TOL = 0.5                    # |sqrt(x²+y²+z²) - 1| > ACC_TOL indica acelerometría inválida
INTERNAL_GAP_THRESHOLD_S = 600   # 10 min: máximo de gaps internos acumulados dentro de la ventana válida
EDGE_TRUNC_THRESHOLD_S = 3600    # 1 hora: a partir de este truncamiento de extremo la noche se lista como problemática (igual se trunca siempre)

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
    def label_length_mismatch():
        '''
        Compara la cantidad de epochs etiquetadas por expert_label y
        dreem_label en cada noche. La ventana válida del test de calidad
        se define con expert_label, así que verificamos en qué noches
        ambos etiquetados difieren en extensión temporal.

        Devuelve una lista de dicts (una por noche con discrepancia).
        '''
        mismatches = []
        for patient in PATIENCE_NUMBERS:
            patient_dir = DATA_PATH / f'Bidslab{patient:02d}'
            for night_dir in sorted(patient_dir.iterdir()):
                if not night_dir.is_dir():
                    continue

                mat = sio.loadmat(night_dir / 'labels.mat')
                n_expert = len(mat['expert_label'].flatten())
                n_dreem = len(mat['dreem_label'].flatten())
                if n_expert != n_dreem:
                    mismatches.append({
                        'patient': patient,
                        'night': int(night_dir.name),
                        'n_expert': n_expert,
                        'n_dreem': n_dreem,
                        'diff_epochs': n_dreem - n_expert,
                        'diff_s': (n_dreem - n_expert) * 30,
                    })
        return mismatches

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

    @staticmethod
    def quality_report(gap_threshold: float = 60.0, acc_tol: float = 0.5,
                        ihr_max: float = 200.0, edge_trim: int = 10,
                        save_path: Path = None):
        '''
        Recorre todas las noches de todos los pacientes y registra,
        por cada una, gaps temporales en hr/motion, cobertura de las
        labels respecto a la duración total del registro, y muestras
        de acelerometría/IHR con valores potencialmente inválidos.

        Devuelve un DataFrame (resumen, una fila por noche) y guarda
        el detalle completo (incluyendo listas de gaps) en un JSON.
        '''
        nights = []
        for patient in PATIENCE_NUMBERS:
            patient_dir = DATA_PATH / f'Bidslab{patient:02d}'
            for night_dir in sorted(patient_dir.iterdir()):
                if night_dir.is_dir():
                    nights.append((patient, night_dir))

        records = []
        pbar = tqdm(nights, desc='Analizando noches')
        for patient, night_dir in pbar:
            night = night_dir.name
            pbar.set_description(f'Paciente {patient:02d} - Noche {night}')

            hr = pd.read_csv(night_dir / 'hr.csv', header=None, names=['Timestamp', 'hr'])
            motion = pd.read_csv(night_dir / 'motion.csv')
            mat = sio.loadmat(night_dir / 'labels.mat')

            expert_labels = mat['expert_label'].flatten()
            rec_start = mat['recStart'][0]

            # gaps temporales
            hr_diffs = hr['Timestamp'].diff().dropna()
            hr_gaps_mask = hr_diffs > gap_threshold
            hr_gaps = [
                {'start': float(hr['Timestamp'].iloc[i - 1]), 'end': float(hr['Timestamp'].iloc[i]), 'duration': float(hr_diffs.loc[i])}
                for i in hr_diffs[hr_gaps_mask].index
            ]

            motion_diffs = motion['Timestamp'].diff().dropna()
            motion_gaps_mask = motion_diffs > gap_threshold
            motion_gaps = [
                {'start': float(motion['Timestamp'].iloc[i - 1]), 'end': float(motion['Timestamp'].iloc[i]), 'duration': float(motion_diffs.loc[i])}
                for i in motion_diffs[motion_gaps_mask].index
            ]

            # cobertura de labels
            hr_span = float(hr['Timestamp'].iloc[-1] - hr['Timestamp'].iloc[0])
            label_span = float(len(expert_labels) * 30)
            label_coverage = label_span / hr_span if hr_span > 0 else np.nan

            # calidad de la señal dentro de la ventana etiquetada 
            # rec_start en hora local (America/New_York), hr/motion en Unix/UTC
            start = pd.Timestamp(str(rec_start), tz='America/New_York').timestamp()
            label_end = start + label_span

            hr_start = float(hr['Timestamp'].iloc[0])
            hr_end = float(hr['Timestamp'].iloc[-1])

            # ventana válida: intersección entre la ventana etiquetada y el rango de señal de hr
            valid_start = max(start, hr_start)
            valid_end = min(label_end, hr_end)

            # un gap cuya señal sólo reanuda FUERA de la ventana etiquetada no es un
            # gap interno: es el borde donde terminan (o empiezan) los datos continuos
            # de la noche, y lo que "reanuda" es grabación diurna ajena al sueño.
            # Recortamos la ventana válida a ese tramo continuo para que esos huecos
            # cuenten como truncamiento de extremo y no como gap interno.
            ts = hr['Timestamp'].values
            g0 = ts[:-1][np.diff(ts) > gap_threshold]  # último sample antes de cada gap
            g1 = ts[1:][np.diff(ts) > gap_threshold]    # primer sample tras cada gap
            out_right = g0[(g0 < valid_end) & (g1 > label_end)]
            if len(out_right):
                valid_end = min(valid_end, float(out_right.min()))
            out_left = g1[(g1 > valid_start) & (g0 < start)]
            if len(out_left):
                valid_start = max(valid_start, float(out_left.max()))

            # diferencias al inicio/fin entre señal y labels (se truncan siempre)
            leading_trunc_s = max(0.0, valid_start - start)
            trailing_trunc_s = max(0.0, label_end - valid_end)

            # gaps de hr dentro de la ventana válida, recortando cada gap a esa ventana:
            # sólo se cuenta la porción que cae en [valid_start, valid_end].
            gap_start = hr['Timestamp'].shift(1)
            gap_end = hr['Timestamp']
            gap_overlap = (np.minimum(gap_end, valid_end) - np.maximum(gap_start, valid_start)).clip(lower=0)
            internal_gap_s = float(gap_overlap.loc[hr_gaps_mask.index][hr_gaps_mask].sum())

            # valores NaN en las señales 
            hr_nan_count = int(hr[['Timestamp', 'hr']].isna().any(axis=1).sum())
            motion_nan_count = int(motion[['Timestamp', 'x', 'y', 'z']].isna().any(axis=1).sum())

            # acelerometría: sqrt(x^2 + y^2 + z^2) debería ser approx 1g
            acc_norm = np.sqrt(motion['x']**2 + motion['y']**2 + motion['z']**2)
            acc_invalid = (acc_norm - 1).abs() > acc_tol
            acc_invalid_count = int(acc_invalid.sum())
            acc_invalid_frac = float(acc_invalid.mean())

            # IHR valores nulos o absurdamente altos (sin contar bordes)
            if len(hr) > 2 * edge_trim:
                hr_vals = hr['hr'].iloc[edge_trim:-edge_trim]
            else:
                hr_vals = hr['hr']
            ihr_invalid = (hr_vals <= 0) | (hr_vals > ihr_max)
            ihr_invalid_count = int(ihr_invalid.sum())
            ihr_invalid_frac = float(ihr_invalid.mean()) if len(hr_vals) > 0 else np.nan

            records.append({
                'patient': patient,
                'night': int(night),
                'n_hr_samples': len(hr),
                'n_motion_samples': len(motion),
                'n_label_epochs': len(expert_labels),
                'hr_span_s': hr_span,
                'label_span_s': label_span,
                'label_coverage_frac': label_coverage,
                'hr_span_h': hr_span / 3600,
                'hr_nan_count': hr_nan_count,
                'motion_nan_count': motion_nan_count,
                'n_hr_gaps': len(hr_gaps),
                'max_hr_gap_s': max((g['duration'] for g in hr_gaps), default=0.0),
                'total_hr_gap_s': sum(g['duration'] for g in hr_gaps),
                'n_motion_gaps': len(motion_gaps),
                'max_motion_gap_s': max((g['duration'] for g in motion_gaps), default=0.0),
                'total_motion_gap_s': sum(g['duration'] for g in motion_gaps),
                'acc_invalid_count': acc_invalid_count,
                'acc_invalid_frac': acc_invalid_frac,
                'ihr_invalid_count': ihr_invalid_count,
                'ihr_invalid_frac': ihr_invalid_frac,
                'leading_trunc_s': leading_trunc_s,
                'trailing_trunc_s': trailing_trunc_s,
                'valid_start_s': valid_start,
                'valid_end_s': valid_end,
                'internal_gap_s': internal_gap_s,
                'hr_gaps': hr_gaps,
                'motion_gaps': motion_gaps,
            })

        if save_path is None:
            save_path = ANALYSIS_DIR / 'quality_report.json'
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)

        df = pd.DataFrame(records).drop(columns=['hr_gaps', 'motion_gaps'])
        return df

    @staticmethod
    def problematic_nights(quality_df: pd.DataFrame,
                           internal_gap_threshold: float = INTERNAL_GAP_THRESHOLD_S,
                           edge_trunc_threshold: float = EDGE_TRUNC_THRESHOLD_S):
        '''
        A partir del DataFrame devuelto por `quality_report`, identifica
        noches problemáticas y deja el registro en `analysis/problematic_nights.json`.

        Una noche se lista como problemática si tiene:
        - `internal_gap_s > internal_gap_threshold`: gaps de señal dentro de
          la ventana válida que rompen la continuidad temporal. Es el único
          criterio de *descarte/reparación* (se decide al construir el dataset).
        - `leading_trunc_s` o `trailing_trunc_s > edge_trunc_threshold`:
          truncamiento de extremo significativo. NO se descarta: la solución
          es recortar labels/señal a la ventana válida (no puede haber epochs
          etiquetadas sin HR). Se listan para ilustrar esa modificación.

        Devuelve una lista de dicts {patient, night, internal_gap_s,
        leading_trunc_s, trailing_trunc_s, valid_start_s, valid_end_s, hr_span_h}.
        '''
        bad = quality_df[
            (quality_df['internal_gap_s'] > internal_gap_threshold) |
            (quality_df['leading_trunc_s'] > edge_trunc_threshold) |
            (quality_df['trailing_trunc_s'] > edge_trunc_threshold)
        ]
        nights = bad[['patient', 'night', 'internal_gap_s', 'leading_trunc_s', 'trailing_trunc_s',
                      'valid_start_s', 'valid_end_s', 'hr_span_h']].to_dict('records')

        with open('../analysis/problematic_nights.json', 'w', encoding='utf-8') as f:
            json.dump({
                'criterion': 'internal_gap_s > internal_gap_threshold OR leading_trunc_s/trailing_trunc_s > edge_trunc_threshold',
                'internal_gap_threshold_s': INTERNAL_GAP_THRESHOLD_S,
                'edge_trunc_threshold_s': EDGE_TRUNC_THRESHOLD_S,
                'description': (
                    'Ventana valida = interseccion entre la ventana etiquetada '
                    '[recStart, recStart + label_span_s] y el rango de senial continua de '
                    'hr.csv (un gap que solo reanuda fuera del etiquetado se trata como borde, '
                    'no como gap interno). leading_trunc_s/trailing_trunc_s son las diferencias '
                    'entre labels y senial al inicio/fin; se truncan SIEMPRE al construir el '
                    'dataset (recortando a la ventana valida), y se listan como problematicas '
                    'cuando superan edge_trunc_threshold para ilustrar ese recorte. internal_gap_s '
                    'es la suma de gaps de hr.csv (>60s) dentro de la ventana valida, recortados a '
                    'esa ventana, y es el unico criterio de descarte/reparacion. hr_span_h es la '
                    'duracion total de la senial de hr en horas (informativa, no criterio).'
                ),
                'n_total_nights': len(quality_df),
                'n_problematic': len(nights),
                'problematic': [
                    {
                        'patient': int(d['patient']),
                        'night': int(d['night']),
                        'internal_gap_s': float(d['internal_gap_s']),
                        'leading_trunc_s': float(d['leading_trunc_s']),
                        'trailing_trunc_s': float(d['trailing_trunc_s']),
                        'valid_start_s': float(d['valid_start_s']),
                        'valid_end_s': float(d['valid_end_s']),
                        'hr_span_h': float(d['hr_span_h']),
                    }
                    for d in nights
                ],
            }, f, indent=2)

        return nights

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
