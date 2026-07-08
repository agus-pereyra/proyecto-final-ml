'''
Carga, control de calidad y recorte del dataset BIDSleep.

Expone `EDA` (métodos estáticos para leer cada noche, computar el reporte de
calidad, definir la ventana válida y resolver los gaps internos) y `DataSet`
(Dataset de PyTorch para servir épocas desde un manifiesto). Todo el
recorte/limpieza se hace en memoria: nunca se modifican los archivos del dataset.
Las constantes de módulo (umbrales de gap, tolerancias, etc.) fijan los criterios
de calidad usados por `quality_report`/`problematic_nights`.
'''

import json
import pandas as pd
import scipy.io as sio
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from plotly.subplots import make_subplots

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
MIN_CLEAN_PREFIX_FRAC = 0.6      # prefijo contiguo cubierto mínimo para recortar la cola (en vez de descartar) una noche con gaps internos
EDGE_TRUNC_THRESHOLD_S = 3600    # 1 hora: truncamiento de extremo a partir del cual la noche se grafica en el overview
ALIGN_TOL_S = 30                 # 1 epoch: por debajo de esto señal y labels se consideran ya alineados (no se lista como modificada)

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
    Métodos estáticos para el análisis exploratorio y el procesamiento de los
    datos: carga por noche, reporte de calidad, ventana válida y resolución de
    gaps internos. No mantiene estado; cada método opera sobre los archivos del
    dataset (leídos, nunca escritos) o sobre el `quality_report.json` derivado.
    '''
    @staticmethod
    def load_night(patient: int, night: int):
        '''
        Carga una noche cruda (sin recortar a la ventana válida), filtrando filas
        con timestamps corruptos (<=1e9).
        In:  patient, night.
        Out: (hr, motion, dreem_labels, expert_labels, rec_start); rec_start en
             hora local America/New_York.
        '''
        path = DATA_PATH / f'Bidslab{patient:02d}' / f'{night}'
        hr = pd.read_csv(path / 'hr.csv', header=None, names=['Timestamp', 'hr'])
        motion = pd.read_csv(path / 'motion.csv')

        # algunas noches traen filas corruptas (p. ej. P42 N4 termina con un
        # Timestamp = 1.0 y x/y/z = NaN). Descartamos timestamps no plausibles
        # (NaN o fuera del rango Unix esperable) para limpiar todo el pipeline.
        hr = hr[hr['Timestamp'] > 1e9].reset_index(drop=True)
        motion = motion[motion['Timestamp'] > 1e9].reset_index(drop=True)

        # pasaje timestamps de unix a segundos
        hr['datetime'] = pd.to_datetime(hr['Timestamp'], unit='s')
        motion['datetime'] = pd.to_datetime(motion['Timestamp'], unit='s')

        mat = sio.loadmat(path / 'labels.mat')

        dreem_labels = mat['dreem_label'].flatten()
        expert_labels = mat['expert_label'].flatten()
        rec_start = mat['recStart'][0]
        
        return hr, motion, dreem_labels, expert_labels, rec_start

    @staticmethod
    def valid_windows(report_path: Path = None):
        '''
        Lee el `quality_report.json` y devuelve un dict
        {(patient, night): (valid_start_s, valid_end_s)} con la ventana
        válida de cada noche (intersección señal/labels ya calculada por
        `quality_report`). Es la fuente de verdad del recorte en memoria;
        tanto la extracción de features como el armado de secuencias la usan.
        '''
        if report_path is None:
            report_path = ANALYSIS_DIR / 'quality_report.json'
        with open(report_path, encoding='utf-8') as f:
            records = json.load(f)
        return {(r['patient'], r['night']): (r['valid_start_s'], r['valid_end_s'])
                for r in records}

    @staticmethod
    def load_night_clean(patient: int, night: int,
                         valid_start: float = None, valid_end: float = None):
        '''
        Carga una noche y la recorta en memoria a su ventana válida, sin
        tocar los archivos. `valid_start`/`valid_end` deberían venir de
        `valid_windows()` (quality_report); si se omiten, se usa la
        intersección simple entre la ventana etiquetada y el rango de hr.

        Recorta hr y motion a [valid_start, valid_end) y las labels a las
        epochs cuyo inicio cae dentro de esa ventana, devolviendo además el
        timestamp de inicio de la primera epoch conservada (ya alineado).

        Devuelve (hr, motion, dreem, expert, aligned_start).
        '''
        hr, motion, dreem, expert, rec_start = EDA.load_night(patient, night)
        start = pd.Timestamp(str(rec_start), tz='America/New_York').timestamp()

        if valid_start is None:
            valid_start = max(start, float(hr['Timestamp'].iloc[0]))
        if valid_end is None:
            valid_end = min(start + len(expert) * 30, float(hr['Timestamp'].iloc[-1]))

        hr = hr[(hr['Timestamp'] >= valid_start) & (hr['Timestamp'] < valid_end)].reset_index(drop=True)
        motion = motion[(motion['Timestamp'] >= valid_start) & (motion['Timestamp'] < valid_end)].reset_index(drop=True)

        i0 = int(max(0, np.ceil((valid_start - start) / 30)))
        i1 = int(np.floor((valid_end - start) / 30))
        expert = expert[i0:i1]
        dreem = dreem[i0:i1]
        aligned_start = start + i0 * 30

        return hr, motion, dreem, expert, aligned_start

    @staticmethod
    def class_distribution():
        '''
        Cuenta las épocas de cada etapa (0..5) sobre todas las noches, para
        experto y Dreem.
        In:  (nada).
        Out: {'expert': array[6], 'dreem': array[6]} (conteos por clase).
        '''
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
        '''
        Concatena las etiquetas de expert y dreem de todas las noches (recortando
        cada una al largo común de ambos, que en 2 noches difieren).
        In:  (nada).
        Out: (expert, dreem), dos arrays 1D alineados época a época.
        '''
        expert_labels = []
        dreem_labels = []

        for patient in PATIENCE_NUMBERS:
            patient_dir = DATA_PATH / f'Bidslab{patient:02d}'
            for night_dir in patient_dir.iterdir():
                if not night_dir.is_dir():
                    continue

                mat = sio.loadmat(night_dir / 'labels.mat')
                exp = mat['expert_label'].flatten()
                dre = mat['dreem_label'].flatten()
                # en 2 noches expert y dreem difieren en largo; se recortan al
                # común para mantener la comparación alineada epoch a epoch
                m = min(len(exp), len(dre))
                expert_labels.append(exp[:m])
                dreem_labels.append(dre[:m])

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

        In:  gap_threshold, acc_tol, ihr_max, edge_trim, save_path (todos opc.).
        Out: nada; escribe `analysis/quality_report.json` con un registro por noche
             (ventana válida, gaps, coberturas, conteos de inválidos, etc.).
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
            # descartar filas con timestamps corruptos (misma limpieza que load_night; p. ej.
            # Bidslab42/4/motion.csv termina con Timestamp=1.0). Necesario ahora que la ventana
            # usa también los bordes de motion: sin esto mo_end=1.0 rompería la ventana.
            hr = hr[hr['Timestamp'] > 1e9].reset_index(drop=True)
            motion = motion[motion['Timestamp'] > 1e9].reset_index(drop=True)
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
            mo_start = float(motion['Timestamp'].iloc[0])
            mo_end = float(motion['Timestamp'].iloc[-1])

            # ventana válida: intersección entre la ventana etiquetada y el rango de señal
            # de AMBAS señales (hr y acelerometría). Los modelos usan IHR + acelerometría,
            # así que un tramo con una sola de las dos NO es válido (p. ej. la acelerometría
            # que se corta antes que el IHR al final de la noche).
            valid_start = max(start, hr_start, mo_start)
            valid_end = min(label_end, hr_end, mo_end)

            # un gap cuya señal sólo reanuda FUERA de la ventana etiquetada no es un
            # gap interno: es el borde donde terminan (o empiezan) los datos continuos
            # de la noche, y lo que "reanuda" es grabación diurna ajena al sueño.
            # Recortamos la ventana válida a ese tramo continuo (para hr y para motion).
            def _trim_edges(ts_arr, vs, ve):
                d = np.diff(ts_arr)
                g0 = ts_arr[:-1][d > gap_threshold]  # último sample antes de cada gap
                g1 = ts_arr[1:][d > gap_threshold]   # primer sample tras cada gap
                out_right = g0[(g0 < ve) & (g1 > label_end)]
                if len(out_right):
                    ve = min(ve, float(out_right.min()))
                out_left = g1[(g1 > vs) & (g0 < start)]
                if len(out_left):
                    vs = max(vs, float(out_left.max()))
                return vs, ve
            valid_start, valid_end = _trim_edges(hr['Timestamp'].values, valid_start, valid_end)
            valid_start, valid_end = _trim_edges(motion['Timestamp'].values, valid_start, valid_end)

            # diferencias al inicio/fin entre señal y labels (se truncan siempre)
            # labels sin señal (se recortan las labels):
            leading_trunc_s = max(0.0, valid_start - start)
            trailing_trunc_s = max(0.0, label_end - valid_end)
            # señal sin label, fuera de la ventana etiquetada (se recorta la señal):
            head_excess_s = max(0.0, start - hr_start)
            tail_excess_s = max(0.0, hr_end - label_end)

            # gaps internos dentro de la ventana válida, recortando cada gap a esa ventana
            # (sólo la porción que cae en [valid_start, valid_end]). Se calcula por señal:
            # un hueco de IHR o de acelerometría invalida ese tramo por igual.
            def _internal_gap(ts_series, gaps_mask):
                gs = ts_series.shift(1)
                ge = ts_series
                ov = (np.minimum(ge, valid_end) - np.maximum(gs, valid_start)).clip(lower=0)
                return float(ov.loc[gaps_mask.index][gaps_mask].sum())
            internal_gap_s = _internal_gap(hr['Timestamp'], hr_gaps_mask)
            motion_internal_gap_s = _internal_gap(motion['Timestamp'], motion_gaps_mask)

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
                'head_excess_s': head_excess_s,
                'tail_excess_s': tail_excess_s,
                'valid_start_s': valid_start,
                'valid_end_s': valid_end,
                'internal_gap_s': internal_gap_s,
                'motion_internal_gap_s': motion_internal_gap_s,
                'hr_gaps': hr_gaps,
                'motion_gaps': motion_gaps,
            })

        if save_path is None:
            save_path = ANALYSIS_DIR / 'quality_report.json'
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        return

    @staticmethod
    def problematic_nights(quality_df: pd.DataFrame,
                           align_tol: float = ALIGN_TOL_S,
                           internal_gap_threshold: float = INTERNAL_GAP_THRESHOLD_S):
        '''
        A partir del DataFrame devuelto por `quality_report`, lista todas las
        noches que requieren alguna modificación al construir el dataset y deja
        el registro en `analysis/problematic_nights.json`. Sólo las noches ya
        alineadas (señal y labels sobre el mismo período, dentro de `align_tol`)
        quedan fuera.

        Cada noche se etiqueta con las modificaciones que necesita:
        - `signal_excess`: hay señal fuera de la ventana etiquetada
          (`head_excess_s`/`tail_excess_s`). Solución: recortar la señal.
        - `label_trunc`: hay labels sin señal al inicio/fin
          (`leading_trunc_s`/`trailing_trunc_s`). Solución: recortar las labels.
        - `internal_gap`: hay gaps de señal dentro de la ventana válida
          (`internal_gap_s > internal_gap_threshold`). Solución: descartar o
          reparar la noche (se decide al construir el dataset).

        Todas las modificaciones se aplican en memoria, sin tocar los archivos.

        Devuelve la lista de dicts (una por noche modificada).
        '''
        q = quality_df.copy()
        if 'motion_internal_gap_s' not in q.columns:
            q['motion_internal_gap_s'] = 0.0
        needs_signal_trim = (q['head_excess_s'] > align_tol) | (q['tail_excess_s'] > align_tol)
        needs_label_trim = (q['leading_trunc_s'] > align_tol) | (q['trailing_trunc_s'] > align_tol)
        # solo el gap interno de IHR dispara reparación/descarte (rompe la secuencia). Los
        # gaps de acelerometría se resuelven dropeando épocas, no la noche (ver
        # internal_gap_resolution); motion_internal_gap_s queda como campo informativo.
        needs_gap_fix = q['internal_gap_s'] > internal_gap_threshold
        bad = q[needs_signal_trim | needs_label_trim | needs_gap_fix]

        cols = ['patient', 'night', 'leading_trunc_s', 'trailing_trunc_s',
                'head_excess_s', 'tail_excess_s', 'internal_gap_s', 'motion_internal_gap_s',
                'valid_start_s', 'valid_end_s', 'hr_span_h']
        def mods(d):
            m = []
            if d['head_excess_s'] > align_tol or d['tail_excess_s'] > align_tol:
                m.append('signal_excess')
            if d['leading_trunc_s'] > align_tol or d['trailing_trunc_s'] > align_tol:
                m.append('label_trunc')
            if d['internal_gap_s'] > internal_gap_threshold:
                m.append('internal_gap')
            return m

        nights = bad[cols].to_dict('records')
        for d in nights:
            d['modifications'] = mods(d)

        with open('../analysis/problematic_nights.json', 'w', encoding='utf-8') as f:
            json.dump({
                'criterion': 'cualquier desalineacion senial/labels > align_tol, o internal_gap_s > internal_gap_threshold',
                'align_tol_s': ALIGN_TOL_S,
                'internal_gap_threshold_s': INTERNAL_GAP_THRESHOLD_S,
                'edge_trunc_threshold_s': EDGE_TRUNC_THRESHOLD_S,
                'description': (
                    'Lista de noches que requieren modificacion al construir el dataset. '
                    'Ventana valida = interseccion entre la ventana etiquetada [recStart, '
                    'recStart + label_span_s] y el rango de senial continua de hr.csv (un gap '
                    'que solo reanuda fuera del etiquetado se trata como borde, no como gap '
                    'interno). Modificaciones: signal_excess = hay senial fuera del etiquetado '
                    '(head/tail_excess), se recorta la senial; label_trunc = hay labels sin '
                    'senial al inicio/fin (leading/trailing_trunc), se recortan las labels; '
                    'internal_gap = gaps de hr dentro de la ventana valida (>internal_gap_threshold), '
                    'se descarta o repara. Todo se aplica en memoria, sin tocar los archivos. '
                    'hr_span_h es la duracion total de la senial de hr en horas (informativa).'
                ),
                'n_total_nights': len(quality_df),
                'n_problematic': len(nights),
                'problematic': [
                    {
                        'patient': int(d['patient']),
                        'night': int(d['night']),
                        'modifications': mods(d),
                        'leading_trunc_s': float(d['leading_trunc_s']),
                        'trailing_trunc_s': float(d['trailing_trunc_s']),
                        'head_excess_s': float(d['head_excess_s']),
                        'tail_excess_s': float(d['tail_excess_s']),
                        'internal_gap_s': float(d['internal_gap_s']),
                        'motion_internal_gap_s': float(d.get('motion_internal_gap_s', 0.0)),
                        'valid_start_s': float(d['valid_start_s']),
                        'valid_end_s': float(d['valid_end_s']),
                        'hr_span_h': float(d['hr_span_h']),
                    }
                    for d in nights
                ],
            }, f, indent=2)

        return nights

    @staticmethod
    def internal_gap_resolution(quality_df: pd.DataFrame = None,
                                internal_gap_threshold: float = INTERNAL_GAP_THRESHOLD_S,
                                min_clean_prefix_frac: float = MIN_CLEAN_PREFIX_FRAC):
        '''
        Decide cómo tratar cada noche con gaps internos (`internal_gap_s >
        internal_gap_threshold`) según DÓNDE caen las epochs sin cobertura de hr
        dentro de su ventana válida. Una epoch está "cubierta" si su ventana de
        30 s contiene al menos un sample de hr.

        - `trim_tail`: las epochs sin cobertura están agrupadas al final. El
          prefijo contiguo cubierto `[valid_start, valid_start + first_gap)`
          abarca >= `min_clean_prefix_frac` de la noche; se conserva ese prefijo
          (100% cubierto, sin huecos internos → apto para secuencias) y se
          descarta la cola recortando `valid_end` a `new_valid_end`.
        - `discard`: los gaps son interiores o dispersos (el prefijo limpio es
          menor a `min_clean_prefix_frac`). Un hueco en el medio rompe la
          continuidad temporal (perjudicial para modelos secuenciales tipo
          LSTM), por lo que la noche se descarta entera.

        No se colapsan los gaps: el diagnóstico muestra que no son artefactos de
        timestamps sino dropouts reales de medición (saltos de hasta ~60 bpm a
        través del hueco), y colapsar el tiempo desalinearía las labels (ancladas
        a tiempo absoluto, 30 s por epoch).

        Si `quality_df` es None se lee de `quality_report.json`. Devuelve
        `{(patient, night): {'action', 'new_valid_end', 'clean_prefix_frac',
        'n_uncovered', 'n_epochs'}}`.
        '''
        if quality_df is None:
            with open(ANALYSIS_DIR / 'quality_report.json', encoding='utf-8') as f:
                quality_df = pd.DataFrame(json.load(f)).drop(
                    columns=['hr_gaps', 'motion_gaps'], errors='ignore')

        # La decisión trim_tail/discard se basa SOLO en la continuidad del IHR: un gap
        # contiguo de IHR rompe la secuencia y hay que reparar/descartar. Los dropouts de
        # acelerometría (dispersos) NO descartan la noche: los resuelve el filtro por época
        # de feature_extraction / build_night_sequences (se dropea la época, no la noche), y
        # el recorte de acc al borde ya lo hizo la ventana válida (quality_report).
        ig = quality_df[quality_df['internal_gap_s'] > internal_gap_threshold]
        res = {}
        for _, r in ig.iterrows():
            p, n = int(r['patient']), int(r['night'])
            hr, _, _, expert, al = EDA.load_night_clean(
                p, n, r['valid_start_s'], r['valid_end_s'])
            ts = hr['Timestamp'].values
            n_ep = len(expert)
            starts = al + np.arange(n_ep) * 30
            covered = np.searchsorted(ts, starts + 30) > np.searchsorted(ts, starts)
            uncov = ~covered
            first_gap = int(np.argmax(uncov)) if uncov.any() else n_ep
            frac = first_gap / n_ep if n_ep else 0.0
            if frac >= min_clean_prefix_frac:
                action, new_end = 'trim_tail', float(al + first_gap * 30)
            else:
                action, new_end = 'discard', None
            res[(p, n)] = {
                'action': action,
                'new_valid_end': new_end,
                'clean_prefix_frac': frac,
                'n_uncovered': int(uncov.sum()),
                'n_epochs': n_ep,
            }
        return res

    @staticmethod
    def plot_motion_comparison_3d(patient: int, night: int, step: int = 1000):
        '''
        Scatter 3D interactivo (plotly) de la acelerometría (x, y, z) de una
        noche, coloreada por etapa, en dos paneles: etiquetas del experto vs.
        Dreem. `step` submuestrea las filas para aligerar el render. Muestra la
        figura; no devuelve nada.
        '''
        # 1. Cargar datos
        path = DATA_PATH / f'Bidslab{patient:02d}' / f'{night}'
        motion = pd.read_csv(path / 'motion.csv')
        mat = sio.loadmat(path / 'labels.mat')
        
        # 2. Definir paleta de colores fija
        color_map = {
            'Wake': '#d62728', 'N1': '#ff7f0e', 'N2': '#bcbd22', 
            'N3': '#2ca02c', 'REM': '#1f77b4', 'Unknown': '#7f7f7f'
        }
        
        expert_labels = mat['expert_label'].flatten()
        dreem_labels = mat['dreem_label'].flatten()
        
        # 3. Sincronización
        start_time = motion['Timestamp'].iloc[0]
        motion['epoch_idx'] = ((motion['Timestamp'] - start_time) // 30).astype(int)
        
        # Asignar etiquetas evitando errores de índice
        def get_label(idx, labels):
            return STAGES_LABELS.get(labels[idx], 'Unknown') if idx < len(labels) else 'Unknown'

        motion['expert_stage'] = motion['epoch_idx'].apply(lambda x: get_label(x, expert_labels))
        motion['dreem_stage'] = motion['epoch_idx'].apply(lambda x: get_label(x, dreem_labels))
        
        df_plot = motion.iloc[::step, :]
        
        # 4. Crear figura
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Expert Labels', 'Dreem Labels')
        )
        
        # 5. Agregar trazos (filtramos solo etapas presentes en los datos para evitar errores)
        for stage in STAGES_LABELS.values():
            # Expert
            d_exp = df_plot[df_plot['expert_stage'] == stage]
            if not d_exp.empty:
                fig.add_trace(
                    dict(type='scatter3d', x=d_exp['x'], y=d_exp['y'], z=d_exp['z'], 
                         mode='markers', name=stage, legendgroup=stage,
                         marker=dict(size=2, color=color_map.get(stage, 'black'))),
                    row=1, col=1
                )
            
            # Dreem
            d_dre = df_plot[df_plot['dreem_stage'] == stage]
            if not d_dre.empty:
                fig.add_trace(
                    dict(type='scatter3d', x=d_dre['x'], y=d_dre['y'], z=d_dre['z'], 
                         mode='markers', name=stage, legendgroup=stage, 
                         showlegend=False,
                         marker=dict(size=2, color=color_map.get(stage, 'black'))),
                    row=1, col=2
                )
        
        fig.update_layout(title=f'Classification comparison: patient {patient} - night {night} - step = {step}', height=600)
        fig.show()

class DataSet(torch.utils.data.Dataset):
    '''Sirve épocas desde un manifiesto de tuplas (path, epoch_idx, label).'''
    def __init__(self, manifest):
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        path, epoch_idx, label = self.manifest[idx]
        epoch = np.load(path, mmap_mode='r')[epoch_idx]
        return torch.tensor(epoch, dtype=torch.float32), label
