import json
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from tqdm.auto import tqdm

try:
    from data import EDA, DATA_PATH, PATIENCE_NUMBERS, ANALYSIS_DIR
except ImportError:
    from src.data import EDA, DATA_PATH, PATIENCE_NUMBERS, ANALYSIS_DIR

# --- parámetros de las features ---
ACC_IMMOBILITY_TOL = 0.05   # | ||a|| - 1| por debajo de esto => muestra "inmóvil" [g]
MOVE_ENMO_THRESHOLD = 0.05  # ENMO medio por encima de esto => época con "movimiento" [g]
PNN_THRESHOLD_MS = 50.0     # umbral de pNN50 sobre intervalos RR [ms]
LAGS = (1, 2)               # desfasajes para las features de contexto [épocas]
ROLL_WINDOW = 5             # ventana centrada (±2 épocas) para estadísticas móviles


def _epoch_hr_features(hr, ts):
    '''Features intra-época del IHR. `hr` en bpm, `ts` timestamps (s).'''
    hr = hr[(hr > 0) & np.isfinite(hr)]
    n = len(hr)
    if n == 0:
        return dict(hr_mean=0.0, hr_std=0.0, hr_median=0.0, hr_iqr=0.0,
                    hr_rmssd=0.0, hr_pnn50=0.0, hr_slope=0.0, hr_ptp=0.0,
                    n_beats=0)

    hr_mean = float(np.mean(hr))
    hr_std = float(np.std(hr))                       # SDNN aproximado
    hr_median = float(np.median(hr))
    hr_iqr = float(np.percentile(hr, 75) - np.percentile(hr, 25))
    hr_ptp = float(np.ptp(hr))

    # HRV sobre intervalos RR (ms) derivados del IHR: RR = 60000 / bpm
    if n > 1:
        rr = 60000.0 / hr
        drr = np.diff(rr)
        hr_rmssd = float(np.sqrt(np.mean(drr ** 2)))
        hr_pnn50 = float(np.mean(np.abs(drr) > PNN_THRESHOLD_MS))
        t = ts[:n] - ts[0]
        hr_slope = float(np.polyfit(t, hr, 1)[0]) if np.ptp(t) > 0 else 0.0
    else:
        hr_rmssd = hr_pnn50 = hr_slope = 0.0

    return dict(hr_mean=hr_mean, hr_std=hr_std, hr_median=hr_median, hr_iqr=hr_iqr,
                hr_rmssd=hr_rmssd, hr_pnn50=hr_pnn50, hr_slope=hr_slope, hr_ptp=hr_ptp,
                n_beats=n)


def _epoch_accel_features(mag):
    '''
    Features intra-época de la acelerometría, todas a partir de la magnitud
    del vector ||a|| = sqrt(x²+y²+z²) 
    '''
    n = len(mag)
    if n == 0:
        return dict(enmo_mean=0.0, enmo_std=0.0, acc_std=0.0, acc_ptp=0.0,
                    immobility_frac=1.0, jerk_std=0.0)

    enmo = np.maximum(mag - 1.0, 0.0)  # aceleración dinámica neta (sin gravedad)
    return dict(
        enmo_mean=float(np.mean(enmo)),
        enmo_std=float(np.std(enmo)),
        acc_std=float(np.std(mag)),
        acc_ptp=float(np.ptp(mag)),
        immobility_frac=float(np.mean(np.abs(mag - 1.0) < ACC_IMMOBILITY_TOL)),
        jerk_std=float(np.std(np.diff(mag))) if n > 1 else 0.0,
    )


def _add_temporal_features(dfn, base_cols, lags=LAGS, roll=ROLL_WINDOW):
    '''
    Agrega features de contexto entre épocas (calculadas dentro de la noche,
    respetando el orden temporal): lags/leads, diferencia con la época previa,
    estadísticas móviles centradas y épocas desde el último movimiento.

    Los valores en los bordes (lags/leads/delta sin vecino) quedan como NaN;
    XGBoost los maneja de forma nativa.
    '''
    new = {}
    for c in base_cols:
        s = dfn[c]
        for l in lags:
            new[f'{c}_lag{l}'] = s.shift(l)
            new[f'{c}_lead{l}'] = s.shift(-l)
        new[f'{c}_delta1'] = s.diff()
        new[f'{c}_rmean'] = s.rolling(roll, center=True, min_periods=1).mean()
        new[f'{c}_rstd'] = s.rolling(roll, center=True, min_periods=1).std().fillna(0.0)

    # épocas transcurridas desde el último movimiento grande
    move = (dfn['enmo_mean'] > MOVE_ENMO_THRESHOLD).values
    since = np.empty(len(move))
    cnt = len(move)  # sin movimiento previo => valor grande
    for i, m in enumerate(move):
        cnt = 0 if m else cnt + 1
        since[i] = cnt
    new['epochs_since_move'] = since

    return pd.concat([dfn, pd.DataFrame(new, index=dfn.index)], axis=1)


def feature_extraction(output_path: Path = None, skip_internal_gap: bool = True):
    '''
    Convierte cada noche a una tabla de épocas (30 s) con features de IHR y
    acelerometría para modelos tabulares (XGBoost).

    Las features de acelerometría se calculan sobre la magnitud del vector
    (invariante a la orientación del reloj). Además de las features intra-época
    se agregan features de contexto entre épocas (lags, deltas, ventanas
    móviles). El recorte a la ventana válida se aplica en memoria vía
    `EDA.load_night_clean` (fuente de verdad: `quality_report.json`), sin tocar
    los CSV. Las noches con gaps internos (`internal_gap`) se descartan.
    '''
    root_dir = Path(__file__).resolve().parent.parent
    if output_path is None:
        output_path = root_dir / 'data' / 'epoch_features.csv'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)  # se reescribe de cero en cada corrida

    # ventanas válidas y noches a descartar, desde el reporte de calidad
    valid_windows = EDA.valid_windows()
    skip = set()
    if skip_internal_gap:
        with open(ANALYSIS_DIR / 'problematic_nights.json', encoding='utf-8') as f:
            prob = json.load(f)
        skip = {(e['patient'], e['night']) for e in prob['problematic']
                if 'internal_gap' in e['modifications']}

    nights = []
    for patient in PATIENCE_NUMBERS:
        patient_dir = DATA_PATH / f'Bidslab{patient:02d}'
        for night_dir in sorted(patient_dir.iterdir()):
            if night_dir.is_dir() and (patient, int(night_dir.name)) not in skip:
                nights.append((patient, int(night_dir.name)))

    header_written = False
    pbar = tqdm(nights, unit='night')
    for patient, night in pbar:
        pbar.set_description(f'P{patient:02d}-N{night}')

        vs, ve = valid_windows.get((patient, night), (None, None))
        hr, motion, dreem, expert, start = EDA.load_night_clean(patient, night, vs, ve)

        hr_ts = hr['Timestamp'].values
        hr_val = hr['hr'].values
        mo_ts = motion['Timestamp'].values
        mag = np.sqrt(motion['x'].values ** 2 + motion['y'].values ** 2 + motion['z'].values ** 2)

        n_ep = len(expert)
        rows = []
        for i in range(n_ep):
            t0 = start + i * 30
            t1 = t0 + 30
            hm = (hr_ts >= t0) & (hr_ts < t1)
            am = (mo_ts >= t0) & (mo_ts < t1)

            feat = {}
            feat.update(_epoch_hr_features(hr_val[hm], hr_ts[hm]))
            feat.update(_epoch_accel_features(mag[am]))
            feat['epoch_frac'] = i / n_ep if n_ep > 0 else 0.0
            rows.append(feat)

        dfn = pd.DataFrame(rows)
        base_cols = [c for c in dfn.columns if c != 'epoch_frac']
        dfn = _add_temporal_features(dfn, base_cols)

        dfn.insert(0, 'epoch', np.arange(n_ep))
        dfn.insert(0, 'night', night)
        dfn.insert(0, 'subject', patient)
        dfn['label'] = [int(x) for x in expert]   # etiqueta del experto (target)
        dfn['dreem'] = [int(x) for x in dreem]    # etiqueta de Dreem (referencia)

        dfn.to_csv(output_path, mode='w' if not header_written else 'a',
                   header=not header_written, index=False)
        header_written = True

        del hr, motion, dreem, dfn, rows
        gc.collect()
