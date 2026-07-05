import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

try:
    from data import EDA
except ImportError:
    from src.data import EDA


def get_cnn_dataset(output_dir='../data_extraction/processed_data',
                    n_patients=None, n_steps=150):
    '''
    Arma los tensores por época para la CNN a partir de la alineación
    "fuente de verdad" de `data.py` (recStart convertido con timezone +
    ventana válida de `quality_report`), NO del primer sample de señal.

    Para cada noche:
    - `EDA.valid_windows()` da la ventana válida [valid_start, valid_end).
    - `EDA.internal_gap_resolution()` decide las noches con gap interno:
      `discard` se saltea entera; `trim_tail` recorta `valid_end` al prefijo
      contiguo cubierto.
    - `EDA.load_night_clean()` recorta hr/motion/labels a esa ventana y
      devuelve `aligned_start`, el timestamp de inicio de la primera época
      conservada. La época `i` es [aligned_start + i*30, aligned_start +
      (i+1)*30) y le corresponde `expert[i]` por construcción.

    Cada época se lleva a longitud fija (`n_steps` timesteps, 5 Hz) con 4
    canales [HR, mag_mean, mag_std, enmo_mean]. Se guarda un `.npz` por
    paciente con X (N, n_steps, 4) float32 e y (N,) int8.
    '''
    windows = EDA.valid_windows()
    gap_res = EDA.internal_gap_resolution()

    # agrupar las noches de cada paciente (las claves son (patient, night))
    nights_by_patient = defaultdict(list)
    for patient, night in sorted(windows):
        nights_by_patient[patient].append(night)

    patients = sorted(nights_by_patient)
    if n_patients:
        patients = patients[:n_patients]

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    bin_dur = 30 / n_steps   # 0.2 s por bin

    for patient in tqdm(patients, desc="Procesando pacientes"):
        patient_X, patient_y = [], []

        for night in nights_by_patient[patient]:
            valid_start, valid_end = windows[(patient, night)]

            # noches con gap interno: descartar o recortar la cola
            res = gap_res.get((patient, night))
            if res is not None:
                if res['action'] == 'discard':
                    continue
                if res['action'] == 'trim_tail' and res['new_valid_end'] is not None:
                    valid_end = res['new_valid_end']

            # carga alineada: hr/motion/labels ya recortados a la ventana válida
            hr, motion, _dreem, expert, aligned_start = EDA.load_night_clean(
                patient, night, valid_start, valid_end)

            for i, label in enumerate(expert):
                if label > 4:
                    continue  # se descarta Unknown (5)

                epoch_start = aligned_start + i * 30
                epoch_end = epoch_start + 30

                hr_win = hr[(hr['Timestamp'] >= epoch_start) & (hr['Timestamp'] < epoch_end)]
                acc_win = motion[(motion['Timestamp'] >= epoch_start) & (motion['Timestamp'] < epoch_end)]

                # tolerancia relajada para HR (baja frecuencia) y mínimo para Acc
                if len(hr_win) < 2 or len(acc_win) < 10:
                    continue

                # HR: interpolada sobre la grilla densa (solo tiene ~5 puntos reales)
                grid = np.linspace(epoch_start, epoch_end, n_steps, dtype=np.float32)
                hr_fixed = np.interp(grid, hr_win['Timestamp'], hr_win['hr'])

                # Acelerometría: magnitud invariante a orientación (x/y/z dependen de
                # cómo esté puesto el reloj). Por cada bin guardamos media y std de la
                # magnitud: el std es lo que codifica el movimiento y se pierde al promediar.
                acc_win = acc_win.copy()
                mag = np.sqrt(acc_win['x']**2 + acc_win['y']**2 + acc_win['z']**2)
                acc_win['mag'] = mag
                acc_win['enmo'] = np.maximum(mag - 1.0, 0.0)
                acc_win['bin'] = ((acc_win['Timestamp'] - epoch_start) / bin_dur).astype(int).clip(0, n_steps - 1)

                grouped = acc_win.groupby('bin')
                mag_mean = grouped['mag'].mean().reindex(range(n_steps), method='ffill').fillna(0).values
                mag_std  = grouped['mag'].std().reindex(range(n_steps), method='ffill').fillna(0).values
                enmo_mean = grouped['enmo'].mean().reindex(range(n_steps), method='ffill').fillna(0).values

                # Canales: [HR, mag_mean, mag_std, enmo_mean]
                patient_X.append(np.column_stack([hr_fixed, mag_mean, mag_std, enmo_mean]))
                patient_y.append(int(label))

        # guardado condicional por paciente
        if patient_X:
            file_path = os.path.join(output_dir, f"Bidslab{patient:02d}.npz")
            np.savez_compressed(file_path, X=np.array(patient_X, dtype=np.float32), y=np.array(patient_y, dtype=np.int8))
            print(f"-> Bidslab{patient:02d}: Guardado con {len(patient_X)} epochs.")
        else:
            print(f"-> Bidslab{patient:02d}: No se encontraron epochs válidos tras el filtrado.")

    print(f"--- Proceso Finalizado. Archivos listos en: {output_dir} ---")
