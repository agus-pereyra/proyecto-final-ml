import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import os
from tqdm import tqdm 

def get_cnn_dataset(data_root='../data/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0', 
                    output_dir='../data_extraction/processed_data', 
                    n_patients=None,
                    problem_file='../analysis/problematic_nights.json'):
    
    # 1. Cargar reporte de noches problemáticas
    problem_map = {}
    if os.path.exists(problem_file):
        with open(problem_file, 'r') as f:
            problems = json.load(f)
            if isinstance(problems, list):
                problem_map = {p['night_id']: p.get('modifications', []) for p in problems}
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = Path(data_root)
    patient_dirs = sorted([x for x in data_path.glob('Bidslab*') if x.is_dir()])
    if n_patients: 
        patient_dirs = patient_dirs[:n_patients]
    
    for p_dir in tqdm(patient_dirs, desc="Procesando pacientes"):
        patient_X, patient_y = [], []
        
        for night_dir in sorted([x for x in p_dir.iterdir() if x.is_dir()]):
            night_id = f"{p_dir.name}/{night_dir.name}"
            mods = problem_map.get(night_id, [])
            
            # Si tiene un gap interno grave, descartamos la noche
            if 'internal_gap' in mods:
                continue
            
            # Carga de archivos
            hr_df = pd.read_csv(night_dir / 'hr.csv', names=['ts', 'hr'], dtype=np.float64)
            acc_df = pd.read_csv(night_dir / 'motion.csv', dtype=np.float64)
            mat_data = loadmat(night_dir / 'labels.mat')
            labels = mat_data['expert_label'].flatten()
            
            # Sincronización real basada en el inicio de la señal
            real_start = max(hr_df['ts'].min(), acc_df['Timestamp'].min())
            real_end = min(hr_df['ts'].max(), acc_df['Timestamp'].max())
            
            for i, label in enumerate(labels):
                if label > 4: continue # Solo etapas de sueño válidas
                
                epoch_start = real_start + (i * 30)
                epoch_end = epoch_start + 30
                
                # Solución para 'signal_excess': cortamos si superamos el fin de grabación
                if epoch_end > real_end: break
                
                hr_win = hr_df[(hr_df['ts'] >= epoch_start) & (hr_df['ts'] < epoch_end)]
                acc_win = acc_df[(acc_df['Timestamp'] >= epoch_start) & (acc_df['Timestamp'] < epoch_end)]
                
                # --- AQUÍ ESTÁ EL CAMBIO CLAVE ---
                # Tolerancia relajada para HR (baja frecuencia) y requerimiento para Acc
                if len(hr_win) < 2 or len(acc_win) < 10: continue
                
                # 2. Procesamiento (Interpolación y Binning)
                grid = np.linspace(epoch_start, epoch_end, 30, dtype=np.float32)
                hr_fixed = np.interp(grid, hr_win['ts'], hr_win['hr'])
                
                acc_win = acc_win.copy()
                acc_win['sec'] = ((acc_win['Timestamp'] - epoch_start) // 1).astype(int)
                acc_fixed = acc_win.groupby('sec')[['x', 'y', 'z']].mean().reindex(range(30), method='ffill').fillna(0).values
                
                patient_X.append(np.column_stack([hr_fixed, acc_fixed]))
                patient_y.append(int(label))
        
        # 3. Guardado condicional
        if patient_X:
            file_path = os.path.join(output_dir, f"{p_dir.name}.npz")
            np.savez_compressed(file_path, X=np.array(patient_X, dtype=np.float32), y=np.array(patient_y, dtype=np.int8))
            print(f"-> {p_dir.name}: Guardado con {len(patient_X)} epochs.")
        else:
            print(f"-> {p_dir.name}: No se encontraron epochs válidos tras el filtrado.")

    print(f"--- Proceso Finalizado. Archivos listos en: {output_dir} ---")