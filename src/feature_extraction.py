import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from datetime import datetime
import gc

def extract_features(signal):
    if len(signal) == 0 or np.isnan(signal).any():
        return [0, 0]
    return [np.mean(signal), np.std(signal)]

def extract_features(signal):
    """Extrae features estadísticas enriquecidas."""
    signal = pd.to_numeric(pd.Series(signal), errors='coerce').dropna().values
    if len(signal) == 0:
        return [0, 0, 0, 0, 0] # Ajustado a 5 features
    
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    ptp_val = np.ptp(signal)
    energy = np.sum(signal**2) / len(signal) # Energía normalizada
    p90 = np.percentile(signal, 90)
    
    return [mean_val, std_val, ptp_val, energy, p90]

def feature_extraction():
    
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    output_path = root_dir / "data_extraction/dataset_extraction.csv"
    
    cols = ['subject', 'night', 'hr_mean', 'hr_std', 
            'x_mean', 'x_std', 'y_mean', 'y_std', 'z_mean', 'z_std', 'label']
    
    features_per_sensor = ['_mean', '_std', '_ptp', '_energy', '_p90']
    sensors = ['hr', 'x', 'y', 'z']
    cols = ['subject', 'night'] + [s + f for s in sensors for f in features_per_sensor] + ['label']

    pd.DataFrame(columns=cols).to_csv(output_path, index=False)

    # Obtenemos los pacientes UNA SOLA VEZ
    pacientes = [p for p in data_dir.iterdir() if p.is_dir()]
    print(f"DEBUG: Encontré {len(pacientes)} pacientes.")
    
    for p_path in pacientes:
        subject = str(p_path.name)
        noches = [n for n in p_path.iterdir() if n.is_dir()]
        
        for n_path in noches:
            night = str(n_path.name)
            mat_path = n_path / "labels.mat"
            
            if not mat_path.exists() or not (n_path / 'hr.csv').exists():
                continue

            print(f"Procesando: {subject} - Noche {night}")

            try:
                # low_memory=False soluciona el DtypeWarning
                hr_df = pd.read_csv(n_path / 'hr.csv', header=None, names=['ts', 'hr'], low_memory=False)
                motion_df = pd.read_csv(n_path / 'motion.csv', names=['ts', 'x', 'y', 'z'], low_memory=False)
                
                # Convertimos columnas a numérico explícitamente
                hr_df['ts'] = pd.to_numeric(hr_df['ts'], errors='coerce')
                motion_df['ts'] = pd.to_numeric(motion_df['ts'], errors='coerce')
            except Exception as e:
                print(f"Error procesando {n_path}: {e}")
                continue
            
            mat_data = loadmat(mat_path)
            clave_experto = next((k for k in mat_data.keys() if 'expert' in k.lower()), None)
            if clave_experto is None: continue
            
            expert_labels = mat_data[clave_experto].flatten()
            
            raw_rec_start = mat_data['recStart'][0] 
            dt_object = datetime.strptime(str(raw_rec_start), '%Y-%m-%d %H:%M:%S')
            rec_start = dt_object.timestamp()
            
            batch_data = []
            for i in range(len(expert_labels)):
                
                start_t = rec_start + (i * 30)
                end_t = start_t + 30
                
                hr_data = hr_df[(hr_df['ts'] >= start_t) & (hr_df['ts'] < end_t)]['hr']
                # Convertimos a numérico, forzando a que lo que no sea número se vuelva NaN
                hr_win = pd.to_numeric(hr_data, errors='coerce').values
                
                # Repetimos la lógica para motion
                x_data = motion_df[(motion_df['ts'] >= start_t) & (motion_df['ts'] < end_t)]['x']
                y_data = motion_df[(motion_df['ts'] >= start_t) & (motion_df['ts'] < end_t)]['y']
                z_data = motion_df[(motion_df['ts'] >= start_t) & (motion_df['ts'] < end_t)]['z']
                
                feat_hr = extract_features(pd.to_numeric(hr_data, errors='coerce').values)
                feat_x = extract_features(pd.to_numeric(x_data, errors='coerce').values)
                feat_y = extract_features(pd.to_numeric(y_data, errors='coerce').values)
                feat_z = extract_features(pd.to_numeric(z_data, errors='coerce').values)
                
                batch_data.append([subject, night] + feat_hr + feat_x + feat_y + feat_z + [expert_labels[i]])

            pd.DataFrame(batch_data).to_csv(output_path, mode='a', header=False, index=False)
            del hr_df, motion_df, batch_data
            gc.collect()

    print(f"Dataset generado exitosamente en {output_path}")