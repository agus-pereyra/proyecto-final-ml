import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from scipy.io import loadmat

def get_cnn_dataset(data_root='../data/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0', n_patients=None):
    """
    Recorre la estructura de carpetas, alinea HR y Acelerometría a 1Hz
    y genera el tensor para la CNN 1D.
    
    :param n_patients: Int. Cantidad de pacientes a procesar (None procesa todos).
    """
    data_path = Path(data_root)
    all_X = []
    all_y = []

    # Buscamos y ordenamos los directorios de pacientes
    patient_dirs = sorted(list(data_path.glob('Bidslab*')))
    
    # Aplicamos el filtro de cantidad de pacientes
    if n_patients is not None:
        patient_dirs = patient_dirs[:n_patients]
    
    for p_dir in tqdm(patient_dirs, desc=f"Procesando {len(patient_dirs)} pacientes"):
        for night_dir in sorted(p_dir.iterdir()):
            if night_dir.is_dir():
                # 1. Cargar archivos (Asegúrate de que los nombres coincidan con tus archivos)
                hr_df = pd.read_csv(night_dir / 'hr.csv', names=['ts', 'hr'])
                acc_df = pd.read_csv(night_dir / 'motion.csv') 
                labels_df = pd.read_csv(night_dir / 'expert_labels.csv')
                
                # 2. Conversión a datetime
                hr_df['ts'] = pd.to_datetime(hr_df['ts'], unit='s')
                acc_df['Timestamp'] = pd.to_datetime(acc_df['Timestamp'], unit='s')
                labels_df['Timestamp'] = pd.to_datetime(labels_df['Timestamp'], unit='s')
                
                # 3. Resample a 1Hz
                # 3. Resample a 1Hz (cambia '1S' por '1s')
                hr_res = hr_df.set_index('ts').resample('1s').mean().interpolate(method='linear')
                acc_res = acc_df.set_index('Timestamp').resample('1s').mean()
                
                # 4. Merge
                df_merged = pd.merge(hr_res, acc_res, left_index=True, right_index=True).dropna()
                
                # 5. Ventaneo de 30 segundos
                data_matrix = df_merged[['hr', 'x', 'y', 'z']].values
                num_epochs = len(data_matrix) // 30
                
                for i in range(num_epochs):
                    
                    epoch_data = data_matrix[i*30 : (i+1)*30, :] # (30, 4)
                    epoch_label = labels_df.iloc[i]['stage'] # Asumiendo 1 fila = 1 época
                    
                    all_X.append(epoch_data.T)
                    all_y.append(epoch_label)

    return np.array(all_X), np.array(all_y)

def get_cnn_dataset(data_root='../data/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0', n_patients=None):
    data_path = Path(data_root)
    all_X = []
    all_y = []

    patient_dirs = sorted(list(data_path.glob('Bidslab*')))
    if n_patients is not None:
        patient_dirs = patient_dirs[:n_patients]
    
    for p_dir in tqdm(patient_dirs, desc="Procesando pacientes"):
        for night_dir in sorted(p_dir.iterdir()):
            if night_dir.is_dir():
                # 1. Cargar sensores (HR y Motion)
                hr_df = pd.read_csv(night_dir / 'hr.csv', names=['ts', 'hr'])
                acc_df = pd.read_csv(night_dir / 'motion.csv')
                mat_data = loadmat(night_dir / 'labels.mat')
                
                recStart = float(mat_data['recStart'][0][0])
                expert_labels = mat_data['expert_label'].flatten()
                
                # 2. Resampling manual sin datetime
                # Creamos una grilla de tiempo con pasos de 1 segundo
                # El inicio es recStart, el fin es el último timestamp
                t_min = recStart
                t_max = max(hr_df['ts'].max(), acc_df['Timestamp'].max())
                time_grid = np.arange(t_min, t_max, 1.0)
                
                # Interpolación manual para HR
                hr_res = np.interp(time_grid, hr_df['ts'], hr_df['hr'])
                
                # Para Acelerometría, agrupamos por el segundo redondeado
                acc_df['ts_sec'] = acc_df['Timestamp'].apply(np.floor)
                acc_res = acc_df.groupby('ts_sec')[['x', 'y', 'z']].mean()
                
                # Alineamos ambas en un DataFrame usando la grilla de tiempo
                df_merged = pd.DataFrame({'hr': hr_res}, index=time_grid)
                df_merged = df_merged.join(acc_res, how='inner').dropna()
                
                # 3. Ventaneo
                data_matrix = df_merged[['hr', 'x', 'y', 'z']].values
                
                for i in range(len(expert_labels)):
                    start_idx = i * 30
                    end_idx = (i + 1) * 30
                    
                    if end_idx <= len(data_matrix):
                        label = expert_labels[i]
                        if label <= 4:
                            epoch_data = data_matrix[start_idx:end_idx, :]
                            all_X.append(epoch_data.T)
                            all_y.append(label)

    return np.array(all_X), np.array(all_y)