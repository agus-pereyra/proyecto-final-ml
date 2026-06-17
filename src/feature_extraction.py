from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.io import loadmat

def extract_features(signal):
    return [np.mean(signal), np.std(signal), np.ptp(signal), stats.skew(signal)]

def feature_extraction():
    
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    output_dir = root_dir / "data_extraction"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "dataset_final.csv"
    
    all_data = []
    
    # Buscamos todos los labels.mat
    mat_files = list(data_dir.glob("**/labels.mat"))
    
    for mat_path in mat_files:
        folder = mat_path.parent
        
        # Carga de señales
        hr_data = pd.read_csv(folder / 'hr.csv').values.flatten()
        motion_data = pd.read_csv(folder / 'motion.csv').values
        
        mat_data = loadmat(mat_path)
    
        claves = [k for k in mat_data.keys() if 'label' in k.lower() or 'stage' in k.lower()]
        if not claves:
            print(f"Saltando {mat_path}: no se encontraron claves de etiquetas válidas.")
            continue
        

        clave_experto = next((k for k in claves if 'expert' in k.lower()), claves[0])
        expert_labels = mat_data[clave_experto].flatten()

        
        subject = folder.parent.name
        night = folder.name
        
        n_epochs = len(expert_labels)
        for i in range(n_epochs):
            # Tu lógica de extracción
            row = [subject, night]
            row.extend(extract_features(hr_data[i*30:(i+1)*30]))
            row.extend(extract_features(motion_data[i*30:(i+1)*30, 0]))
            row.extend(extract_features(motion_data[i*30:(i+1)*30, 1]))
            row.extend(extract_features(motion_data[i*30:(i+1)*30, 2]))
            row.append(expert_labels[i])
            all_data.append(row)
    
    # Guardado
    cols = ['subject', 'night', 'hr_mean', 'hr_std', 'hr_ptp', 'hr_skew', 
            'x_mean', 'x_std', 'x_ptp', 'x_skew', 'y_mean', 'y_std', 'y_ptp', 'y_skew',
            'z_mean', 'z_std', 'z_ptp', 'z_skew', 'label']
    
    pd.DataFrame(all_data, columns=cols).to_csv(output_path, index=False)
    print(f"Dataset generado en: {output_path}")