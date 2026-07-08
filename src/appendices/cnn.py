import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from data import EDA
except ImportError:
    from src.data import EDA


# ---------------------------------------------------------------------------
# Modelo CNN 1D intra-época + entrenamiento
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()

        # Bloque 1: patrones locales (150 -> 75 timesteps)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
        )

        # Bloque 2: patrones medios (75 -> 37 timesteps)
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
        )

        # Bloque 3: representación global de la época (37 -> 1)
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (batch, 150, 4) -> (batch, 4, 150) para Conv1d
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def train_model(model, train_loader, val_loader, class_weights,
                epochs=50, lr=0.001, patience=10,
                model_path='../../models/best_cnn.pth'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping en epoch {epoch+1}.")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model, history


# ---------------------------------------------------------------------------
# Procesamiento del dataset: señal cruda -> tensores [150, 4] por época (.npz)
# (antes en process_dataset_cnn.py)
# ---------------------------------------------------------------------------
def get_cnn_dataset(output_dir='../../data_extraction/processed_data',
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


# ---------------------------------------------------------------------------
# Split por paciente + StandardScaler + DataLoaders
# (antes en split_dataset_cnn.py)
# ---------------------------------------------------------------------------
class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_split(processed_dir='../../data_extraction/processed_data',
                   val_size=5, test_size=4, random_seed=42):

    rng = np.random.default_rng(random_seed)

    npz_files = sorted(Path(processed_dir).glob('*.npz'))
    indices = np.arange(len(npz_files))
    rng.shuffle(indices)

    test_idx  = indices[:test_size]
    val_idx   = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    def load_split(idx_list):
        Xs, ys = [], []
        for i in idx_list:
            data = np.load(npz_files[i])
            Xs.append(data['X'])
            ys.append(data['y'].astype(np.int64))
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = load_split(train_idx)
    X_val,   y_val   = load_split(val_idx)
    X_test,  y_test  = load_split(test_idx)

    n_train, T, C = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(n_train, T, C)
    X_val   = scaler.transform(X_val.reshape(-1, C)).reshape(len(X_val), T, C)
    X_test  = scaler.transform(X_test.reshape(-1, C)).reshape(len(X_test), T, C)

    classes = np.arange(5)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)

    print(f"Train : {len(y_train):>6} epochs | {len(train_idx)} pacientes")
    print(f"Val   : {len(y_val):>6} epochs | {len(val_idx)} pacientes")
    print(f"Test  : {len(y_test):>6} epochs | {len(test_idx)} pacientes")
    print(f"Class weights: { {i: round(w, 3) for i, w in enumerate(weights)} }")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, weights


def get_dataloaders(processed_dir='../../data_extraction/processed_data',
                    batch_size=64, val_size=5, test_size=4, random_seed=42):

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, weights = \
        load_and_split(processed_dir, val_size, test_size, random_seed)

    train_loader = DataLoader(SleepDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SleepDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SleepDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, weights
