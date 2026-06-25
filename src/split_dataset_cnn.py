import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader

class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_split(processed_dir='../data_extraction/processed_data',
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

def get_dataloaders(processed_dir='../data_extraction/processed_data',
                    batch_size=64, val_size=5, test_size=4, random_seed=42):

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, weights = \
        load_and_split(processed_dir, val_size, test_size, random_seed)

    train_loader = DataLoader(SleepDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SleepDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SleepDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, weights