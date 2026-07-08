'''
MLP (red neuronal feedforward) sobre las features tabulares por época
(epoch_features.csv). Preprocesa lo que una red densa necesita y XGBoost no:
imputa los NaN de contexto y estandariza.
'''

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score

# columnas de metadata / target que NO son features
META_COLS = ['subject', 'night', 'epoch', 'label', 'dreem']


class TabularDataset(Dataset):
    '''Envuelve la matriz de features X [N, F] y las etiquetas y [N].'''
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    '''
    Red densa para clasificación por época. Cada bloque oculto es
    Linear -> BatchNorm -> ReLU -> Dropout; la última capa proyecta a `num_classes`.
    `hidden_dims` son los tamaños de las capas ocultas.
    '''
    def __init__(self, input_dim, num_classes=5, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        d_in = input_dim
        for d_out in hidden_dims:
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), nn.Dropout(dropout)]
            d_in = d_out
        layers.append(nn.Linear(d_in, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        '''In: x [B, F]. Out: logits [B, num_classes] (sin softmax).'''
        return self.net(x)


def get_dataloaders(csv_path=None, batch_size=256, random_state=42, weight_mode='sqrt'):
    '''
    Lee el CSV de features, descarta Unknown (clase 5), particiona por paciente
    (dev/test 80/20 y dev -> train/val 80/20), imputa NaN + estandariza (fiteado
    sólo en train) y arma los DataLoaders. Devuelve los pesos de clase, las
    etiquetas de Dreem del test (para comparar) y la dimensión de entrada. Si
    `csv_path` es None, busca el CSV en `data_extraction/` y `data/`.

    `weight_mode` controla los pesos de clase de la loss:
    - 'balanced': pesos proporcionales a 1/frecuencia (maximiza F1-macro, pero
      sobre-predice las clases chicas y baja accuracy/kappa).
    - 'sqrt': raíz de los balanced, un punto intermedio (mejor accuracy/kappa
      manteniendo recall razonable en las minoritarias).
    - 'none': sin pesos (maximiza accuracy/kappa, ignora las clases chicas).
    '''
    if csv_path is None:
        candidates = ['../data_extraction/epoch_features.csv', '../data/epoch_features.csv']
        csv_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    print(f"CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df['label'] != 5].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    y = df['label'].values.astype(np.int64)
    groups = df['subject'].values
    dreem = df['dreem'].values

    # partición por paciente para evitar fuga de información (las épocas de una
    # misma noche están muy correlacionadas)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    tr_rel, val_rel = next(gss.split(X[dev_idx], y[dev_idx], groups[dev_idx]))
    train_idx, val_idx = dev_idx[tr_rel], dev_idx[val_rel]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    dreem_test = dreem[test_idx]

    # los _lag/_lead/_delta/_rmean/_rstd tienen NaN en los bordes de cada noche;
    # una red densa no los maneja como XGBoost, así que se imputan con la mediana
    # y luego se estandariza. Ambos fiteados sólo en train.
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val = scaler.transform(imputer.transform(X_val))
    X_test = scaler.transform(imputer.transform(X_test))

    balanced = compute_class_weight('balanced', classes=np.arange(5), y=y_train)
    if weight_mode == 'balanced':
        weights = balanced
    elif weight_mode == 'sqrt':
        weights = np.sqrt(balanced)
    elif weight_mode == 'none':
        weights = np.ones_like(balanced)
    else:
        raise ValueError(f"weight_mode inválido: {weight_mode!r} (usar 'balanced', 'sqrt' o 'none')")

    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    print(f"Features: {len(feature_cols)} | train: {len(y_train)}  val: {len(y_val)}  test: {len(y_test)}")
    print(f"Class weights ({weight_mode}): { {i: round(float(w), 3) for i, w in enumerate(weights)} }")

    return train_loader, val_loader, test_loader, weights, dreem_test, len(feature_cols)


def train_model(model, train_loader, val_loader, class_weights,
                epochs=100, lr=1e-3, patience=12, weight_decay=1e-4,
                model_path=None):
    '''
    Entrena el MLP (Adam + CrossEntropy ponderada por clase, `ReduceLROnPlateau` y
    early stopping sobre la val loss, restaurando el mejor epoch).
    In:  model, train_loader, val_loader, class_weights [5]; epochs/lr/patience/... opc.
    Out: (model entrenado, history) con history = train_loss/val_loss/val_acc por epoch.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_state = None
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping en epoch {epoch+1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, loader, device=None):
    '''Corre el modelo sobre un loader (sin shuffle) y devuelve (y_true, y_pred)
    como arrays de numpy, en el mismo orden que el loader.'''
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    ys, preds = [], []
    for X_batch, y_batch in loader:
        outputs = model(X_batch.to(device))
        preds.append(outputs.argmax(dim=1).cpu().numpy())
        ys.append(y_batch.numpy())
    return np.concatenate(ys), np.concatenate(preds)


@torch.no_grad()
def permutation_importance(model, csv_path=None, n_repeats=3, top=15,
                           random_state=42, device=None):
    '''
    Importancia de features por PERMUTACIÓN sobre el test: baraja cada feature y
    mide cuánto cae el Cohen's Kappa (caída grande = feature importante). Es el
    análogo neuronal del feature importance del XGBoost. Reproduce el mismo split
    y preprocesamiento que `get_dataloaders`. Devuelve [(feature, importancia)]
    ordenado de mayor a menor (top N).
    '''
    if csv_path is None:
        candidates = ['../data_extraction/epoch_features.csv', '../data/epoch_features.csv']
        csv_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    df = pd.read_csv(csv_path)
    df = df[df['label'] != 5].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    y = df['label'].values.astype(np.int64)
    groups = df['subject'].values

    # mismo split por paciente que get_dataloaders
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    tr_rel, _ = next(gss.split(X[dev_idx], y[dev_idx], groups[dev_idx]))
    train_idx = dev_idx[tr_rel]

    # mismo preprocesamiento (fiteado sólo en train)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(X[train_idx]))
    X_test = scaler.transform(imputer.transform(X[test_idx])).astype(np.float32)
    y_test = y[test_idx]

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    def kappa_of(Xmat):
        logits = model(torch.tensor(Xmat, dtype=torch.float32, device=device))
        return cohen_kappa_score(y_test, logits.argmax(1).cpu().numpy())

    base = kappa_of(X_test)
    rng = np.random.default_rng(random_state)
    imp = np.zeros(len(feature_cols))
    for j in range(X_test.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            col = Xp[:, j].copy()
            rng.shuffle(col)
            Xp[:, j] = col
            drops.append(base - kappa_of(Xp))
        imp[j] = float(np.mean(drops))

    order = np.argsort(imp)[::-1][:top]
    return [(feature_cols[j], imp[j]) for j in order]
