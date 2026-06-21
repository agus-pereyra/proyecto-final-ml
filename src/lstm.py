from dataclasses import dataclass, field
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

# DIMENSIONES
#   B = batch size: cantidad de noches (secuencias) en el batch.
#   T = timesteps: cantidad de épocas de una noche (aprox. 800-900, variable por noche).
#   F = cantidad de features por época (=122).
#   F_enc = features que salen del encoder intra-época.
#   C = N_CLASSES = 5: clases a predecir (Wake, N1, N2, N3, REM).

N_CLASSES = 5
UNKNOWN = 5  # usado como ignore_index y como pad de labels

META_COLS = ['subject', 'night', 'epoch', 'label', 'dreem'] # features "meta-datos" 

# colapso a 4 clases para comparar con paper original: Wake / Light(N1+N2) / Deep(N3) / REM
COLLAPSE_4 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
COLLAPSE_4_NAMES = ['Wake', 'Light', 'Deep', 'REM']
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

@dataclass
class Config:
    features_path: str = '../data/epoch_features.csv'

    # arquitectura
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True  # flag LSTM <-> BiLSTM

    # optimización
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 60
    grad_clip: float = 5.0
    use_class_weights: bool = True

    # split por sujeto (sujetos disjuntos), fracciones sobre el total de sujetos
    val_frac: float = 0.15
    test_frac: float = 0.15

    seed: int = 36631
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path: str = '../src/models/best_lstm.pt' # checkpoint path

    # se completan en runtime
    feature_cols: list = field(default=None)
    input_size: int = None

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class NightSequenceDataset(Dataset):
    '''
    Agrupa el DataFrame de features por (subject, night) y devuelve una secuencia
    por noche: (features[T, F], labels[T]).

    Guarda el DataFrame y los índices de cada grupo, y materializa el
    tensor de cada noche recién en `__getitem__` (carga lazy). Las features se estandarizan con
    (mean, std) y los NaN de borde (lags/leads sin vecino) se imputan a 0, es decir, a
    la media post-estandarización.
    '''
    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 mean: np.ndarray = None, std: np.ndarray = None):
        self.feature_cols = feature_cols
        self.mean = mean
        self.std = std
        # orden estable por época dentro de cada noche
        df = df.sort_values(['subject', 'night', 'epoch'])
        self.df = df
        # lista de (key, posiciones iloc) -> carga lazy por noche
        self.groups = list(df.groupby(['subject', 'night']).indices.items())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        _, pos = self.groups[i]
        rows = self.df.iloc[pos]
        feats = rows[self.feature_cols].to_numpy(dtype=np.float32)
        labels = rows['label'].to_numpy(dtype=np.int64)

        if self.mean is not None:
            feats = (feats - self.mean) / self.std
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # .copy() garantiza arrays escribibles (torch.from_numpy se queja si no lo son)
        return torch.from_numpy(feats.copy()), torch.from_numpy(labels.copy())

def collate_nights(batch):
    '''
    Padding a la noche más larga del batch. Devuelve:
      feats   [B, T_max, F]  (padding con 0.0)
      labels  [B, T_max]     (padding con UNKNOWN -> ignorado por la loss)
      lengths [B]            (largo real de cada noche, en CPU para pack_padded_sequence)
    '''
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)

    feats = nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=UNKNOWN)
    return feats, labels, lengths

class EpochEncoder(nn.Module):
    '''
    Interfaz del encoder intra-época. Mapea la representación cruda de cada época
    a un vector de features: [B, T, *] -> [B, T, out_dim].

    A IMPLEMENTAR para el híbrido CNN1D -> LSTM 
    (recibe señal cruda por época [B, T, C, L] y devuelve [B, T, out_dim] 
    '''
    out_dim: int

    def forward(self, x):
        raise NotImplementedError

class IdentityEncoder(EpochEncoder):
    '''Pasa las features pre-computadas tal cual; out_dim = input_size.'''
    def __init__(self, input_size: int):
        super().__init__()
        self.out_dim = input_size

    def forward(self, x):
        return x

class SleepStager(nn.Module):
    '''
    encoder -> LSTM (con packing para ignorar el padding) -> Linear por timestep.

    Configurable vía Config: hidden_size, num_layers, dropout, bidirectional. El
    mismo código sirve para LSTM y BiLSTM (sólo cambia `bidirectional`).
    '''
    def __init__(self, cfg: Config, encoder: EpochEncoder = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else IdentityEncoder(cfg.input_size)
        self.lstm = nn.LSTM(
            input_size=self.encoder.out_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        out_mult = 2 if cfg.bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size * out_mult, N_CLASSES),
        )

    def forward(self, feats, lengths):
        # feats [B, T, F] -> encoder -> [B, T, F_enc]
        x = self.encoder(feats)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                     total_length=feats.shape[1])
        return self.head(out)  # [B, T, N_CLASSES]

def split_subjects(df: pd.DataFrame, cfg: Config):
    '''Particiona los SUJETOS (no noches ni épocas) en train/val/test disjuntos.'''
    subjects = np.array(sorted(df['subject'].unique()))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = int(round(n * cfg.test_frac))
    n_val = int(round(n * cfg.val_frac))
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test:n_test + n_val])
    train_s = set(subjects[n_test + n_val:])

    masks = {name: df['subject'].isin(s)
             for name, s in [('train', train_s), ('val', val_s), ('test', test_s)]}
    return {name: df[m].copy() for name, m in masks.items()}, \
           {'train': train_s, 'val': val_s, 'test': test_s}

def feature_stats(train_df: pd.DataFrame, feature_cols: list):
    '''Media/desvío por feature sobre el train (ignorando NaN). std=0 -> 1.'''
    arr = train_df[feature_cols].to_numpy(dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def class_weights(train_df: pd.DataFrame, device: str):
    '''
    Pesos inversos a la frecuencia de cada clase 0..4 en el train (Unknown excluido),
    normalizados a media 1. Se pasan a CrossEntropyLoss para el desbalance.
    '''
    labels = train_df['label'].to_numpy()
    labels = labels[labels != UNKNOWN]
    counts = np.bincount(labels, minlength=N_CLASSES)[:N_CLASSES].astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_CLASSES * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)

def _collapse4(y):
    return np.vectorize(COLLAPSE_4.get)(y)

def compute_metrics(y_true, y_pred):
    '''
    Métricas por época sobre las predicciones ya filtradas (sin padding ni Unknown).
    Devuelve un dict con kappa (principal), macro F1, accuracy y matriz de confusión
    a 5 clases, más sus versiones colapsadas a 4 (Wake/Light/Deep/REM, SLAMSS-IFS).
    '''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    m = {
        'kappa': cohen_kappa_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', labels=range(N_CLASSES)),
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion': confusion_matrix(y_true, y_pred, labels=range(N_CLASSES)),
    }
    t4, p4 = _collapse4(y_true), _collapse4(y_pred)
    m['kappa_4'] = cohen_kappa_score(t4, p4)
    m['macro_f1_4'] = f1_score(t4, p4, average='macro', labels=range(4))
    m['accuracy_4'] = accuracy_score(t4, p4)
    m['confusion_4'] = confusion_matrix(t4, p4, labels=range(4))
    return m


@torch.no_grad()
def collect_predictions(model, loader, device):
    '''Corre el modelo y junta (y_true, y_pred) de las épocas válidas (label != UNKNOWN).'''
    model.eval()
    ys, ps = [], []
    for feats, labels, lengths in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats, lengths)               # [B, T, C]
        pred = logits.argmax(-1)                      # [B, T]
        valid = labels != UNKNOWN
        ys.append(labels[valid].cpu().numpy())
        ps.append(pred[valid].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

def make_loaders(cfg: Config):
    '''
    Carga las features, splitea por sujeto, estandariza con stats del train y arma
    los DataLoaders. Devuelve (loaders, stats, splits_subjects).
    '''
    df = pd.read_csv(cfg.features_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    cfg.feature_cols = feature_cols
    cfg.input_size = len(feature_cols)

    splits, subj = split_subjects(df, cfg)
    mean, std = feature_stats(splits['train'], feature_cols)

    loaders = {}
    for name, part in splits.items():
        ds = NightSequenceDataset(part, feature_cols, mean, std)
        loaders[name] = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=(name == 'train'),
            collate_fn=collate_nights,
        )
    return loaders, (mean, std), subj, splits['train']

def train(cfg: Config, encoder: EpochEncoder = None):
    '''
    Entrena la LSTM inter-época y guarda el mejor checkpoint por Kappa de validación.
    Devuelve (model, history, test_metrics).
    '''
    set_seed(cfg.seed)
    device = cfg.device

    loaders, (mean, std), subj, train_df = make_loaders(cfg)
    print(f'sujetos -> train {len(subj["train"])} | val {len(subj["val"])} | test {len(subj["test"])}')

    model = SleepStager(cfg, encoder).to(device)
    weight = class_weights(train_df, device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=UNKNOWN)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_kappa = -np.inf

    pbar = tqdm(range(1, cfg.epochs + 1), desc='entrenando', unit='epoch')
    for epoch in pbar:
        model.train()
        total = 0.0
        for feats, labels, lengths in loaders['train']:
            feats, labels = feats.to(device), labels.to(device)
            optim.zero_grad()
            logits = model(feats, lengths) # [B, T, C]
            # CrossEntropyLoss espera [N, C] y [N]; aplanamos timesteps
            loss = criterion(logits.reshape(-1, N_CLASSES), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            total += loss.item() * feats.shape[0]

        train_loss = total / len(loaders['train'].dataset)
        y_val, p_val = collect_predictions(model, loaders['val'], device)
        val_m = compute_metrics(y_val, p_val)
        history.append({'epoch': epoch, 'train_loss': train_loss, **val_m})

        is_best = val_m['kappa'] > best_kappa
        if is_best:
            best_kappa = val_m['kappa']
            torch.save({
                'model_state': model.state_dict(),
                'config': cfg,
                'mean': mean, 'std': std,
                'val_kappa': best_kappa, 'epoch': epoch,
            }, cfg.ckpt_path)

        pbar.set_postfix({
            'loss': f'{train_loss:.4f}',
            'val_kappa': f'{val_m["kappa"]:.4f}',
            'macroF1': f'{val_m["macro_f1"]:.4f}',
            'acc': f'{val_m["accuracy"]:.4f}',
            'best': f'{best_kappa:.4f}{"*" if is_best else ""}',
        })

    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    y_test, p_test = collect_predictions(model, loaders['test'], device)
    test_m = compute_metrics(y_test, p_test)
    print(f'\nTEST (mejor ckpt, val kappa {ckpt["val_kappa"]:.4f}, epoch {ckpt["epoch"]}):')
    print(f'  kappa {test_m["kappa"]:.4f} | macroF1 {test_m["macro_f1"]:.4f} | acc {test_m["accuracy"]:.4f}')
    print(f'  4-clases: kappa {test_m["kappa_4"]:.4f} | macroF1 {test_m["macro_f1_4"]:.4f} | acc {test_m["accuracy_4"]:.4f}')

    return model, history, test_m

if __name__ == '__main__':
    cfg = Config()
    train(cfg)
