from dataclasses import dataclass, field
from pathlib import Path
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
#   F = cantidad de features por época (=122 en modo tabular).
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
class ConfigLSTM:
    '''
    Config unificada de los modelos secuenciales (LSTM tabular e híbrido): fija
    modo, arquitectura, optimización y split. Con la misma seed/fracciones ambos
    modelos comparten el split por sujeto. Cada campo está anotado abajo;
    `feature_cols`/`input_size` se completan al armar los loaders.
    '''
    # modo de operación
    hybrid: bool = False  # False = LSTM sobre features tabulares; True = CNN1D->BiLSTM sobre señal cruda
    features_path: str = '../data_extraction/epoch_features.csv'
    sequences_dir: str = '../data_extraction/sequences'
    feature_dim: int = 128   # out_dim del CNNEpochEncoder

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
    amp: bool = None  # mixed precision; None -> se activa solo si device=='cuda'
    patience: int = None  # early stopping: corta si el kappa de val no mejora en N epochs; None -> sin ES

    # split por sujeto (sujetos disjuntos), fracciones sobre el total de sujetos
    val_frac: float = 0.15
    test_frac: float = 0.15

    seed: int = 36631
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path: str = '../models/best_lstm.pt' # checkpoint path

    # se completan en runtime
    feature_cols: list = field(default=None)
    input_size: int = None

def set_seed(seed: int):
    '''Fija la seed de random, numpy y torch (CPU+CUDA) para reproducibilidad.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class NightSequenceDataset(Dataset):
    '''
    Modo TABULAR. Agrupa el DataFrame de features por (subject, night) y devuelve una
    secuencia por noche: (features[T, F], labels[T]).

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
      feats   [B, T_max, ...]  (padding con 0.0)  -- [B,T,F] tabular o [B,T,150,4] crudo
      labels  [B, T_max]       (padding con UNKNOWN -> ignorado por la loss)
      lengths [B]              (largo real de cada noche, en CPU para pack_padded_sequence)
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
    '''
    out_dim: int

    def forward(self, x):
        raise NotImplementedError

class IdentityEncoder(EpochEncoder):
    '''Pasa las features pre-computadas tal cual; out_dim = input_size (modo tabular).'''
    def __init__(self, input_size: int):
        super().__init__()
        self.out_dim = input_size

    def forward(self, x):
        return x

def _conv_block(in_ch, out_ch, pool=True):
    '''
    Bloque conv 1D (misma arquitectura que src/CNN.py) con GroupNorm en vez de BatchNorm
    (ver CNNEpochEncoder). El último bloque cierra con AdaptiveAvgPool1d(1) para colapsar
    la época a un vector.
    '''
    layers = [
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.ReLU(),
    ]
    layers.append(nn.MaxPool1d(kernel_size=2) if pool else nn.AdaptiveAvgPool1d(1))
    layers.append(nn.Dropout(0.2))
    return nn.Sequential(*layers)

class CNNEpochEncoder(EpochEncoder):
    '''
    Encoder intra-época del modo HÍBRIDO (misma arquitectura que src/CNN.py: 4->32->64->128)
    pero con **GroupNorm en vez de BatchNorm**. Motivo: `collate_nights` paddea las noches
    cortas con épocas de ceros; la LSTM ignora ese padding (pack_padded_sequence +
    ignore_index) pero la CNN lo procesa igual. Con BatchNorm las estadísticas del batch
    mezclaban épocas reales con relleno y variaban según la fracción de padding -> normalización
    inestable. GroupNorm normaliza **por época** (independiente del batch y del padding).

        forward: feats [B, T, 150, 4] -> [B, T, out_dim]  (out_dim = feature_dim = 128)
    '''
    def __init__(self, feature_dim=128):
        super().__init__()
        self.block1 = _conv_block(4, 32)              # 150 -> 75
        self.block2 = _conv_block(32, 64)             # 75 -> 37
        self.block3 = _conv_block(64, feature_dim, pool=False)  # 37 -> 1 (AdaptiveAvgPool)
        self.out_dim = feature_dim

    def forward(self, x):
        B, T, L, C = x.shape
        x = x.reshape(B * T, L, C).permute(0, 2, 1)   # [B*T, C, L] para Conv1d
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)                            # [B*T, out_dim, 1]
        return x.flatten(1).reshape(B, T, -1)         # [B, T, out_dim]

class LSTM(nn.Module):
    '''
    encoder -> LSTM (con packing para ignorar el padding) -> Linear por timestep.

    Configurable vía ConfigLSTM: hidden_size, num_layers, dropout, bidirectional. El
    mismo código sirve para LSTM y BiLSTM (sólo cambia `bidirectional`) y para ambos modos
    (el encoder decide si consume features tabulares o señal cruda).
    '''
    def __init__(self, cfg: ConfigLSTM, encoder: EpochEncoder = None):
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
        '''In: feats [B, T, ...], lengths [B] (largo real de cada noche, para el
        packing). Out: logits [B, T, N_CLASSES].'''
        x = self.encoder(feats)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                     total_length=feats.shape[1])
        return self.head(out)  # [B, T, N_CLASSES]

def partition_subjects(subjects: np.ndarray, cfg: ConfigLSTM):
    '''
    Reparte un array de IDs de sujetos en train/val/test disjuntos (sets), de forma
    determinística dada la seed. Es la ÚNICA fuente del split: standalone e híbrido la
    usan sobre la misma lista de sujetos -> particiones idénticas -> modelos comparables.
    '''
    subjects = np.array(sorted(subjects))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = int(round(n * cfg.test_frac))
    n_val = int(round(n * cfg.val_frac))
    test_s = set(subjects[:n_test].tolist())
    val_s = set(subjects[n_test:n_test + n_val].tolist())
    train_s = set(subjects[n_test + n_val:].tolist())
    return {'train': train_s, 'val': val_s, 'test': test_s}

def split_subjects(df: pd.DataFrame, cfg: ConfigLSTM):
    '''Wrapper tabular: parte los SUJETOS del DataFrame y devuelve (dfs por split, sets).'''
    subj = partition_subjects(df['subject'].unique(), cfg)
    masks = {name: df['subject'].isin(s) for name, s in subj.items()}
    return {name: df[m].copy() for name, m in masks.items()}, subj

def feature_stats(train_df: pd.DataFrame, feature_cols: list):
    '''Media/desvío por feature sobre el train (ignorando NaN). std=0 -> 1.'''
    arr = train_df[feature_cols].to_numpy(dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def class_weights(labels, device: str):
    '''
    Pesos inversos a la frecuencia de cada clase 0..4 en el train (Unknown excluido),
    normalizados a media 1, para el desbalance. `labels` es un array 1D de etiquetas.
    '''
    labels = np.asarray(labels)
    labels = labels[labels != UNKNOWN]
    counts = np.bincount(labels, minlength=N_CLASSES)[:N_CLASSES].astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_CLASSES * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)

def _collapse4(y):
    '''Colapsa etiquetas de 5 a 4 clases (Wake / Light=N1+N2 / Deep=N3 / REM) vía COLLAPSE_4.'''
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


@torch.no_grad()
def collect_probabilities(model, loader, device):
    '''
    Como collect_predictions pero devuelve (y_true, y_score) con y_score las
    probabilidades softmax por clase [N, N_CLASSES] de las épocas válidas
    (label != UNKNOWN). Alimenta metrics.roc_pr_curves para LSTM e híbrido.
    '''
    model.eval()
    ys, probs = [], []
    for feats, labels, lengths in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats, lengths)                   # [B, T, C]
        p = torch.softmax(logits, dim=-1)                # [B, T, C]
        valid = labels != UNKNOWN
        ys.append(labels[valid].cpu().numpy())
        probs.append(p[valid].cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, use_amp=False):
    '''
    Una sola pasada por el loader que devuelve (loss_promedio, y_true, y_pred). Usada
    en cada epoch para las curvas de validación (loss + métricas) sin iterar dos veces.
    '''
    model.eval()
    ys, ps = [], []
    total, n = 0.0, 0
    for feats, labels, lengths in loader:
        feats, labels = feats.to(device), labels.to(device)
        with torch.autocast(device_type='cuda', enabled=use_amp):
            logits = model(feats, lengths)               # [B, T, C]
            loss = criterion(logits.reshape(-1, N_CLASSES), labels.reshape(-1))
        total += loss.item() * feats.shape[0]
        n += feats.shape[0]
        pred = logits.argmax(-1)
        valid = labels != UNKNOWN
        ys.append(labels[valid].cpu().numpy())
        ps.append(pred[valid].cpu().numpy())
    return total / max(n, 1), np.concatenate(ys), np.concatenate(ps)


def _labels_from_files(files):
    '''Concatena las etiquetas de una lista de .npz (para los pesos de clase del híbrido).'''
    return np.concatenate([np.load(f)['y'].astype(np.int64) for f in files])


def make_loaders(cfg: ConfigLSTM):
    '''
    Modo TABULAR. Carga las features, splitea por sujeto, estandariza con stats del train
    y arma los DataLoaders. Devuelve (loaders, (mean, std), subj, train_labels).
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
    return loaders, (mean, std), subj, splits['train']['label'].to_numpy()


def make_hybrid_loaders(cfg: ConfigLSTM):
    '''
    Modo HÍBRIDO. Lee los .npz de señal cruda por noche (sequences/), splitea POR SUJETO
    con `partition_subjects` (mismo criterio y seed que el tabular -> mismo split),
    estandariza por canal con stats del train y arma los DataLoaders. Devuelve
    (loaders, (mean, std), subj, train_labels).
    '''
    try:
        from sequence_data import NightSignalDataset, channel_stats
    except ImportError:
        from src.sequence_data import NightSignalDataset, channel_stats

    files = sorted(Path(cfg.sequences_dir).glob('*.npz'))
    pid = lambda f: int(f.stem.replace('Bidslab', ''))
    subj = partition_subjects(np.array(sorted({pid(f) for f in files})), cfg)
    split_files = {name: [f for f in files if pid(f) in s] for name, s in subj.items()}

    mean, std = channel_stats(split_files['train'])
    loaders = {}
    for name, fs in split_files.items():
        ds = NightSignalDataset(fs, mean, std)
        loaders[name] = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=(name == 'train'),
            collate_fn=collate_nights,
        )
    return loaders, (mean, std), subj, _labels_from_files(split_files['train'])


def _build(cfg: ConfigLSTM, encoder: EpochEncoder = None):
    '''Arma loaders + encoder según el modo. Devuelve (loaders, (mean,std), subj, train_labels, encoder).'''
    if cfg.hybrid:
        loaders, stats, subj, train_labels = make_hybrid_loaders(cfg)
        if encoder is None:
            encoder = CNNEpochEncoder(cfg.feature_dim)
    else:
        loaders, stats, subj, train_labels = make_loaders(cfg)
        # encoder queda como se pasó (IdentityEncoder por defecto dentro de LSTM)
    return loaders, stats, subj, train_labels, encoder


def plot_history(history, title='', ax=None):
    '''
    Curvas de entrenamiento de UN modelo: (izq) loss train vs val por epoch,
    (der) métricas de validación (kappa/macro-F1/accuracy). Una línea punteada roja marca la
    época del **mejor kappa de val** (el checkpoint elegido; con early stopping el corte ocurre
    `patience` épocas después). Devuelve los ejes.
    '''
    import matplotlib.pyplot as plt
    ep = [h['epoch'] for h in history]
    best_epoch = max(history, key=lambda h: h['kappa'])['epoch']  # época del checkpoint guardado
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h['train_loss'] for h in history], label='train')
    ax[0].plot(ep, [h['val_loss'] for h in history], label='val')
    ax[0].axvline(best_epoch, color='red', ls='--', lw=1, label=f'mejor epoch ({best_epoch})')
    ax[0].set_xlabel('época'); ax[0].set_ylabel('loss'); ax[0].set_title(f'{title} — loss'); ax[0].legend()
    ax[1].plot(ep, [h['kappa'] for h in history], label='kappa')
    ax[1].plot(ep, [h['macro_f1'] for h in history], label='macro-F1')
    ax[1].plot(ep, [h['accuracy'] for h in history], label='accuracy')
    ax[1].axvline(best_epoch, color='red', ls='--', lw=1)
    ax[1].set_xlabel('época'); ax[1].set_title(f'{title} — validación'); ax[1].legend()
    return ax


def train(cfg: ConfigLSTM, encoder: EpochEncoder = None, epoch_callback=None, verbose=True):
    '''
    ÚNICA función de entrenamiento. Corre en modo standalone (cfg.hybrid=False) o híbrido
    (cfg.hybrid=True); en ambos casos el mismo loop, con barra de progreso tqdm (sin prints
    sueltos por epoch). Guarda el mejor checkpoint por Kappa de validación (con config,
    mean/std, subj) y al final lo recarga y reporta test.

    `epoch_callback(epoch, val_metrics)`: hook opcional llamado al final de cada epoch (lo usa
    el search para reportar el kappa de val al pruner de Optuna y cortar trials malos temprano;
    si lanza una excepción, el loop la propaga).

    `verbose` (default True): muestra la barra tqdm y el reporte de test. El search lo pone en
    False para no ensuciar el output (una barra + un bloque de test por trial es ilegible).

    Con `cfg.amp` (o device=='cuda') usa mixed precision (autocast + GradScaler): ~1.5-2x en la
    CNN del híbrido y menos memoria.

    Con `cfg.patience` (int) hace **early stopping**: corta si el kappa de val no mejora en esas
    epochs seguidas (el mejor checkpoint ya quedó guardado). `None` = entrena `cfg.epochs` fijas.

    Devuelve (model, history, test_metrics). `history` incluye por epoch: train_loss,
    val_loss y las métricas de validación (kappa, macro_f1, accuracy) -> curvas train/val.
    '''
    set_seed(cfg.seed)
    device = cfg.device
    use_amp = cfg.amp if cfg.amp is not None else (device == 'cuda')

    loaders, (mean, std), subj, train_labels, encoder = _build(cfg, encoder)

    model = LSTM(cfg, encoder).to(device)
    weight = class_weights(train_labels, device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=UNKNOWN)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    ckpt_dir = Path(cfg.ckpt_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_kappa = -np.inf
    best_epoch = 0                       # última epoch que mejoró el kappa de val
    modo = 'Hybrid' if cfg.hybrid else 'LSTM'
    pbar = tqdm(range(1, cfg.epochs + 1), desc=f'Training ({modo})', unit='epoch',
                disable=not verbose)
    for epoch in pbar:
        model.train()
        total, n = 0.0, 0
        for feats, labels, lengths in loaders['train']:
            feats, labels = feats.to(device), labels.to(device)
            optim.zero_grad()
            with torch.autocast(device_type='cuda', enabled=use_amp):
                logits = model(feats, lengths)                       # [B, T, C]
                loss = criterion(logits.reshape(-1, N_CLASSES), labels.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optim)                                    # unscale antes del clip
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            total += loss.item() * feats.shape[0]
            n += feats.shape[0]

        train_loss = total / max(n, 1)
        val_loss, y_val, p_val = evaluate_loader(model, loaders['val'], criterion, device, use_amp)
        val_m = compute_metrics(y_val, p_val)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'kappa': val_m['kappa'], 'macro_f1': val_m['macro_f1'],
                        'accuracy': val_m['accuracy']})

        is_best = val_m['kappa'] > best_kappa
        if is_best:
            best_kappa = val_m['kappa']
            best_epoch = epoch
            torch.save({
                'model_state': model.state_dict(),
                'config': cfg,
                'mean': mean, 'std': std, 'subj': subj,
                'val_kappa': best_kappa, 'epoch': epoch,
            }, cfg.ckpt_path)

        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_kappa': f'{val_m["kappa"]:.4f}',
            'best': f'{best_kappa:.4f}{"*" if is_best else ""}',
        })

        if epoch_callback is not None:
            epoch_callback(epoch, val_m)

        # early stopping: sin mejora del kappa de val
        if cfg.patience is not None and epoch - best_epoch >= cfg.patience:
            pbar.set_description(
                f'Early Stop ({modo}) - epoch {epoch} (mejor {best_kappa:.4f} @ {best_epoch})')
            pbar.close()
            break

    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    y_test, s_test = collect_probabilities(model, loaders['test'], device)
    p_test = s_test.argmax(1)
    test_m = compute_metrics(y_test, p_test)
    test_m['y_true'], test_m['y_pred'], test_m['y_score'] = y_test, p_test, s_test
    if verbose:
        print(f'\nTEST ({modo}) — mejor ckpt (val kappa {ckpt["val_kappa"]:.4f}, epoch {ckpt["epoch"]}):')
        print(f'  kappa {test_m["kappa"]:.4f} | macroF1 {test_m["macro_f1"]:.4f} | acc {test_m["accuracy"]:.4f}')
        print(f'  4-clases: kappa {test_m["kappa_4"]:.4f} | macroF1 {test_m["macro_f1_4"]:.4f} | acc {test_m["accuracy_4"]:.4f}')
    return model, history, test_m


def evaluate(cfg: ConfigLSTM, encoder: EpochEncoder = None):
    '''
    Reconstruye el modelo desde `cfg.ckpt_path` SIN entrenar y lo evalúa en test (mismo
    split por sujeto, misma seed). Contraparte de `train` para la rama TRAIN_NEW=False del
    notebook. Devuelve (model, test_metrics).
    '''
    set_seed(cfg.seed)
    device = cfg.device

    loaders, _stats, _subj, _labels, encoder = _build(cfg, encoder)
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    model = LSTM(cfg, encoder).to(device)
    model.load_state_dict(ckpt['model_state'])

    y_test, s_test = collect_probabilities(model, loaders['test'], device)
    p_test = s_test.argmax(1)
    test_m = compute_metrics(y_test, p_test)
    test_m['y_true'], test_m['y_pred'], test_m['y_score'] = y_test, p_test, s_test
    modo = 'híbrido CNN1D->BiLSTM' if cfg.hybrid else 'LSTM tabular'
    print(f'cargado {cfg.ckpt_path} ({modo}, val kappa {ckpt["val_kappa"]:.4f}, epoch {ckpt["epoch"]})')
    print(f'TEST -> kappa {test_m["kappa"]:.4f} | macroF1 {test_m["macro_f1"]:.4f} | acc {test_m["accuracy"]:.4f}')
    return model, test_m


@torch.no_grad()
def predict_night(model, cfg: ConfigLSTM, subject: int, night: int, device=None):
    '''
    Predice UNA noche completa y devuelve (epochs, y_pred) alineados al índice de época
    de la ventana limpia (el mismo que usan epoch_features.csv y build_night_sequences).
    Sirve para superponer las predicciones sobre las señales (plots.night_prediction_overview).
    Funciona en ambos modos: tabular (lee el CSV) o híbrido (lee el .npz de señal cruda).
    La estandarización sale del checkpoint (mean/std de train, sin leakage).
    '''
    device = device or cfg.device
    model.eval()
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    mean, std = ckpt['mean'], ckpt['std']

    if cfg.hybrid:
        try:
            from sequence_data import NightSignalDataset  # noqa: F401 (solo por consistencia de import)
        except ImportError:
            pass
        f = Path(cfg.sequences_dir) / f'Bidslab{subject:02d}.npz'
        d = np.load(f)
        if 'epoch' not in d.files:
            raise KeyError("los .npz no tienen 'epoch': re-generá sequences con build_night_sequences")
        m = d['night_id'] == night
        epochs = d['epoch'][m]
        X = ((d['X'][m] - mean) / std).astype(np.float32)
        order = np.argsort(epochs)                          # orden temporal por época
        feats = torch.from_numpy(X[order])[None].to(device)  # [1, T, 150, 4]
        epochs = epochs[order]
    else:
        df = pd.read_csv(cfg.features_path)
        sub = df[(df['subject'] == subject) & (df['night'] == night)].sort_values('epoch')
        feature_cols = cfg.feature_cols or [c for c in df.columns if c not in META_COLS]
        X = sub[feature_cols].to_numpy(dtype=np.float32)
        X = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        feats = torch.from_numpy(X)[None].to(device)         # [1, T, F]
        epochs = sub['epoch'].to_numpy()

    lengths = torch.tensor([feats.shape[1]], dtype=torch.long)
    pred = model(feats, lengths)[0].argmax(-1).cpu().numpy()
    return epochs, pred


if __name__ == '__main__':
    train(ConfigLSTM())                       # LSTM tabular
    train(ConfigLSTM(hybrid=True, ckpt_path='../models/best_hybrid.pt'))  # híbrido
