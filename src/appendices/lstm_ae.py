'''
Autoencoder LSTM (apéndice): representación NO supervisada para sleep staging.

Un `SeqAutoencoder` (BiLSTM encoder -> bottleneck latente por época -> LSTM
decoder) se entrena a reconstruir las features por noche, sin usar las etiquetas.
Luego `extract_embeddings` guarda el embedding z_t por época, que un XGBoost usa
como input. Reutiliza el pipeline por noche de `src.lstm` (dataset, collate,
split por sujeto y stats de train).
'''

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

# Reutilizamos el pipeline ya construido para la LSTM inter-época: el dataset por
# noche (con estandarización + imputación de NaN de borde), el collate con padding,
# el split por sujeto y las stats de train. NO se reimplementa nada de eso acá.
from src.lstm import (
    NightSequenceDataset,
    collate_nights,
    split_subjects,
    feature_stats,
    set_seed,
    UNKNOWN,
    META_COLS,
)

# DIMENSIONES
#   B = batch size: cantidad de noches (secuencias) en el batch.
#   T = timesteps: cantidad de épocas de una noche (variable por noche).
#   F = features por época (=122): también es la dim de reconstrucción.
#   L = latent_dim: tamaño del embedding por época (bottleneck).


@dataclass
class ConfigAE:
    '''Configuración (dataclass) del autoencoder LSTM.

    Atributos:
        features_path: ruta al epoch_features.csv de entrada.
        hidden_size: tamaño del estado oculto de las LSTM.
        latent_dim: tamaño del bottleneck = embedding por época.
        num_layers: nº de capas de las LSTM.
        dropout: dropout entre capas de las LSTM.
        batch_size, lr, weight_decay, epochs, grad_clip: hiperparámetros de optimización.
        val_frac, test_frac: fracciones del split por sujeto (sujetos disjuntos).
        seed: semilla de reproducibilidad.
        device: dispositivo de cómputo ('cuda'/'cpu').
        ckpt_path: ruta del checkpoint del mejor modelo.
        embeddings_path: ruta del parquet de embeddings.
        feature_cols, input_size: se completan en runtime al armar los loaders.
    '''
    features_path: str = '../../data_extraction/epoch_features.csv'

    # arquitectura
    hidden_size: int = 128
    latent_dim: int = 32       # tamaño del bottleneck = embedding por época
    num_layers: int = 2
    dropout: float = 0.3

    # optimización
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 60
    grad_clip: float = 5.0

    # split por sujeto (sujetos disjuntos), fracciones sobre el total de sujetos
    val_frac: float = 0.15
    test_frac: float = 0.15

    seed: int = 36631
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path: str = '../../models/best_ae.pt'
    embeddings_path: str = '../../data_extraction/ae_embeddings.parquet'

    # se completan en runtime
    feature_cols: list = field(default=None)
    input_size: int = None


class SeqAutoencoder(nn.Module):
    '''Autoencoder secuencial determinístico (representación NO supervisada) sobre secuencias
    de features por noche. El encoder es un BiLSTM que concatena forward/backward y proyecta a
    `latent` (z_t, embedding por época); el decoder es una LSTM unidireccional sobre z_t que
    reconstruye las features. No usa etiquetas: el bottleneck `latent_dim` fuerza a comprimir
    cada época (y su contexto temporal) en un vector de dimensión baja.

    Args (__init__):
        cfg: ConfigAE con la arquitectura (input_size, hidden_size, latent_dim, num_layers,
            dropout).

    Atributos:
        encoder_lstm: BiLSTM del encoder.
        to_latent: Linear(2*hidden -> latent) que produce z_t.
        decoder_lstm: LSTM unidireccional del decoder.
        to_recon: Linear(hidden -> F) que produce la reconstrucción.

    Métodos:
        encode: features [B, T, F] -> embedding z [B, T, L].
        decode: z [B, T, L] -> reconstrucción x_hat [B, T, F].
        forward: devuelve (x_hat, z).
    '''
    def __init__(self, cfg: ConfigAE):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.to_latent = nn.Linear(cfg.hidden_size * 2, cfg.latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.to_recon = nn.Linear(cfg.hidden_size, cfg.input_size)

    def _run_lstm(self, lstm, x, lengths, total_length):
        '''pack -> LSTM -> unpad, para que el padding no contamine los estados.

        Args:
            lstm: capa LSTM a aplicar (encoder o decoder).
            x: entrada [B, T, *].
            lengths: longitud real [B] de cada secuencia.
            total_length: T al que re-paddear la salida.

        Returns:
            Salida de la LSTM [B, T, H] con el padding restaurado.
        '''
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                     total_length=total_length)
        return out

    def encode(self, feats, lengths):
        '''Codifica las features por época en el embedding latente.

        Args:
            feats: features por época [B, T, F].
            lengths: longitud real [B] de cada noche.

        Returns:
            Embedding z [B, T, L] (uno por época).
        '''
        T = feats.shape[1]
        h = self._run_lstm(self.encoder_lstm, feats, lengths, T)  # [B, T, 2*hidden]
        return self.to_latent(h)                                  # [B, T, L]

    def decode(self, z, lengths):
        '''Reconstruye las features por época a partir del embedding.

        Args:
            z: embedding por época [B, T, L].
            lengths: longitud real [B] de cada noche.

        Returns:
            Reconstrucción x_hat [B, T, F].
        '''
        T = z.shape[1]
        g = self._run_lstm(self.decoder_lstm, z, lengths, T)      # [B, T, hidden]
        return self.to_recon(g)                                   # [B, T, F]

    def forward(self, feats, lengths):
        '''Codifica y reconstruye una secuencia de features por noche.

        Args:
            feats: features por época [B, T, F].
            lengths: longitud real [B] de cada noche.

        Returns:
            Tupla (x_hat [B, T, F], z [B, T, L]).
        '''
        z = self.encode(feats, lengths)
        x_hat = self.decode(z, lengths)
        return x_hat, z


def masked_mse(x_hat, x, lengths):
    '''MSE de reconstrucción promediando sobre features y sobre los timesteps válidos. Los
    timesteps de padding (>= length) no contribuyen a la loss.

    Args:
        x_hat: reconstrucción [B, T, F].
        x: entrada estandarizada [B, T, F].
        lengths: longitud real [B] de cada noche.

    Returns:
        Escalar (tensor) con el MSE enmascarado.
    '''
    T = x.shape[1]
    mask = torch.arange(T, device=x.device)[None, :] < lengths.to(x.device)[:, None]  # [B, T]
    se = ((x_hat - x) ** 2).mean(dim=-1)   # [B, T] promedio sobre F
    return (se * mask).sum() / mask.sum().clamp(min=1)


def make_loaders(cfg: ConfigAE):
    '''Carga las features, splitea por sujeto, estandariza con stats del train y arma los
    DataLoaders. A diferencia de la LSTM supervisada, el AE NO descarta las épocas Unknown
    (label=5): entran al entrenamiento no supervisado.

    Args:
        cfg: configuración del autoencoder (se completan cfg.feature_cols y cfg.input_size).

    Returns:
        Tupla (loaders, (mean, std), subj, splits), donde loaders es un dict train/val/test.
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
        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=(name == 'train'),
            collate_fn=collate_nights,
        )
    return loaders, (mean, std), subj, splits


@torch.no_grad()
def eval_recon(model, loader, device):
    '''Reconstruction loss promedio (ponderada por noches) sobre un loader.

    Args:
        model: autoencoder entrenado.
        loader: DataLoader a evaluar.
        device: dispositivo de cómputo.

    Returns:
        Reconstruction loss promedio (float).
    '''
    model.eval()
    total, n = 0.0, 0
    for feats, _labels, lengths in loader:
        feats = feats.to(device)
        x_hat, _z = model(feats, lengths)
        total += masked_mse(x_hat, feats, lengths).item() * feats.shape[0]
        n += feats.shape[0]
    return total / max(n, 1)


def train(cfg: ConfigAE):
    '''Entrena el autoencoder LSTM y guarda el mejor checkpoint por VAL reconstruction loss
    (mínima). La extracción de embeddings es un paso aparte (extract_embeddings), llamado
    desde el notebook.

    Args:
        cfg: configuración del autoencoder.

    Returns:
        Tupla (model con el mejor checkpoint cargado, history) con train_recon/val_recon por
        epoch.
    '''
    set_seed(cfg.seed)
    device = cfg.device

    loaders, (mean, std), subj, _splits = make_loaders(cfg)
    print(f'sujetos -> train {len(subj["train"])} | val {len(subj["val"])} | test {len(subj["test"])}')

    model = SeqAutoencoder(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_val = np.inf

    pbar = tqdm(range(1, cfg.epochs + 1), desc='entrenando AE', unit='epoch')
    for epoch in pbar:
        model.train()
        total, n = 0.0, 0
        for feats, _labels, lengths in loaders['train']:
            feats = feats.to(device)
            optim.zero_grad()
            x_hat, _z = model(feats, lengths)
            loss = masked_mse(x_hat, feats, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            total += loss.item() * feats.shape[0]
            n += feats.shape[0]

        train_loss = total / max(n, 1)
        val_loss = eval_recon(model, loaders['val'], device)
        history.append({'epoch': epoch, 'train_recon': train_loss, 'val_recon': val_loss})

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'config': cfg,
                'mean': mean, 'std': std,
                'val_recon': best_val, 'epoch': epoch,
            }, cfg.ckpt_path)

        pbar.set_postfix({
            'train_recon': f'{train_loss:.4f}',
            'val_recon': f'{val_loss:.4f}',
            'best': f'{best_val:.4f}{"*" if is_best else ""}',
        })

    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_loss = eval_recon(model, loaders['test'], device)
    print(f'\nTEST (mejor ckpt, val recon {ckpt["val_recon"]:.4f}, epoch {ckpt["epoch"]}):')
    print(f'  recon loss {test_loss:.4f}')

    return model, history


@torch.no_grad()
def extract_embeddings(model, cfg, mean, std):
    '''Extrae el embedding del bottleneck z_t por época para TODOS los sujetos (sin leakage: el
    AE se entrenó solo con sujetos de train). Alinea cada z_t con (subject, night, epoch, label,
    dreem) y lo guarda en cfg.embeddings_path (parquet). Reutiliza NightSequenceDataset para la
    estandarización + imputación de NaN de borde, así el preprocesamiento es idéntico al del
    entrenamiento; procesa una noche por vez (batch de 1 -> sin padding).

    Args:
        model: autoencoder entrenado.
        cfg: configuración del autoencoder.
        mean: media por feature (stats del train) para estandarizar.
        std: desvío por feature (stats del train) para estandarizar.

    Returns:
        DataFrame con (subject, night, epoch, emb_0..emb_{L-1}, label, dreem). Escribe además
        el parquet.
    '''
    device = cfg.device
    model.eval()

    df = pd.read_csv(cfg.features_path)
    feature_cols = cfg.feature_cols if cfg.feature_cols is not None \
        else [c for c in df.columns if c not in META_COLS]

    ds = NightSequenceDataset(df, feature_cols, mean, std)
    emb_cols = [f'emb_{i}' for i in range(cfg.latent_dim)]

    rows_out = []
    for i in tqdm(range(len(ds)), desc='extrayendo embeddings', unit='noche'):
        feats, _labels = ds[i]                       # feats [T, F] estandarizada+imputada
        _key, pos = ds.groups[i]
        meta = ds.df.iloc[pos][['subject', 'night', 'epoch', 'label', 'dreem']]

        lengths = torch.tensor([feats.shape[0]], dtype=torch.long)
        z = model.encode(feats[None].to(device), lengths)[0]  # [T, L]

        emb = pd.DataFrame(z.cpu().numpy(), columns=emb_cols, index=meta.index)
        rows_out.append(pd.concat([meta.reset_index(drop=True),
                                   emb.reset_index(drop=True)], axis=1))

    out = pd.concat(rows_out, ignore_index=True)
    out = out[['subject', 'night', 'epoch'] + emb_cols + ['label', 'dreem']]
    out.to_parquet(cfg.embeddings_path, index=False)
    print(f'embeddings -> {cfg.embeddings_path}  ({out.shape[0]} épocas, {cfg.latent_dim} dims)')
    return out


if __name__ == '__main__':
    cfg = ConfigAE()
    model, history = train(cfg)
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=False)
    extract_embeddings(model, cfg, ckpt['mean'], ckpt['std'])
