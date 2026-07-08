'''
Búsqueda bayesiana de hiperparámetros (Optuna) para los modelos secuenciales de
lstm.py (LSTM tabular e híbrido CNN1D->BiLSTM).

Cada estudio maximiza el Cohen's Kappa de validación (sampler TPE + MedianPruner)
y persiste en `models/searchs/<name>.db` (SQLite, resume entre corridas) y
`<name>.csv` (log legible con toda la config + métricas de val, ordenado por
kappa desc). `run_search` corre trials; `best_config` levanta la config ganadora.
'''

from dataclasses import fields, replace
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
import optuna

try:
    from lstm import ConfigLSTM, train
except ImportError:
    from src.lstm import ConfigLSTM, train

SEARCH_DIR = Path('../models/searchs')

# campos de ConfigLSTM que se completan en runtime, se excluyen del CSV.
_RUNTIME_FIELDS = {'feature_cols', 'input_size'}

# métricas de val que se loguean por trial
_METRIC_COLS = ['kappa', 'macro_f1', 'accuracy']

# campos que forman parte del espacio de búsqueda
_INT_TUNABLE = {'hidden_size', 'num_layers', 'batch_size', 'feature_dim'}
_FLOAT_TUNABLE = {'dropout', 'lr', 'weight_decay'}
_TUNABLE = _INT_TUNABLE | _FLOAT_TUNABLE

def default_space(trial: optuna.Trial, hybrid: bool) -> dict:
    '''
    Espacio de búsqueda default. Devuelve un dict de hiperparámetros para `replace` sobre
    la config base. El híbrido agrega `feature_dim` (dimensión de salida del CNNEpochEncoder).
    '''
    # num_layers arranca en 2: con 1 capa rindió consistentemente peor en ambos modelos (search).
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', choices=[64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', low=2, high=3),
        'dropout': trial.suggest_float('dropout', low=0.1, high=0.5),
        'lr': trial.suggest_float('lr', low=1e-4, high=5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', low=1e-6, high=1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', choices=[4, 8, 16]),
    }
    if hybrid:
        # feature_dim sin 64: el 256 (y en menor medida 128) dominaron el top del híbrido.
        params['feature_dim'] = trial.suggest_categorical('feature_dim', choices=[128, 256])
    return params

def run_search(base_cfg: ConfigLSTM, name: str, n_trials: int, space=default_space,
               search_epochs=None, show_progress: bool = True) -> optuna.Study:
    '''
    Corre `n_trials` de búsqueda bayesiana sobre `base_cfg` y persiste el resultado.

    - `base_cfg`: config base; sus campos NO sampleados (seed, val_frac/test_frac, epochs,
      paths, hybrid, ...) quedan fijos -> el split por sujeto es idéntico entre trials y modelos.
    - `name`: nombre del estudio y de los archivos (`lstm` / `hybrid`).
    - `search_epochs`: si se pasa, cada trial entrena esas epochs (más cortas que el final) para
      abaratar el search; el ranking por kappa de val se mantiene. `None` = usa `base_cfg.epochs`.
    - persiste en `models/searchs/<name>.db` (SQLite, resume) y `models/searchs/<name>.csv`
      (log legible con toda la config + kappa/macro_f1/accuracy de val, ordenado por kappa desc).

    Usa un **MedianPruner** que corta trials cuyo kappa de val va por debajo de la mediana de
    los previos al mismo epoch. Se puede llamar varias veces (en distintas ejecuciones): el
    estudio resume y el CSV acumula. Devuelve el `optuna.Study`.
    '''
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SEARCH_DIR / f'{name}.csv'
    storage = f'sqlite:///{(SEARCH_DIR / f"{name}.db").as_posix()}'

    study = optuna.create_study(
        direction='maximize',
        study_name=name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )

    n_before = len(study.trials)
    # Seed del sampler = seed base + trials ya existentes. Así cada corrida (resume) usa un RNG
    # distinto pero determinístico -> no re-propone los mismos trials de arranque entre ejecuciones
    # (que con entrenamiento determinístico serían evaluaciones duplicadas), y el TPE igual usa
    # todo el historial del .db.
    study.sampler = optuna.samplers.TPESampler(seed=base_cfg.seed + n_before)
    with tempfile.TemporaryDirectory() as tmp_dir:
        study.optimize(_make_objective(base_cfg, space, tmp_dir, search_epochs),
                       n_trials=n_trials, show_progress_bar=show_progress)

    # trials completados en esta corrida se agregan al CSV
    new_rows = [_trial_row(t, base_cfg) for t in study.trials[n_before:]
                if t.state == optuna.trial.TrialState.COMPLETE]
    if new_rows:
        _append_and_sort(csv_path, new_rows)
    else:
        warnings.warn('ningún trial nuevo completado; el CSV queda igual.')

    best = study.best_trial
    print(f'\n[{name}] mejor kappa de val: {best.value:.4f} (trial {best.number})')
    print(f'  params: {best.params}')
    print(f'  log: {csv_path}  |  storage: {storage}')
    return study

def best_config(base_cfg: ConfigLSTM, name: str) -> ConfigLSTM:
    '''
    Devuelve una `ConfigLSTM` con los mejores hiperparámetros registrados en el CSV del search
    (mayor kappa de val), sobre la config base. Útil para re-entrenar la config ganadora con
    todas las épocas / guardarla en el checkpoint real.
    '''
    csv_path = SEARCH_DIR / f'{name}.csv'
    top = pd.read_csv(csv_path).sort_values('kappa', ascending=False).iloc[0]
    # solo los hiperparámetros del espacio de búsqueda; el resto queda como en base_cfg.
    # castea a los tipos de la dataclass (el CSV los trae como numpy float/int).
    tuned = {c: (int(top[c]) if c in _INT_TUNABLE else float(top[c]))
             for c in _TUNABLE if c in top.index}
    return replace(base_cfg, **tuned)

def _config_row(cfg: ConfigLSTM) -> dict:
    '''Campos de la config necesarios para reconstruirla (excluye los de runtime).'''
    return {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name not in _RUNTIME_FIELDS}

def _best_val_metrics(history: list) -> dict:
    '''Métricas de val del epoch con mayor kappa (el mismo criterio con que train guarda el ckpt).'''
    best = max(history, key=lambda h: h['kappa'])
    return {k: float(best[k]) for k in _METRIC_COLS}

def _make_objective(base_cfg: ConfigLSTM, space, tmp_dir: str, search_epochs=None):
    '''
    Construye la `objective` de Optuna: cada trial samplea del `space`, entrena
    (checkpoint temporal, `search_epochs` si se dio), reporta el kappa de val por
    epoch al MedianPruner y devuelve el mejor kappa de val (a maximizar).
    '''
    def objective(trial: optuna.Trial) -> float:
        params = space(trial, base_cfg.hybrid)
        # checkpoint temporal por trial: no pisar el best_lstm.pt / best_hybrid.pt reales.
        ckpt = str(Path(tmp_dir) / f'trial_{trial.number}.pt')
        overrides = dict(params)
        if search_epochs is not None:
            overrides['epochs'] = search_epochs  # trials más cortos que el entrenamiento final
        cfg = replace(base_cfg, ckpt_path=ckpt, **overrides)

        # reporta el kappa de val por epoch al pruner -> corta trials sin futuro (MedianPruner).
        def on_epoch(epoch, val_m):
            trial.report(val_m['kappa'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, history, _ = train(cfg, epoch_callback=on_epoch, verbose=False)
        m = _best_val_metrics(history)
        trial.set_user_attr('macro_f1', m['macro_f1'])
        trial.set_user_attr('accuracy', m['accuracy'])
        return m['kappa']  # objetivo = kappa de validación (maximizar)

    return objective

def _trial_row(trial: optuna.Trial, base_cfg: ConfigLSTM) -> dict:
    '''Reconstruye la config completa del trial y le adjunta las métricas de val.'''
    cfg = replace(base_cfg, **trial.params)
    row = _config_row(cfg)
    row['kappa'] = float(trial.value)
    row['macro_f1'] = float(trial.user_attrs['macro_f1'])
    row['accuracy'] = float(trial.user_attrs['accuracy'])
    return row

def _append_and_sort(csv_path: Path, new_rows: list):
    '''Agrega los trials nuevos al CSV (sin reescribir los viejos) y ordena por kappa desc.'''
    df_new = pd.DataFrame(new_rows)
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
    else:
        df = df_new
    df = df.sort_values('kappa', ascending=False).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df