'''
Módulo dedicado a visualizaciones de datos y métricas
'''

from pathlib import Path
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from src.data import EDA, GAP_THRESHOLD_S, ACC_TOL

DATA_DIR = Path(__file__).parent.parent / 'report' if '__file__' in dir() else Path('../report')
FIG_DIR = DATA_DIR / 'figures'

STAGE_COLORS = {
    0: 'lightgray',  # Wake
    1: 'lightblue',  # N1
    2: 'lightgreen', # N2
    3: 'lightcoral', # N3
    4: 'plum',       # REM
    5: 'white',      # Unknown
}

STAGE_NAMES = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Unknown'}

def _draw_night_overview(night_data: pd.DataFrame, fontsize_scale: float = 1.0):
    '''Dibuja una noche completa: IHR, su frecuencia de muestreo, acelerometría (x/y/z) y los
    hipnogramas de Expert y Dreem, sin guardar ni mostrar.

    Args:
        night_data: tupla (hr, motion, dreem_labels, expert_labels, rec_start) de EDA.load_night.
        fontsize_scale: factor de escala de las fuentes.

    Returns:
        La figura de matplotlib.
    '''
    hr, motion, dreem_labels, expert_labels, rec_start = night_data

    # descarta timestamps no plausibles (filas corruptas del dataset)
    hr = hr[hr['Timestamp'] > 1e9]
    motion = motion[motion['Timestamp'] > 1e9]

    label_fs = 14 * fontsize_scale
    tick_fs = 12 * fontsize_scale
    legend_fs = 14 * fontsize_scale
    stage_legend_fs = 12 * fontsize_scale

    # ejes: IHR, frecuencia instantánea del IHR (más bajo), acelerometría, expert, dreem
    fig, ax = plt.subplots(5, 1, figsize=(14, 8), sharex=True,
                           gridspec_kw={'height_ratios': [2, 0.9, 2, 0.6, 0.6]})

    # eje x en horas, comenzando desde 0
    t_min = min(hr['Timestamp'].min(), motion['Timestamp'].min())
    hr_hours = (hr['Timestamp'] - t_min) / 3600
    motion_hours = (motion['Timestamp'] - t_min) / 3600

    ax[0].plot(hr_hours, hr['hr'], color='tab:red')

    # frecuencia de muestreo del IHR = 1/Δt entre timestamps consecutivos (~0.2 Hz, irregular)
    ts = hr['Timestamp'].values
    dt = np.diff(ts)
    valid = dt > 0  # Δt=0 (timestamps duplicados) daría frecuencia infinita
    inst_freq = 1.0 / dt[valid]
    ax[1].plot(hr_hours.values[1:][valid], inst_freq, color='tab:orange', linewidth=0.6)
    ax[1].set_yscale('log')
    f_min, f_med, f_max = inst_freq.min(), np.median(inst_freq), inst_freq.max()
    ax[1].set_ylim(f_min, f_max)
    ax[1].set_yticks([f_min, f_med, f_max])
    ax[1].set_yticklabels([f'{f_min:.2g}', f'{f_med:.2g}', f'{f_max:.2g}'])

    ax[2].plot(motion_hours, motion['x'], label='x', color='tab:red')
    ax[2].plot(motion_hours, motion['y'], label='y', color='tab:green')
    ax[2].plot(motion_hours, motion['z'], label='z', color='tab:blue')

    ax[0].set_ylabel('IHR\n[bpm]', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=2)
    ax[1].set_ylabel('IHR\nFreq [Hz]', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=2)
    ax[2].set_ylabel('Accel [g]', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=2)
    # yticks a la derecha, ylabel a la izquierda
    ax[0].tick_params(axis='both', labelsize=tick_fs, left=False, labelleft=False, right=True, labelright=True)
    ax[1].tick_params(axis='both', labelsize=tick_fs, left=False, labelleft=False, right=True, labelright=True)
    ax[2].tick_params(axis='both', labelsize=tick_fs, left=False, labelleft=False, right=True, labelright=True)

    epoch_len = 30  # segundos
    # recStart en hora local (America/New_York); hr/motion en Unix/UTC
    start = pd.Timestamp(str(rec_start), tz='America/New_York').timestamp()

    for i, label in enumerate(expert_labels):
        t0 = (start + i * epoch_len - t_min) / 3600
        t1 = t0 + epoch_len / 3600
        ax[3].axvspan(t0, t1, color=STAGE_COLORS[label], zorder=0)

    for i, label in enumerate(dreem_labels):
        t0 = (start + i * epoch_len - t_min) / 3600
        t1 = t0 + epoch_len / 3600
        ax[4].axvspan(t0, t1, color=STAGE_COLORS[label], zorder=0)

    ax[3].set_yticks([])
    ax[3].tick_params(axis='both', labelsize=tick_fs)
    ax[3].set_ylabel('Expert', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=2)

    ax[4].set_yticks([])
    ax[4].tick_params(axis='both', labelsize=tick_fs)
    ax[4].set_ylabel('Dreem', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=2)

    legend_patches = [mpatches.Patch(color=c, label=STAGE_NAMES[s]) for s, c in STAGE_COLORS.items()]
    ax[4].legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -1.6), ncol=6, fontsize=stage_legend_fs)

    ax[2].legend(loc='upper right', ncol=3, fontsize=legend_fs, handlelength=1, handletextpad=0.4)
    ax[0].grid(True, alpha=0.4, linestyle='--')
    ax[1].grid(True, alpha=0.4, linestyle='--')
    ax[2].grid(True, alpha=0.4, linestyle='--')

    ax[0].set_ylim(hr['hr'].min(), hr['hr'].max())
    ax[2].set_ylim(motion[['x', 'y', 'z']].min().min(), motion[['x', 'y', 'z']].max().max())

    ax[4].set_xlabel('Time [h]', fontsize=tick_fs)

    t_max = max(hr['Timestamp'].max(), motion['Timestamp'].max())
    ax[4].set_xlim(0, (t_max - t_min) / 3600)

    fig.align_ylabels(ax)
    plt.tight_layout()

    return fig

def night_overview(night_data: pd.DataFrame, patient_nr: int = None, night_nr: int = None):
    '''Guarda el overview de una noche en report/figures/ y lo muestra.

    Args:
        night_data: tupla (hr, motion, dreem_labels, expert_labels, rec_start) de EDA.load_night.
        patient_nr: número de paciente; si se pasa, se usa en el nombre del archivo y el título.
        night_nr: número de noche; si se pasa, se usa en el nombre del archivo y el título.

    Returns:
        None. Escribe el PNG en disco y muestra la figura.
    '''
    save_fig = _draw_night_overview(night_data, fontsize_scale=1.5)
    file_name = f'night-overview-{patient_nr}-{night_nr}' if (patient_nr is not None and night_nr is not None) else 'night-overview'
    save_fig.savefig(FIG_DIR / f'{file_name}.png')
    plt.close(save_fig)

    show_fig = _draw_night_overview(night_data, fontsize_scale=1.0)
    if (patient_nr is not None and night_nr is not None):
        show_fig.suptitle(f'Night {night_nr} overview of Patient {patient_nr:02d}', fontsize=16)
        show_fig.tight_layout()

    plt.show()

def _resolved_valid_window(patient: int, night: int):
    '''Ventana válida de la noche con la MISMA resolución de gaps internos que usan
    feature_extraction y build_night_sequences (trim_tail recorta valid_end).

    Args:
        patient: número de paciente.
        night: número de noche.

    Returns:
        Tupla (valid_start_s, valid_end_s) en segundos Unix.
    '''
    vs, ve = EDA.valid_windows()[(patient, night)]
    res = EDA.internal_gap_resolution().get((patient, night))
    if res and res.get('action') == 'trim_tail' and res.get('new_valid_end') is not None:
        ve = res['new_valid_end']
    return vs, ve

def _draw_night_prediction_overview(patient: int, night: int, predictions: dict,
                                    fontsize_scale: float = 1.0, signal_ratio: float = 2.5):
    '''Como night_overview pero centrado en las ETIQUETAS: dibuja IHR + acelerometría y, debajo,
    un hipnograma (banda de colores por época) para Expert, Dreem y cada modelo. Las épocas sin
    predicción quedan Unknown (blanco).

    Args:
        patient: número de paciente.
        night: número de noche.
        predictions: {nombre: (epochs, y_pred)} como los devuelve lstm.predict_night (épocas
            alineadas al índice de la ventana limpia).
        fontsize_scale: factor de escala de las fuentes.
        signal_ratio: alto relativo de los ax de señales (IHR/accel) frente a las bandas de
            labels (fijas en 0.55); bajarlo achata las señales sin tocar los hipnogramas.

    Returns:
        La figura de matplotlib.
    '''
    vs, ve = _resolved_valid_window(patient, night)
    hr, motion, dreem, expert, start = EDA.load_night_clean(patient, night, vs, ve)
    hr = hr[hr['Timestamp'] > 1e9]
    motion = motion[motion['Timestamp'] > 1e9]

    n_ep = len(expert)
    rows = {'Expert': np.asarray(expert, dtype=int), 'Dreem': np.asarray(dreem, dtype=int)}
    for name, (epochs, preds) in predictions.items():
        arr = np.full(n_ep, 5, dtype=int)           # 5 = Unknown por defecto (blanco)
        epochs = np.asarray(epochs); preds = np.asarray(preds)
        keep = (epochs >= 0) & (epochs < n_ep)
        arr[epochs[keep]] = preds[keep]
        rows[name] = arr

    label_fs = 13 * fontsize_scale
    tick_fs = 11 * fontsize_scale
    legend_fs = 11 * fontsize_scale

    n_bands = len(rows)
    band_h = 0.55
    height_ratios = [signal_ratio, signal_ratio] + [band_h] * n_bands
    fig_h = 2 * signal_ratio + band_h * n_bands
    fig, ax = plt.subplots(2 + n_bands, 1, figsize=(14, fig_h), sharex=True,
                           gridspec_kw={'height_ratios': height_ratios})

    hr_h = (hr['Timestamp'] - start) / 3600
    mo_h = (motion['Timestamp'] - start) / 3600
    ax[0].plot(hr_h, hr['hr'], color='tab:red')
    ax[0].set_ylabel('IHR [bpm]', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=2)
    ax[1].plot(mo_h, motion['x'], label='x', color='tab:red')
    ax[1].plot(mo_h, motion['y'], label='y', color='tab:green')
    ax[1].plot(mo_h, motion['z'], label='z', color='tab:blue')
    ax[1].set_ylabel('Accel [g]', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=2)
    ax[1].legend(loc='upper right', ncol=3, fontsize=legend_fs, handlelength=1, handletextpad=0.4)
    for a in (ax[0], ax[1]):
        a.grid(True, alpha=0.4, linestyle='--')
        # yticks a la derecha, ylabel a la izquierda
        a.tick_params(axis='both', labelsize=tick_fs, left=False, labelleft=False,
                      right=True, labelright=True)

    for r, (name, labels) in enumerate(rows.items()):
        a = ax[2 + r]
        for i, lab in enumerate(labels):
            a.axvspan(i * 30 / 3600, (i + 1) * 30 / 3600, color=STAGE_COLORS[int(lab)], zorder=0)
        a.set_yticks([])
        a.set_ylabel(name, fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=2)
        a.tick_params(axis='both', labelsize=tick_fs)

    legend_patches = [mpatches.Patch(color=c, label=STAGE_NAMES[s]) for s, c in STAGE_COLORS.items()]
    ax[-1].legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -1.4),
                  ncol=6, fontsize=legend_fs)
    ax[-1].set_xlabel('Time [h]', fontsize=tick_fs)

    t_max = max(hr['Timestamp'].max(), motion['Timestamp'].max())
    ax[0].set_xlim(0, (t_max - start) / 3600)
    fig.align_ylabels(ax)
    plt.tight_layout()
    return fig

def night_prediction_overview(patient: int, night: int, predictions: dict, save: bool = False):
    '''Dibuja el overview de una noche con las predicciones de los modelos superpuestas como
    hipnogramas.

    Args:
        patient: número de paciente.
        night: número de noche.
        predictions: {nombre: (epochs, y_pred)} (salida de lstm.predict_night). Ej.:
            {'LSTM tabular': (e1, p1), 'Híbrido': (e2, p2)}.
        save: si es True, guarda el PNG en report/figures/ además de mostrarlo.

    Returns:
        None. Muestra la figura (y opcionalmente la guarda).
    '''
    if save:
        save_fig = _draw_night_prediction_overview(patient, night, predictions,
                                                   fontsize_scale=1.5, signal_ratio=1.8)
        save_fig.tight_layout()
        save_fig.savefig(FIG_DIR / f'night-predictions-{patient}-{night}.png', dpi=200,
                         bbox_inches='tight')
        plt.close(save_fig)

    fig = _draw_night_prediction_overview(patient, night, predictions, fontsize_scale=1.0)
    fig.suptitle(f'Night {night} — Patient {patient:02d}: expert / dreem / predicciones', fontsize=15)
    fig.tight_layout()
    plt.show()

def _raw_vs_clean_overview(patient: int, night: int, fontsize_scale: float = 1.0, title: bool = True):
    '''Overview estilo night_overview que muestra una noche CRUDA y el efecto del procesamiento:
    IHR y acelerometría con la ventana válida sombreada (verde) y lo removido en gris, más tres
    hipnogramas: Expert (cruda), Dreem (cruda) y Expert (procesada) donde las épocas descartadas
    (fuera de la ventana o sin cobertura de ambas señales: IHR<2 o acc<10) quedan en blanco.
    Reconstruye la decisión de procesamiento en vivo (ventana válida resuelta + filtro por época),
    así refleja el pipeline sin depender de los CSV/npz ya generados.

    Args:
        patient: número de paciente.
        night: número de noche.
        fontsize_scale: factor de escala de las fuentes.
        title: si es True, agrega un suptitle con la ventana y las épocas conservadas.

    Returns:
        La figura de matplotlib.
    '''
    hr, motion, dreem, expert, rec_start = EDA.load_night(patient, night)
    hr = hr[hr['Timestamp'] > 1e9]
    motion = motion[motion['Timestamp'] > 1e9]
    start = pd.Timestamp(str(rec_start), tz='America/New_York').timestamp()

    vs, ve = _resolved_valid_window(patient, night)
    gap_res = EDA.internal_gap_resolution().get((patient, night))
    discarded = bool(gap_res and gap_res.get('action') == 'discard')

    # validez por época (mismo criterio que feature_extraction / build_night_sequences)
    n_ep = len(expert)
    starts = start + np.arange(n_ep) * 30
    hr_ts, mo_ts = hr['Timestamp'].values, motion['Timestamp'].values
    hr_cnt = np.searchsorted(hr_ts, starts + 30) - np.searchsorted(hr_ts, starts)
    mo_cnt = np.searchsorted(mo_ts, starts + 30) - np.searchsorted(mo_ts, starts)
    in_win = (starts >= vs) & (starts < ve)
    kept = in_win & (hr_cnt >= 2) & (mo_cnt >= 10) & (not discarded)

    label_fs = 13 * fontsize_scale
    tick_fs = 11 * fontsize_scale
    legend_fs = 11 * fontsize_scale

    fig, ax = plt.subplots(5, 1, figsize=(14, 7), sharex=True,
                           gridspec_kw={'height_ratios': [2.2, 2.2, 0.6, 0.6, 0.6]})

    t_min = min(float(hr['Timestamp'].min()), float(motion['Timestamp'].min()))
    # la vista se limita al alcance real de la noche (IHR / etiquetas): motion suele traer
    # muestras espurias muy posteriores (hasta decenas de horas) que el procesamiento remueve.
    label_end = start + n_ep * 30
    t_max = label_end  # la ventana etiquetada define la noche; señal más allá es excedente/espuria
    to_h = lambda t: (t - t_min) / 3600
    vs_h, ve_h, end_h = to_h(vs), to_h(ve), to_h(t_max)

    ax[0].plot(to_h(hr['Timestamp']), hr['hr'], color='tab:red')
    ax[0].set_ylabel('IHR [bpm]', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=8)
    ax[0].set_yticks([])
    hr_view = hr[to_h(hr['Timestamp']).between(0, to_h(t_max))]
    if len(hr_view):
        ax[0].set_ylim(float(hr_view['hr'].min()), float(hr_view['hr'].max()))
    ax[1].plot(to_h(motion['Timestamp']), motion['x'], label='x', color='tab:red')
    ax[1].plot(to_h(motion['Timestamp']), motion['y'], label='y', color='tab:green')
    ax[1].plot(to_h(motion['Timestamp']), motion['z'], label='z', color='tab:blue')
    ax[1].set_ylabel('Accel [g]', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=8)
    ax[1].set_yticks([])
    # límites Y según lo que cae dentro de la vista (evita que las muestras espurias los estiren)
    mo_view = motion[to_h(motion['Timestamp']).between(0, end_h)]
    if len(mo_view):
        lo = float(mo_view[['x', 'y', 'z']].min().min())
        hi = float(mo_view[['x', 'y', 'z']].max().max())
        ax[1].set_ylim(lo, hi)
    ax[1].legend(loc='upper right', ncol=3, fontsize=legend_fs, handlelength=1, handletextpad=0.4)

    # ventana válida (verde) y tramos removidos (gris) sobre las señales
    for a in (ax[0], ax[1]):
        if not discarded and ve_h > vs_h:
            a.axvspan(vs_h, ve_h, color='mediumseagreen', alpha=0.13, zorder=0)
            a.axvline(vs_h, color='seagreen', linewidth=1.1, zorder=1)
            a.axvline(ve_h, color='seagreen', linewidth=1.1, zorder=1)
            if vs_h > 0:
                a.axvspan(0, vs_h, color='gray', alpha=0.18, zorder=0)
            if end_h > ve_h:
                a.axvspan(ve_h, end_h, color='gray', alpha=0.18, zorder=0)
        else:  # noche descartada: todo removido
            a.axvspan(0, end_h, color='gray', alpha=0.22, zorder=0)
        a.grid(True, alpha=0.4, linestyle='--')
        a.tick_params(axis='both', labelsize=tick_fs)

    # gaps de señal (Δt > GAP_THRESHOLD_S) dentro de la vista, en rojo: marcan las
    # discontinuidades que motivan el recorte/descarte (p. ej. el gap interno de IHR de P42 N4).
    def _mark_gaps(a, ts_arr):
        if len(ts_arr) < 2:
            return
        d = np.diff(ts_arr)
        for i in np.where(d > GAP_THRESHOLD_S)[0]:
            a0, a1 = max(ts_arr[i], t_min), min(ts_arr[i + 1], t_max)
            if a1 > a0:
                a.axvspan(to_h(a0), to_h(a1), color='red', alpha=0.35, zorder=1)
    _mark_gaps(ax[0], hr_ts)
    _mark_gaps(ax[1], mo_ts)

    def _band(a, labels, mask=None):
        for i, lab in enumerate(labels):
            col = STAGE_COLORS[int(lab)] if (mask is None or mask[i]) else 'white'
            a.axvspan((start + i * 30 - t_min) / 3600, (start + (i + 1) * 30 - t_min) / 3600,
                      color=col, zorder=0)
        a.set_yticks([])
        a.tick_params(axis='both', labelsize=tick_fs)

    _band(ax[2], expert)
    ax[2].set_ylabel('Expert\n(raw)', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=8)
    _band(ax[3], dreem)
    ax[3].set_ylabel('Dreem\n(raw)', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=8)
    _band(ax[4], expert, kept)
    ax[4].set_ylabel('Expert\n(proccesed)', fontsize=label_fs, rotation=0, ha='right', va='center', labelpad=8)

    legend_patches = [mpatches.Patch(color=c, label=STAGE_NAMES[s]) for s, c in STAGE_COLORS.items()]
    legend_patches.append(mpatches.Patch(facecolor='red', alpha=0.35, label='gap señal (Δt>60s)'))
    ax[4].legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -1.4),
                 ncol=7, fontsize=legend_fs)
    ax[4].set_xlabel('Time [h]', fontsize=tick_fs)
    ax[0].set_xlim(0, end_h)
    fig.align_ylabels(ax)
    plt.tight_layout()

    if title:
        n_kept = int(kept.sum())
        estado = 'DESCARTADA' if discarded else f'{n_kept}/{n_ep} épocas conservadas'
        fig.suptitle(f'Cruda vs procesada — P{patient:02d} N{night}  ·  ventana '
                     f'[{vs_h:.2f}, {ve_h:.2f}] h  ·  {estado}', fontsize=14)
        fig.tight_layout()
    return fig

def raw_vs_clean_overview(patient: int, night: int, save: bool = False):
    '''Dibuja para una noche la señal cruda y el resultado del procesamiento (ventana válida +
    filtro por época): night_overview con la ventana sombreada y el hipnograma Expert antes
    (cruda) y después (procesada).

    Args:
        patient: número de paciente.
        night: número de noche.
        save: si es True, guarda el PNG en report/figures/ además de mostrarlo.

    Returns:
        None. Muestra la figura (y opcionalmente la guarda).
    '''
    if save:
        save_fig = _raw_vs_clean_overview(patient, night, fontsize_scale=1.3, title=False)
        save_fig.savefig(FIG_DIR / f'raw-vs-clean-{patient}-{night}.png', dpi=200, bbox_inches='tight')
        plt.close(save_fig)

    _raw_vs_clean_overview(patient, night, fontsize_scale=1.3)
    plt.show()

def _draw_class_distribution(distribution: dict, fontsize_scale: float = 1.0):
    '''Barras del porcentaje de épocas por etapa, Expert vs Dreem.

    Args:
        distribution: {'expert': counts, 'dreem': counts}, conteos por etapa (0..5).
        fontsize_scale: factor de escala de las fuentes.

    Returns:
        Tupla (fig, ax) de matplotlib.
    '''
    label_fs = 12 * fontsize_scale
    tick_fs = 0 * fontsize_scale
    legend_fs = 11 * fontsize_scale

    expert_pct = 100 * distribution['expert'] / distribution['expert'].sum()
    dreem_pct = 100 * distribution['dreem'] / distribution['dreem'].sum()

    classes = list(STAGE_NAMES.values())
    colors = list(STAGE_COLORS.values())
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, expert_pct, width, color=colors, edgecolor='black', hatch='', label='Expert')
    ax.bar(x + width/2, dreem_pct, width, color=colors, edgecolor='black', hatch='//', label='Dreem')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=tick_fs)
    ax.set_ylabel('Epochs percentage [%]', fontsize=label_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)

    legend_patches = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Expert'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Dreem'),
    ]
    ax.legend(handles=legend_patches, fontsize=legend_fs)
    ax.grid(True, axis='y', alpha=0.4, linestyle='--')

    plt.tight_layout()

    return fig, ax

def class_distribution(distribution: dict):
    '''Guarda la distribución de etapas en report/figures/ y la muestra con título.

    Args:
        distribution: {'expert': counts, 'dreem': counts}, conteos por etapa (0..5).

    Returns:
        None. Escribe el PNG en disco y muestra la figura.
    '''
    save_fig, _ = _draw_class_distribution(distribution, fontsize_scale=1.3)
    save_fig.savefig(FIG_DIR / 'stages-distribution.png')
    plt.close(save_fig)

    show_fig, show_ax = _draw_class_distribution(distribution, fontsize_scale=1.0)
    show_ax.set_title('Sleep Stages Distribution')
    show_fig.tight_layout()

    plt.show()

def _draw_confusion_matrix(cm: np.ndarray, fontsize_scale: float = 1.0):
    '''Heatmap de la matriz de confusión, normalizada por fila y anotada con conteo y porcentaje.

    Args:
        cm: matriz de confusión (filas = Expert, columnas = Dreem), tamaño 6x6.
        fontsize_scale: factor de escala de las fuentes.

    Returns:
        Tupla (fig, ax) de matplotlib.
    '''
    label_fs = 12 * fontsize_scale
    tick_fs = 11 * fontsize_scale
    annot_fs = 11 * fontsize_scale

    classes = list(STAGE_NAMES.values())
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=tick_fs)
    ax.set_yticklabels(classes, fontsize=tick_fs)
    ax.set_xlabel('Dreem', fontsize=label_fs)
    ax.set_ylabel('Expert', fontsize=label_fs)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                    ha='center', va='center', color=color, fontsize=annot_fs)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=tick_fs)

    plt.tight_layout()

    return fig, ax

def confusion_matrix(cm: np.ndarray):
    '''Guarda la matriz de confusión Expert vs Dreem en report/figures/ y la muestra.

    Args:
        cm: matriz de confusión (filas = Expert, columnas = Dreem), tamaño 6x6.

    Returns:
        None. Escribe el PNG en disco y muestra la figura.
    '''
    save_fig, _ = _draw_confusion_matrix(cm, fontsize_scale=1.3)
    save_fig.savefig(FIG_DIR / 'label-confusion-matrix.png')
    plt.close(save_fig)

    show_fig, show_ax = _draw_confusion_matrix(cm, fontsize_scale=1.0)
    show_ax.set_title('Expert vs Dreem Confusion Matrix')
    show_fig.tight_layout()

    plt.show()

def problematic_nights_overview(json_path: Path = None, ncols: int = 4):
    '''Grilla con las noches problemáticas de problematic_nights.json: por noche, IHR y
    desviación de la acelerometría con la ventana válida (verde), los extremos truncados
    (naranja) y los gaps internos (rojo). Muestra sólo las que pierden labels (truncamiento
    de extremo o gap interno por encima del umbral).

    Args:
        json_path: ruta al problematic_nights.json; por defecto analysis/problematic_nights.json.
        ncols: número de columnas de la grilla.

    Returns:
        None. Escribe el PNG en report/figures/ y muestra la figura.
    '''
    ANALYSIS_DIR = Path(__file__).parent.parent / 'analysis'
    if json_path is None:
        json_path = ANALYSIS_DIR / 'problematic_nights.json'

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    internal_gap_threshold = data['internal_gap_threshold_s']
    edge_trunc_threshold = data['edge_trunc_threshold_s']

    # el overview muestra sólo las noches que pierden labels o tienen gaps
    # (truncamiento de extremo significativo o gap interno); las de puro exceso
    # de señal sobrante se recortan trivialmente y quedan sólo en el JSON.
    discarded = [
        e for e in data['problematic']
        if e['leading_trunc_s'] > edge_trunc_threshold
        or e['trailing_trunc_s'] > edge_trunc_threshold
        or e['internal_gap_s'] > internal_gap_threshold
    ]

    n = len(discarded)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(ncols * 4.5, nrows * 3.2))

    outer = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.15, wspace=0.05,
                              top=1, bottom=0.01, left=0.01, right=1)

    for idx, entry in enumerate(discarded):
        col = idx % ncols
        row = idx // ncols

        # gridspec interno: espacio mínimo entre IHR y acc dentro del bloque
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col],
            height_ratios=[3, 1], hspace=0.08,
        )
        ax_hr = fig.add_subplot(inner[0])
        ax_acc = fig.add_subplot(inner[1])

        patient = entry['patient']
        night = entry['night']
        internal_gap_s = entry['internal_gap_s']
        leading_trunc_s = entry['leading_trunc_s']
        trailing_trunc_s = entry['trailing_trunc_s']
        valid_start = entry['valid_start_s']
        valid_end = entry['valid_end_s']

        hr, motion, _, expert_labels, rec_start = EDA.load_night(patient, night)

        t_min = float(hr['Timestamp'].min())
        t_max = float(hr['Timestamp'].max())
        hr_hours = (hr['Timestamp'] - t_min) / 3600
        motion_hours = (motion['Timestamp'] - t_min) / 3600

        start = pd.Timestamp(str(rec_start), tz='America/New_York').timestamp()
        label_span_s = len(expert_labels) * 30
        label_start_h = (start - t_min) / 3600
        label_end_h = (start + label_span_s - t_min) / 3600

        valid_start_h = (valid_start - t_min) / 3600
        valid_end_h = (valid_end - t_min) / 3600

        def draw_windows(ax):
            # ventana válida (lo que se conserva): verde visible + bordes marcados
            ax.axvspan(valid_start_h, valid_end_h, color='mediumseagreen', alpha=0.22, zorder=0)
            ax.axvline(valid_start_h, color='seagreen', linewidth=1.1, zorder=1)
            ax.axvline(valid_end_h, color='seagreen', linewidth=1.1, zorder=1)
            # extremos etiquetados que se truncan (labels sin señal): naranja
            if valid_start_h > label_start_h:
                ax.axvspan(label_start_h, valid_start_h, color='darkorange', alpha=0.3, zorder=1)
            if label_end_h > valid_end_h:
                ax.axvspan(valid_end_h, label_end_h, color='darkorange', alpha=0.3, zorder=1)

        # IHR
        ax_hr.plot(hr_hours, hr['hr'], color='steelblue', linewidth=0.5, zorder=2)
        draw_windows(ax_hr)

        # gaps internos dentro de la ventana válida (criterio de reparación/descarte): rojo
        hr_ts = hr['Timestamp'].values
        diffs = np.diff(hr_ts)
        for i, d in enumerate(diffs):
            if d > GAP_THRESHOLD_S:
                g0, g1 = hr_ts[i], hr_ts[i + 1]
                if g1 > valid_start and g0 < valid_end:
                    ax_hr.axvspan((max(g0, valid_start) - t_min) / 3600,
                                  (min(g1, valid_end) - t_min) / 3600,
                                  color='red', alpha=0.45, zorder=2)

        # Acelerometría (desviación de la norma respecto a 1g)
        acc_norm = np.sqrt(motion['x']**2 + motion['y']**2 + motion['z']**2)
        acc_dev = (acc_norm - 1).abs()
        ax_acc.plot(motion_hours, acc_dev, color='dimgray', linewidth=0.4, zorder=2)
        ax_acc.axhline(ACC_TOL, color='red', linewidth=0.6, linestyle='--', zorder=3)
        ax_acc.fill_between(motion_hours, ACC_TOL, acc_dev,
                            where=(acc_dev > ACC_TOL), color='red', alpha=0.35, zorder=1)
        draw_windows(ax_acc)

        # criterios marcados (solo significativos)
        failing = []
        if internal_gap_s > internal_gap_threshold:
            failing.append(f'gap={internal_gap_s/60:.0f}min')
        if leading_trunc_s > edge_trunc_threshold:
            failing.append(f'lead={leading_trunc_s/3600:.1f}h')
        if trailing_trunc_s > edge_trunc_threshold:
            failing.append(f'trail={trailing_trunc_s/3600:.1f}h')

        ax_hr.set_title(f'P{patient:02d} N{night}', fontsize=10, pad=2)
        ax_hr.text(0.5, 0.96, '  '.join(failing), transform=ax_hr.transAxes,
                   color='red', fontsize=8.5, ha='center', va='top', fontweight='bold')

        x_end = (t_max - t_min) / 3600
        hour_ticks = list(range(1, int(x_end) + 1))
        tick_labels = [str(h) if h == int(x_end) else '' for h in hour_ticks]

        for ax in (ax_hr, ax_acc):
            ax.set_yticks([])
            ax.set_xticks(hour_ticks)
            ax.set_xticklabels(tick_labels, fontsize=7.5)
            ax.tick_params(axis='x', length=2, pad=1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

    save_path = FIG_DIR / 'problematic-nights-overview.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()