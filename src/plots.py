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
    hr, motion, dreem_labels, expert_labels, rec_start = night_data

    # algunas noches traen filas corruptas Descartamos timestamps no plausibles
    # (NaN o fuera del rango Unix esperable) para que no deformen el eje.
    hr = hr[hr['Timestamp'] > 1e9]
    motion = motion[motion['Timestamp'] > 1e9]

    label_fs = 14 * fontsize_scale
    tick_fs = 12 * fontsize_scale
    legend_fs = 14 * fontsize_scale
    stage_legend_fs = 12 * fontsize_scale

    # ejes: IHR, frecuencia instantánea del IHR (más bajo), acelerometría, expert, dreem
    fig, ax = plt.subplots(5, 1, figsize=(14, 8.4), sharex=True,
                           gridspec_kw={'height_ratios': [3, 0.9, 3, 0.6, 0.6]})

    # eje x en horas, comenzando desde 0
    t_min = min(hr['Timestamp'].min(), motion['Timestamp'].min())
    hr_hours = (hr['Timestamp'] - t_min) / 3600
    motion_hours = (motion['Timestamp'] - t_min) / 3600

    ax[0].plot(hr_hours, hr['hr'], color='tab:red')

    # frecuencia instantánea de muestreo del IHR: 1/Δt entre timestamps consecutivos.
    # El muestreo no es equiespaciado; fluctúa en torno a ~0.2 Hz (ver paper).
    ts = hr['Timestamp'].values
    dt = np.diff(ts)
    valid = dt > 0  # timestamps duplicados (Δt=0) darían frecuencia infinita
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

    ax[0].set_ylabel('IHR [bpm]', fontsize=label_fs)
    ax[1].set_ylabel('IHR Freq [Hz]', fontsize=label_fs)
    ax[2].set_ylabel('Accelerometry [g]', fontsize=label_fs)
    ax[0].tick_params(axis='both', labelsize=tick_fs)
    ax[1].tick_params(axis='both', labelsize=tick_fs)
    ax[2].tick_params(axis='both', labelsize=tick_fs)

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
    ax[3].set_ylabel('Expert', fontsize=label_fs)

    ax[4].set_yticks([])
    ax[4].tick_params(axis='both', labelsize=tick_fs)
    ax[4].set_ylabel('Dreem', fontsize=label_fs)

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
    save_fig = _draw_night_overview(night_data, fontsize_scale=1.1)
    file_name = f'night-overview-{patient_nr}-{night_nr}' if (patient_nr is not None and night_nr is not None) else 'night-overview'
    save_fig.savefig(FIG_DIR / f'{file_name}.png')
    plt.close(save_fig)

    show_fig = _draw_night_overview(night_data, fontsize_scale=1.0)
    if (patient_nr is not None and night_nr is not None):
        show_fig.suptitle(f'Night {night_nr} overview of Patient {patient_nr:02d}', fontsize=16)
        show_fig.tight_layout()

    plt.show()

def _draw_class_distribution(distribution: dict, fontsize_scale: float = 1.0):
    label_fs = 12 * fontsize_scale
    tick_fs = 11 * fontsize_scale
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
    save_fig, _ = _draw_class_distribution(distribution, fontsize_scale=1.3)
    save_fig.savefig(FIG_DIR / 'stages-distribution.png')
    plt.close(save_fig)

    show_fig, show_ax = _draw_class_distribution(distribution, fontsize_scale=1.0)
    show_ax.set_title('Sleep Stages Distribution')
    show_fig.tight_layout()

    plt.show()

def _draw_confusion_matrix(cm: np.ndarray, fontsize_scale: float = 1.0):
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
    save_fig, _ = _draw_confusion_matrix(cm, fontsize_scale=1.3)
    save_fig.savefig(FIG_DIR / 'label-confusion-matrix.png')
    plt.close(save_fig)

    show_fig, show_ax = _draw_confusion_matrix(cm, fontsize_scale=1.0)
    show_ax.set_title('Expert vs Dreem Confusion Matrix')
    show_fig.tight_layout()

    plt.show()

def problematic_nights_overview(json_path: Path = None, ncols: int = 4):

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