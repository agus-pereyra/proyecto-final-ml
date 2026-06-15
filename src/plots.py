'''
Módulo dedicado a visualizaciones de datos y métricas
'''

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

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

    label_fs = 14 * fontsize_scale
    tick_fs = 12 * fontsize_scale
    legend_fs = 14 * fontsize_scale
    stage_legend_fs = 12 * fontsize_scale

    fig, ax = plt.subplots(4, 1, figsize=(14, 7.6), sharex=True, gridspec_kw={'height_ratios': [3, 3, 0.6, 0.6]})

    ax[0].plot(hr['Timestamp'], hr['hr'], color='tab:red')
    ax[1].plot(motion['Timestamp'], motion['x'], label='x', color='tab:red')
    ax[1].plot(motion['Timestamp'], motion['y'], label='y', color='tab:green')
    ax[1].plot(motion['Timestamp'], motion['z'], label='z', color='tab:blue')

    ax[0].set_ylabel('IHR [bpm]', fontsize=label_fs)
    ax[1].set_ylabel('Accelerometry [g]', fontsize=label_fs)
    ax[0].tick_params(axis='both', labelsize=tick_fs)
    ax[1].tick_params(axis='both', labelsize=tick_fs)

    epoch_len = 30  # segundos
    # recStart en hora local (America/New_York); hr/motion en Unix/UTC
    start = pd.Timestamp(str(rec_start), tz='America/New_York').timestamp()

    for i, label in enumerate(expert_labels):
        t0 = start + i * epoch_len
        t1 = t0 + epoch_len
        ax[2].axvspan(t0, t1, color=STAGE_COLORS[label], zorder=0)

    for i, label in enumerate(dreem_labels):
        t0 = start + i * epoch_len
        t1 = t0 + epoch_len
        ax[3].axvspan(t0, t1, color=STAGE_COLORS[label], zorder=0)

    ax[2].set_yticks([])
    ax[2].tick_params(axis='both', labelsize=tick_fs)
    ax[2].set_ylabel('Expert', fontsize=tick_fs)

    ax[3].set_yticks([])
    ax[3].tick_params(axis='both', labelsize=tick_fs)
    ax[3].set_ylabel('Dreem', fontsize=tick_fs)

    legend_patches = [mpatches.Patch(color=c, label=STAGE_NAMES[s]) for s, c in STAGE_COLORS.items()]
    ax[3].legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -1.6), ncol=6, fontsize=stage_legend_fs)

    ax[1].legend(loc='upper right', ncol=3, fontsize=legend_fs, handlelength=1, handletextpad=0.4)
    ax[0].grid(True, alpha=0.4, linestyle='--')
    ax[1].grid(True, alpha=0.4, linestyle='--')

    ax[3].set_xlabel('Time [s]', fontsize=tick_fs)

    t_min = min(hr['Timestamp'].min(), motion['Timestamp'].min())
    t_max = max(hr['Timestamp'].max(), motion['Timestamp'].max())
    ax[3].set_xlim(t_min, t_max)

    plt.tight_layout()

    return fig

def night_overview(night_data: pd.DataFrame, patient_nr: int = None, night_nr: int = None):
    save_fig = _draw_night_overview(night_data, fontsize_scale=1.3)
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