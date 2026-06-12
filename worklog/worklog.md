# Worklog — Proyecto Final I302

## 1. Dataset

### 1.1. Sitios Web Explorados

**Datasets tabulares / general**
- Papers With Code – Datasets — Datasets con papers y métricas de referencia asociadas.
- OpenML — Repositorio de datasets para benchmarking con resultados comparativos públicos.
- Kaggle Datasets — Datasets de competencias con notebooks y baselines de referencia.
- UCI ML Repository — Colección clásica de datasets para investigación en ML.
- Hugging Face Datasets — Repositorio creciente con datasets vinculados a papers recientes.
- Zenodo — Repositorio del CERN: cada dataset tiene DOI y referencia directa a su paper.
- Harvard Dataverse — Repositorio de Harvard con datos de investigación de todas las disciplinas.

**Datos científicos / clínicos**
- PhysioNet — Datos clínicos reales con papers asociados (UCI hospitalaria, señales fisiológicas).
- NIH GDC — Datos genómicos y clínicos de investigación oncológica.
- CDC BRFSS — Encuestas de salud poblacional de EE.UU., ampliamente usadas en ML clínico.

**Datos geoespaciales / ambientales**
- NASA Earthdata — Datos climáticos, atmosféricos y oceánicos de misiones de la NASA.
- NOAA Climate Data — Datos meteorológicos y climáticos históricos reales.
- PANGAEA — Datos de ciencias del mar y la tierra asociados a publicaciones científicas.

**Datos socioeconómicos**
- Global Health Observatory – OMS — Indicadores de salud pública por país.
- World Bank Open Data — Indicadores económicos y sociales por país.
- Our World in Data — Datos globales sobre pobreza, energía, educación y salud.
- IPUMS — Microdatos censales de todo el mundo, incluyendo América Latina.
- Data.gov — Más de 290k datasets de agencias federales de EE.UU.
- datos.gob.ar — Portal de datos abiertos del gobierno argentino.
- EU Open Data Portal — Datos abiertos de la Unión Europea.

**Imágenes**
- Roboflow Universe — Miles de datasets de visión anotados, muchos con papers.
- Google Open Images — 9M imágenes anotadas con labels jerárquicos.
- NIH Chest X-ray — Radiografías con 14 patologías, benchmark médico clásico.
- ISIC Archive — Imágenes dermatológicas para detección de melanoma (HAM10000).
- EuroSAT — Imágenes satelitales Sentinel-2, clasificación de uso del suelo.
- CheXpert — Radiografías de tórax de Stanford, muy citado en papers.

**Audio**
- DCASE Challenge — Detección de sonidos ambientales, papers anuales.
- UrbanSound8K — 8,732 clips de sonidos urbanos, 10 clases.
- ESC-50 — 2,000 clips ambientales, 50 clases, paper propio.
- AudioSet (Google) — 2M clips de YouTube, 527 categorías de sonido.
- RAVDESS — Emociones en el habla y canción, 24 actores.

**Series temporales**
- UCR Time Series Archive — 128 datasets de series temporales con benchmarks.
- UEA Time Series Archive — Versión multivariada del UCR, ideal para LSTM.
- Climate Change AI Datasets — Series climáticas con papers de sostenibilidad.

---

### 1.2. Datasets Candidatos Evaluados

| Dataset | Descripción | n | m | Objetivo | Decisión |
|---|---|---|---|---|---|
| phenomene | Sonidos nasales y orales. Amplitudes de armónicos. | 5.404 | 5 | Clasificación Binaria | Descartado |
| cardiotocography | Cardiotocogramas (CTG) fetales. | 2.126 | 36 | Clasificación Multiclase (10 o 3) | Descartado |
| steam-games-dataset | Información de juegos publicados en Steam. | 124.146 | 41 | Clasificación / Regresión | Descartado |
| volcanoes-c1 | Imágenes SAR de la superficie de Venus. | 28.626 | — | Clasificación Multiclase (4) | Descartado |
| **BIDSleep** | IHR y acelerometría para sleep staging. 47 adultos, hasta 7 noches. | 253 noches | 4 señales | Clasificación Multiclase (5) | ✅ **Elegido** |
| hillel-yaffe-glaucoma | Imágenes de fondos de ojos para detección de glaucomas. | 747 | — | Clasificación Binaria | Descartado |
| HAM10000 (ISIC 2018) | Imágenes dermatológicas, 7 tipos de lesiones cutáneas. | ~10.000 | — | Clasificación Multiclase (7) | Descartado |
| Sismo (USGS) | Registros sísmicos con magnitud, ubicación, profundidad. | — | 22 | Clasificación / Regresión | **Rechazado**: todas las features describen el evento ya ocurrido, no son predictoras |

---

### 1.3. Dataset Elegido: BIDSleep

**Nombre completo:** A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels

**Fuente:** PhysioNet (acceso abierto)
**URL:** https://physionet.org/content/bidsleep-dataset/1.0.0/
**DOI:** https://doi.org/10.13026/a0sy-7t69
**Fecha de publicación:** 12 de mayo de 2026 (muy reciente)

**Descripción:**
El dataset contiene registros de 47 adultos sanos sin historial de trastornos del sueño, con hasta 7 noches de grabación por sujeto (253 noches en total). Para cada noche se registraron dos tipos de señales temporales mediante un Apple Watch con la aplicación BIDSleep: frecuencia cardíaca instantánea (IHR) y acelerometría en 3 ejes (x, y, z). Las señales de referencia (EEG) fueron capturadas con el Dreem 2 headband.

**Estructura de archivos por noche:**
- `hr.csv` — IHR en bpm, ~0.2 Hz de muestreo, timestamps Unix
- `motion.csv` — Acelerometría 3 ejes, timestamps Unix
- `labels.mat` — Variables: `recStart`, `dreem_label`, `expert_label`

**Etiquetas (estándar AASM, epochs de 30 segundos):**

| Valor | Etapa |
|---|---|
| 0 | Wake |
| 1 | N1 (sueño liviano) |
| 2 | N2 (sueño intermedio) |
| 3 | N3 (sueño profundo) |
| 4 | REM |
| 5 | Unknown (excluir del entrenamiento) |

**Dos tipos de etiquetado:**
- `dreem_label`: asignado automáticamente por el algoritmo del dispositivo Dreem
- `expert_label`: anotación manual por un experto en medicina del sueño → **ground truth**

**Observaciones sobre los datos:**
- Una noche de un sujeto tiene aproximadamente 800–900 epochs
- El desbalance de clases es **esperable fisiológicamente** (N2 suele ser mayoritaria, N1 minoritaria) pero debe cuantificarse sobre el dataset completo — no confirmar con datos de un solo sujeto
- El split train/test debe realizarse **por sujeto** para evitar data leakage
- Normalización debe hacerse **por sujeto o por noche**, no global

**Lectura de labels.mat en Python:**
```python
import scipy.io as sio
import numpy as np

mat = sio.loadmat('labels.mat')
dreem_labels  = mat['dreem_label'].flatten()   # shape: (n_epochs,)
expert_labels = mat['expert_label'].flatten()  # shape: (n_epochs,)
rec_start     = mat['recStart'][0]

clases = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Unknown'}
total = len(expert_labels)
for clase, nombre in clases.items():
    n = np.sum(expert_labels == clase)
    print(f"{nombre}: {n} epochs ({n/total*100:.1f}%)")
```

---

## 2. Modelos Propuestos

### Pregunta de investigación
¿Cuánto valor agrega modelar la secuencia temporal explícitamente, respecto a features hand-crafted o patrones locales, para la clasificación automática de etapas del sueño?

### Progresión de modelos

```
Baseline:    XGBoost / MLP + Feature Engineering   →  Murphy Cap. 18 / Cap. 13
Intermedio:  CNN 1D                                 →  Murphy Cap. 14 / Bishop & Bishop Cap. 10
Principal:   LSTM                                   →  Murphy Cap. 15
Mejora:      BiLSTM                                 →  Schuster & Paliwal (1997)
```

---

### 2.1. Baseline: XGBoost / MLP + Feature Engineering

**Idea:** Extraer features estadísticas de cada epoch de 30s para obtener una representación tabular (~40 features por epoch). Cada fila = un epoch.

**Features a extraer por señal (IHR, acc_x, acc_y, acc_z):**
- Dominio temporal: media, std, min, max, skewness, kurtosis, percentiles 25/75
- Dominio frecuencial: densidad espectral de potencia (PSD), potencia por banda
- Específicas de IHR: RMSSD (variabilidad de frecuencia cardíaca)
- Acelerometría: magnitud total √(x²+y²+z²)

**Ventajas:**
- Enteramente dentro del programa de la materia
- Interpretable: feature importance directo de XGBoost
- Rápido de entrenar

**Pipeline:**
```
señal cruda → extracción de ~40 features por epoch → vector tabular → XGBoost/MLP → clase
```

**Bibliografía:** Murphy (2022, Cap. 18) para XGBoost; Murphy (2022, Cap. 13) para MLP

---

### 2.2. Intermedio: CNN 1D

**Idea:** Aplicar filtros convolucionales sobre la dimensión temporal de cada epoch para detectar patrones locales automáticamente, sin feature engineering manual.

**Justificación de CNN 1D vs CNN 2D:**
El input tiene forma `(300 timesteps, 4 señales)` — un tensor 1D + canales. Los filtros 1D se deslizan sobre el eje temporal integrando las 4 señales simultáneamente, lo cual captura co-ocurrencias fisiológicas (ej. movimiento brusco reflejado en los 3 ejes del acelerómetro e IHR). CNN 2D asumiría continuidad espacial entre las señales, lo que carece de interpretación fisiológica.

**Ventajas sobre LSTM:**
- Significativamente más rápida de entrenar
- Captura bien patrones locales (picos, transiciones bruscas)

**Desventaja:**
- No modela dependencias temporales de largo plazo dentro del epoch

**Pipeline:**
```
señal cruda (300, 4) → Conv1D → MaxPool → Conv1D → MaxPool → Flatten → Dense(5) → Softmax
```

**Bibliografía:** Murphy (2022, Cap. 14); Bishop & Bishop (2023, Cap. 10)

---

### 2.3. Principal: LSTM

**Idea:** Red neuronal recurrente con celdas de memoria explícita que procesa la secuencia temporal del epoch directamente, aprendiendo dependencias de largo plazo.

**Por qué LSTM es la elección natural:**
Las fases del sueño presentan dependencias temporales dentro de cada epoch: patrones de HR y movimiento evolucionan a lo largo de los 30 segundos de formas características para cada etapa. La LSTM captura estas dependencias mediante su mecanismo de forget/input/output gates sin necesidad de feature engineering manual.

**Arquitectura:**
```python
class SleepLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64,
                 num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, timesteps, features)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]       # último timestep
        out = self.dropout(out)
        return self.classifier(out)    # (batch, 5)
```

**Pipeline:**
```
señal cruda (300, 4) → LSTM → Dropout → Dense(5) → Softmax
```

**Split correcto (evitar data leakage):**
```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))
```

**Métricas:**
```python
from sklearn.metrics import classification_report, cohen_kappa_score
kappa = cohen_kappa_score(y_test, y_pred)  # estándar en sleep staging
```

**Bibliografía:** Murphy (2022, Cap. 15: *Neural Networks for Sequences*)

---

### 2.4. Mejora: BiLSTM

**Idea:** Extensión de LSTM que procesa cada epoch en ambas direcciones temporales (pasado → futuro y futuro → pasado), concatenando los estados ocultos resultantes.

**Justificación de bidireccionalidad:**
Al momento de clasificar un epoch, los 30 segundos completos están disponibles (clasificación offline, no tiempo real). Procesar la secuencia en ambas direcciones permite capturar contexto futuro dentro del epoch, lo cual puede ser relevante para patrones fisiológicos simétricos. La comparación LSTM vs BiLSTM permite cuantificar si esta información bidireccional es estadísticamente significativa a escala de 30 segundos.

**Diferencia de implementación respecto a LSTM:**
```python
self.lstm = nn.LSTM(
    ...
    bidirectional=True   # ← único cambio
)
self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 por bidireccional
```

**Bibliografía:** Schuster & Paliwal (1997)

---

## 3. Extensiones Planificadas

1. **Comparación sistemática de los 3 enfoques** — mismas métricas y mismo esquema de validación cruzada por sujeto
2. **Análisis exploratorio del dataset via feature engineering** — feature importance de XGBoost para identificar señales más discriminativas por fase
3. **Análisis de desbalance de clases** — comparar class weights, SMOTE, undersampling
4. **Comparación LSTM vs BiLSTM** — cuantificar valor de la bidireccionalidad intra-epoch
5. **Arquitectura híbrida CNN 1D + LSTM** — motivada por SLAMSS-IFS; CNN extrae features locales, LSTM modela la secuencia resultante
6. **Clasificación inter-época** — segunda LSTM sobre secuencia de epochs consecutivos, incorporando la prior fisiológica de que ciertas transiciones entre fases son improbables; motivada por el componente "inter-epoch learning" de SLAMSS-IFS

---

## 4. Métricas de Evaluación

| Métrica | Uso |4
|---|---|
| **Cohen's Kappa** | Principal — estándar en sleep staging, descuenta el azar |
| **F1-score macro** | Complementaria — pondera por igual todas las clases |
| **Accuracy** | Referencia — no usar como métrica principal por sensibilidad al desbalance |
| **Cohen's Kappa (dreem vs expert)** | Baseline del dispositivo comercial — referencia de dificultad del problema |

---

## 5. Bibliografía del Proyecto

**Dataset**
Song, T. (2026). *A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels* (version 1.0.0). PhysioNet. https://doi.org/10.13026/a0sy-7t69

**Paper oficial del dataset (SLAMSS-IFS)**
Song, T.-A., Zhang, Y., Zhou, Z., Hou, L., Malekzadeh, M., Behzad, A., & Dutta, J. (2026). AI-driven sleep staging using instantaneous heart rate and accelerometry: Insights from an Apple Watch study. *IEEE Transactions on Biomedical Engineering*, *73*(4), 1596–1608. https://doi.org/10.1109/TBME.2025.3612158
Versión abierta: https://pmc.ncbi.nlm.nih.gov/articles/PMC12931632/

**Paper base SLAMSS**
Song, T.-A., Roy Chowdhury, S., Malekzadeh, M., Harrison, S., Hoge, T. B., Redline, S., Stone, K. L., Saxena, R., & Purcell, S. M. (2023). AI-driven sleep staging from actigraphy and heart rate. *PLOS ONE*, *18*(5), e0285703. https://doi.org/10.1371/journal.pone.0285703

**BiLSTM**
Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, *45*(11), 2673–2681. https://doi.org/10.1109/78.650093

**Bibliografía de la materia**
Murphy, K. P. (2022). *Probabilistic machine learning: An introduction*. MIT Press.
Bishop, C. M., & Bishop, H. (2023). *Deep learning: Foundations and concepts*. Springer.