<p align="center"><em>Proyecto Final</em></p>

<h1 align="center"><strong>Predicción de Etapas del Sueño sobre Señales Fisiológicas con RNNs</strong></h1>

<p align="center"><em>I302 - Aprendizaje Automático y Aprendizaje Profundo</em></p>

<p><strong>Agustín Patruno y Agustín Pereyra</strong></p>

<hr/>

## Setup

1) Decarga del dataset

En Linux / macOS / Git Bash:
```bash
wget -r -N -c -np https://physionet.org/files/bidsleep-dataset/1.0.0/
```

En Windows, instalar wget previamente (por ejemplo con `winget install -e --id GNU.Wget2`) y luego ejecutar el mismo comando desde PowerShell o Git Bash.

O directamente desde la web [https://physionet.org/content/bidsleep-dataset/1.0.0/#files-panel](https://physionet.org/content/bidsleep-dataset/1.0.0/#files-panel)

2) Descomprimir la carpeta directamente en [data/](data/) de manera que quede estructurado de la siguiente manera

```text
data/
└── <extracted_dir>/
    ├── Bidslab00/
    ├── Bidslab01/
    └── ...
```

3) Instalación de requisitos

Crear un entorno con conda:
```bash
conda create -n <env_name> python=3.12
conda activate <env_name>
```

O con venv:
```bash
python -m venv <env_name>
source <env_name>/bin/activate  # Linux / macOS / Git Bash
<env_name>\Scripts\activate      # Windows (PowerShell/cmd)
```

Luego instalar las dependencias:
```bash
pip install -r requirements.txt
```

> **Nota:** `requirements.txt` instala `torch` con soporte CUDA 12.1 (requiere GPU NVIDIA con drivers actualizados). Si no contás con GPU NVIDIA, instalar primero el resto de las dependencias y luego la versión CPU de `torch`:
> ```bash
> pip install numpy matplotlib pandas scipy
> pip install torch==2.5.1
> ```