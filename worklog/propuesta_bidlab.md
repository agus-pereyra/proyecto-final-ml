# A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels

## 1. Título del Proyecto

Clasificador de Etapas del Sueño sobre Señales Fisiológicas con RNNs

## 2. Conjunto de Datos

**(a) Descripción del Conjunto de Datos**

El dataset BIDSleep (PhysioNet, 2023) contiene registros polisomnográficos de 47 adultos sanos, con entre 3 a 7 noches de grabación por sujeto (un total de 253 noches). Para cada noche se registraron dos tipos de sañales temporales: (1) frecuencia cardíaca instantánea (IHR) y (2) acelerometría  en 3 ejes (x, y, z) obtenidas por un smartwatch de Apple (Apple Watch) mediante una aplicación desarrollada específicamente para BIDSleep . Las noches se segmentan en épocas de 30 segundos (estándar clínico AASM), resultando cada noche de aproximadamente 800 a 900 épocas.
Provee 2 tipos de etiqueta por época: "dreem_label" es el asignado automáticamente por el dispositivo Dreem (headband, algoritmo automático), y "expert_label", asignado por un experto medicina del sueño (anotación manual), la cual sería utilizada como ground truth. Las etiquetas corresponden a 5 etapas del sueño: Wake (0), N1 (1), N2 (2), N3 (3), REM (4), Deconocido (5).

**(b) Acceso al Conjunto de Datos**

[https://physionet.org/content/bidsleep-dataset/1.0.0/#files-panel](https://physionet.org/content/bidsleep-dataset/1.0.0/#files-panel)

## 3. Tarea Básica

**Objetivo**: dado una época de 30 segundos compuesto por IHR y acelerometría (x, y, z), el modelo debe predecir a cuál de las cinco etapas del sueño pertenece: Wake, N1, N2, N3 o REM. Esta tarea tiene relevancia clínica directa, dado que el monitoreo del sueño es fundamental para el diagnóstico de trastornos como la apnea, el insomnio y otras patologías relacionadas, y los métodos actuales son costosos e incómodos para el paciente.

**Problema**:  Clasificación Multiclase Supervisada. Teniendo 2 tipos de etiquetado se puede evaluar el modelo contra el ground truth (expert_label), contra el dispositivo automático (dreem_label) y contra una medida como el coeficiente de Kappa de Cohen (grado de acuerdo entre ambas clasificaciones). Además el dataset tiene un potencial desbalanceo de clases dado que en otros datasets de "sleep staging" se observó a N2 como la clase mayoritaria y N1 como una clase minoritaria.

**Arquitectura**: proponemos 3 enfoques en orden de motivación personal y complejidad esperada, que podrían implementarse los 3 y compararse, o bien reducirse según disponibilidad de tiempo/complejidad.

(1) Red Neuronal Recurrente LSTM y BiLSTM (Schuster & Paliwal, 1997): es la arquitectura que naturalmente se adecúa al problema. Captura dependencias temporales aprendiendo automáticamente patrones relevantes sin necesidad de feature engineering. Además la extensión BiLSTM (Bidirectional LSTM) procesa las épocas en ambas direcciones temporales. LSTM está fundamentado en Murphy (2022, Cap.15), por lo que a pesar de no haber dado el tema como tal en la teóricas, estamos motivados en trabajar en un tipo de dato nunca trabajado antes (secuencial) y explorar estos tipos de arquitecturas.

(2) Red Neuronal Convolucional 1D: mediante filtros convolucionales es posible detectar patrones locales temporales en la señal, otra vez sin necesitar obligatoriamente de feature engineering. La ventaja sobre LSTM es que es un modelo potencialmente mucho más rápido para entrenar, pero su desventaja es que no capta dependencias temporales de "largo plazo" como LSTM. Otra vez no es un modelo dado explícitamente en las clases teóricas pero en ciertas clases tutoriales fueron mostradas capas convolucionales de redes y está también fundamentado en Murphy (2022, Cap. 14) y además en Bishop & Bishop (2023, Cap. 10). Se utilizaría CNN 1D dado que la secuencia correspondería a un tensor 1D + canales (4 señales), es decir solo tienen dimensión temporal.

(3) XGBoost / MLP con Feature Engineering: Este será el baseline clásico, en el que tendremos que extraer features estadísticas por época en el dominio temporal y frecuencial para obtener una representación tabular. Nuestra idea es que esta pueda funcionar como punto de partida de comparación con alguno de los modelos más complejos, dándonos chances de obtener interpretaciones como el feature importance y además poder evaluar el valor agregado del modelado de la secuencia temporal explícita en la RNN.

## 4. Ideas para Extensiones

- Comparación de los 3 enfoques propuestos
- Análisis extra del dataset a partir del feature engineering + MLP/XGBoost
- Análisis de desbalance de clases
- Comparación LSTM vs BiLSTM (análisis bidireccionalidad dentro de una época de 30 segundos)
- Arquitectura híbrida CNN 1D + LSTM (el paper citado por la página del dataset original plantea un modelo que entre otras cosas realiza una combinación de CNN y LSTM -- mejora de SLAMSS que utiliza además capas como Attention, y Encoder y Decoder de LSTM --  llamado SLAMSS-IFS)
- Clasificación inter-época: incorporar contexto entre épocas consecutivas teniendo la creencia prior de que las fases del sueño siguen patrones fisiológicos. (La "I" de SLAMSS-IFS viene de "inter-epoch learning", por lo que nos da motivaciones para intentar replicar alguna forma de realizar esto)

## 5. Justificación

El problema integra la mayor parte de los temas dados en la materia. El "pipeline" completo incluye normalización, preprocesamiento de datos, feature engineering, tratamiento de desbalance de clases, selección y ajuste de hiperparámetros, evaluación con métricas (F1-Score, o por ejemplo la propuesta de Cohen's Kappa) y comparación entre familias de modelos.

El dataset es muy reciente (Mayo de 2026) y responde a un criterio de relevancia clínica real. El monitoreo del sueño es fundamental en ciertos diagnósticos y los métodos actuales requieren herramientas y equipamiento costoso (además de condiciones hospitalarias). Los modelos propuestos operarían sobre señales simples obtenidas con dispositivos comunes.

El proyecto presenta una oportunidad de explorar tanto con tipos de datos como con arquitecturas que extienden los contenidos del curso, pero sin irse del hilo principal. Además el trabajo con señales fisiológicas es fundamental para el resto de la carrera y además en lo personal la rama IA aplicada a la medicina/biología es la que voy a intentar abocar en mi futuro como profesional.

La existencia de un paper asociado (aunque aún no pudimos obtener el acceso a él, se requiere una cuenta de una institución asociada a IEEE Xplore) que propone un modelo mucho más complejo permite situar resultados propios en un contexto académico y comparar contra un benchmark establecido.

## 6. Bibliografía

Dataset
Song, T. (2026). A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels (version 1.0.0). PhysioNet. [https://doi.org/10.13026/a0sy-7t69](https://doi.org/10.13026/a0sy-7t69)

Paper oficial del dataset (acceso restringido)
Song, T.-A., Zhang, Y., Zhou, Z., Hou, L., Malekzadeh, M., Behzad, A., & Dutta, J. (2025). AI-driven sleep staging using instantaneous heart rate and accelerometry: Insights from an Apple Watch study. IEEE Transactions on Biomedical Engineering. [https://doi.org/10.1109/TBME.2025.3612158](https://doi.org/10.1109/TBME.2025.3612158)

Artículo del paper oficial
Song, T. A., Zhang, Y., Zhou, Z., Hou, L., Malekzadeh, M., Behzad, A., & Dutta, J. (2026). AI-Driven Sleep Staging Using Instantaneous Heart Rate and Accelerometry: Insights From an Apple Watch Study. IEEE transactions on bio-medical engineering, 73(4), 1596–1608. [https://doi.org/10.1109/TBME.2025.3612158](https://doi.org/10.1109/TBME.2025.3612158)
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12931632/#S3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12931632/#S3)

Paper del modelo SLAMSS (en el que se basa el paper oficial del dataset)
Song, T.-A., Roy Chowdhury, S., Malekzadeh, M., Harrison, S., Hoge, T. B., Redline, S., Stone, K. L., Saxena, R., & Purcell, S. M. (2023). AI-Driven sleep staging from actigraphy and heart rate. PLOS ONE, 18(5), e0285703. [https://doi.org/10.1371/journal.pone.0285703](https://doi.org/10.1371/journal.pone.0285703)

Paper original BiLSTM
Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 45(11), 2673–2681. [https://doi.org/10.1109/78.650093](https://doi.org/10.1109/78.650093)
[https://deeplearning.cs.cmu.edu/F20/document/readings/Bidirectional%20Recurrent%20Neural%20Networks.pdf](https://deeplearning.cs.cmu.edu/F20/document/readings/Bidirectional%20Recurrent%20Neural%20Networks.pdf)

Bibliografía de la materia utilizado
Murphy, K. P. (2022). Probabilistic machine learning: An introduction. MIT Press.
