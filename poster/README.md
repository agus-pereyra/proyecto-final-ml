# Póster — Proyecto Final

Póster A1 vertical (594×841 mm) del proyecto de estadificación del sueño con SmartWatch.
Hay **dos versiones** del mismo póster (mismo contenido y paleta); elegí la que prefieras
para imprimir/editar.

| Archivo | Qué es |
|---|---|
| `poster.html` | Diseño **web** (HTML/CSS autocontenido, figuras embebidas). El más pulido. |
| `poster-web.pdf` | PDF A1 generado desde `poster.html`. |
| `poster.tex` | Versión **LaTeX** (`tikzposter`), editable y compilable. |
| `poster.pdf` | PDF A1 generado al compilar `poster.tex`. |

Ambos PDF son **una sola página tamaño A1 vertical** (1684×2384 pt).

## Editar / regenerar

### Versión LaTeX (`poster.tex`)
```bash
pdflatex poster.tex      # compilar dos veces -> poster.pdf
pdflatex poster.tex
```
- Requiere una distribución LaTeX (MiKTeX/TeX Live) con el paquete `tikzposter`.
- Usa las figuras de `../report/figures/` (no las copia: las incluye por ruta relativa).
- Paleta y textos están arriba del archivo, fáciles de tocar.
- **Ojo:** al compilar, `poster.tex` genera `poster.pdf`. Si no querés pisar nada, compilá
  con otro nombre de salida: `pdflatex -jobname poster_v2 poster.tex`.

### Versión web (`poster.html`)
Abrí `poster.html` en el navegador (Chrome/Edge). Para regenerar el PDF A1:
```
Imprimir -> Guardar como PDF -> tamaño A1, vertical, márgenes 0, escala 100 %,
con "Gráficos de fondo" activado.
```

## Para imprimir
Renombrá el PDF elegido como `Pereyra_Patruno_Poster_PF.pdf`, revisalo **al 100 % de zoom**
(que las figuras no se vean borrosas a tamaño A1) e imprimí en **A1 vertical**.

## Diseño
- Concepto: *el descenso de la noche* — el hipnograma como firma gráfica.
- Paleta (3 colores): midnight `#0E1330` · ivory `#F7F4EE` · ámbar `#E8912A`.
- Figuras usadas: `raw-vs-clean-20-1`, `night-predictions-31-1`, `lstm-confussion-4`,
  `xgboost-shap` (y `udesa-logo` en la versión web), todas de `../report/figures/`.
