# Listwise Learning to Rank con AdaRank — consultas LOINC

Machine Learned Ranking, MUIA (UPM). Training set de relevancia (escala 0–3) para las tres consultas
del profesor sobre 67 documentos LOINC, ampliado con 289 términos nuevos y 5 consultas más, y AdaRank
entrenado y evaluado con leave-one-query-out.

| Configuración | Archivo | Consultas × documentos |
|---|---|---|
| official | `data/train.txt` | 3 × 67 (entregable obligatorio) |
| terms | `data/train_terms.txt` | 3 × 356 (más términos) |
| extended | `data/train_extended.txt` | 8 × 356 (más términos y consultas) |

## Estructura

```
data/
  loinc_dataset-v2.xlsx     Excel original del profesor (3 hojas × 67 documentos)
  dataset_v0.csv            primera versión del dataset (escala 0–2), solo como referencia
  dataset_extended.csv      términos nuevos elegidos por el equipo (sin código ni etiqueta)
  extended_terms.csv        los mismos términos con su loinc_num (generado por fetch_loinc_codes.py)
  overrides.csv             correcciones manuales a la regla (vacío por ahora)
  relevance_labels.csv      etiquetas 0–3 + dimensiones + justificación  (generado)
  features.csv              f1–f5 por par consulta–documento            (generado)
  train.txt                 LETOR, 3 consultas × 67 docs, 201 líneas     (generado)
  train_terms.txt           LETOR, 3 consultas × 356 docs, 1068 líneas   (generado)
  train_extended.txt        LETOR, 8 consultas × 356 docs, 2848 líneas   (generado)
src/
  queries.py                consultas y regla de relevancia 0–3
  fetch_loinc_codes.py      (una vez) añade loinc_num a los términos nuevos vía la API de LOINC de la NLM
  label.py                  aplica la regla -> relevance_labels.csv
  build_dataset.py          features + LETOR + comprobaciones
  evaluate.py               leave-one-query-out, AdaRank frente a baselines -> results/
  export_tables.py          tablas LaTeX -> tex/tables/
adarank.py, metrics.py, utils.py   AdaRank y NDCG (adaptado de github.com/rueycheng/AdaRank)
test.py                     CLI original: entrena y evalúa sobre ficheros LETOR
AdaRank.ipynb               notebook de demostración
results/                    resultados de la evaluación (generado)
tex/                        informe, guía de etiquetado, diccionario de features y slides (LaTeX)
```

## Reproducir

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
python src/build_dataset.py          # etiquetas, features, LETOR + comprobaciones
python src/evaluate.py               # LOQO en las tres configuraciones (o --config official|terms|extended)
python src/export_tables.py          # regenera tex/tables/*.tex
python test.py data/train.txt "" data/train.txt   # CLI original
```

## Compilar los documentos (Overleaf)

1. Sube la carpeta `tex/` completa como proyecto de Overleaf (*New project → Upload project*, en .zip).
2. Compila con pdfLaTeX cada documento principal: `report.tex` (informe, máximo 4 páginas), `labeling_guidelines.tex`,
   `feature_dictionary.tex` y `presentation.tex` (Beamer, tema metropolis).

Las tablas de `tex/tables/` se generan con `src/export_tables.py`. No se editan a mano.
