# Listwise Learning to Rank con AdaRank — consultas LOINC

Machine Learned Ranking, MUIA (UPM). Training set de relevancia (escala 0–3) para tres consultas
sobre 67 documentos LOINC, y AdaRank entrenado y evaluado con leave-one-query-out.

## Estructura

```
data/
  loinc_dataset-v2.xlsx     Excel original del profesor (3 hojas × 67 documentos)
  dataset_v0.csv            primera versión del dataset (escala 0–2), solo como referencia
  overrides.csv             correcciones manuales a la regla (vacío por ahora)
  relevance_labels.csv      etiquetas 0–3 + dimensiones + justificación  (generado)
  features.csv              f1–f5 por par consulta–documento            (generado)
  train.txt                 LETOR, 3 consultas oficiales, 201 líneas     (generado)
  train_extended.txt        LETOR, 6 consultas (parte opcional)          (generado)
src/
  queries.py                consultas y regla de relevancia 0–3
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
python src/evaluate.py               # LOQO con las 3 consultas oficiales
python src/evaluate.py --extended    # LOQO con 6 consultas
python src/export_tables.py          # regenera tex/tables/*.tex
python test.py data/train.txt "" data/train.txt   # CLI original
```

## Compilar los documentos (Overleaf)

1. Sube la carpeta `tex/` completa como proyecto de Overleaf (*New project → Upload project*, en .zip).
2. Compila con pdfLaTeX cada documento principal: `report.tex` (informe, máximo 4 páginas), `labeling_guidelines.tex`,
   `feature_dictionary.tex` y `presentation.tex` (Beamer, tema metropolis).

Las tablas de `tex/tables/` se generan con `src/export_tables.py`. No se editan a mano.
