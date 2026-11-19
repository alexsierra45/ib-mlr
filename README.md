# Listwise Learning to Rank con AdaRank — consultas LOINC

Machine Learned Ranking, MUIA (UPM). Training set de relevancia (escala 0–3) para las tres consultas
del profesor sobre 67 documentos LOINC, ampliado de forma sistemática con 12 consultas y una colección
de 1222 documentos obtenida por *pooling* sobre la tabla oficial de LOINC. AdaRank entrenado y evaluado
con leave-one-query-out.

| Configuración | Archivo | Consultas × documentos | Pares |
|---|---|---|---|
| official | `data/train.txt` | 3 × 67 (entregable obligatorio) | 201 |
| terms | `data/train_terms.txt` | 3 × 1222 (más términos) | 3666 |
| extended | `data/train_extended.txt` | 15 × 1222 (más términos y consultas) | 18 330 |

## Estructura

```
data/
  loinc_dataset-v2.xlsx     Excel original del profesor (3 hojas × 67 documentos)
  collection.csv            colección de 1222 documentos (Excel + pooling sobre LOINC)  (generado)
  overrides.csv             correcciones manuales a la regla (vacío por ahora)
  relevance_labels.csv      etiquetas 0–3 + dimensiones + justificación de los 18 330 pares (generado)
  features.csv              f1–f5 por par y configuración                                (generado)
  train.txt, train_terms.txt, train_extended.txt   LETOR                                  (generado)
  dataset_v0.csv            primera versión del dataset (escala 0–2), solo como referencia
  dataset_extended.csv      primera ampliación manual (piloto, sustituida por el pooling)
src/
  queries.py                las 15 consultas y la regla de relevancia 0–3
  build_collection.py       selección de consultas y pooling sobre Loinc.csv -> collection.csv
  label.py                  aplica la regla -> relevance_labels.csv
  text.py                   tokenización común
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

`data/collection.csv` ya está incluido. Para regenerarlo hace falta la tabla oficial de LOINC
(`Loinc.csv`, descarga gratuita con registro en https://loinc.org/downloads/), que **no** se incluye en el
repositorio por su tamaño (84 MB) y licencia:

```bash
python src/build_collection.py ruta/a/Loinc.csv   # por defecto, ./Loinc.csv
```

## Compilar los documentos (Overleaf)

1. Sube la carpeta `tex/` completa como proyecto de Overleaf (*New project → Upload project*, en .zip).
2. Compila con pdfLaTeX cada documento principal: `report.tex` (informe, máximo 4 páginas), `labeling_guidelines.tex`,
   `feature_dictionary.tex` y `presentation.tex` (Beamer, tema metropolis).

Las tablas de `tex/tables/` se generan con `src/export_tables.py`. No se editan a mano.

## Licencia de LOINC

This material contains content from LOINC® (http://loinc.org). LOINC is copyright © Regenstrief Institute, Inc.
and the Logical Observation Identifiers Names and Codes (LOINC) Committee and is available at no cost under the
license at http://loinc.org/license. LOINC® is a registered United States trademark of Regenstrief Institute, Inc.
