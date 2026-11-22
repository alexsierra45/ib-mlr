"""
Analiza los resultados del ranking e identifica falsos positivos y falsos negativos.
Contrasta el código LOINC en los resultados con la etiqueta otorgada en el archivo
de entrenamiento y recupera las columnas originales en collection.csv.

Uso:  python src/falses_check.py [--config official|terms|extended] (por defecto: terms)

Documentos a contrastar:
  official   data/train.txt           3 consultas x 67 documentos del Excel (3 folds)
             results/ranking_official.csv

  terms      data/train_terms.txt     3 consultas x colección completa      (3 folds)
             ranking_terms.csv

  extended   data/train_extended.txt  15 consultas x colección completa     (15 folds)
             ranking_extended.csv

  collection.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_dataset import FEATURES, CONFIGS  # noqa: E402

RESULTS = {
    'official': (True, True, os.path.join(ROOT, 'results', 'ranking_official.csv')),
    'terms': (True, False, os.path.join(ROOT, 'results', 'ranking_terms.csv')),
    'extended': (False, False, os.path.join(ROOT, 'results', 'ranking_extended.csv')),
}

COLLECTION_FILE = os.path.join(ROOT, 'data', 'collection.csv')

FALSE_POS_OUT = os.path.join(ROOT, 'results', 'false_positives.csv')
FALSE_NEG_OUT = os.path.join(ROOT, 'results', 'false_negatives.csv')

TOP_K = 10
MIN_RELEVANCE = 2
MAX_RANK_RELEVANT = 10

# ---------------------------------------------------------------------------
# LOAD LETOR FILE
# ---------------------------------------------------------------------------

def load_train_txt(path):
    """Carga el archivo .txt empleado por Adarank en el entrenamiento."""
    rows = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data, _, comment = line.partition('#')
            tokens = data.split()
            relevance = int(tokens[0])
            qid = int(tokens[1].split(':')[1])
            row = {
                'qid': qid,
                'loinc_num': None,
                'train_relevance': relevance,
            }

            for i, feature in enumerate(FEATURES, start=1):
                row[feature] = np.nan

            for token in tokens[2:]:
                index, value = token.split(':')
                feature_index = int(index)

                if 1 <= feature_index <= len(FEATURES):
                    feature_name = FEATURES[feature_index - 1]
                    row[feature_name] = float(value)

            if 'docid =' in comment:
                row['loinc_num'] = (
                    comment.split('docid =')[1].split()[0]
                )
            rows.append(row)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

def load_dataset(dataset_name):
    """Carga el dataset correspondiente al entrenamiento a analizar."""
    if dataset_name not in CONFIGS:
        raise ValueError(f"Dataset desconocido: {dataset_name}")

    train_path = CONFIGS[dataset_name][2]
    ranking_path = RESULTS[dataset_name][2]

    print(f"\nCargando dataset: {dataset_name}")
    print(f"LETOR:   {train_path}")
    print(f"Ranking: {ranking_path}")
    print(f"Collection: {COLLECTION_FILE}")
    train = load_train_txt(train_path)
    ranking = pd.read_csv(ranking_path)
    collection = pd.read_csv(COLLECTION_FILE)

    # Normalize identifiers
    for data in (train, ranking, collection):
        data['loinc_num'] = data['loinc_num'].astype(str).str.strip()
    ranking['qid'] = pd.to_numeric(ranking['qid']).astype(int)
    ranking['rank'] = pd.to_numeric(ranking['rank'])
    ranking['relevance'] = pd.to_numeric(ranking['relevance'])
    ranking['score'] = pd.to_numeric(ranking['score'])
    train['qid'] = pd.to_numeric(train['qid']).astype(int)

    # Merge ranking with LETOR features
    df = ranking.merge(
        train,
        on=['qid', 'loinc_num'],
        how='left',
        validate='one_to_one'
    )
    # Merge original LOINC attributes
    df = df.merge(
        collection,
        on='loinc_num',
        how='left',
        validate='many_to_one'
    )

    if df['train_relevance'].isna().any():
        missing = df[df['train_relevance'].isna()]
        raise ValueError(
            f"{len(missing)} las entradas del ranking no coincidieron "
            "con el archivo LETOR."
        )

    mismatch = df[df['relevance'] != df['train_relevance']]
    if not mismatch.empty:
        print(
            f"WARNING: {len(mismatch)} relevance mismatches "
            "entre ranking y LETOR."
        )

    return df


# ---------------------------------------------------------------------------
# IDENTIFY SIGNIFICANT ERRORS
# ---------------------------------------------------------------------------

def identify_errors(df):
    """ A partir de los parámetros de control, identifica los falsos positivos
    y falsos negativos. Es decir, codigos mal rankeados y fuera del rango de 
    tolerancia"""
    false_positives = df[
        (df['relevance'] < MIN_RELEVANCE) &
        (df['rank'] <= TOP_K)
    ].copy()

    false_negatives = df[
        (df['relevance'] >= MIN_RELEVANCE) &
        (df['rank'] > MAX_RANK_RELEVANT)
    ].copy()

    return false_positives, false_negatives

def display_errors(data, title):
    """Imprime los errores encontrados."""
    print('\n' + '=' * 120)
    print(title)
    print('=' * 120)
    if data.empty:
        print('No se encontraron errores significativos.')
        return

    columns = [
        'qid',
        'loinc_num',
        'relevance',
        'rank',
        'score',
        *FEATURES,
        'long_common_name',
        'component',
        'system',
        'property',
        'common_test_rank',
        'scale',
        'loinc_class',
    ]
    columns = [c for c in columns if c in data.columns]

    for qid, group in data.groupby('qid'):
        print(f'\nQUERY ID: {qid}')
        print('-' * 120)
        print(
            group[columns]
            .sort_values('rank')
            .to_string(index=False)
        )

def display_summary(df, false_positives, false_negatives):
    """Imprime un resumen del análisis."""
    print('\n' + '=' * 80)
    print('RESUMEN DEL ANALISIS')
    print('=' * 80)
    for qid in sorted(df['qid'].unique()):
        fp = (false_positives['qid'] == qid).sum()
        fn = (false_negatives['qid'] == qid).sum()
        print(
            f'Query {qid}: '
            f'{fp} falsos positivos | '
            f'{fn} falsos megativos'
        )

def main():

    parser = argparse.ArgumentParser(
        description='Analizar errores en LOINC ranking.'
    )
    parser.add_argument(
        '--config',
        choices=['official', 'terms', 'extended'],
        default='terms',
        help='Dataset a analizar.'
    )
    args = parser.parse_args()

    df = load_dataset(args.config)
    false_positives, false_negatives = identify_errors(df)

    display_errors(false_positives, 'FALSOS POSITIVOS')
    display_errors(false_negatives, 'FALSOS NEGATIVOS')
    display_summary(df, false_positives, false_negatives)

    # Export diagnostic results
    false_positives.to_csv(
        FALSE_POS_OUT,
        index=False
    )
    false_negatives.to_csv(
        FALSE_NEG_OUT,
        index=False
    )
    print('\nArchivos del análisis generados correctamente en /results.')

if __name__ == '__main__':
    main()