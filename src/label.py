"""
Genera data/relevance_labels.csv aplicando la regla 0-3 de src/queries.py
a los 67 documentos LOINC del Excel del profesor.

Uso:  python src/label.py
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from queries import QUERIES, relevance  # noqa: E402

XLSX = os.path.join(ROOT, 'data', 'loinc_dataset-v2.xlsx')
OVERRIDES = os.path.join(ROOT, 'data', 'overrides.csv')
OUT = os.path.join(ROOT, 'data', 'relevance_labels.csv')

DOC_COLS = ['loinc_num', 'long_common_name', 'component', 'system', 'property']


def load_sheets():
    """Devuelve {nombre_hoja: DataFrame} con las 5 columnas LOINC (cabecera en la fila 3)."""
    sheets = pd.read_excel(XLSX, sheet_name=None, header=2, dtype=str)
    for name, df in sheets.items():
        assert list(df.columns) == DOC_COLS, (name, list(df.columns))
        sheets[name] = df.apply(lambda col: col.str.strip())
    return sheets


def load_documents(sheets):
    """Los 67 documentos únicos. Comprueba que las tres hojas contienen el mismo conjunto."""
    codes = [set(df['loinc_num']) for df in sheets.values()]
    assert all(c == codes[0] for c in codes), 'las hojas no contienen los mismos códigos'
    first = next(iter(sheets.values()))
    assert first['loinc_num'].is_unique
    return first.reset_index(drop=True)


def load_overrides():
    if not os.path.exists(OVERRIDES):
        return {}
    ov = pd.read_csv(OVERRIDES, dtype={'loinc_num': str})
    return {(int(r.qid), r.loinc_num): (int(r.relevance), r.justification) for r in ov.itertuples()}


def build_labels():
    sheets = load_sheets()
    docs = load_documents(sheets)
    overrides = load_overrides()

    rows = []
    for q in QUERIES:
        # Conservamos el orden de la hoja del profesor para las consultas oficiales
        order = sheets[q['sheet']] if q['sheet'] else docs
        for d in order.itertuples(index=False):
            label, analyte, specimen, measure, why = relevance(q, d.component, d.system, d.property)
            key = (q['qid'], d.loinc_num)
            overridden = key in overrides
            if overridden:
                new_label, just = overrides[key]
                why = '{} | OVERRIDE {} -> {}: {}'.format(why, label, new_label, just)
                label = new_label
            rows.append({
                'qid': q['qid'], 'query': q['query'], 'official': q['official'],
                'loinc_num': d.loinc_num, 'long_common_name': d.long_common_name,
                'component': d.component, 'system': d.system, 'property': d.property,
                'analyte': analyte, 'specimen': specimen, 'measure': measure,
                'relevance': label, 'override': overridden, 'rationale': why,
            })
    return pd.DataFrame(rows), docs


def main():
    labels, _ = build_labels()
    labels.to_csv(OUT, index=False, encoding='utf-8')
    print('Escrito', os.path.relpath(OUT, ROOT), labels.shape)
    print(labels.pivot_table(index='query', columns='relevance', values='loinc_num',
                             aggfunc='count', fill_value=0, sort=False))


if __name__ == '__main__':
    main()
