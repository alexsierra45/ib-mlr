"""
Genera data/relevance_labels.csv aplicando la regla 0-3 de src/queries.py.

Colección de documentos:
  - 67 documentos LOINC del Excel del profesor (doc_source = 'excel')
  - términos nuevos del equipo, data/extended_terms.csv (doc_source = 'ampliacion'),
    sin repetir los que ya están en el Excel

Todas las consultas se juzgan contra TODA la colección (como en el Excel, donde las tres
consultas comparten los mismos 67 documentos). Los subconjuntos de entrenamiento se
definen en src/build_dataset.py.

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
EXTENDED = os.path.join(ROOT, 'data', 'extended_terms.csv')
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
    """Colección completa: 67 documentos del Excel + términos nuevos no repetidos."""
    codes = [set(df['loinc_num']) for df in sheets.values()]
    assert all(c == codes[0] for c in codes), 'las hojas no contienen los mismos códigos'
    excel = next(iter(sheets.values())).reset_index(drop=True).assign(doc_source='excel')
    assert excel['loinc_num'].is_unique

    ext = pd.read_csv(EXTENDED, dtype=str).apply(lambda col: col.str.strip())
    assert ext['loinc_num'].notna().all(), 'faltan códigos LOINC en extended_terms.csv'
    ext = ext[~ext['loinc_num'].isin(excel['loinc_num'])].drop_duplicates('loinc_num')
    ext = ext[DOC_COLS].assign(doc_source='ampliacion')
    return pd.concat([excel, ext], ignore_index=True)


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
        # Consultas oficiales: los 67 del Excel en el orden de su hoja, después los términos nuevos
        if q['sheet']:
            excel_order = sheets[q['sheet']]['loinc_num']
            order = pd.concat([docs.set_index('loinc_num').loc[excel_order].reset_index(),
                               docs[docs.doc_source == 'ampliacion']], ignore_index=True)
        else:
            order = docs
        for d in order.itertuples(index=False):
            label, analyte, specimen, measure, why = relevance(
                q, d.component, d.system, d.property, d.long_common_name)
            key = (q['qid'], d.loinc_num)
            overridden = key in overrides
            if overridden:
                new_label, just = overrides[key]
                why = '{} | OVERRIDE {} -> {}: {}'.format(why, label, new_label, just)
                label = new_label
            rows.append({
                'qid': q['qid'], 'query': q['query'], 'official': q['official'],
                'loinc_num': d.loinc_num, 'doc_source': d.doc_source,
                'long_common_name': d.long_common_name,
                'component': d.component, 'system': d.system, 'property': d.property,
                'analyte': analyte, 'specimen': specimen, 'measure': measure,
                'relevance': label, 'override': overridden, 'rationale': why,
            })
    return pd.DataFrame(rows), docs


def main():
    labels, docs = build_labels()
    labels.to_csv(OUT, index=False, encoding='utf-8')
    print('Escrito', os.path.relpath(OUT, ROOT), labels.shape,
          '| documentos:', docs.doc_source.value_counts().to_dict())
    print(labels.pivot_table(index='query', columns='relevance', values='loinc_num',
                             aggfunc='count', fill_value=0, sort=False))


if __name__ == '__main__':
    main()
