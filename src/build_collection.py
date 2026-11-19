"""
Construye de forma sistemática las consultas adicionales y la colección de documentos
a partir de la tabla oficial de LOINC (Loinc.csv, no incluida en el repositorio por su tamaño
y licencia; se descarga gratis en https://loinc.org/downloads/).

1. Selección estratificada de consultas (sin elección manual de las pruebas):
   se recorren las pruebas de laboratorio por COMMON_TEST_RANK (ranking oficial de LOINC de
   las pruebas más frecuentes) y se toma la PRIMERA de cada estrato
   (dominio clínico = CLASS, grupo de muestra = SYSTEM agrupado) aún no cubierto, hasta 12:
     - se excluyen las entradas que no son pruebas de un paciente (clases SPEC, MISC, PANEL,
       BLDBK, CHAL; componentes genéricos como "Observation"), los índices por célula
       (propiedades Ent*) y las pruebas sin muestra concreta (XXX, Isolate)
     - analito simple (sin "/") y no repetido, ni ya cubierto por las consultas oficiales
   Así las consultas varían en dominio, muestra y tipo de medida. El TEXTO de cada consulta
   se redacta a mano en src/queries.py, al estilo de las consultas del profesor; aquí se
   comprueba que las pruebas de origen coinciden.

2. Pooling (como en TREC/LETOR), igual para las 15 consultas, sobre las pruebas de
   laboratorio activas de LOINC:
     - 40 candidatos por BM25 de la consulta sobre LONG_COMMON_NAME   (distractores léxicos)
     - 20 candidatos cuyo COMPONENT contiene el analito, por frecuencia (variantes del analito)
     - 300 términos de laboratorio aleatorios compartidos (semilla fija) (negativos fáciles)
     - los 67 documentos del Excel del profesor
   Se limitan los casi-duplicados: máximo 2 por (componente sin números, sistema, propiedad).

Escribe data/collection.csv, que es lo que usa el resto del pipeline (no necesita Loinc.csv).

Uso:  python src/build_collection.py [ruta/a/Loinc.csv]
"""

import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from queries import QUERIES  # noqa: E402
from text import tokenize  # noqa: E402
from label import load_sheets  # noqa: E402

LOINC = os.path.join(ROOT, 'Loinc.csv')
OUT = os.path.join(ROOT, 'data', 'collection.csv')

N_BM25, N_COMPONENT, N_RANDOM, MAX_DUP, SEED = 40, 20, 300, 2, 42
N_NEW_QUERIES = 12
OFFICIAL_ANALYTES = {'glucose', 'bilirubin', 'leukocytes'}
NON_TEST_CLASSES = ('SPEC', 'MISC', 'PANEL', 'BLDBK', 'CHAL')
GENERIC_COMPONENTS = {'observation', 'specimen', 'age', 'service comment'}

COLS = {'LOINC_NUM': 'loinc_num', 'LONG_COMMON_NAME': 'long_common_name', 'COMPONENT': 'component',
        'SYSTEM': 'system', 'PROPERTY': 'property', 'SCALE_TYP': 'scale', 'CLASS': 'loinc_class',
        'COMMON_TEST_RANK': 'common_test_rank'}


def load_universe(path):
    """Pruebas de laboratorio (CLASSTYPE 1) activas de LOINC."""
    d = pd.read_csv(path, dtype=str, low_memory=False, usecols=list(COLS) + ['CLASSTYPE', 'STATUS'])
    d = d[(d.CLASSTYPE == '1') & (d.STATUS == 'ACTIVE')].rename(columns=COLS)[list(COLS.values())]
    d = d.fillna('')
    d['common_test_rank'] = pd.to_numeric(d['common_test_rank'], errors='coerce').fillna(0).astype(int)
    return d.reset_index(drop=True)


def specimen_group(system):
    """Agrupa el SYSTEM de LOINC en grandes tipos de muestra (None = sin muestra concreta)."""
    s = system.split('^')[0]
    if s.startswith('Bld') or s == 'RBC':
        return 'sangre'
    if s in ('Ser', 'Plas', 'Ser/Plas', 'Ser/Plas/Bld', 'PPP', 'PPP/Bld', 'PlasV', 'PlasA', 'Ser/Bld'):
        return 'suero/plasma'
    if s.startswith('Urine'):
        return 'orina'
    if s == 'CSF':
        return 'LCR'
    if 'Respiratory' in s or s in ('Nph', 'Thrt', 'Nose', 'Sputum', 'Bronchial'):
        return 'respiratoria'
    if s == 'Stool':
        return 'heces'
    if s in ('Tiss', 'Bld/Tiss'):
        return 'tejido'
    if s in ('XXX', 'Specimen', 'Isolate'):
        return None
    return 'otros fluidos'


def select_queries(universe):
    """La prueba más frecuente de cada estrato (dominio clínico x grupo de muestra), hasta 12."""
    top = universe[universe.common_test_rank > 0].sort_values('common_test_rank')
    chosen, strata, seen = [], set(), set(OFFICIAL_ANALYTES)
    for r in top.itertuples():
        comp = r.component.lower()
        if r.loinc_class.startswith(NON_TEST_CLASSES) or comp in GENERIC_COMPONENTS:
            continue
        if r.property.startswith('Ent') or r.scale in ('-', 'Doc', 'Nar'):
            continue
        group = specimen_group(r.system)
        stratum = (r.loinc_class.split('.')[0], group)
        if group is None or stratum in strata or comp in seen or '/' in comp:
            continue
        chosen.append(r)
        strata.add(stratum)
        seen.add(comp)
        if len(chosen) == N_NEW_QUERIES:
            break
    return chosen


def dup_key(r):
    return (re.sub(r'[\d.]+', '#', r.component.lower()), r.system, r.property)


def take(ranked, n, exclude, counts):
    """Toma n documentos de `ranked` respetando el límite de casi-duplicados."""
    out = []
    for r in ranked.itertuples():
        if len(out) == n:
            break
        if r.loinc_num in exclude or counts.get(dup_key(r), 0) >= MAX_DUP:
            continue
        counts[dup_key(r)] = counts.get(dup_key(r), 0) + 1
        out.append(r.loinc_num)
        exclude.add(r.loinc_num)
    return out


def bm25_matrix(texts, k1=1.2, b=0.75):
    vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    tf = vec.fit_transform(texts).tocsc().astype(float)
    n = tf.shape[0]
    dl = np.asarray(tf.sum(axis=1)).ravel()
    df = np.diff(tf.indptr)
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    return vec, tf, dl, dl.mean(), idf, k1, b


def bm25_scores(query, index):
    vec, tf, dl, avgdl, idf, k1, b = index
    scores = np.zeros(tf.shape[0])
    for t in set(tokenize(query)):
        j = vec.vocabulary_.get(t)
        if j is None:
            continue
        col = tf[:, j]
        rows, vals = col.nonzero()[0], col.data
        scores[rows] += idf[j] * vals * (k1 + 1) / (vals + k1 * (1 - b + b * dl[rows] / avgdl))
    return scores


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else LOINC
    universe = load_universe(path)
    print('Universo: {} pruebas de laboratorio activas'.format(len(universe)))

    # 1. Selección sistemática de consultas y comprobación contra src/queries.py
    chosen = select_queries(universe)
    new_queries = [q for q in QUERIES if not q['official']]
    print('Consultas seleccionadas (estratificación dominio x muestra, por COMMON_TEST_RANK):')
    for r, q in zip(chosen, new_queries):
        print('  rank {:3d}  {:9s} {:9s} {:13s} {:6s} {:30s} -> qid {:2d} "{}"'.format(
            r.common_test_rank, r.loinc_num, r.loinc_class.split('.')[0], specimen_group(r.system),
            r.property, r.component[:30], q['qid'], q['query']))
        assert q['source_loinc'] == r.loinc_num, (q['query'], q['source_loinc'], r.loinc_num)
    assert len(chosen) == len(new_queries) == N_NEW_QUERIES

    # 2. Pooling
    index = bm25_matrix(universe['long_common_name'].tolist())
    excel = next(iter(load_sheets().values()))
    selected, source = set(excel['loinc_num']), {c: 'excel' for c in excel['loinc_num']}
    counts = {}
    for q in QUERIES:
        ranked = universe.assign(s=bm25_scores(q['query'], index))
        ranked = ranked[ranked.s > 0].sort_values(['s', 'common_test_rank'], ascending=[False, True])
        for c in take(ranked, N_BM25, selected, counts):
            source[c] = 'pool_bm25'
        terms = [t for t in tokenize(' '.join(q['analyte_terms']))]
        has = universe['component'].map(lambda s: bool(set(tokenize(s)) & set(terms)))
        ranked = universe[has].assign(r=lambda d: d.common_test_rank.replace(0, 10 ** 6)).sort_values(
            ['r', 'loinc_num'])
        for c in take(ranked, N_COMPONENT, selected, counts):
            source[c] = 'pool_component'
    rest = universe[~universe.loinc_num.isin(selected)]
    for c in rest.sample(n=N_RANDOM, random_state=SEED)['loinc_num']:
        source[c] = 'random'

    coll = universe[universe.loinc_num.isin(source) & ~universe.loinc_num.isin(excel['loinc_num'])].copy()
    # Los documentos del Excel se toman del propio Excel (mismas columnas, por si difiere la versión)
    ex = excel.assign(common_test_rank=excel['loinc_num'].map(
        universe.set_index('loinc_num')['common_test_rank']).fillna(0).astype(int))
    coll = pd.concat([ex, coll], ignore_index=True)
    coll['doc_source'] = coll['loinc_num'].map(source)
    coll.to_csv(OUT, index=False, encoding='utf-8')
    print('Colección: {} documentos {}'.format(len(coll), coll.doc_source.value_counts().to_dict()))
    print('Escrito', os.path.relpath(OUT, ROOT))


if __name__ == '__main__':
    main()
