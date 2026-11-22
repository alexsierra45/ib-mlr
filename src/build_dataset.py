"""
Construye el training set Listwise en tres configuraciones:

  official   3 consultas oficiales x 67 documentos del Excel      -> data/train.txt          (entregable obligatorio)
  terms      3 consultas oficiales x colección completa (pooling) -> data/train_terms.txt    (más términos)
  extended   15 consultas          x colección completa (pooling) -> data/train_extended.txt (más términos y consultas)

Además:
  data/relevance_labels.csv   etiquetas 0-3 + justificación de todos los pares (vía src/label.py)
  data/features.csv           f1..f5 por par y configuración (columna `config`)

Las features se calculan SIN mirar la etiqueta y son deliberadamente más gruesas que la
regla de etiquetado (ver tex/feature_dictionary.tex). El corpus de TF-IDF/BM25 es la
colección de documentos de cada configuración.

Uso:  python src/build_dataset.py
"""

import math
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from label import build_labels, OUT as LABELS_OUT  # noqa: E402
from queries import get_query  # noqa: E402
from text import stem, tokenize  # noqa: E402

FEATURES_OUT = os.path.join(ROOT, 'data', 'features.csv')

CONFIGS = {
    # nombre: (solo consultas oficiales, solo documentos del Excel, archivo LETOR)
    'official': (True, True, os.path.join(ROOT, 'data', 'train.txt')),
    'terms': (True, False, os.path.join(ROOT, 'data', 'train_terms.txt')),
    'extended': (False, False, os.path.join(ROOT, 'data', 'train_extended.txt')),
}

FEATURES = ['f1_tfidf_cosine', 'f2_bm25', 'f3_synonym_overlap', 'f4_component_match', 'f5_system_match', 'f6_property_match'] 


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def f1_tfidf_cosine(query, vectorizer, doc_matrix):
    qv = vectorizer.transform([query])
    return (doc_matrix @ qv.T).toarray().ravel()  # vectores ya normalizados L2 -> coseno


def f2_bm25(query, docs_tokens, k1=1.2, b=0.75):
    """BM25 clásico sobre long_common_name, normalizado por el máximo de la consulta."""
    n = len(docs_tokens)
    avgdl = np.mean([len(d) for d in docs_tokens])
    df = {}
    for d in docs_tokens:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = np.zeros(n)
    for t in set(tokenize(query)):
        if t not in df:
            continue
        idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
        for i, d in enumerate(docs_tokens):
            tf = d.count(t)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
    return scores / scores.max() if scores.max() > 0 else scores


def f3_synonym_overlap(q, doc_raw):
    """Fracción de términos de la consulta presentes en el documento (todos sus campos),
    aceptando sinónimos (p. ej. white -> leukocytes, blood -> bld)."""
    text = doc_raw.lower()
    toks = set(tokenize(text))
    terms = tokenize(q['query'])
    hits = 0
    for t in terms:
        alts = [t] + q['synonyms'].get(t, []) + q['synonyms'].get(t + 's', [])
        if any(stem(a) in toks if a.isalnum() else a in text for a in alts):
            hits += 1
    return hits / len(terms)


def f4_component_match(q, component):
    """Proporción de analitos de la consulta en `component`."""
    toks = set(tokenize(component))
    analytes = set(tokenize(' '.join(q['analyte_terms'])))

    if not analytes:
        return 0.0

    matches = analytes.intersection(toks)

    return len(matches) / len(analytes)


def f5_system_match(q, system):
    """1 si el término de muestra de la consulta aparece literalmente en `system` (sin graduar)."""
    toks = set(re.findall(r'[a-z]+', system.lower()))
    return float(any(t in toks for t in q['specimen_terms']))


def f6_property_match(q, property_value):
    """1 si existe councidencia con la propiedad"""
    prop = str(property_value).strip()

    if prop in q['property_ok']:
        return 1.0

    return 0.0

def popularity_factor(common_test_rank, max_rank=20000):
    """Asigna un factor de popularidad en funcion  de el valor de common_test_rank
    segun una escala logarítmica"""
    if common_test_rank is None:
        return 0.0

    try:
        rank = float(common_test_rank)
    except (ValueError, TypeError):
        return 0.0

    if not np.isfinite(rank) or rank <= 0:
        return 0.0

    rank = min(rank, max_rank)

    popularity = 1.0 - (
        np.log1p(rank) / np.log1p(max_rank)
    )

    return 2.0 * popularity - 1.0

def adjust_feature(feature_value, common_test_rank, alpha=0.05):
    factor = popularity_factor(common_test_rank)
    return feature_value * (1.0 + alpha * factor)

def compute_features(labels):
    """Features de los pares de `labels`; el corpus son los documentos únicos de `labels`."""
    docs = labels.drop_duplicates('loinc_num').set_index('loinc_num')
    names = docs['long_common_name'].tolist()
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    doc_matrix = vectorizer.fit_transform(names)
    docs_tokens = [tokenize(t) for t in names]
    pos = {code: i for i, code in enumerate(docs.index)}

    out = []
    for qid, sub in labels.groupby('qid', sort=False):
        q = get_query(qid)
        f1 = f1_tfidf_cosine(q['query'], vectorizer, doc_matrix)
        f2 = f2_bm25(q['query'], docs_tokens)
        for r in sub.itertuples(index=False):
            i = pos[r.loinc_num]
            raw = ' '.join([r.long_common_name, r.component, r.system, r.property])
            out.append({
                'qid': r.qid, 'loinc_num': r.loinc_num,
                FEATURES[0]: adjust_feature(round(float(f1[i]), 6), r.common_test_rank, alpha=0.1),
                FEATURES[1]: round(float(f2[i]), 6),
                FEATURES[2]: adjust_feature(round(f3_synonym_overlap(q, raw), 6), r.common_test_rank, alpha=0.1),
                FEATURES[3]: f4_component_match(q, r.component),
                FEATURES[4]: f5_system_match(q, r.system),
                FEATURES[5]: f6_property_match(q, r.property),
            })
    return pd.DataFrame(out)


def subset(labels, config):
    only_official, only_excel, _ = CONFIGS[config]
    sel = labels
    if only_official:
        sel = sel[sel['official']]
    if only_excel:
        sel = sel[sel['doc_source'] == 'excel']
    return sel


def build_config(labels, config):
    sel = subset(labels, config)
    feats = compute_features(sel)
    data = sel.merge(feats, on=['qid', 'loinc_num'], validate='one_to_one', sort=False)
    return data.assign(config=config)


# ---------------------------------------------------------------------------
# LETOR
# ---------------------------------------------------------------------------

def write_letor(df, path):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for r in df.itertuples(index=False):
            feats = ' '.join('{}:{:.6g}'.format(j + 1, getattr(r, f)) for j, f in enumerate(FEATURES))
            fh.write('{} qid:{} {} # docid = {} query = {}\n'.format(
                r.relevance, r.qid, feats, r.loinc_num, r.query))


# ---------------------------------------------------------------------------
# Comprobaciones
# ---------------------------------------------------------------------------

EXPECTED = {  # casos de referencia de tex/labeling_guidelines.tex
    # Documentos del Excel
    (1, '74774-1'): 3, (1, '14749-6'): 2, (1, '15076-3'): 1, (1, '14764-5'): 1,
    (1, '35184-1'): 1, (1, '933-2'): 0,
    (2, '1975-2'): 3, (2, '1968-7'): 2, (2, '1971-1'): 2, (2, '14423-8'): 1,
    (2, '33870-7'): 1, (2, '54439-5'): 0,
    (3, '26464-8'): 3, (3, '26474-7'): 2, (3, '26484-6'): 2, (3, '26478-8'): 1,
    # Documentos de la colección (pooling sobre LOINC)
    (1, '2339-0'): 3,     # Glucose [Mass/volume] in Blood
    (1, '27353-2'): 1,    # Glucose mean value ... Estimated from glycated hemoglobin
    (2, '1970-3'): 2,     # Bilirubin.delta
    # Consultas de la ampliación: la prueba de origen es 3 ...
    (4, '2160-0'): 3, (5, '789-8'): 3, (6, '11556-8'): 3, (7, '5902-2'): 3, (8, '33903-6'): 3,
    (9, '104451-0'): 3, (10, '630-4'): 3, (13, '56888-1'): 3, (14, '11253-2'): 3, (15, '39004-7'): 3,
    # ... relacionados ...
    (7, '6301-6'): 2,     # INR (por nombre: LOINC codifica la prueba en el método)
    # ... y distractores difíciles
    (7, '3173-2'): 0,     # aPTT: otra prueba de coagulación
    (8, '13009-6'): 0,    # Methyl ethyl ketone: disolvente, no cuerpo cetónico
    (9, '41458-1'): 0,    # SARS coronavirus RNA: SARS-CoV de 2003
    (15, '14049-1'): 0,   # Epidermal growth factor: el ligando, no el receptor
}


def check(labels, built):
    n_docs = labels['loinc_num'].nunique()
    n_excel = labels.loc[labels.doc_source == 'excel', 'loinc_num'].nunique()
    assert n_excel == 67, n_excel
    for qid, g in labels.groupby('qid'):
        assert len(g) == n_docs and g['loinc_num'].is_unique, qid
        assert g['relevance'].max() == 3, 'la consulta {} no tiene ningún documento 3'.format(qid)
    assert labels['relevance'].isin([0, 1, 2, 3]).all()
    assert labels['rationale'].notna().all()
    lookup = labels.set_index(['qid', 'loinc_num'])['relevance']
    for key, val in EXPECTED.items():
        assert lookup[key] == val, (key, lookup[key], val)

    sizes = {}
    for config, data in built.items():
        assert np.isfinite(data[FEATURES].to_numpy()).all(), config
        per_q = data.groupby('qid').size()
        assert per_q.nunique() == 1, (config, per_q.to_dict())  # misma lista para cada consulta
        sizes[config] = '{} consultas x {} docs = {}'.format(len(per_q), per_q.iloc[0], len(data))
    assert len(built['official']) == 201
    print('OK: {} casos de referencia verificados'.format(len(EXPECTED)))
    for config, s in sizes.items():
        print('   {:9s} {}'.format(config, s))


def main():
    labels, _ = build_labels()
    labels.to_csv(LABELS_OUT, index=False, encoding='utf-8')
    built = {c: build_config(labels, c) for c in CONFIGS}
    check(labels, built)

    feats = pd.concat([d[['config', 'qid', 'query', 'loinc_num'] + FEATURES] for d in built.values()])
    feats.to_csv(FEATURES_OUT, index=False, encoding='utf-8')
    for config, data in built.items():
        write_letor(data, CONFIGS[config][2])
    for p in [LABELS_OUT, FEATURES_OUT] + [c[2] for c in CONFIGS.values()]:
        print('Escrito', os.path.relpath(p, ROOT))


if __name__ == '__main__':
    main()
