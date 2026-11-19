"""
Construye el training set Listwise:

  data/relevance_labels.csv   etiquetas 0-3 + justificación (vía src/label.py)
  data/features.csv           una fila por par consulta-documento con f1..f5
  data/train.txt              formato LETOR/SVMlight, solo las 3 consultas oficiales (201 líneas)
  data/train_extended.txt     LETOR con las 6 consultas (parte opcional)

Las features se calculan SIN mirar la etiqueta y son deliberadamente más
gruesas que la regla de etiquetado (ver docs/feature_dictionary.tex).

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
from queries import QUERIES  # noqa: E402

FEATURES_OUT = os.path.join(ROOT, 'data', 'features.csv')
LETOR_OUT = os.path.join(ROOT, 'data', 'train.txt')
LETOR_EXT_OUT = os.path.join(ROOT, 'data', 'train_extended.txt')

FEATURES = ['f1_tfidf_cosine', 'f2_bm25', 'f3_synonym_overlap', 'f4_component_match', 'f5_system_match']

STOPWORDS = {'in', 'of', 'or', 'by', 'the', 'for', 'and', 'from', 'to', 'a'}


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

def stem(tok):
    """Stemming mínimo: plural inglés regular (cells -> cell, leukocytes -> leukocyte)."""
    return tok[:-1] if len(tok) > 3 and tok.endswith('s') and not tok.endswith('ss') else tok


def tokenize(text):
    return [stem(t) for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def f1_tfidf_cosine(query, docs_text, vectorizer, doc_matrix):
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
    """1 si algún término de analito de la consulta aparece como palabra en `component`."""
    toks = set(tokenize(component))
    return float(any(stem(t) in toks for t in q['analyte_terms']))


def f5_system_match(q, system):
    """1 si el término de muestra de la consulta aparece literalmente en `system` (sin graduar)."""
    toks = set(re.findall(r'[a-z]+', system.lower()))
    return float(any(t in toks for t in q['specimen_terms']))


def compute_features(labels, docs):
    docs = docs.set_index('loinc_num')
    names = docs['long_common_name'].tolist()
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    doc_matrix = vectorizer.fit_transform(names)
    docs_tokens = [tokenize(t) for t in names]
    pos = {code: i for i, code in enumerate(docs.index)}

    out = []
    for q in QUERIES:
        f1 = f1_tfidf_cosine(q['query'], names, vectorizer, doc_matrix)
        f2 = f2_bm25(q['query'], docs_tokens)
        sub = labels[labels['qid'] == q['qid']]
        for r in sub.itertuples(index=False):
            i = pos[r.loinc_num]
            raw = ' '.join([r.long_common_name, r.component, r.system, r.property])
            out.append({
                'qid': r.qid, 'query': r.query, 'loinc_num': r.loinc_num,
                FEATURES[0]: round(float(f1[i]), 6),
                FEATURES[1]: round(float(f2[i]), 6),
                FEATURES[2]: round(f3_synonym_overlap(q, raw), 6),
                FEATURES[3]: f4_component_match(q, r.component),
                FEATURES[4]: f5_system_match(q, r.system),
            })
    return pd.DataFrame(out)


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

EXPECTED = {  # casos de referencia de docs/labeling_guidelines.tex
    (1, '74774-1'): 3, (1, '14749-6'): 2, (1, '15076-3'): 1, (1, '14764-5'): 1,
    (1, '35184-1'): 1, (1, '933-2'): 0,
    (2, '1975-2'): 3, (2, '1968-7'): 2, (2, '1971-1'): 2, (2, '14423-8'): 1,
    (2, '33870-7'): 1, (2, '54439-5'): 0,
    (3, '26464-8'): 3, (3, '26474-7'): 2, (3, '26484-6'): 2, (3, '26478-8'): 1,
}


def check(data, n_docs=67):
    official = data[data['official']]
    assert len(official) == 3 * n_docs, len(official)
    codes = None
    for qid, g in data.groupby('qid'):
        assert len(g) == n_docs, (qid, len(g))
        assert g['loinc_num'].is_unique, qid
        s = set(g['loinc_num'])
        codes = codes or s
        assert s == codes, qid
    assert data['relevance'].isin([0, 1, 2, 3]).all()
    assert data['rationale'].notna().all()
    assert np.isfinite(data[FEATURES].to_numpy()).all()
    for qid, g in data.groupby('qid'):
        assert g['relevance'].max() == 3, 'la consulta {} no tiene ningún documento 3'.format(qid)
    lookup = data.set_index(['qid', 'loinc_num'])['relevance']
    for key, val in EXPECTED.items():
        assert lookup[key] == val, (key, lookup[key], val)
    print('OK: {} pares ({} oficiales), {} por consulta, {} casos de referencia verificados'.format(
        len(data), len(official), n_docs, len(EXPECTED)))


def main():
    labels, docs = build_labels()
    labels.to_csv(LABELS_OUT, index=False, encoding='utf-8')
    feats = compute_features(labels, docs)
    feats.to_csv(FEATURES_OUT, index=False, encoding='utf-8')

    data = labels.merge(feats, on=['qid', 'query', 'loinc_num'], validate='one_to_one')
    check(data)
    write_letor(data[data['official']], LETOR_OUT)
    write_letor(data, LETOR_EXT_OUT)
    for p in (LABELS_OUT, FEATURES_OUT, LETOR_OUT, LETOR_EXT_OUT):
        print('Escrito', os.path.relpath(p, ROOT))


if __name__ == '__main__':
    main()
