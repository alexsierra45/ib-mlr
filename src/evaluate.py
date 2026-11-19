"""
Evaluación leave-one-query-out (LOQO) de AdaRank frente a baselines de una sola feature.

Con pocas consultas no hay partición train/test fiable: cada fold entrena
con todas menos una y evalúa en la restante. Los resultados son demostrativos.

Uso:  python src/evaluate.py [--config official|terms|extended|all]
  official   data/train.txt           3 consultas x 67 documentos  (3 folds)
  terms      data/train_terms.txt     3 consultas x 356 documentos (3 folds)
  extended   data/train_extended.txt  8 consultas x 356 documentos (8 folds)

Escribe results/loqo_<config>.csv, results/coef_<config>.csv y results/ranking_<config>.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from adarank import AdaRank  # noqa: E402
from metrics import NDCGScorer  # noqa: E402
from utils import load_docno  # noqa: E402
from build_dataset import FEATURES, CONFIGS  # noqa: E402

KS = (1, 3, 5, 10)


def ndcg_all(y, pred, qid):
    return {'NDCG@{}'.format(k): NDCGScorer(k=k)(y, pred, qid) for k in KS}


def fit_adarank(X, y, qid):
    return AdaRank(max_iter=100, estop=10, scorer=NDCGScorer(k=10)).fit(X, y, qid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='all', choices=list(CONFIGS) + ['all'])
    args = ap.parse_args()
    for config in (CONFIGS if args.config == 'all' else [args.config]):
        evaluate(config)


def evaluate(tag):
    path = CONFIGS[tag][2]

    X, y, qid = load_svmlight_file(path, query_id=True, n_features=len(FEATURES))
    docno = load_docno(path, letor=True)
    X = X.tocsr()
    out_dir = os.path.join(ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for test_q in np.unique(qid):
        tr, te = qid != test_q, qid == test_q
        model = fit_adarank(X[tr], y[tr], qid[tr])
        systems = {'AdaRank': model.predict(X[te], qid[te])}
        for j, f in enumerate(FEATURES):
            systems['solo ' + f] = X[te][:, j].toarray().ravel()
        for name, pred in systems.items():
            scores = ndcg_all(y[te], pred, qid[te])
            rows.append({'test_qid': int(test_q), 'system': name,
                         **{k: float(v[0]) for k, v in scores.items()}})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'loqo_{}.csv'.format(tag)), index=False)
    summary = res.groupby('system', sort=False)[['NDCG@{}'.format(k) for k in KS]].mean()
    print('\n=== {}: LOQO ({} consultas, {} pares) — media por sistema'.format(tag, len(np.unique(qid)), len(y)))
    print(summary.round(4).to_string())

    # Modelo final con todas las consultas: coeficientes y ranking resultante
    model = fit_adarank(X, y, qid)
    coef = pd.DataFrame({'feature': FEATURES, 'coef': model.coef_})
    coef.to_csv(os.path.join(out_dir, 'coef_{}.csv'.format(tag)), index=False)
    print('\nCoeficientes (modelo entrenado con todas las consultas):')
    print(coef.round(4).to_string(index=False))

    pred = model.predict(X, qid)
    rank = pd.DataFrame({'qid': qid.astype(int), 'loinc_num': docno, 'relevance': y.astype(int),
                         'score': pred})
    rank['rank'] = rank.groupby('qid')['score'].rank(ascending=False, method='first').astype(int)
    rank.sort_values(['qid', 'rank']).to_csv(os.path.join(out_dir, 'ranking_{}.csv'.format(tag)),
                                             index=False)


if __name__ == '__main__':
    main()
