"""
Genera las tablas LaTeX (booktabs) que usan el informe, las guías y las slides,
para que los números del PDF salgan siempre de los datos.

Requiere haber ejecutado antes:
  python src/build_dataset.py
  python src/evaluate.py
  python src/evaluate.py --extended

Uso:  python src/export_tables.py      ->  tex/tables/*.tex
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_dataset import FEATURES  # noqa: E402

OUT = os.path.join(ROOT, 'tex', 'tables')
KS = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10']
FEAT_TEX = {f: r'$f_{}$'.format(i + 1) for i, f in enumerate(FEATURES)}
FEAT_NAME = {
    'f1_tfidf_cosine': 'coseno TF-IDF', 'f2_bm25': 'BM25', 'f3_synonym_overlap': 'solap. sinónimos',
    'f4_component_match': 'analito en component', 'f5_system_match': 'muestra en system',
}


def esc(s):
    s = str(s)
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                 ('^', r'\^{}'), ('~', r'\~{}')]:
        s = s.replace(a, b)
    return s


def write(name, body):
    path = os.path.join(OUT, name)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('% Generado por src/export_tables.py — no editar a mano\n' + body)
    print('Escrito', os.path.relpath(path, ROOT))


def tabular(cols, header, rows):
    lines = [r'\begin{tabular}{' + cols + '}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for r in rows:
        lines.append(r if isinstance(r, str) else ' & '.join(r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def system_label(s):
    if s == 'AdaRank':
        return r'\textbf{AdaRank}'
    f = s.replace('solo ', '')
    return 'solo {} ({})'.format(FEAT_TEX[f], FEAT_NAME[f])


def main():
    os.makedirs(OUT, exist_ok=True)
    labels = pd.read_csv(os.path.join(ROOT, 'data', 'relevance_labels.csv'), dtype={'loinc_num': str})

    # 1. Distribución de etiquetas
    rows = []
    for off in (True, False):
        sub = labels[labels.official == off]
        for q, g in sub.groupby('query', sort=False):
            c = g.relevance.value_counts()
            rows.append([esc(q) + ('' if off else r'$^\dagger$')] +
                        [str(c.get(k, 0)) for k in (3, 2, 1, 0)] + [str(len(g))])
        if off:
            rows.append(r'\midrule')
    write('label_distribution.tex',
          tabular('lrrrrr', ['Consulta', '3', '2', '1', '0', 'Total'], rows))

    # 2. Documentos relevantes (etiqueta > 0) de las consultas oficiales
    pos = labels[(labels.official) & (labels.relevance > 0)].sort_values(
        ['qid', 'relevance'], ascending=[True, False])
    rows, last = [], None
    for r in pos.itertuples():
        if last is not None and r.qid != last:
            rows.append(r'\midrule')
        q = esc(r.query) if r.qid != last else ''
        last = r.qid
        rows.append([q, esc(r.loinc_num), esc(r.long_common_name), esc(r.system), esc(r.property),
                     str(r.relevance)])
    write('positive_labels.tex', tabular('llp{6.2cm}llc',
                                         ['Consulta', 'LOINC', 'long\\_common\\_name', 'system',
                                          'property', 'Rel.'], rows))

    # 3. Resultados LOQO (media) — oficial y ampliado
    rows = []
    for tag, title in (('official', '3 consultas oficiales (3 folds)'),
                       ('extended', r'6 consultas, ampliado$^\dagger$ (6 folds)')):
        res = pd.read_csv(os.path.join(ROOT, 'results', 'loqo_{}.csv'.format(tag)))
        mean = res.groupby('system', sort=False)[KS].mean()
        rows.append(r'\multicolumn{5}{l}{\emph{' + title + r'}} \\')
        for s, r in mean.iterrows():
            rows.append([system_label(s)] + ['{:.3f}'.format(v) for v in r])
        if tag == 'official':
            rows.append(r'\midrule')
    write('loqo_results.tex', tabular('lrrrr', ['Sistema'] + KS, rows))

    # 4. LOQO por fold: AdaRank vs baseline f1 (oficial)
    res = pd.read_csv(os.path.join(ROOT, 'results', 'loqo_official.csv'))
    qname = labels.drop_duplicates('qid').set_index('qid')['query']
    rows = []
    for (q, s), r in res[res.system.isin(['AdaRank', 'solo f1_tfidf_cosine'])].groupby(
            ['test_qid', 'system'], sort=False):
        rows.append([esc(qname[q]), system_label(s)] + ['{:.3f}'.format(r[k].iloc[0]) for k in KS])
    write('loqo_per_fold.tex', tabular('llrrrr', ['Consulta de test', 'Sistema'] + KS, rows))

    # 5. Coeficientes del modelo final
    rows = []
    for tag in ('official', 'extended'):
        c = pd.read_csv(os.path.join(ROOT, 'results', 'coef_{}.csv'.format(tag))).set_index('feature')
        rows.append(c['coef'])
    body = tabular('l' + 'r' * len(FEATURES), ['Entrenado con'] + [FEAT_TEX[f] for f in FEATURES],
                   [['3 consultas'] + ['{:.3f}'.format(v) for v in rows[0]],
                    [r'6 consultas$^\dagger$'] + ['{:.3f}'.format(v) for v in rows[1]]])
    write('coefficients.tex', body)

    # 6. Ranking del modelo final (top-6 por consulta oficial)
    rk = pd.read_csv(os.path.join(ROOT, 'results', 'ranking_official.csv'), dtype={'loinc_num': str})
    names = labels.drop_duplicates('loinc_num').set_index('loinc_num')['long_common_name']
    rows, last = [], None
    for r in rk[rk['rank'] <= 6].itertuples():
        if last is not None and r.qid != last:
            rows.append(r'\midrule')
        rows.append([esc(qname[r.qid]) if r.qid != last else '', str(r.rank),
                     esc(names[r.loinc_num]), str(r.relevance), '{:.2f}'.format(r.score)])
        last = r.qid
    write('ranking_top.tex', tabular('lrp{7cm}cr', ['Consulta', 'Pos.', 'Documento', 'Rel.', 'Score'],
                                     rows))

    # 7. Ejemplo de líneas LETOR
    lines = open(os.path.join(ROOT, 'data', 'train.txt'), encoding='utf-8').read().splitlines()
    wanted = ['74774-1', '14749-6', '15076-3', '1975-2', '26464-8', '26478-8', '933-2']
    ex = [l for w in wanted for l in lines if 'docid = {} '.format(w) in l and 'qid:{} '.format(
        1 if w in ('74774-1', '14749-6', '15076-3', '933-2') else 2 if w == '1975-2' else 3) in l]
    ex = [l.split(' query =')[0] for l in ex]
    write('letor_example.tex', '\\begin{lstlisting}\n' + '\n'.join(ex) + '\n\\end{lstlisting}\n')


if __name__ == '__main__':
    main()
