"""
Genera las tablas LaTeX (booktabs) que usan el informe, las guías y las slides,
para que los números del PDF salgan siempre de los datos.

Requiere haber ejecutado antes:
  python src/build_dataset.py
  python src/evaluate.py

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
CONFIG_NAME = {'official': 'Oficial', 'terms': '+ términos', 'extended': '+ términos y consultas'}
CONFIG_TEX = {}  # se rellena en main() con los tamaños reales, p. ej. "Oficial (3x67)"


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
    lines = [r'\begin{tabular}{' + cols + '}', r'\toprule']
    lines += header if isinstance(header, list) and header and header[0].endswith('\\\\') else \
        [' & '.join(header) + r' \\']
    lines.append(r'\midrule')
    for r in rows:
        lines.append(r if isinstance(r, str) else ' & '.join(r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def system_label(s):
    if s == 'AdaRank':
        return r'\textbf{AdaRank}'
    f = s.replace('solo ', '')
    return 'solo {} ({})'.format(FEAT_TEX[f], FEAT_NAME[f])


def ranking_rows(rk, names, qname, top):
    rows, last = [], None
    for r in rk[rk['rank'] <= top].itertuples():
        if last is not None and r.qid != last:
            rows.append(r'\midrule')
        rows.append([esc(qname[r.qid]) if r.qid != last else '', str(r.rank),
                     esc(names[r.loinc_num]), str(r.relevance), '{:.2f}'.format(r.score)])
        last = r.qid
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    labels = pd.read_csv(os.path.join(ROOT, 'data', 'relevance_labels.csv'), dtype={'loinc_num': str})
    qname = labels.drop_duplicates('qid').set_index('qid')['query']
    names = labels.drop_duplicates('loinc_num').set_index('loinc_num')['long_common_name']
    n_all = labels['loinc_num'].nunique()
    n_official = labels.loc[labels.official, 'qid'].nunique()
    sizes = {'official': (n_official, 67), 'terms': (n_official, n_all),
             'extended': (labels.qid.nunique(), n_all)}
    for c, (nq, nd) in sizes.items():
        CONFIG_TEX[c] = r'{} ({}$\times${})'.format(CONFIG_NAME[c], nq, nd)

    # 1. Distribución de etiquetas: oficial (67 docs) y colección ampliada
    rows = []
    blocks = [(labels[labels.official & (labels.doc_source == 'excel')], '67 documentos del Excel'),
              (labels, 'Colección completa: {} documentos (Excel + pooling sobre LOINC)'.format(n_all))]
    for i, (sub, title) in enumerate(blocks):
        if i:
            rows.append(r'\midrule')
        rows.append(r'\multicolumn{6}{l}{\emph{' + title + r'}} \\')
        for qid, g in sub.groupby('qid', sort=False):
            c = g.relevance.value_counts()
            mark = '' if g.official.iloc[0] else r'$^\dagger$'
            rows.append([esc(qname[qid]) + mark] + [str(c.get(k, 0)) for k in (3, 2, 1, 0)] + [str(len(g))])
    write('label_distribution.tex', tabular('lrrrrr', ['Consulta', '3', '2', '1', '0', 'Total'], rows))

    # 2. Documentos relevantes (etiqueta > 0) de las consultas oficiales, documentos del Excel
    pos = labels[labels.official & (labels.doc_source == 'excel') & (labels.relevance > 0)].sort_values(
        ['qid', 'relevance'], ascending=[True, False])
    rows, last = [], None
    for r in pos.itertuples():
        if last is not None and r.qid != last:
            rows.append(r'\midrule')
        rows.append([esc(r.query) if r.qid != last else '', esc(r.loinc_num), esc(r.long_common_name),
                     esc(r.system), esc(r.property), str(r.relevance)])
        last = r.qid
    write('positive_labels.tex', tabular('llp{6.2cm}llc',
                                         ['Consulta', 'LOINC', 'long\\_common\\_name', 'system',
                                          'property', 'Rel.'], rows))

    # 3. Ejemplos de etiquetado de documentos de la colección (decisiones representativas)
    examples = [
        (6, '11556-8'), (6, '2708-6'),                    # gasometría: pO2 exacta; saturación relacionada
        (7, '5902-2'), (7, '6301-6'), (7, '3173-2'),     # PT; INR; aPTT (otra prueba)
        (8, '33903-6'), (8, '13009-6'),                  # cetonas; metil etil cetona (disolvente)
        (9, '104451-0'), (9, '95209-3'), (9, '41458-1'), # SARS-CoV-2 RNA; antígeno combinado; SARS-CoV 2003
        (10, '630-4'), (10, '105917-9'),                 # urocultivo; cultivo de lavado broncoalveolar
        (13, '56888-1'), (13, '31201-7'),                # HIV Ab+Ag; solo anticuerpos
        (14, '11253-2'), (14, '77348-1'),                # tacrolimus; valle (condición)
        (15, '39004-7'), (15, '14049-1'),                # EGFR; EGF (el ligando)
    ]
    lab = labels.set_index(['qid', 'loinc_num'])
    rows, last = [], None
    for key in examples:
        r = lab.loc[key]  # KeyError si el término deja de existir: mejor fallar que omitirlo
        if last is not None and key[0] != last:
            rows.append(r'\midrule')
        rows.append([esc(qname[key[0]]) if key[0] != last else '', esc(r.long_common_name),
                     esc(r.analyte), esc(r.specimen), esc(r.measure), str(r.relevance)])
        last = key[0]
    write('extended_examples.tex', tabular('lp{6.6cm}lllc',
                                           ['Consulta', 'Documento de la colección', 'Analito', 'Muestra', 'Medida', 'Rel.'],
                                           rows))

    # 4. Resultados LOQO (media) en las tres configuraciones: NDCG@5 y NDCG@10
    res = {c: pd.read_csv(os.path.join(ROOT, 'results', 'loqo_{}.csv'.format(c))).groupby(
        'system', sort=False)[KS].mean() for c in CONFIG_TEX}
    header = [r'& \multicolumn{2}{c}{' + CONFIG_TEX['official'] + r'} & \multicolumn{2}{c}{' +
              CONFIG_TEX['terms'] + r'} & \multicolumn{2}{c}{' + CONFIG_TEX['extended'] + r'} \\',
              r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
              r'Sistema & @5 & @10 & @5 & @10 & @5 & @10 \\']
    rows = []
    for s in res['official'].index:
        vals = []
        for c in CONFIG_TEX:
            vals += ['{:.3f}'.format(res[c].loc[s, 'NDCG@5']), '{:.3f}'.format(res[c].loc[s, 'NDCG@10'])]
        rows.append([system_label(s)] + vals)
    write('loqo_results.tex', tabular('lrrrrrr', header, rows))

    # 5. LOQO por consulta de test: AdaRank vs f1, NDCG@10 en las tres configuraciones
    per = {c: pd.read_csv(os.path.join(ROOT, 'results', 'loqo_{}.csv'.format(c))) for c in CONFIG_TEX}
    rows = []
    for qid in per['extended'].test_qid.unique():
        vals = []
        for c in CONFIG_TEX:
            r = per[c][per[c].test_qid == qid].set_index('system')
            vals += ['{:.3f}'.format(r.loc[s, 'NDCG@10']) if qid in per[c].test_qid.values else '--'
                     for s in ('AdaRank', 'solo f1_tfidf_cosine')]
        mark = '' if qid <= 3 else r'$^\dagger$'
        rows.append([esc(qname[qid]) + mark] + vals)
    header = [r'& \multicolumn{2}{c}{' + CONFIG_TEX['official'] + r'} & \multicolumn{2}{c}{' +
              CONFIG_TEX['terms'] + r'} & \multicolumn{2}{c}{' + CONFIG_TEX['extended'] + r'} \\',
              r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
              r'Consulta de test & AdaR. & $f_1$ & AdaR. & $f_1$ & AdaR. & $f_1$ \\']
    write('loqo_per_fold.tex', tabular('lrrrrrr', header, rows))

    # 6. Coeficientes del modelo final en cada configuración
    rows = []
    for c in CONFIG_TEX:
        coef = pd.read_csv(os.path.join(ROOT, 'results', 'coef_{}.csv'.format(c))).set_index('feature')['coef']
        rows.append([CONFIG_TEX[c]] + ['{:.3f}'.format(v) for v in coef])
    write('coefficients.tex', tabular('l' + 'r' * len(FEATURES),
                                      ['Entrenado con'] + [FEAT_TEX[f] for f in FEATURES], rows))

    # 7. Ranking del modelo final: oficial (top-6) y WBC con términos nuevos (top-6)
    rk = pd.read_csv(os.path.join(ROOT, 'results', 'ranking_official.csv'), dtype={'loinc_num': str})
    write('ranking_top.tex', tabular('lrp{7cm}cr', ['Consulta', 'Pos.', 'Documento', 'Rel.', 'Score'],
                                     ranking_rows(rk, names, qname, 6)))
    rk = pd.read_csv(os.path.join(ROOT, 'results', 'ranking_terms.csv'), dtype={'loinc_num': str})
    rk = rk[rk.qid == 3]
    rows = ranking_rows(rk, names, qname, 6)
    best = rk[rk.relevance == 3].iloc[0]
    rows += [r'\multicolumn{5}{c}{$\vdots$} \\',
             ['', str(best['rank']), esc(names[best.loinc_num]), '3', '{:.2f}'.format(best.score)]]
    write('ranking_terms_wbc.tex', tabular('lrp{7cm}cr', ['Consulta', 'Pos.', 'Documento', 'Rel.', 'Score'],
                                           rows))

    # 8. Ejemplo de líneas LETOR (oficial)
    lines = open(os.path.join(ROOT, 'data', 'train.txt'), encoding='utf-8').read().splitlines()
    wanted = [(1, '74774-1'), (1, '14749-6'), (1, '15076-3'), (2, '1975-2'), (3, '26464-8'),
              (3, '26478-8'), (1, '933-2')]
    ex = [l.split(' query =')[0] for q, w in wanted for l in lines
          if 'docid = {} '.format(w) in l and 'qid:{} '.format(q) in l]
    write('letor_example.tex', '\\begin{lstlisting}\n' + '\n'.join(ex) + '\n\\end{lstlisting}\n')


if __name__ == '__main__':
    main()
