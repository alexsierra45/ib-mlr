"""
Especificación de las consultas y de la regla de relevancia 0-3.

Cada consulta define, a partir de las columnas LOINC del documento:
  - analito exacto / relacionado (sobre `component`)
  - muestra exacta / compatible (sobre `system`); cualquier otra es incompatible
  - propiedades adecuadas (sobre `property`); el resto se considera medida desviada
  - sinónimos de la consulta (solo se usan en las features, nunca en la etiqueta)

Las tres consultas oficiales del profesor tienen qid 1-3. Las consultas
adicionales (parte opcional) se marcan con official=False.
"""

import re

QUERIES = [
    {
        'qid': 1,
        'query': 'glucose in blood',
        'sheet': 'glucose in blood',
        'official': True,
        # "Glucose" o "Glucose^<condición>" (fasting, post-carga...)
        'analyte_exact': r'^glucose(\^.*)?$',
        'analyte_related': None,
        'specimen_exact': {'Bld', 'BldC', 'BldV', 'BldA', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Ser/Plas', 'Ser', 'Plas'},
        'property_ok': {'MCnc', 'SCnc', 'MSCnc'},
        'analyte_terms': ['glucose'],
        'specimen_terms': ['blood', 'bld'],
        'synonyms': {'blood': ['bld'], 'glucose': ['glu']},
    },
    {
        'qid': 2,
        'query': 'bilirubin in plasma',
        'sheet': 'bilirubin in plasma',
        'official': True,
        # El "bilirubin" sin fracción del nombre se interpreta como bilirrubina total
        'analyte_exact': r'^bilirubin(\.total)?$',
        # Fracciones: directa (glucuronidada) e indirecta (no glucuronidada)
        'analyte_related': r'^bilirubin\.(glucuronidated|non-glucuronidated|direct|indirect)',
        'specimen_exact': {'Plas', 'Ser/Plas', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Ser', 'Bld'},
        'property_ok': {'MCnc', 'SCnc', 'MSCnc'},
        'analyte_terms': ['bilirubin'],
        'specimen_terms': ['plasma', 'plas'],
        'synonyms': {'plasma': ['plas'], 'bilirubin': ['bili']},
    },
    {
        'qid': 3,
        'query': 'white blood cells count',
        'sheet': 'White blood cells count',
        'official': True,
        'analyte_exact': r'^leukocytes$',
        # Subpoblaciones de leucocitos (también como fracción /100 leukocytes)
        'analyte_related': r'^(lymphocytes|monocytes|neutrophils|eosinophils|basophils|granulocytes)\b',
        'specimen_exact': {'Bld'},
        'specimen_compatible': set(),
        # "count" pide un recuento por volumen
        'property_ok': {'NCnc'},
        'analyte_terms': ['leukocytes', 'white', 'wbc'],
        'specimen_terms': ['blood', 'bld'],
        'synonyms': {'white': ['leukocytes', 'wbc'], 'cells': ['leukocytes'],
                     'count': ['#/volume', 'ncnc'], 'blood': ['bld']},
    },
    # ---------------- Consultas adicionales (parte opcional) ----------------
    {
        'qid': 4,
        'query': 'calcium in serum',
        'sheet': None,
        'official': False,
        # Calcio total; el "corrected for albumin" lleva ^ y cuenta como condición
        'analyte_exact': r'^calcium(\^.*)?$',
        'analyte_related': r'^calcium\.ionized$',
        'specimen_exact': {'Ser', 'Ser/Plas', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Plas', 'Bld'},
        'property_ok': {'MCnc', 'SCnc', 'MSCnc'},
        'analyte_terms': ['calcium'],
        'specimen_terms': ['serum', 'ser'],
        'synonyms': {'serum': ['ser'], 'calcium': ['ca']},
    },
    {
        'qid': 5,
        'query': 'cholesterol in plasma',
        'sheet': None,
        'official': False,
        'analyte_exact': r'^cholesterol$',
        'analyte_related': r'^cholesterol\.',
        'specimen_exact': {'Plas', 'Ser/Plas', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Ser', 'Bld'},
        'property_ok': {'MCnc', 'SCnc', 'MSCnc'},
        'analyte_terms': ['cholesterol'],
        'specimen_terms': ['plasma', 'plas'],
        'synonyms': {'plasma': ['plas'], 'cholesterol': ['chol']},
    },
    {
        'qid': 6,
        'query': 'chloride in blood',
        'sheet': None,
        'official': False,
        'analyte_exact': r'^chloride$',
        'analyte_related': None,
        'specimen_exact': {'Bld', 'BldC', 'BldV', 'BldA', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Ser/Plas', 'Ser', 'Plas'},
        'property_ok': {'MCnc', 'SCnc', 'MSCnc'},
        'analyte_terms': ['chloride'],
        'specimen_terms': ['blood', 'bld'],
        'synonyms': {'blood': ['bld'], 'chloride': ['cl']},
    },
]

OFFICIAL_QUERIES = [q for q in QUERIES if q['official']]


def get_query(qid):
    return next(q for q in QUERIES if q['qid'] == qid)


# ---------------------------------------------------------------------------
# Regla de relevancia
# ---------------------------------------------------------------------------

def judge_analyte(q, component):
    c = component.strip().lower()
    if re.search(q['analyte_exact'], c):
        return 'exacto'
    if q['analyte_related'] and re.search(q['analyte_related'], c):
        return 'relacionado'
    return 'ninguno'


def judge_specimen(q, system):
    s = system.strip()
    if s in q['specimen_exact']:
        return 'exacta'
    if s in q['specimen_compatible']:
        return 'compatible'
    return 'incompatible'


def judge_measure(q, component, prop):
    """Devuelve ('adecuada'|'desviada', motivo)."""
    reasons = []
    if prop.strip() not in q['property_ok']:
        reasons.append('propiedad {}'.format(prop.strip()))
    if '^' in component:
        reasons.append('condición "{}"'.format(component.split('^', 1)[1].strip('^')))
    return ('desviada' if reasons else 'adecuada'), '; '.join(reasons)


def relevance(q, component, system, prop):
    """Aplica la regla 0-3. Devuelve (label, analyte, specimen, measure, rationale)."""
    analyte = judge_analyte(q, component)
    specimen = judge_specimen(q, system)
    measure, why = judge_measure(q, component, prop)

    if analyte == 'ninguno':
        return 0, analyte, specimen, measure, 'analito distinto ({}) -> 0'.format(component)

    base = 3 if analyte == 'exacto' else 2
    steps = ['analito {} ({}): base {}'.format(analyte, component, base)]
    label = base
    if specimen == 'compatible':
        label -= 1
        steps.append('muestra compatible pero distinta ({}): -1'.format(system))
    elif specimen == 'incompatible':
        label -= 2
        steps.append('muestra incompatible ({}): -2'.format(system))
    else:
        steps.append('muestra exacta ({})'.format(system))
    if measure == 'desviada':
        label -= 1
        steps.append('medida desviada ({}): -1'.format(why))
    if label < 1:
        steps.append('mínimo 1 por compartir analito')
        label = 1
    return label, analyte, specimen, measure, '; '.join(steps) + ' -> {}'.format(label)
