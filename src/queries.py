"""
Especificación de las consultas y de la regla de relevancia 0-3.

Cada consulta define, a partir de las columnas LOINC del documento:
  - analito exacto / relacionado / excluido (sobre `component`)
  - muestra exacta / compatible (sobre `system`); cualquier otra es incompatible
  - propiedades adecuadas (sobre `property`); el resto se considera medida desviada
  - términos y sinónimos de la consulta (solo se usan en las features, nunca en la etiqueta)

Consultas:
  qid 1-3  oficiales (Excel del profesor)
  qid 4-5  añadidas por el equipo junto con sus términos (data/dataset_extended.csv)
  qid 6-8  añadidas sobre los documentos del Excel
"""

import re

# Muestras de sangre: total (y sus variantes arterial/venosa/capilar) y especiales
BLOOD = {'Bld', 'BldC', 'BldV', 'BldA', 'BldMV', 'Ser/Plas/Bld'}
BLOOD_SPECIAL = {'BldCo', 'Bld^Fetus', 'Bld.dot'}  # cordón, feto, gota seca: población o formato especial
SERUM_OR_PLASMA = {'Ser/Plas', 'Ser', 'Plas'}
QUANT = {'MCnc', 'SCnc', 'MSCnc', '{Measurement}'}

QUERIES = [
    {
        'qid': 1, 'query': 'glucose in blood', 'sheet': 'glucose in blood', 'official': True,
        'origin': 'profesor',
        # "Glucose" o "Glucose^<condición>" (ayunas, post-carga...)
        'analyte_exact': r'^glucose(\^.*)?$',
        # Glucosa media estimada a partir de la HbA1c
        'analyte_related': r'^estimated average glucose$',
        'specimen_exact': BLOOD,
        'specimen_compatible': SERUM_OR_PLASMA | BLOOD_SPECIAL,
        'property_ok': QUANT,
        'analyte_terms': ['glucose'],
        'specimen_terms': ['blood', 'bld'],
        'synonyms': {'blood': ['bld'], 'glucose': ['glu']},
    },
    {
        'qid': 2, 'query': 'bilirubin in plasma', 'sheet': 'bilirubin in plasma', 'official': True,
        'origin': 'profesor',
        # "bilirubin" sin fracción se interpreta como bilirrubina total
        'analyte_exact': r'^bilirubin(\.total)?$',
        # Fracciones (directa/conjugada, indirecta, delta), paneles y aclaramiento
        'analyte_related': r'^bilirubin\.(glucuronidated|non-glucuronidated|albumin bound|direct|indirect)'
                           r'|^(neonatal )?bilirubin\b.*panel|^bilirubin renal clearance',
        'specimen_exact': {'Plas', 'Ser/Plas', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Ser', 'Bld'},
        'property_ok': QUANT,
        'analyte_terms': ['bilirubin'],
        'specimen_terms': ['plasma', 'plas'],
        'synonyms': {'plasma': ['plas'], 'bilirubin': ['bili']},
    },
    {
        'qid': 3, 'query': 'white blood cells count', 'sheet': 'White blood cells count', 'official': True,
        'origin': 'profesor',
        'analyte_exact': r'^leukocytes$',
        # Subpoblaciones de leucocitos, como recuento o como proporción "X cells/Leukocytes"
        'analyte_related': r'^(lymphocytes|monocytes|neutrophils|eosinophils|basophils|granulocytes)\b'
                           r'|cells[^/]*/leukocytes$|^leukocytes\+',
        'specimen_exact': BLOOD - {'Ser/Plas/Bld'},
        'specimen_compatible': BLOOD_SPECIAL,
        # "count" pide un recuento por volumen
        'property_ok': {'NCnc'},
        'analyte_terms': ['leukocytes', 'white', 'wbc'],
        'specimen_terms': ['blood', 'bld'],
        'synonyms': {'white': ['leukocytes', 'wbc'], 'cells': ['leukocytes'],
                     'count': ['#/volume', 'ncnc'], 'blood': ['bld']},
    },
    # ------------- Consultas del equipo, con sus propios términos nuevos -------------
    {
        'qid': 4, 'query': 'platelets in blood', 'sheet': None, 'official': False, 'origin': 'equipo',
        'analyte_exact': r'^platelets?(\^.*)?$',
        # Subtipos (large, giant, reticulated...), índices y pruebas funcionales de las plaquetas
        'analyte_related': r'platelet',
        # Registros de transfusión de concentrados de plaquetas: no son pruebas de laboratorio
        'analyte_exclude': r'transfuse|given|units|pooled|concentrate',
        'specimen_exact': BLOOD - {'Ser/Plas/Bld'},
        'specimen_compatible': BLOOD_SPECIAL,
        # En sangre, la medida estándar de plaquetas es el recuento por volumen
        'property_ok': {'NCnc'},
        'analyte_terms': ['platelets', 'platelet'],
        'specimen_terms': ['blood', 'bld'],
        'synonyms': {'platelets': ['platelet', 'plt', 'thrombocytes'], 'blood': ['bld']},
    },
    {
        'qid': 5, 'query': 'creatinine in serum', 'sheet': None, 'official': False, 'origin': 'equipo',
        'analyte_exact': r'^creatinine(\^.*)?$',
        # Cocientes con creatinina, aclaramiento, paneles...
        'analyte_related': r'creatinine',
        # Conjuntos de códigos administrativos (protocolos PhenX, value sets HEDIS)
        'analyte_exclude': r'phenx|hedis',
        'specimen_exact': {'Ser', 'Ser/Plas', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Plas', 'Bld'},
        'property_ok': QUANT,
        'analyte_terms': ['creatinine'],
        'specimen_terms': ['serum', 'ser'],
        'synonyms': {'serum': ['ser'], 'creatinine': ['crea', 'scr']},
    },
    # ------------- Consultas adicionales sobre los documentos del Excel -------------
    {
        'qid': 6, 'query': 'calcium in serum', 'sheet': None, 'official': False, 'origin': 'equipo',
        'analyte_exact': r'^calcium(\^.*)?$',
        'analyte_related': r'^calcium\.ionized$',
        'specimen_exact': {'Ser', 'Ser/Plas', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Plas', 'Bld'},
        'property_ok': QUANT,
        'analyte_terms': ['calcium'],
        'specimen_terms': ['serum', 'ser'],
        'synonyms': {'serum': ['ser'], 'calcium': ['ca']},
    },
    {
        'qid': 7, 'query': 'cholesterol in plasma', 'sheet': None, 'official': False, 'origin': 'equipo',
        'analyte_exact': r'^cholesterol$',
        'analyte_related': r'^cholesterol\.',
        'specimen_exact': {'Plas', 'Ser/Plas', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Ser', 'Bld'},
        'property_ok': QUANT,
        'analyte_terms': ['cholesterol'],
        'specimen_terms': ['plasma', 'plas'],
        'synonyms': {'plasma': ['plas'], 'cholesterol': ['chol']},
    },
    {
        'qid': 8, 'query': 'chloride in blood', 'sheet': None, 'official': False, 'origin': 'equipo',
        'analyte_exact': r'^chloride$',
        'analyte_related': None,
        'specimen_exact': BLOOD,
        'specimen_compatible': SERUM_OR_PLASMA | BLOOD_SPECIAL,
        'property_ok': QUANT,
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

# Valores derivados, no medidos directamente (p. ej. glucosa media estimada desde la HbA1c)
DERIVED = r'estimated from|by calculation'


def judge_analyte(q, component):
    c = component.strip().lower()
    if q.get('analyte_exclude') and re.search(q['analyte_exclude'], c):
        return 'ninguno'
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


def judge_measure(q, component, prop, long_name=''):
    """Devuelve ('adecuada'|'desviada', motivo)."""
    reasons = []
    if prop.strip() not in q['property_ok']:
        reasons.append('propiedad {}'.format(prop.strip()))
    if '^' in component:
        reasons.append('condición "{}"'.format(component.split('^', 1)[1].strip('^')))
    if 'panel' in component.lower():
        reasons.append('panel de varias pruebas')
    m = re.search(DERIVED, long_name.lower())
    if m:
        reasons.append('valor derivado ("{}")'.format(m.group(0)))
    return ('desviada' if reasons else 'adecuada'), '; '.join(reasons)


def relevance(q, component, system, prop, long_name=''):
    """Aplica la regla 0-3. Devuelve (label, analyte, specimen, measure, rationale)."""
    analyte = judge_analyte(q, component)
    specimen = judge_specimen(q, system)
    measure, why = judge_measure(q, component, prop, long_name)

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
