"""
Especificación de las consultas y de la regla de relevancia 0-3.

Cada consulta define, a partir de las columnas LOINC del documento:
  - analito exacto / relacionado / excluido (sobre `component`)
  - muestra exacta / compatible (sobre `system`); cualquier otra es incompatible
  - propiedades adecuadas (sobre `property`); el resto se considera medida desviada
  - términos y sinónimos de la consulta (solo se usan en las features, nunca en la etiqueta)

Consultas:
  qid 1-3   oficiales (Excel del profesor), con especificación escrita a mano
  qid 4-15  las 12 pruebas de laboratorio más frecuentes de LOINC que cumplen los criterios
            de src/build_collection.py, con especificación generada por make_query()
"""

import re

# Muestras de sangre: total (y sus variantes arterial/venosa/capilar) y especiales
BLOOD = {'Bld', 'BldC', 'BldV', 'BldA', 'BldMV', 'BldP', 'Ser/Plas/Bld'}
BLOOD_SPECIAL = {'BldCo', 'BldCoA', 'BldCoV', 'Bld^Fetus', 'Bld.dot'}  # cordón, feto, gota seca
SERUM_OR_PLASMA = {'Ser/Plas', 'Ser', 'Plas'}
QUANT = {'MCnc', 'SCnc', 'MSCnc', '{Measurement}'}
# Registros administrativos que contienen el nombre del analito pero no son pruebas
ADMIN = r'transfuse|given|units|available|phenx|hedis|value set'

OFFICIAL = [
    {
        'qid': 1, 'query': 'glucose in blood', 'sheet': 'glucose in blood',
        # "Glucose" o "Glucose^<condición>" (ayunas, post-carga...)
        'analyte_exact': r'^glucose(\^.*)?$',
        # Glucosa media estimada a partir de la HbA1c
        'analyte_related': r'^estimated average glucose$',
        'analyte_exclude': ADMIN,
        'specimen_exact': BLOOD,
        'specimen_compatible': SERUM_OR_PLASMA | BLOOD_SPECIAL,
        'property_ok': QUANT,
        'analyte_terms': ['glucose'],
        'specimen_terms': ['blood', 'bld'],
        'synonyms': {'blood': ['bld'], 'glucose': ['glu']},
    },
    {
        'qid': 2, 'query': 'bilirubin in plasma', 'sheet': 'bilirubin in plasma',
        # "bilirubin" sin fracción se interpreta como bilirrubina total
        'analyte_exact': r'^bilirubin(\.total)?$',
        # Fracciones (directa/conjugada, indirecta, delta), paneles y aclaramiento
        'analyte_related': r'^bilirubin\.(glucuronidated|non-glucuronidated|albumin bound|direct|indirect)'
                           r'|^(neonatal )?bilirubin\b.*panel|^bilirubin renal clearance',
        'analyte_exclude': ADMIN,
        'specimen_exact': {'Plas', 'Ser/Plas', 'Ser/Plas/Bld'},
        'specimen_compatible': {'Ser', 'Bld'},
        'property_ok': QUANT,
        'analyte_terms': ['bilirubin'],
        'specimen_terms': ['plasma', 'plas'],
        'synonyms': {'plasma': ['plas'], 'bilirubin': ['bili']},
    },
    {
        'qid': 3, 'query': 'white blood cells count', 'sheet': 'White blood cells count',
        'analyte_exact': r'^leukocytes$',
        # Subpoblaciones de leucocitos, como recuento o como proporción "X cells/Leukocytes"
        'analyte_related': r'^(lymphocytes|monocytes|neutrophils|eosinophils|basophils|granulocytes)\b'
                           r'|cells[^/]*/leukocytes$|^leukocytes\+',
        'analyte_exclude': ADMIN,
        'specimen_exact': BLOOD - {'Ser/Plas/Bld'},
        'specimen_compatible': BLOOD_SPECIAL,
        # "count" pide un recuento por volumen
        'property_ok': {'NCnc'},
        'analyte_terms': ['leukocytes', 'white', 'wbc'],
        'specimen_terms': ['blood', 'bld'],
        'synonyms': {'white': ['leukocytes', 'wbc'], 'cells': ['leukocytes'],
                     'count': ['#/volume', 'ncnc'], 'blood': ['bld']},
    },
]


def make_query(qid, component, system, prop, source_loinc, synonyms, related_extra=None):
    """Especificación genérica a partir de la prueba LOINC más frecuente del analito.

    - Analito exacto: el mismo componente (singular o plural), con o sin condición ^.
    - Relacionado: el analito como término propio seguido de fracción (.), condición (^)
      o en un cociente (/), p. ej. "Calcium.ionized", "Urea nitrogen/Creatinine"; su
      aclaramiento renal; o un panel que lo incluye ("Sodium & Potassium panel").
      `related_extra` añade variantes propias del analito (p. ej. Hemoglobin A1c).
    - Muestra: la de la prueba de origen (suero/plasma o sangre) y sus compatibles.
    - Propiedad: la de la prueba de origen (concentración, recuento o actividad).
    """
    name = component.lower()
    base = name[:-1] if name.endswith('s') else name
    word = re.escape(base) + 's?'
    if system == 'Ser/Plas':
        specimen, exact = 'serum', {'Ser', 'Ser/Plas', 'Ser/Plas/Bld'}
        compatible = {'Plas'} | (BLOOD - {'Ser/Plas/Bld'})
        specimen_terms = ['serum', 'ser']
    else:
        specimen, exact = 'blood', BLOOD - ({'Ser/Plas/Bld'} if prop == 'NCnc' else set())
        compatible = BLOOD_SPECIAL
        specimen_terms = ['blood', 'bld']
    if prop == 'NCnc':
        query, prop_ok = '{} count in {}'.format(name, specimen), {'NCnc'}
        synonyms = dict(synonyms, count=['#/volume', 'ncnc'])
    elif prop == 'CCnc':
        query, prop_ok = '{} in {}'.format(name, specimen), {'CCnc', '{Measurement}'}
    else:
        query, prop_ok = '{} in {}'.format(name, specimen), QUANT
    synonyms = dict(synonyms, **{specimen: specimen_terms[1:]})
    return {
        'qid': qid, 'query': query, 'sheet': None, 'source_loinc': source_loinc,
        'analyte_exact': r'^{}(\^.*)?$'.format(word),
        'analyte_related': r'(^|/){w}([./^]|$)|^{w} renal clearance|(^|& ){w}\b[^/]*panel'.format(w=word)
                           + ('|' + related_extra if related_extra else ''),
        'analyte_exclude': ADMIN,
        'specimen_exact': exact, 'specimen_compatible': compatible,
        'property_ok': prop_ok,
        'analyte_terms': re.findall(r'[a-z0-9]+', name),
        'specimen_terms': specimen_terms,
        'synonyms': synonyms,
    }


# Salida de src/build_collection.py (selección por COMMON_TEST_RANK); allí se comprueba que coincide.
#            qid  component                   system      property  LOINC     sinónimos (solo features)
SELECTED = [
    make_query(4, 'Creatinine', 'Ser/Plas', 'MCnc', '2160-0', {'creatinine': ['crea', 'scr']}),
    make_query(5, 'Sodium', 'Ser/Plas', 'SCnc', '2951-2', {'sodium': ['na']}),
    make_query(6, 'Potassium', 'Ser/Plas', 'SCnc', '2823-3', {'potassium': ['k']}),
    make_query(7, 'Chloride', 'Ser/Plas', 'SCnc', '2075-0', {'chloride': ['cl']}),
    make_query(8, 'Urea nitrogen', 'Ser/Plas', 'MCnc', '3094-0', {'urea': ['bun'], 'nitrogen': ['bun']}),
    make_query(9, 'Calcium', 'Ser/Plas', 'MCnc', '17861-6', {'calcium': ['ca']}),
    make_query(10, 'Carbon dioxide', 'Ser/Plas', 'SCnc', '2028-9', {'carbon': ['co2'], 'dioxide': ['co2']}),
    make_query(11, 'Erythrocytes', 'Bld', 'NCnc', '789-8', {'erythrocytes': ['rbc', 'red']}),
    make_query(12, 'Hemoglobin', 'Bld', 'MCnc', '718-7', {'hemoglobin': ['hgb', 'hb']},
               related_extra=r'^hemoglobin [a-z0-9]+$'),  # variantes: A1c, F, A2...
    make_query(13, 'Platelets', 'Bld', 'NCnc', '777-3', {'platelets': ['platelet', 'plt', 'thrombocytes']}),
    make_query(14, 'Protein', 'Ser/Plas', 'MCnc', '2885-2', {'protein': ['prot']}),
    make_query(15, 'Alanine aminotransferase', 'Ser/Plas', 'CCnc', '1742-6',
               {'alanine': ['alt', 'sgpt'], 'aminotransferase': ['alt', 'sgpt']}),
]

for _q in OFFICIAL:
    _q['official'] = True
for _q in SELECTED:
    _q['official'] = False

QUERIES = OFFICIAL + SELECTED
OFFICIAL_QUERIES = OFFICIAL


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
