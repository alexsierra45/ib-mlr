"""
Especificación de las consultas y de la regla de relevancia 0-3.

Cada consulta define, a partir de las columnas LOINC del documento:
  - analito exacto / relacionado / excluido (sobre `component`; o sobre el nombre largo si
    `analyte_field` = 'name', para pruebas que LOINC distingue por el método, como la coagulación)
  - muestra exacta / compatible (sobre `system`); cualquier otra es incompatible
  - propiedades adecuadas (sobre `property`); el resto se considera medida desviada
  - términos y sinónimos de la consulta (solo se usan en las features, nunca en la etiqueta)

Consultas:
  qid 1-3   oficiales (Excel del profesor)
  qid 4-15  selección estratificada de src/build_collection.py (prueba más frecuente de cada
            dominio clínico x grupo de muestra); el texto de la consulta se redacta a mano
"""

import re

# Muestras de sangre: total (y sus variantes arterial/venosa/capilar/central) y especiales
BLOOD = {'Bld', 'BldC', 'BldV', 'BldA', 'BldMV', 'BldCV', 'BldP', 'Ser/Plas/Bld'}
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


URINE = {'Urine', 'Urine sed'}
RESPIRATORY = {'Respiratory System Specimen', 'Respiratory system specimen.upper', 'Nph', 'Nose', 'Thrt',
               'Nasopharyngeal', 'Oropharyngeal wash'}
RESPIRATORY_OTHER = {'Respiratory system specimen.lower', 'Sputum', 'Bronchial', 'BAL', 'Saliva'}
PRESENCE = {'PrThr', 'ACnc'}


def spec(qid, query, source_loinc, exact, related, specimen_exact, specimen_compatible, property_ok,
         analyte_terms, specimen_terms, synonyms, exclude=ADMIN, analyte_field='component'):
    """Especificación de una consulta de la ampliación (mismo esquema que las oficiales)."""
    return {
        'qid': qid, 'query': query, 'sheet': None, 'source_loinc': source_loinc,
        'analyte_field': analyte_field,
        'analyte_exact': exact, 'analyte_related': related, 'analyte_exclude': exclude,
        'specimen_exact': specimen_exact, 'specimen_compatible': specimen_compatible,
        'property_ok': property_ok,
        'analyte_terms': analyte_terms, 'specimen_terms': specimen_terms, 'synonyms': synonyms,
    }


def analyte_patterns(name):
    """Patrones genéricos de analito exacto y relacionado para un componente simple.

    Exacto: el mismo componente (singular o plural), con o sin condición ^.
    Relacionado: el analito como término propio seguido de fracción (.), condición (^) o en un
    cociente (/) -"Calcium.ionized", "Urea nitrogen/Creatinine"-, su aclaramiento renal o un
    panel que lo incluye ("Sodium & Potassium panel"). "Protein C" no es "protein".
    """
    base = name[:-1] if name.endswith('s') else name
    w = re.escape(base) + 's?'
    return (r'^{}(\^.*)?$'.format(w),
            r'(^|/){w}([./^]|$)|^{w} renal clearance|(^|& ){w}\b[^/]*panel'.format(w=w))


# Prueba de origen (salida de src/build_collection.py, allí se comprueba) y consulta redactada a mano.
SELECTED = [
    # Química, suero/plasma -- 2160-0 Creatinine [Mass/volume] in Serum or Plasma
    spec(4, 'creatinine in serum', '2160-0', *analyte_patterns('creatinine'),
         {'Ser', 'Ser/Plas', 'Ser/Plas/Bld'}, {'Plas'} | (BLOOD - {'Ser/Plas/Bld'}), QUANT,
         ['creatinine'], ['serum', 'ser'], {'creatinine': ['crea', 'scr'], 'serum': ['ser']}),
    # Hematología, sangre -- 789-8 Erythrocytes [#/volume] in Blood by Automated count
    spec(5, 'red blood cells count', '789-8', *analyte_patterns('erythrocytes'),
         BLOOD - {'Ser/Plas/Bld'}, BLOOD_SPECIAL, {'NCnc'},
         ['erythrocytes', 'erythrocyte'], ['blood', 'bld'],
         {'red': ['erythrocyte', 'rbc'], 'cells': ['erythrocyte'], 'count': ['#/volume', 'ncnc'],
          'blood': ['bld']}),
    # Química, sangre (gasometría) -- 11556-8 Oxygen [Partial pressure] in Blood
    spec(6, 'oxygen partial pressure in blood', '11556-8',
         r'^oxygen(\^.*)?$', r'^oxygen\b',  # contenido, saturación, capacidad, cocientes
         BLOOD, BLOOD_SPECIAL, {'PPres'},
         ['oxygen'], ['blood', 'bld'],
         {'oxygen': ['po2', 'o2'], 'partial': ['ppres'], 'pressure': ['ppres'], 'blood': ['bld']}),
    # Coagulación, plasma -- 5902-2 Prothrombin time (PT). LOINC distingue las pruebas de
    # coagulación por el MÉTODO (el componente es siempre "Coagulation"), así que el analito
    # se juzga sobre el nombre largo: PT exacto; INR y cocientes de PT, relacionados.
    spec(7, 'prothrombin time in plasma', '5902-2',
         r'^prothrombin time \(pt\)(?! actual)', r'\binr\b|prothrombin time',  # PT actual/normal: cociente
         {'PPP', 'Plas'}, {'Bld', 'BldC', 'PPP/Bld'}, {'Time', 'RelTime'},
         ['coagulation', 'prothrombin'], ['plasma', 'ppp', 'plas'],
         {'prothrombin': ['pt', 'inr', 'coagulation'], 'time': ['pt'], 'plasma': ['ppp', 'plas']},
         analyte_field='name'),
    # Química, orina -- 2514-8 Ketones [Presence] in Urine by Test strip
    spec(8, 'ketones in urine', '2514-8',
         r'^ketones(\^.*)?$', r'^(acetoacetate|beta hydroxybutyrate)\b',  # cuerpos cetónicos concretos
         URINE, set(), PRESENCE | QUANT,  # la consulta no especifica cualitativo/cuantitativo
         ['ketones', 'ketone'], ['urine'],
         {'ketones': ['acetoacetate', 'hydroxybutyrate']}),
    # Microbiología, respiratoria -- 94500-6 SARS-CoV-2 RNA [Presence] in Respiratory system specimen
    spec(9, 'SARS-CoV-2 RNA in respiratory specimen', '94500-6',
         # Relacionado: genes diana, antígeno, anticuerpos y tests combinados o paneles que lo incluyen.
         # "SARS coronavirus RNA" (SARS-CoV de 2003) no lo es.
         r'^sars coronavirus 2 rna$', r'sars coronavirus 2|sars-cov-2',
         RESPIRATORY, RESPIRATORY_OTHER, {'PrThr', 'Prid'},
         ['sars', 'coronavirus'], ['respiratory', 'nph', 'thrt'],
         {'sars': ['coronavirus', 'covid'], 'cov': ['coronavirus', 'covid'],
          'respiratory': ['nph', 'nasopharyngeal', 'thrt']}),
    # Microbiología, orina -- 630-4 Bacteria identified in Urine by Culture
    spec(10, 'bacteria in urine culture', '630-4',
         r'^bacteria(\^.*)?$', r'^bacteria\b',  # paneles de sensibilidad, recuentos...
         {'Urine'}, {'Urine sed'}, {'Prid'},
         ['bacteria'], ['urine'],
         {'bacteria': ['organism'], 'culture': ['prid']}),
    # Uroanálisis, orina -- 5799-2 Leukocyte esterase [Presence] in Urine by Test strip
    spec(11, 'leukocyte esterase in urine', '5799-2',
         r'^leukocyte esterase$', r'^leukocyte esterase\b',
         URINE, set(), PRESENCE | {'CCnc'},
         ['leukocyte', 'esterase'], ['urine'],
         {'leukocyte': ['wbc'], 'esterase': ['le']}),
    # Serología, suero -- 30539-1 Neutrophil cytoplasmic Ab.perinuclear [Presence] in Serum
    spec(12, 'perinuclear ANCA in serum', '30539-1',
         r'^neutrophil cytoplasmic ab\.perinuclear$', r'^neutrophil cytoplasmic ab\b',  # c-ANCA, atípico...
         {'Ser', 'Ser/Plas'}, {'Plas'}, PRESENCE | {'Titr'},
         ['neutrophil', 'cytoplasmic'], ['serum', 'ser'],
         {'anca': ['neutrophil', 'cytoplasmic'], 'perinuclear': ['perinuclear'], 'serum': ['ser']}),
    # Microbiología, suero/plasma -- 56888-1 HIV 1+2 Ab+HIV1 p24 Ag [Presence] in Serum or Plasma
    spec(13, 'HIV antibody and antigen in serum', '56888-1',
         r'^hiv 1\+2 ab\+hiv ?1 p24 ag$', r'^hiv\b',  # solo anticuerpos, carga viral, genotipo...
         {'Ser', 'Ser/Plas', 'Ser/Plas/Bld'}, {'Plas', 'Bld', 'Bld.dot'}, PRESENCE,
         ['hiv'], ['serum', 'ser'],
         {'antibody': ['ab'], 'antigen': ['ag', 'p24'], 'serum': ['ser']}),
    # Tóxicos/fármacos, sangre -- 11253-2 Tacrolimus [Mass/volume] in Blood
    spec(14, 'tacrolimus in blood', '11253-2', *analyte_patterns('tacrolimus'),
         {'Bld'}, {'Ser/Plas', 'Bld.dot'}, QUANT,
         ['tacrolimus'], ['blood', 'bld'],
         {'tacrolimus': ['fk506'], 'blood': ['bld']}),
    # Patología, tejido -- 39004-7 Epidermal growth factor receptor Ag [Presence] in Tissue
    spec(15, 'EGFR antigen in tissue', '39004-7',
         r'^epidermal growth factor receptor ag$', r'^(epidermal growth factor receptor|egfr gene)\b',
         {'Tiss'}, {'Bld/Tiss'}, PRESENCE | {'Arb'},
         ['epidermal', 'egfr'], ['tissue', 'tiss'],
         {'egfr': ['epidermal', 'receptor'], 'antigen': ['ag'], 'tissue': ['tiss']}),
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
    by_name = q.get('analyte_field') == 'name'
    analyte_text = long_name if by_name else component
    analyte = judge_analyte(q, analyte_text)
    specimen = judge_specimen(q, system)
    measure, why = judge_measure(q, component, prop, long_name)

    if analyte == 'ninguno':
        return 0, analyte, specimen, measure, 'analito distinto ({}) -> 0'.format(analyte_text)

    base = 3 if analyte == 'exacto' else 2
    steps = ['analito {} ({}{}): base {}'.format(analyte, analyte_text, ', por nombre' if by_name else '', base)]
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
