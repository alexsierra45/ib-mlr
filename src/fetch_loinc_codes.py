"""
Script de un solo uso: añade el código LOINC a los términos nuevos de
data/dataset_extended.csv (que solo trae nombre, component, system y property).

Busca cada long_common_name en la API pública de LOINC de la NLM
(clinicaltables.nlm.nih.gov) y se queda con la coincidencia EXACTA de nombre.
Escribe data/extended_terms.csv, que es lo que usa el pipeline (no depende de la red).

Uso:  python src/fetch_loinc_codes.py
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'dataset_extended.csv')
OUT = os.path.join(ROOT, 'data', 'extended_terms.csv')
API = 'https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search?terms={}&maxList=100&df=LOINC_NUM,LONG_COMMON_NAME'


def lookup(name):
    # La búsqueda falla con algunos símbolos (p. ej. "[#]"): si no hay coincidencia,
    # se reintenta sin ellos, comparando siempre contra el nombre completo.
    for terms in (name, re.sub(r'\[[^\]]*\]', ' ', name)):
        url = API.format(urllib.parse.quote(terms))
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        for code, long_name in data[3] or []:
            if long_name.strip().lower() == name.strip().lower():
                return code
    return None


def main():
    ext = pd.read_csv(SRC, encoding='utf-8-sig', dtype=str).apply(lambda c: c.str.strip())
    ext = ext.rename(columns={'query': 'source_query'})
    codes = []
    for i, name in enumerate(ext['long_common_name']):
        codes.append(lookup(name))
        if (i + 1) % 25 == 0:
            print('  {}/{}'.format(i + 1, len(ext)))
        time.sleep(0.1)
    ext.insert(0, 'loinc_num', codes)
    missing = ext['loinc_num'].isna().sum()
    print('Encontrados {}/{} códigos ({} sin coincidencia exacta)'.format(len(ext) - missing, len(ext), missing))
    ext.to_csv(OUT, index=False, encoding='utf-8')
    print('Escrito', os.path.relpath(OUT, ROOT))


if __name__ == '__main__':
    main()
