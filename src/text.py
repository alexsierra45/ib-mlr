"""Preprocesado de texto común (features y pooling)."""

import re

STOPWORDS = {'in', 'of', 'or', 'by', 'the', 'for', 'and', 'from', 'to', 'a'}


def stem(tok):
    """Stemming mínimo: plural inglés regular (cells -> cell, leukocytes -> leukocyte)."""
    return tok[:-1] if len(tok) > 3 and tok.endswith('s') and not tok.endswith('ss') else tok


def tokenize(text):
    return [stem(t) for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOPWORDS]
