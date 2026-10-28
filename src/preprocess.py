"""
src/preprocess.py
-----------------
Reusable text cleaning utilities for Medical Text Classification.
"""

import re


def clean_medical_text(text: str) -> str:
    """
    Cleans medical abstract text while preserving domain terminology.
    - Lowercases text
    - Strips URLs
    - Cleans non-semantic punctuation (brackets, quotes, semicolons)
    - Preserves hyphenated words (e.g., anti-inflammatory, neuron-specific)
    - Preserves decimal numbers (e.g., 1.5) and alphanumeric terms (e.g., cd4, mac16)
    - Collapses multiple spaces
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase text
    cleaned = text.lower()

    # 2. Remove URLs if any
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # 3. Replace brackets, quotes, slashes, and symbols with spaces
    cleaned = re.sub(r'[\(\)\[\]\{\}\"\'\\\/<>,;:!+*?=~_]', " ", cleaned)

    # 4. Remove standalone hyphens, keeping hyphenated words
    cleaned = re.sub(r"(?<!\w)-|-(?!\w)", " ", cleaned)

    # 5. Remove sentence-ending periods, keeping decimals
    cleaned = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", cleaned)

    # 6. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
