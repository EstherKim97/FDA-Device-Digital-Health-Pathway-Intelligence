"""
utils.py
Shared text normalization and similarity utilities.
Centralizes logic that was previously duplicated across app.py and scoring.py.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
    "that", "this", "is", "are", "be", "by", "from", "as", "at", "it",
    "software", "system", "device"
}


# ---------------------------------------------------------------------------
# Basic normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set:
    return set(normalize_text(text).split())


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Simple token-overlap (Jaccard) similarity. Returns 0.0–1.0."""
    a = tokenize(text_a)
    b = tokenize(text_b)
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    total = len(a | b)
    return overlap / total if total > 0 else 0.0


def tfidf_similarity(query: str, candidates: list[str]) -> list[float]:
    """
    TF-IDF cosine similarity between a query string and a list of candidate strings.
    Returns a list of float scores (0.0–1.0) in the same order as candidates.
    Falls back to Jaccard if sklearn is unavailable or the corpus is empty.
    """
    if not candidates:
        return []

    corpus = [normalize_text(query)] + [normalize_text(c) for c in candidates]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        # cosine_similarity returns shape (1, len(candidates))
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return [round(float(s), 4) for s in scores]
    except Exception:
        # Graceful fallback to Jaccard
        return [jaccard_similarity(query, c) for c in candidates]


def contains_any(text: str, keywords: list[str]) -> bool:
    text = normalize_text(text)
    return any(word in text for word in keywords)


# ---------------------------------------------------------------------------
# Query building helpers
# ---------------------------------------------------------------------------

def extract_terms(keyword: str, intended_use: str) -> list[str]:
    text = normalize_text(f"{keyword} {intended_use}")
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]


def build_query_variants(keyword: str, intended_use: str) -> list[str]:
    words = extract_terms(keyword, intended_use)
    variants = []

    if keyword.strip():
        variants.append(keyword.strip())

    short_text = " ".join(words[:8])
    if short_text:
        variants.append(short_text)

    for i in range(len(words) - 1):
        variants.append(f"{words[i]} {words[i+1]}")

    variants.extend(words[:8])

    seen: set = set()
    unique = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            unique.append(v)

    return unique[:10]
