"""
fda_client.py
Handles all FDA open-API data retrieval and preprocessing.

Improvements over original:
- Removed duplicate `return results[:limit]` line
- Added structured exception logging instead of bare `except: return []`
- All search functions wrapped with @st.cache_data (TTL = 1 hour)
- Added `get_510k_url()` helper for building direct FDA links
"""

import streamlit as st
import requests
import logging

from utils import build_query_variants

logger = logging.getLogger(__name__)

CLASS_URL = "https://api.fda.gov/device/classification.json"
K510_URL  = "https://api.fda.gov/device/510k.json"
PMA_URL   = "https://api.fda.gov/device/pma.json"


# ---------------------------------------------------------------------------
# Direct link helpers
# ---------------------------------------------------------------------------

def get_510k_url(k_number: str) -> str:
    """Return a direct FDA accessdata URL for a given K-number."""
    if not k_number:
        return ""
    return f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={k_number}"


def get_pma_url(pma_number: str) -> str:
    """Return a direct FDA accessdata URL for a given PMA number."""
    if not pma_number:
        return ""
    return f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm?id={pma_number}"


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def _fda_fetch(url: str, query: str, limit: int) -> list:
    """
    Low-level GET against an FDA endpoint.
    Returns an empty list on any failure and surfaces a Streamlit warning
    so the user knows what happened instead of silently failing.
    """
    try:
        response = requests.get(
            url,
            params={"search": query, "limit": limit},
            timeout=20
        )
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.Timeout:
        st.warning("FDA API request timed out. Results may be incomplete.")
        logger.warning("FDA timeout: url=%s query=%s", url, query)
        return []
    except requests.exceptions.HTTPError as exc:
        # 404 from FDA just means no results for that query — not a real error
        if exc.response is not None and exc.response.status_code == 404:
            return []
        st.warning(f"FDA API HTTP error: {exc}")
        logger.warning("FDA HTTP error: %s", exc)
        return []
    except Exception as exc:
        st.warning(f"FDA API error: {exc}")
        logger.warning("FDA unexpected error: %s", exc, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Multi-field search
# ---------------------------------------------------------------------------

def _multi_search(url: str, field_names: list, keyword: str, intended_use: str,
                  limit_per_query: int = 10) -> list:
    variants = build_query_variants(keyword, intended_use)
    all_results = []
    seen_ids: set = set()

    for variant in variants:
        field_queries = [f'{field}:"{variant}"' for field in field_names]
        query = " OR ".join(field_queries)
        for row in _fda_fetch(url, query, limit_per_query):
            row_id = str(row)
            if row_id not in seen_ids:
                seen_ids.add(row_id)
                all_results.append(row)

    return all_results


# ---------------------------------------------------------------------------
# Public search functions — all cached for 1 hour
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def search_classification(keyword: str, intended_use: str = "", limit: int = 15) -> list:
    queries = []
    if keyword.strip():
        queries.append(keyword.strip())
    short_phrase = " ".join(intended_use.split()[:4])
    if short_phrase:
        queries.append(short_phrase)

    results = []
    seen: set = set()

    for q in queries:
        query = (
            f'device_name:"{q}" OR '
            f'definition:"{q}" OR '
            f'medical_specialty_description:"{q}"'
        )
        for r in _fda_fetch(CLASS_URL, query, limit):
            key = str(r)
            if key not in seen:
                seen.add(key)
                results.append(r)

    return results[:limit]


@st.cache_data(ttl=3600, show_spinner=False)
def search_510k(keyword: str, intended_use: str = "", product_code: str = "",
                limit: int = 20) -> list:
    direct_results = []
    if product_code:
        direct_results = _fda_fetch(K510_URL, f"product_code:{product_code}", limit)

    broad_results = _multi_search(
        K510_URL,
        ["device_name", "trade_name", "applicant"],
        keyword,
        intended_use,
        limit_per_query=8
    )

    all_results = []
    seen: set = set()
    for row in direct_results + broad_results:
        row_id = str(row)
        if row_id not in seen:
            seen.add(row_id)
            all_results.append(row)

    return all_results[:limit]


@st.cache_data(ttl=3600, show_spinner=False)
def search_pma(keyword: str, intended_use: str = "", product_code: str = "",
               limit: int = 20) -> list:
    direct_results = []
    if product_code:
        direct_results = _fda_fetch(PMA_URL, f"product_code:{product_code}", limit)

    broad_results = _multi_search(
        PMA_URL,
        ["generic_name", "trade_name", "applicant"],
        keyword,
        intended_use,
        limit_per_query=8
    )

    all_results = []
    seen: set = set()
    for row in direct_results + broad_results:
        row_id = str(row)
        if row_id not in seen:
            seen.add(row_id)
            all_results.append(row)

    return all_results[:limit]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def get_class_counts(results: list) -> dict:
    counts = {"1": 0, "2": 0, "3": 0}
    for row in results:
        value = str(row.get("device_class", "")).strip()
        if value in counts:
            counts[value] += 1
    return counts


def get_product_codes(results: list) -> list:
    counts: dict = {}
    for row in results:
        code = row.get("product_code", "")
        if code:
            counts[code] = counts.get(code, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [code for code, _ in ordered[:5]]


def get_denovo_count(results: list) -> int:
    """
    Count De Novo records. Checks k_number prefix AND decision_type field
    as a fallback so detection works even when one field is absent.
    """
    total = 0
    for row in results:
        k_number = str(row.get("k_number", "")).upper()
        decision_type = str(row.get("decision_description", "")).lower()
        if k_number.startswith("DEN") or "de novo" in decision_type:
            total += 1
    return total


def safe_columns(df, wanted: list):
    existing = [c for c in wanted if c in df.columns]
    return df[existing] if existing else df
