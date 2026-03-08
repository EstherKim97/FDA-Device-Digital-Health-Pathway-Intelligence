"""
scoring.py
All pathway scoring, CDS screening, predicate ranking, and recommendation logic.

Improvements over original:
- Scoring weights extracted to WEIGHTS dict at top of file for easy tuning
- TF-IDF cosine similarity (via utils.tfidf_similarity) replaces Jaccard for predicate ranking
- normalize_text / tokenize / contains_any imported from utils.py (no longer duplicated)
- PMA score capped consistently before normalisation
- High-risk keyword detection now requires 2+ signal types to co-occur before boosting PMA score
- rank_predicates removed from app.py (single definition here)
"""

from utils import normalize_text, contains_any, tfidf_similarity, jaccard_similarity


# ---------------------------------------------------------------------------
# Scoring weights — edit these to tune pathway sensitivity
# ---------------------------------------------------------------------------

WEIGHTS = {
    # 510(k)
    "510k_predicate_yes":           35,
    "510k_predicate_maybe":         20,
    "510k_risk_low":                10,
    "510k_risk_moderate":           20,
    "510k_novelty_low":             20,
    "510k_novelty_medium":          10,
    "510k_k_count_cap":             16,   # max points from raw k_count
    "510k_k_count_per_record":       4,
    "510k_class2_cap":              20,
    "510k_class2_per_record":        5,
    "510k_similarity_scale":        20,   # multiplied by similarity score
    "510k_similarity_high_boost":   15,   # sim >= 0.20
    "510k_similarity_mid_boost":     8,   # sim >= 0.10
    "510k_low_sim_denovo_penalty":  10,   # added to De Novo when sim is low

    # De Novo
    "denovo_predicate_no":          35,
    "denovo_predicate_maybe":       15,
    "denovo_risk_moderate":         20,
    "denovo_novelty_high":          25,
    "denovo_novelty_medium":        15,
    "denovo_count_per_record":      10,
    "denovo_count_cap":             20,
    "denovo_class12_per_record":     5,
    "denovo_class12_cap":           20,
    "denovo_low_sim_boost":         10,   # when sim < 0.15
    "denovo_implant_therapy_pen":   15,   # penalty when both implant + therapeutic

    # PMA
    "pma_risk_high":                40,
    "pma_novelty_high":             15,
    "pma_count_per_record":         10,
    "pma_count_cap":                25,
    "pma_class3_per_record":        10,
    "pma_class3_cap":               25,
    "pma_implant_boost":            25,
    "pma_therapeutic_boost":        20,
    "pma_serious_boost":            15,
    # High-risk signals now require 2+ co-occurring signals before applying boosts
    "pma_min_signal_count":          2,
}


# ---------------------------------------------------------------------------
# High-risk keyword detection
# ---------------------------------------------------------------------------

def detect_high_risk_signals(keyword: str, intended_use: str) -> dict:
    combined = f"{keyword} {intended_use}"

    implant_kw = [
        "implant", "implantable", "stimulator", "stimulation",
        "deep brain", "brain stimulation", "neurostimulator",
        "defibrillator", "pacemaker"
    ]
    therapeutic_kw = [
        "treat", "therapy", "deliver electrical stimulation",
        "deliver stimulation", "therapeutic"
    ]
    serious_kw = [
        "parkinson", "essential tremor", "epilepsy",
        "life threatening", "ventricular"
        # Removed bare "brain" — too broad and caused false positives
    ]

    implant_signal    = contains_any(combined, implant_kw)
    therapeutic_signal = contains_any(combined, therapeutic_kw)
    serious_signal    = contains_any(combined, serious_kw)

    signal_count = sum([implant_signal, therapeutic_signal, serious_signal])

    return {
        "implant_signal":    implant_signal,
        "therapeutic_signal": therapeutic_signal,
        "serious_signal":    serious_signal,
        "signal_count":      signal_count,
    }


def infer_product_label(keyword: str, intended_use: str) -> str:
    combined = f"{keyword} {intended_use}".lower()

    if "deep brain" in combined or "brain stimulation" in combined:
        return "deep brain stimulation system"
    if "defibrillator" in combined:
        return "implantable cardiac defibrillator"
    if "pacemaker" in combined:
        return "pacemaker system"
    if "ecg" in combined or "electrocard" in combined:
        return "ECG monitoring device"
    if "retinal" in combined or "diabetic retinopathy" in combined:
        return "retinal diagnostic software"

    return keyword.strip() if keyword.strip() else "proposed product"


# ---------------------------------------------------------------------------
# Predicate ranking — TF-IDF cosine similarity
# ---------------------------------------------------------------------------

def rank_predicates(intended_use: str, k_results: list, top_n: int = 5) -> list:
    """
    Rank 510(k) records against the intended use using TF-IDF cosine similarity.
    Falls back to Jaccard if sklearn is unavailable.
    """
    if not k_results:
        return []

    candidates = [
        f"{row.get('device_name', '')} {row.get('product_code', '')}"
        for row in k_results
    ]

    scores = tfidf_similarity(intended_use, candidates)

    ranked = []
    for row, score in zip(k_results, scores):
        ranked.append({
            "k_number":       row.get("k_number", ""),
            "device_name":    row.get("device_name", ""),
            "applicant":      row.get("applicant", ""),
            "decision_date":  row.get("decision_date", ""),
            "product_code":   row.get("product_code", ""),
            "similarity_score": round(score, 4),
        })

    ranked.sort(key=lambda x: x["similarity_score"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Regulated device score
# ---------------------------------------------------------------------------

def regulated_device_score(
    supports_diagnosis: bool,
    supports_treatment: bool,
    patient_specific: bool,
    independent_review: bool,
    risk_level: str,
    classification_match_count: int
) -> int:
    score = 20

    if supports_diagnosis:   score += 20
    if supports_treatment:   score += 15
    if patient_specific:     score += 15
    if not independent_review: score += 10

    if risk_level == "Moderate":   score += 10
    elif risk_level == "High":     score += 20

    if classification_match_count > 0:
        score += 10

    return min(score, 100)


# ---------------------------------------------------------------------------
# CDS screen
# ---------------------------------------------------------------------------

def cds_screen(
    intended_for_hcp: bool,
    independent_review: bool,
    supports_diagnosis: bool,
    supports_treatment: bool,
    patient_specific: bool
) -> dict:
    """
    Rough CDS-style screening output.
    This is not legal advice; it is a structured signal.
    """
    score = 0
    notes = []

    if intended_for_hcp:
        score += 25
        notes.append("Intended for healthcare professionals.")
    else:
        notes.append("Not limited to healthcare professionals.")

    if independent_review:
        score += 35
        notes.append("Independent review of the basis is possible.")
    else:
        notes.append("Independent review of the basis is not clear.")

    if not patient_specific:
        score += 15
        notes.append("Output is not strongly patient-specific.")
    else:
        notes.append("Patient-specific output increases device sensitivity.")

    if not supports_treatment:
        score += 10
    else:
        notes.append("Treatment support increases regulatory sensitivity.")

    if not supports_diagnosis:
        score += 10
    else:
        notes.append("Diagnostic support increases regulatory sensitivity.")

    score = min(score, 100)

    if score >= 70:
        label = "Possible CDS-friendly profile"
    elif score >= 40:
        label = "Mixed CDS profile"
    else:
        label = "Likely device-like profile"

    return {"score": score, "label": label, "notes": notes}


# ---------------------------------------------------------------------------
# Pathway scores
# ---------------------------------------------------------------------------

def pathway_scores(
    keyword: str,
    intended_use: str,
    predicate_known: str,
    risk_level: str,
    novelty: str,
    class_count_map: dict,
    k_count: int,
    den_count: int,
    pma_count_value: int,
    predicate_similarity_max: float
) -> dict:
    W = WEIGHTS
    risk_flags = detect_high_risk_signals(keyword, intended_use)

    score_510k   = 0
    score_denovo = 0
    score_pma    = 0

    # --- 510(k) ---
    if predicate_known == "Yes":
        score_510k += W["510k_predicate_yes"]
    elif predicate_known == "Maybe":
        score_510k += W["510k_predicate_maybe"]

    if risk_level == "Low":
        score_510k += W["510k_risk_low"]
    elif risk_level == "Moderate":
        score_510k += W["510k_risk_moderate"]

    if novelty == "Low":
        score_510k += W["510k_novelty_low"]
    elif novelty == "Medium":
        score_510k += W["510k_novelty_medium"]

    score_510k += min(k_count * W["510k_k_count_per_record"], W["510k_k_count_cap"])
    score_510k += min(class_count_map["2"] * W["510k_class2_per_record"], W["510k_class2_cap"])
    score_510k += round(predicate_similarity_max * W["510k_similarity_scale"])

    if predicate_similarity_max >= 0.20:
        score_510k += W["510k_similarity_high_boost"]
    elif predicate_similarity_max >= 0.10:
        score_510k += W["510k_similarity_mid_boost"]
    else:
        score_denovo += W["510k_low_sim_denovo_penalty"]

    # --- De Novo ---
    if predicate_known == "No":
        score_denovo += W["denovo_predicate_no"]
    elif predicate_known == "Maybe":
        score_denovo += W["denovo_predicate_maybe"]

    if risk_level == "Moderate":
        score_denovo += W["denovo_risk_moderate"]

    if novelty == "High":
        score_denovo += W["denovo_novelty_high"]
    elif novelty == "Medium":
        score_denovo += W["denovo_novelty_medium"]

    score_denovo += min(den_count * W["denovo_count_per_record"], W["denovo_count_cap"])
    score_denovo += min(
        (class_count_map["1"] + class_count_map["2"]) * W["denovo_class12_per_record"],
        W["denovo_class12_cap"]
    )

    if predicate_similarity_max < 0.15:
        score_denovo += W["denovo_low_sim_boost"]

    # Only penalise De Novo if BOTH implant AND therapeutic signals are present
    if risk_flags["implant_signal"] and risk_flags["therapeutic_signal"]:
        score_denovo -= W["denovo_implant_therapy_pen"]

    # --- PMA ---
    if risk_level == "High":
        score_pma += W["pma_risk_high"]

    if novelty == "High":
        score_pma += W["pma_novelty_high"]

    score_pma += min(pma_count_value * W["pma_count_per_record"], W["pma_count_cap"])
    score_pma += min(class_count_map["3"] * W["pma_class3_per_record"], W["pma_class3_cap"])

    # High-risk boosts only apply when 2+ signal types co-occur
    if risk_flags["signal_count"] >= W["pma_min_signal_count"]:
        if risk_flags["implant_signal"]:
            score_pma += W["pma_implant_boost"]
        if risk_flags["therapeutic_signal"]:
            score_pma += W["pma_therapeutic_boost"]
        if risk_flags["serious_signal"]:
            score_pma += W["pma_serious_boost"]

    raw = {
        "510(k)":  max(score_510k, 0),
        "De Novo": max(score_denovo, 0),
        "PMA":     max(score_pma, 0),
    }

    total = sum(raw.values())
    if total == 0:
        return {"510(k)": 0, "De Novo": 0, "PMA": 0}

    return {k: round(v / total * 100) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Breakthrough score
# ---------------------------------------------------------------------------

def breakthrough_score(serious_condition: bool, unmet_need: bool, novelty: str) -> int:
    score = 0
    if serious_condition: score += 45
    if unmet_need:        score += 35
    if novelty == "High":   score += 20
    elif novelty == "Medium": score += 10
    return min(score, 100)


# ---------------------------------------------------------------------------
# Evidence gaps
# ---------------------------------------------------------------------------

def evidence_gaps(
    predicate_known: str,
    risk_level: str,
    novelty: str,
    classification_match_count: int,
    k_count: int,
    den_count: int,
    pma_count_value: int,
    predicate_similarity_max: float
) -> list:
    gaps = []

    if classification_match_count == 0:
        gaps.append("No strong FDA classification matches were found. Refine the keyword and intended use wording.")

    if predicate_known in ["No", "Maybe"]:
        gaps.append("Predicate mapping is not yet clearly established.")

    if predicate_similarity_max < 0.10:
        gaps.append("The current predicate similarity signal is weak.")

    if novelty == "High" and den_count == 0:
        gaps.append("Novel profile selected, but FDA search shows no strong De Novo-like precedent.")

    if risk_level == "High" and pma_count_value == 0:
        gaps.append("High-risk profile selected, but FDA search shows no clear PMA precedent support.")

    if predicate_known == "Yes" and k_count == 0:
        gaps.append("A predicate is assumed, but FDA search did not retrieve clear 510(k) records.")

    if not gaps:
        gaps.append("No major structural gap detected in this first-pass screen. Next focus: evidence strategy and intended use refinement.")

    return gaps


# ---------------------------------------------------------------------------
# Next-step recommendation
# ---------------------------------------------------------------------------

def next_step_recommendation(pathway_map: dict, cds_result: dict, gaps: list) -> str:
    top_pathway = max(pathway_map, key=pathway_map.get)
    top_score   = pathway_map[top_pathway]

    sorted_items = sorted(pathway_map.items(), key=lambda x: x[1], reverse=True)
    second_pathway, second_score = sorted_items[1]
    margin = top_score - second_score

    if cds_result["score"] >= 70:
        return (
            "Evaluate possible CDS positioning first — the current profile may support a "
            "lower-burden regulatory approach."
        )

    if top_score >= 65 and margin >= 20 and len(gaps) <= 1:
        return (
            f"Proceed with {top_pathway} pathway preparation. "
            "The current ranking shows a strong lead over alternative routes."
        )

    if top_score >= 45 and margin >= 10:
        return (
            f"{top_pathway} is currently the leading pathway, but the margin over "
            f"{second_pathway} remains moderate. Consider a Pre-Sub to confirm the proposed "
            "route and key evidence expectations."
        )

    return (
        "Pre-Sub is recommended — the current pathway ranking remains uncertain and FDA "
        "feedback would help clarify the route."
    )


# ---------------------------------------------------------------------------
# Reason list (explainability)
# ---------------------------------------------------------------------------

def reason_list(
    top_pathway: str,
    predicate_known: str,
    novelty: str,
    risk_level: str,
    class_count_map: dict,
    k_count: int,
    den_count: int,
    pma_count_value: int,
    predicate_similarity_max: float
) -> list:
    reasons = []

    if top_pathway == "510(k)":
        if predicate_known in ["Yes", "Maybe"]:
            reasons.append("Predicate signal exists from intake inputs.")
        if class_count_map["2"] > 0:
            reasons.append("FDA classification search returned Class II signals.")
        if k_count > 0:
            reasons.append("FDA precedent search returned 510(k) records.")
        if predicate_similarity_max >= 0.20:
            reasons.append("Top predicate candidate shows strong TF-IDF cosine similarity.")
        elif predicate_similarity_max >= 0.10:
            reasons.append("Top predicate candidate shows moderate TF-IDF cosine similarity.")

    elif top_pathway == "De Novo":
        if predicate_known == "No":
            reasons.append("No clear predicate was selected.")
        if novelty in ["Medium", "High"]:
            reasons.append("The product appears relatively novel.")
        if den_count > 0:
            reasons.append("FDA precedent search returned De Novo-like records.")
        if predicate_similarity_max < 0.15:
            reasons.append("Predicate similarity is weak, which reduces 510(k) fit.")

    elif top_pathway == "PMA":
        if risk_level == "High":
            reasons.append("High-risk profile was selected.")
        if class_count_map["3"] > 0:
            reasons.append("FDA classification search returned Class III signals.")
        if pma_count_value > 0:
            reasons.append("FDA precedent search returned PMA records.")

    if not reasons:
        reasons.append("Result is being driven by mixed evidence — refine inputs for a clearer signal.")

    return reasons


# ---------------------------------------------------------------------------
# Recommendation summary
# ---------------------------------------------------------------------------

def recommendation_summary(
    top_pathway: str,
    pathway_map: dict,
    class_count_map: dict,
    predicate_similarity_max: float,
    k_count: int,
    den_count: int,
    pma_count_value: int,
    novelty: str,
    risk_level: str
) -> str:
    score = pathway_map.get(top_pathway, 0)
    signals = []

    if predicate_similarity_max >= 0.20:
        signals.append("strong TF-IDF similarity to existing FDA-cleared devices")
    elif predicate_similarity_max >= 0.10:
        signals.append("moderate similarity to existing FDA-cleared devices")
    else:
        signals.append("limited similarity to existing FDA precedents")

    if class_count_map.get("2", 0) > 0:
        signals.append("multiple Class II classification signals")
    if class_count_map.get("3", 0) > 0:
        signals.append("Class III classification signals")
    if k_count > 10:
        signals.append("many 510(k) precedents in FDA records")
    elif k_count > 0:
        signals.append("some 510(k) precedent signals")
    if den_count > 0:
        signals.append("De Novo precedent signals")
    if pma_count_value > 0:
        signals.append("PMA precedent signals")
    if novelty == "High":
        signals.append("high product novelty")
    elif novelty == "Medium":
        signals.append("moderate product novelty")
    if risk_level == "High":
        signals.append("a higher-risk profile")
    elif risk_level == "Moderate":
        signals.append("a moderate-risk profile")

    signal_text = ", ".join(signals)

    if top_pathway == "510(k)":
        return (
            f"510(k) is currently the strongest fit ({score}%) because this product shows "
            f"{signal_text}."
        )
    if top_pathway == "De Novo":
        return (
            f"De Novo appears most appropriate ({score}%) because this product shows "
            f"{signal_text} but does not appear to have a clear predicate route."
        )
    if top_pathway == "PMA":
        return (
            f"PMA is currently the strongest fit ({score}%) because this product shows "
            f"{signal_text}, which is more consistent with a higher-risk submission route."
        )

    return "The recommendation is based on mixed regulatory signals and should be interpreted cautiously."


# ---------------------------------------------------------------------------
# Sensitivity analysis — what-if scoring
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    keyword: str,
    intended_use: str,
    predicate_known: str,
    risk_level: str,
    novelty: str,
    class_count_map: dict,
    k_count: int,
    den_count: int,
    pma_count_value: int,
    predicate_similarity_max: float
) -> list:
    """
    Shows how the top pathway score would change if each key input were altered.
    Returns a list of dicts: {dimension, change_description, new_top_pathway, new_score}
    """
    base = pathway_scores(
        keyword, intended_use, predicate_known, risk_level, novelty,
        class_count_map, k_count, den_count, pma_count_value, predicate_similarity_max
    )
    base_top = max(base, key=base.get)
    base_score = base[base_top]
    rows = []

    scenarios = {
        "If predicate confirmed (Yes)": dict(predicate_known="Yes"),
        "If predicate not found (No)":  dict(predicate_known="No"),
        "If risk raised to High":       dict(risk_level="High"),
        "If risk lowered to Low":       dict(risk_level="Low"),
        "If novelty raised to High":    dict(novelty="High"),
        "If novelty lowered to Low":    dict(novelty="Low"),
    }

    for description, override in scenarios.items():
        kwargs = dict(
            keyword=keyword, intended_use=intended_use,
            predicate_known=predicate_known, risk_level=risk_level,
            novelty=novelty, class_count_map=class_count_map,
            k_count=k_count, den_count=den_count,
            pma_count_value=pma_count_value,
            predicate_similarity_max=predicate_similarity_max
        )
        kwargs.update(override)
        alt = pathway_scores(**kwargs)
        alt_top = max(alt, key=alt.get)
        alt_score = alt[alt_top]
        delta = alt_score - base_score
        rows.append({
            "Scenario":         description,
            "New Top Pathway":  alt_top,
            "New Score (%)":    alt_score,
            "Δ vs Current":     f"{'+' if delta >= 0 else ''}{delta}%",
        })

    return rows


# ---------------------------------------------------------------------------
# Additional regulatory options (strategic overlay)
# ---------------------------------------------------------------------------

def additional_regulatory_options(
    serious_condition: bool,
    unmet_need: bool,
    risk_level: str,
    novelty: str,
    intended_for_hcp: bool,
    patient_specific: bool,
    supports_diagnosis: bool,
    supports_treatment: bool
) -> list:
    options = []

    bt_score = 0
    if serious_condition: bt_score += 45
    if unmet_need:        bt_score += 35
    if novelty == "High":   bt_score += 20
    elif novelty == "Medium": bt_score += 10

    options.append({
        "option": "Breakthrough Device Program",
        "status": "Consider" if bt_score >= 60 else "Low Priority",
        "why": (
            "Serious condition / unmet need signals suggest this may be worth evaluating."
            if bt_score >= 60
            else "Current inputs do not strongly support a breakthrough-style profile."
        )
    })

    options.append({
        "option": "HDE (Humanitarian Device Exemption)",
        "status": "Case-Specific",
        "why": "Relevant only if the target population is a rare disease with very small U.S. patient volume."
    })

    options.append({
        "option": "EUA (Emergency Use Authorization)",
        "status": "Usually Not Applicable",
        "why": "Relevant mainly during declared public health emergencies."
    })

    options.append({
        "option": "IDE (Investigational Device Exemption)",
        "status": "Possible" if risk_level in ["Moderate", "High"] else "Lower Priority",
        "why": (
            "May become relevant if clinical investigation is needed before marketing submission."
            if risk_level in ["Moderate", "High"]
            else "Less likely to be immediately relevant for a lower-risk profile."
        )
    })

    options.append({
        "option": "513(g) Request",
        "status": "Consider if Uncertain",
        "why": "Useful if formal FDA classification clarification is needed."
    })

    return options


# ---------------------------------------------------------------------------
# Broad precedent table
# ---------------------------------------------------------------------------

def build_broad_precedents(
    intended_use: str,
    k_results: list,
    pma_results: list,
    top_n: int = 8
) -> list:
    """
    Combined 510(k) + PMA precedent table ranked by TF-IDF similarity.
    """
    records = []
    texts = []

    for row in k_results:
        k_number = str(row.get("k_number", ""))
        route = "De Novo" if k_number.upper().startswith("DEN") else "510(k)"
        device_name  = row.get("device_name", "")
        product_code = row.get("product_code", "")
        records.append({
            "name":         device_name,
            "company":      row.get("applicant", ""),
            "product_code": product_code,
            "route":        route,
            "decision_date": row.get("decision_date", ""),
        })
        texts.append(f"{device_name} {product_code}")

    for row in pma_results:
        generic_name = row.get("generic_name", "")
        product_code = row.get("product_code", "")
        records.append({
            "name":         generic_name,
            "company":      row.get("applicant", ""),
            "product_code": product_code,
            "route":        "PMA",
            "decision_date": row.get("decision_date", ""),
        })
        texts.append(f"{generic_name} {product_code}")

    if not records:
        return []

    scores = tfidf_similarity(intended_use, texts)
    for rec, score in zip(records, scores):
        rec["similarity_score"] = round(score, 4)

    records.sort(key=lambda x: x["similarity_score"], reverse=True)
    return records[:top_n]


# ---------------------------------------------------------------------------
# Precedent landscape summary
# ---------------------------------------------------------------------------

def precedent_landscape(k_count: int, den_count: int, pma_count_value: int) -> list:
    return [
        {"bucket": "510(k) signals",       "count": k_count},
        {"bucket": "De Novo-like signals",  "count": den_count},
        {"bucket": "PMA signals",           "count": pma_count_value},
    ]
