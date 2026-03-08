"""
app.py
FDA Device / Digital Health Regulatory Pathway Intelligence — Streamlit UI

Improvements over original:
- Removed duplicate rank_predicates definition (now only in scoring.py)
- Fixed indentation bug: additional_options, broad_precedents, landscape_rows
  are now correctly computed INSIDE the spinner block alongside their dependencies
- Added "Strategic Options" tab surfacing additional_regulatory_options output
- Added "Sensitivity Analysis" tab showing what-if pathway scores
- Added clickable FDA links in predicate and precedent tables
- st.cache_data applied to FDA calls (via fda_client.py)
- normalize_text / tokenize imported from utils.py, not redefined here
"""

import streamlit as st
import pandas as pd

from utils import normalize_text, tfidf_similarity

from fda_client import (
    search_classification,
    search_510k,
    search_pma,
    get_class_counts,
    get_product_codes,
    get_denovo_count,
    safe_columns,
    get_510k_url,
    get_pma_url,
)
from scoring import (
    rank_predicates,
    regulated_device_score,
    cds_screen,
    pathway_scores,
    breakthrough_score,
    evidence_gaps,
    next_step_recommendation,
    reason_list,
    recommendation_summary,
    additional_regulatory_options,
    build_broad_precedents,
    precedent_landscape,
    sensitivity_analysis,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FDA Regulatory Pathway Intelligence",
    page_icon="🧭",
    layout="wide"
)

st.markdown(
    """
    <div style="padding: 0.3rem 0 1rem 0;">
        <h1 style="margin-bottom:0.2rem;">🧭 FDA Device / Digital Health Pathway Intelligence</h1>
        <p style="color: #666; margin-top:0;">
        Analyzes a proposed medical product and suggests the most likely FDA regulatory pathway
        based on device characteristics, precedent signals from FDA databases, and TF-IDF similarity
        to previously cleared or approved technologies.
        </p>
        <p style="color: #666; margin-top:0;">
        <b>How to use:</b> Enter a keyword and intended use, configure the profile in the sidebar,
        then click <b>Run Assessment</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Pathway explainers
# ---------------------------------------------------------------------------

st.markdown("### FDA Regulatory Pathways")
c1, c2, c3 = st.columns(3)

with c1:
    st.info("""
**510(k) Clearance**

Used when a device is **substantially equivalent to an existing legally marketed device (predicate)**.

Typical devices:
• ECG monitors
• imaging software
• wearable sensors

Usually **Class II devices**.
""")

with c2:
    st.info("""
**De Novo Classification**

For **novel moderate-risk devices without a predicate**.

After a De Novo authorization, future devices may use **510(k)** referencing it.

Typical devices:
• new AI diagnostics
• novel digital health tools
""")

with c3:
    st.info("""
**PMA (Premarket Approval)**

Required for **high-risk devices (Class III)**.

Usually requires **clinical trials and extensive safety data**.

Typical devices:
• implantable neurostimulators
• pacemakers
• deep brain stimulation systems
""")

st.info("""
**What is a Pre-Sub?**

A Pre-Sub (Q-Submission) is a way to ask FDA for feedback **before** making the formal submission.
It is commonly used when the regulatory pathway, predicate strategy, or evidence requirements are still uncertain.
""")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Product Intake")

    keyword = st.text_input("FDA search keyword", value="retinal imaging ai")

    intended_use = st.text_area(
        "Intended use",
        value="Software that analyzes retinal images to help clinicians identify patients at risk of diabetic retinopathy."
    )

    st.markdown("### Functional Profile")
    supports_diagnosis   = st.checkbox("Supports diagnosis", value=True)
    supports_treatment   = st.checkbox("Supports treatment decisions", value=False)
    patient_specific     = st.checkbox("Patient-specific output", value=True)
    independent_review   = st.checkbox(
        "Clinician can independently review the basis of the recommendation",
        value=False
    )
    intended_for_hcp = st.checkbox("Intended for healthcare professionals", value=True)

    st.markdown("### Regulatory Framing")
    risk_level      = st.selectbox("Risk level",      ["Low", "Moderate", "High"], index=1)
    predicate_known = st.selectbox("Predicate known?", ["Yes", "Maybe", "No"],      index=1)
    novelty         = st.selectbox("Novelty",          ["Low", "Medium", "High"],   index=1)

    st.markdown("### Strategic Overlay")
    serious_condition = st.checkbox("Serious / life-threatening / irreversibly debilitating condition", value=True)
    unmet_need        = st.checkbox("Potential meaningful improvement / unmet need", value=True)

    run_button = st.button("Run Assessment", use_container_width=True)

# ---------------------------------------------------------------------------
# Helper: add clickable FDA link column to a DataFrame
# ---------------------------------------------------------------------------

def add_510k_links(df: pd.DataFrame) -> pd.DataFrame:
    if "k_number" in df.columns:
        df = df.copy()
        df["FDA Link"] = df["k_number"].apply(
            lambda k: f'<a href="{get_510k_url(k)}" target="_blank">View ↗</a>' if k else ""
        )
    return df


def add_pma_links(df: pd.DataFrame) -> pd.DataFrame:
    if "pma_number" in df.columns:
        df = df.copy()
        df["FDA Link"] = df["pma_number"].apply(
            lambda p: f'<a href="{get_pma_url(p)}" target="_blank">View ↗</a>' if p else ""
        )
    return df


def render_html_table(df: pd.DataFrame):
    """Render a DataFrame as HTML so hyperlinks are clickable."""
    st.markdown(
        df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Main assessment
# ---------------------------------------------------------------------------

if not run_button:
    st.info("Complete the sidebar inputs and click **Run Assessment**.")
else:
    with st.spinner("Searching FDA datasets..."):

        # --- FDA data retrieval (all cached) ---
        class_results       = search_classification(keyword, intended_use, limit=15)
        class_count_map     = get_class_counts(class_results)
        top_codes           = get_product_codes(class_results)
        top_code            = top_codes[0] if top_codes else ""

        k_results           = search_510k(keyword, intended_use, top_code, limit=25)
        pma_results         = search_pma(keyword, intended_use, top_code, limit=20)

        # --- Counts ---
        classification_match_count = len(class_results)
        k_count                    = len(k_results)
        den_count                  = get_denovo_count(k_results)
        pma_count_value            = len(pma_results)

        # --- Scoring ---
        ranked_predicates       = rank_predicates(intended_use, k_results, top_n=5)
        predicate_similarity_max = ranked_predicates[0]["similarity_score"] if ranked_predicates else 0.0

        regulated_score_value = regulated_device_score(
            supports_diagnosis, supports_treatment, patient_specific,
            independent_review, risk_level, classification_match_count
        )

        cds_result = cds_screen(
            intended_for_hcp, independent_review,
            supports_diagnosis, supports_treatment, patient_specific
        )

        pathway_map = pathway_scores(
            keyword, intended_use, predicate_known, risk_level, novelty,
            class_count_map, k_count, den_count, pma_count_value, predicate_similarity_max
        )

        breakthrough_value = breakthrough_score(serious_condition, unmet_need, novelty)

        gaps = evidence_gaps(
            predicate_known, risk_level, novelty,
            classification_match_count, k_count, den_count,
            pma_count_value, predicate_similarity_max
        )

        next_step   = next_step_recommendation(pathway_map, cds_result, gaps)
        top_pathway = max(pathway_map, key=pathway_map.get)

        sorted_scores    = sorted(pathway_map.values(), reverse=True)
        confidence_margin = sorted_scores[0] - sorted_scores[1]

        if confidence_margin >= 30:
            confidence_label = "High confidence"
        elif confidence_margin >= 15:
            confidence_label = "Moderate confidence"
        else:
            confidence_label = "Low confidence"

        top_reasons = reason_list(
            top_pathway, predicate_known, novelty, risk_level,
            class_count_map, k_count, den_count, pma_count_value, predicate_similarity_max
        )

        summary_text = recommendation_summary(
            top_pathway, pathway_map, class_count_map,
            predicate_similarity_max, k_count, den_count,
            pma_count_value, novelty, risk_level
        )

        # These were incorrectly OUTSIDE the spinner block in the original
        additional_options = additional_regulatory_options(
            serious_condition, unmet_need, risk_level, novelty,
            intended_for_hcp, patient_specific, supports_diagnosis, supports_treatment
        )

        broad_precedents = build_broad_precedents(
            intended_use, k_results, pma_results, top_n=8
        )

        landscape_rows = precedent_landscape(k_count, den_count, pma_count_value)

        sensitivity_rows = sensitivity_analysis(
            keyword, intended_use, predicate_known, risk_level, novelty,
            class_count_map, k_count, den_count, pma_count_value, predicate_similarity_max
        )

    # -----------------------------------------------------------------------
    # Top metrics
    # -----------------------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Likely Regulated Device", f"{regulated_score_value}%")
    c2.metric("CDS Screen", f'{cds_result["score"]}%')
    c3.metric(
        "Primary Pathway",
        f"{top_pathway} ({pathway_map[top_pathway]}%)",
        confidence_label
    )
    c4.metric("Breakthrough Overlay", f"{breakthrough_value}%")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------

    (
        tab_summary, tab_scores, tab_predicates, tab_precedents,
        tab_gaps, tab_fda, tab_strategic, tab_sensitivity, tab_explain
    ) = st.tabs([
        "Executive Summary",
        "Pathway Scores",
        "Predicate Ranking",
        "Broad Precedents",
        "Evidence Gaps",
        "FDA Matches",
        "Strategic Options",     # NEW — was computed but never displayed
        "Sensitivity Analysis",  # NEW — what-if scoring
        "Explainability",
    ])

    # --- Tab 1: Executive Summary ---
    with tab_summary:
        left, right = st.columns([1.1, 0.9])

        with left:
            st.subheader("Assessment Summary")
            summary_df = pd.DataFrame(
                [
                    ["Keyword",               keyword],
                    ["Risk level",            risk_level],
                    ["Predicate known",       predicate_known],
                    ["Novelty",               novelty],
                    ["Top product codes",     ", ".join(top_codes) if top_codes else "None found"],
                    ["Class distribution",    str(class_count_map)],
                    ["Best predicate similarity", predicate_similarity_max],
                ],
                columns=["Field", "Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.subheader("Recommended Next Step")
            st.info(next_step)
            st.subheader("Why This Recommendation")
            st.write(summary_text)

        with right:
            st.subheader("CDS Screen Result")
            st.write(f'**Label:** {cds_result["label"]}')
            st.progress(cds_result["score"] / 100)

            st.subheader("Pathway Ranking")
            ranking_df = pd.DataFrame(
                list(pathway_map.items()), columns=["Pathway", "Score"]
            ).sort_values("Score", ascending=False)
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    # --- Tab 2: Pathway Scores ---
    with tab_scores:
        st.subheader("Core Pathway Scores")

        PATHWAY_DESC = {
            "510(k)":  "Substantial equivalence to an existing FDA-cleared device.",
            "De Novo": "Novel moderate-risk device with no predicate.",
            "PMA":     "High-risk device requiring clinical evidence."
        }

        score_df = pd.DataFrame(
            [(p, s, PATHWAY_DESC.get(p, "")) for p, s in pathway_map.items()],
            columns=["Pathway", "Score", "Description"]
        ).sort_values("Score", ascending=False)

        st.bar_chart(score_df[["Pathway", "Score"]].set_index("Pathway"))
        st.dataframe(score_df, use_container_width=True, hide_index=True)

        st.subheader("FDA Precedent Landscape")
        landscape_df = pd.DataFrame(landscape_rows)
        st.dataframe(landscape_df, use_container_width=True, hide_index=True)

        st.subheader("Breakthrough Device Overlay")
        st.progress(breakthrough_value / 100)
        st.write(f"{breakthrough_value}%")

    # --- Tab 3: Predicate Ranking ---
    with tab_predicates:
        st.subheader("Top Predicate Candidates")
        st.caption("Ranked by TF-IDF cosine similarity to your intended use. Click links to open FDA records.")
        if ranked_predicates:
            pred_df = pd.DataFrame(ranked_predicates)
            pred_df = add_510k_links(pred_df)
            render_html_table(pred_df)
        else:
            st.write("No predicate candidates ranked from current 510(k) results.")

    # --- Tab 4: Broad Precedents ---
    with tab_precedents:
        st.subheader("Broad Similar Product Precedents")
        st.write(
            "Broader FDA precedent signals across 510(k), De Novo-like, and PMA records, "
            "ranked by TF-IDF cosine similarity to your intended use."
        )
        if broad_precedents:
            broad_df = pd.DataFrame(broad_precedents)
            st.dataframe(broad_df, use_container_width=True, hide_index=True)
        else:
            st.write("No broad precedent signals found from the current FDA search.")

    # --- Tab 5: Evidence Gaps ---
    with tab_gaps:
        st.subheader("Evidence Gaps")
        for item in gaps:
            st.write(f"- {item}")

    # --- Tab 6: FDA Matches ---
    with tab_fda:
        st.subheader("Matched FDA Records")

        st.markdown("#### Classification Matches")
        if class_results:
            class_df = pd.DataFrame(class_results)
            class_df = safe_columns(
                class_df,
                ["device_name", "product_code", "device_class", "medical_specialty_description"]
            )
            st.dataframe(class_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No classification matches found. Try a broader product term such as "
                "'ECG monitor', 'electrocardiograph', or a device category."
            )

        st.markdown("#### 510(k) / De Novo-like Matches")
        if k_results:
            k_df = pd.DataFrame(k_results)
            k_df = safe_columns(
                k_df,
                ["k_number", "device_name", "applicant", "decision_date", "product_code"]
            )
            k_df = add_510k_links(k_df)
            render_html_table(k_df)
        else:
            st.info(
                "No 510(k) / De Novo-like matches found. "
                "This often means the search phrase is too narrow."
            )

        st.markdown("#### PMA Matches")
        if pma_results:
            pma_df = pd.DataFrame(pma_results)
            pma_df = safe_columns(
                pma_df,
                ["pma_number", "generic_name", "applicant", "decision_date", "product_code"]
            )
            pma_df = add_pma_links(pma_df)
            render_html_table(pma_df)
        else:
            st.info("No PMA matches found. That is common for lower- or moderate-risk products.")

    # --- Tab 7: Strategic Options (NEW) ---
    with tab_strategic:
        st.subheader("Strategic Regulatory Options")
        st.caption(
            "These are additional FDA programs and pathways worth evaluating alongside the primary pathway. "
            "This is not legal advice."
        )

        if additional_options:
            opt_df = pd.DataFrame(additional_options)
            # Colour-code status column
            def _status_colour(val):
                colours = {
                    "Consider":             "background-color: #d4edda; color: #155724",
                    "Low Priority":         "background-color: #f8d7da; color: #721c24",
                    "Possible":             "background-color: #fff3cd; color: #856404",
                    "Case-Specific":        "background-color: #cce5ff; color: #004085",
                    "Usually Not Applicable": "background-color: #e2e3e5; color: #383d41",
                    "Consider if Uncertain": "background-color: #cce5ff; color: #004085",
                    "Lower Priority":       "background-color: #f8d7da; color: #721c24",
                }
                return colours.get(val, "")

            styled = opt_df.style.applymap(_status_colour, subset=["status"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.write("No additional regulatory options generated.")

    # --- Tab 8: Sensitivity Analysis (NEW) ---
    with tab_sensitivity:
        st.subheader("What-If Pathway Sensitivity")
        st.caption(
            "Shows how the primary pathway recommendation would shift if key inputs changed. "
            "Helps identify which inputs drive the most uncertainty."
        )

        if sensitivity_rows:
            sens_df = pd.DataFrame(sensitivity_rows)
            st.dataframe(sens_df, use_container_width=True, hide_index=True)
        else:
            st.write("No sensitivity scenarios generated.")

    # --- Tab 9: Explainability ---
    with tab_explain:
        left, right = st.columns(2)

        with left:
            st.subheader("Why the Top Pathway Ranked Highest")
            if top_reasons:
                for item in top_reasons:
                    st.write(f"- {item}")
            else:
                st.info("No explainability reasons were generated for this run.")

            st.subheader("CDS Screen Notes")
            if cds_result.get("notes"):
                for item in cds_result["notes"]:
                    st.write(f"- {item}")
            else:
                st.info("No CDS notes were generated for this run.")

        with right:
            st.subheader("Raw Evidence Counts")
            raw_df = pd.DataFrame(
                [
                    ["Classification matches", classification_match_count],
                    ["510(k) matches",          k_count],
                    ["De Novo-like matches",    den_count],
                    ["PMA matches",             pma_count_value],
                ],
                columns=["Metric", "Count"]
            )
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

            if classification_match_count == 0 and k_count == 0 and pma_count_value == 0:
                st.warning(
                    "The current recommendation is being driven mostly by your structured inputs "
                    "rather than FDA retrieval evidence. Broaden the search keyword to improve "
                    "precedent coverage."
                )

            st.subheader("Intended Use")
            st.write(intended_use)
