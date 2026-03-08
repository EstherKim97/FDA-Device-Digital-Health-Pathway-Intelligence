# FDA Device / Digital Health Pathway Intelligence

A decision-support tool that analyzes potential FDA regulatory pathways for medical devices and digital health products using publicly available FDA data.

This project helps innovators, regulatory professionals, and researchers quickly evaluate whether a product is more likely to follow a 510(k), De Novo, or PMA pathway, while surfacing relevant precedents and regulatory signals.

---

## Project Overview

Medical device innovators frequently face uncertainty when determining the most appropriate FDA regulatory pathway.

<img width="1661" height="812" alt="image" src="https://github.com/user-attachments/assets/b1da89c0-2061-4f5e-8e37-03421dd0170b" />
[Watch Demo Video](DemoVideo.mov)

Traditional regulatory assessment requires extensive manual review of:

- FDA classification databases
- 510(k) clearances
- PMA approvals
- De Novo classifications
- Similar predicate devices

This project builds a data-driven regulatory intelligence dashboard that automates part of that early assessment process.

The system integrates FDA data retrieval with structured scoring logic to generate:

- Likely regulatory pathway
- Precedent analysis
- Predicate candidates
- Evidence gaps
- Regulatory next-step recommendations

The tool is designed for early-stage regulatory exploration, not as a replacement for formal regulatory strategy.

---

## System Architecture
```
User Input
   │
   │ (keyword + intended use + regulatory framing)
   ▼
Streamlit Interface (app.py)
   │
   ├── FDA Data Retrieval
   │       └── fda_client.py
   │
   ├── Similarity + Predicate Ranking
   │       └── text similarity functions
   │
   ├── Regulatory Scoring Engine
   │       └── scoring.py
   │
   └── Results Dashboard
           ├── Pathway Scores
           ├── Predicate Ranking
           ├── Broad Precedents
           ├── Evidence Gaps
           ├── FDA Matches
           └── Explainability
```
---

## Key Features

### 1. Regulatory Pathway Prediction

The application evaluates three FDA device pathways:

| Pathway | Description |
|---|---|
| 510(k) | Demonstrates substantial equivalence to an existing FDA-cleared device |
| De Novo | For novel moderate-risk devices without a predicate |
| PMA | Required for high-risk Class III devices requiring clinical evidence |

The system generates pathway scores using product attributes such as:

- Intended use
- Risk level
- Predicate availability
- Novelty
- FDA precedent signals

All scoring weights are centralised in a `WEIGHTS` dictionary in `scoring.py`, making them easy to inspect and tune.

---

### 2. Predicate Device Ranking

The system searches FDA 510(k) data and ranks potential predicate devices using **TF-IDF cosine similarity** between the user-provided intended use and device name and product code descriptions. This gives meaningfully better ranking than simple word overlap, especially for longer or more technical intended use statements.

---

### 3. FDA Precedent Landscape

The dashboard summarises regulatory signals across FDA databases:

- Classification matches
- 510(k) clearances
- De Novo signals
- PMA approvals

This provides users with a quick understanding of regulatory precedent density.

---

### 4. Evidence Gap Identification

The system identifies areas where regulatory evidence is weak or unclear, such as:

- Limited predicate similarity
- Low classification matches
- High novelty
- Unclear regulatory precedent

These signals inform the recommended next regulatory step.

---

### 5. Regulatory Next-Step Recommendations

Based on the scoring model, the app recommends actions such as:

- Pre-Submission (Pre-Sub) meeting with FDA
- Pursuing 510(k) predicate identification
- Considering a De Novo classification
- Preparing for PMA-level evidence

---

### 6. Strategic Options Panel

A dedicated tab surfaces additional FDA programs worth evaluating alongside the primary pathway, including:

- Breakthrough Device Program
- HDE (Humanitarian Device Exemption)
- EUA (Emergency Use Authorization)
- IDE (Investigational Device Exemption)
- 513(g) Classification Request

Each option is scored and colour-coded based on the product profile inputs.

---

### 7. Sensitivity Analysis

A what-if scoring tab shows how the primary pathway recommendation would shift if key inputs were changed, for example if a predicate were confirmed, if risk were raised to high, or if novelty were lowered. This helps users understand which inputs are driving the most uncertainty in the recommendation.

---

## How the System Works

The application combines four components:

### 1. FDA Data Retrieval

Data is pulled from public FDA datasets including:

- Device Classification Database
- 510(k) Clearance Database
- PMA Approval Database

All API calls are cached for one hour using `@st.cache_data`, so repeated runs within a session do not re-hit the FDA API. Errors are surfaced as Streamlit warnings rather than silently swallowed.

---

### 2. Scoring Model

A structured scoring framework evaluates pathway likelihood based on:

- Product novelty
- Risk classification
- Predicate availability
- FDA precedent density
- TF-IDF cosine similarity to previously cleared devices

Scores are generated for each pathway:

- 510(k)
- De Novo
- PMA

The pathway with the highest score is presented as the most likely regulatory route. PMA high-risk boosts require two or more co-occurring risk signals before applying, reducing false positives from single keyword matches.

---

### 3. TF-IDF Similarity Engine

Predicate ranking and broad precedent ranking use TF-IDF vectorisation with cosine similarity via scikit-learn, with bigram support. This replaces simple token overlap (Jaccard) and produces better rankings for longer or more technical intended use descriptions. A Jaccard fallback is used automatically if scikit-learn is unavailable.

---

### 4. Explainability Layer

To improve transparency, the system explains:

- Why the top pathway ranked highest
- What evidence supports the recommendation
- What uncertainties remain

This avoids black-box regulatory predictions.

---

## Example Use Case

**Keyword**

```
retinal imaging AI
```

**Intended Use**

```
Software that analyzes retinal fundus images to detect diabetic retinopathy.
```

The system will:

1. Search FDA device classifications related to retinal imaging
2. Retrieve similar 510(k) devices
3. Evaluate regulatory precedent
4. Score pathway likelihood using TF-IDF similarity
5. Recommend the most plausible regulatory pathway
6. Surface strategic options such as Breakthrough Device eligibility
7. Show a sensitivity table for what-if input changes

---

## Technology Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| Data processing | Python |
| Data analysis | Pandas |
| Similarity ranking | TF-IDF cosine similarity (scikit-learn) |
| Similarity fallback | Jaccard token overlap |
| Data sources | FDA open datasets |

---

## Project Structure

```
project/
│
├── app.py
├── fda_client.py
├── scoring.py
├── utils.py
├── requirements.txt
└── README.md
```

### app.py

Main Streamlit application. Handles UI layout, sidebar inputs, tab rendering, and orchestration of all scoring and retrieval calls.

### fda_client.py

Handles FDA dataset retrieval, query building, caching, and error handling. Exposes direct FDA accessdata.gov link helpers for predicate records.

### scoring.py

Contains all pathway scoring logic, CDS screening, predicate ranking, sensitivity analysis, and recommendation algorithms. Scoring weights are centralised in a `WEIGHTS` dictionary at the top of the file.

### utils.py

Shared text utilities used across all modules. Contains `normalize_text`, `tokenize`, `tfidf_similarity`, `jaccard_similarity`, `contains_any`, and query variant builders. Centralising these avoids duplication across files.

---

## Intended Users

This tool may be useful for:

- Medical device startups
- Digital health companies
- Regulatory affairs professionals
- Healthcare AI researchers
- Product teams exploring regulatory strategy

---

## Limitations

This tool provides exploratory regulatory intelligence, not regulatory advice.

Important limitations include:

- Simplified scoring logic with manually tuned weights
- Limited FDA dataset coverage from open API endpoints
- TF-IDF similarity does not capture full semantic meaning the way large language model embeddings would
- No clinical evidence assessment
- De Novo detection relies on K-number prefix and decision type fields, which may not always be present

Final regulatory strategy decisions should always involve experienced regulatory professionals and FDA interaction.

---

## Future Improvements

- Sentence-transformer or embedding-based semantic similarity for predicate ranking
- Better FDA dataset filtering by product code and specialty
- Confidence intervals for pathway recommendations
- Stronger predicate retrieval using structured product code matching
- Automated regulatory report generation
- Expanded explainability with per-signal score breakdowns

---

## 👩🏻‍💻 Author

Developed as part of a regulatory intelligence exploration project focused on improving early-stage regulatory pathway assessment for digital health technologies.
