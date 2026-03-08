# FDA-Device-Digital-Health-Pathway-Intelligence

A decision-support tool that analyzes potential FDA regulatory pathways for medical devices and digital health products using publicly available FDA data.

This project helps innovators, regulatory professionals, and researchers quickly evaluate whether a product is more likely to follow a 510(k), De Novo, or PMA pathway, while surfacing relevant precedents and regulatory signals.

## Project Overview

Medical device innovators frequently face uncertainty when determining the most appropriate FDA regulatory pathway.

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
## Key Features

### 1. Regulatory Pathway Prediction

The application evaluates three FDA device pathways:

Pathway	Description
510(k)	Demonstrates substantial equivalence to an existing FDA-cleared device
De Novo	For novel moderate-risk devices without a predicate
PMA	Required for high-risk Class III devices requiring clinical evidence

The system generates pathway scores using product attributes such as:
- Intended use
- Risk level
- Predicate availability
- Novelty
- FDA precedent signals

### 2. Predicate Device Ranking

The system searches FDA 510(k) data and ranks potential predicate devices based on text similarity between:
- user-provided intended use
- device name
- product code descriptions

This helps identify potential substantial equivalence candidates.

### 3. FDA Precedent Landscape

The dashboard summarizes regulatory signals across FDA databases:
- classification matches
- 510(k) clearances
- De Novo signals
- PMA approvals

This provides users with a quick understanding of regulatory precedent density.

### 4. Evidence Gap Identification

The system identifies areas where regulatory evidence is weak or unclear, such as:
- limited predicate similarity
- low classification matches
- high novelty
- unclear regulatory precedent

These signals inform the recommended next regulatory step.

### 5. Regulatory Next-Step Recommendations

Based on the scoring model, the app recommends actions such as:
- Pre-Submission (Pre-Sub) meeting with FDA
- pursuing 510(k) predicate identification
- considering a De Novo classification
- preparing for PMA-level evidence


## How the System Works

The application combines three components:

### 1. FDA Data Retrieval

Data is pulled from public FDA datasets including:
- Device Classification Database
- 510(k) Clearance Database
- PMA Approval Database

These datasets provide signals about historical regulatory decisions.

### 2. Scoring Model

A structured scoring framework evaluates pathway likelihood based on:
- product novelty
- risk classification
- predicate availability
- FDA precedent density
- similarity to previously cleared devices

Scores are generated for each pathway:

510(k)
De Novo
PMA

The pathway with the highest score is presented as the most likely regulatory route.

### 3. Explainability Layer

To improve transparency, the system explains:
- why the top pathway ranked highest
- what evidence supports the recommendation
- what uncertainties remain

This avoids “black box” regulatory predictions.

## Example Use Case

Keyword: retinal imaging AI

Intended Use: Software that analyzes retinal fundus images to detect diabetic retinopathy.

The system will:
	1.	search FDA device classifications related to retinal imaging
	2.	retrieve similar 510(k) devices
	3.	evaluate regulatory precedent
	4.	score pathway likelihood
	5.	recommend the most plausible regulatory pathway

## Technology Stack

Component	Technology
Interface	Streamlit
Data processing	Python
Data analysis	Pandas
Text similarity	Custom token similarity
Data sources	FDA open datasets


## Project Structure

```
project/
│
├── app.py
├── fda_client.py
├── scoring.py
├── requirements.txt
└── README.md
```

app.py    Main Streamlit application.

fda_client.py     Handles FDA dataset retrieval and preprocessing.

scoring.py      Contains pathway scoring logic and recommendation algorithms.


## Intended Users

This tool may be useful for:
	•	medical device startups
	•	digital health companies
	•	regulatory affairs professionals
	•	healthcare AI researchers
	•	product teams exploring regulatory strategy


## Limitations

This tool provides exploratory regulatory intelligence, not regulatory advice.

Important limitations include:
- simplified scoring logic
- limited FDA dataset coverage
- text similarity instead of full semantic search
- no clinical evidence assessment

Final regulatory strategy decisions should always involve experienced regulatory professionals and FDA interaction.

## Future Improvements

Planned improvements include:
- improved semantic similarity models
- better FDA dataset filtering
- confidence scoring for pathway recommendations
- stronger precedent retrieval
- expanded explainability
- automated regulatory report generation

## 👩🏻‍💻 Author

Developed as part of a regulatory intelligence exploration project focused on improving early-stage regulatory pathway assessment for digital health technologies.
