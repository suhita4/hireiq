# AI-Based Hiring Recommendation System

A web-based, explainable AI prototype for ranking job candidates using hybrid entity-based matching and similarity scoring.

This project was developed as part of a B.Sc. Computer Science capstone project and focuses on improving transparency in AI-assisted hiring systems.

---

## Project Overview

Traditional hiring recommendation systems often prioritize ranking accuracy while providing limited explanation of why a candidate is recommended.

This system addresses that gap by:

• Extracting structured entities from job descriptions and resumes
• Applying weighted, interpretable scoring
• Blending entity-based scoring with similarity metrics
• Presenting transparent score breakdowns to support recruiter decision-making

The result is an explainable hiring recommendation prototype designed for human-centered evaluation.

---

## Key Features

• Web-based interface built using Flask
• spaCy-powered entity extraction
• Rule-based detection of:

* Skills
* Job Titles
* Experience
  • Weighted scoring system:
* Skills (60%)
* Job Title (25%)
* Experience (15%)
  • Hybrid blending of:
* Entity-based score
* Similarity score (from prior TF-IDF model)
  • Top-N configurable ranking
  • Visual score breakdown bars
  • Matched skill transparency display
  • Input validation and error handling
  • Health check API endpoint

---

## System Architecture

The system follows a modular architecture:

**Frontend**

* HTML + CSS + JavaScript
* Dynamic rendering of ranked candidates

**Backend**

* Flask REST API
* Entity extraction pipeline
* Scoring engine
* Ranking module

**Data Layer**

* Cleaned resume dataset (sample included)
* Precomputed similarity scores

---

## Project Structure

```
capstone-project/
│
├── app.py
├── ner_pipeline.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── data/
    └── candidates_sample.csv
```

---

## How to Run Locally

1. Clone the repository:

```
git clone https://github.com/suhita4/capstone-project.git
cd capstone-project
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Download spaCy language model:

```
python -m spacy download en_core_web_sm
```

4. Run the application:

```
python app.py
```

5. Open in browser:

```
http://localhost:5000
```

---

## Dataset Note

For submission purposes, a reduced sample dataset is included to demonstrate system functionality.

The system architecture supports larger datasets but sample data is used to maintain repository efficiency and portability.

---

## Testing

The system was validated through:

• Input validation testing
• Entity extraction verification
• Ranking correctness validation
• Top-N filtering tests
• Health endpoint API verification

---

## Limitations

• Rule-based entity extraction (not transformer-based embeddings)
• No automated fairness metrics
• Designed for local execution
• No authentication layer

---

## Future Improvements

• Transformer-based embedding integration
• Bias detection metrics
• Recruiter feedback learning loop
• Cloud deployment
• Structured resume parsing

---

## Author

Suhita Korgaonkar
B.Sc. Computer Science
Capstone Project – AI-Based Recommendation Systems for Hiring
