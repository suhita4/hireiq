"""
app.py
------
Flask web application for the AI-Based Hiring Recommendation System.
Integrates the spaCy NER pipeline to match candidates to job descriptions.

Run:
    pip install flask spacy pandas
    python -m spacy download en_core_web_sm
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from ner_pipeline import recommend_candidates, extract_entities

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load your real cleaned datasets
# ─────────────────────────────────────────────
CANDIDATES_PATH = os.path.join(os.path.dirname(__file__), "data", "candidates_cleaned.csv")
RANKED_PATH     = os.path.join(os.path.dirname(__file__), "data", "ranked_cleaned.csv")

try:
    candidates_df = pd.read_csv(CANDIDATES_PATH)
    print(f"✅ Loaded {len(candidates_df)} candidates")
except FileNotFoundError:
    print(f"❌ candidates_cleaned.csv not found in data/")
    candidates_df = pd.DataFrame(columns=["ID", "clean_resume", "Category", "similarity_score"])

try:
    ranked_df = pd.read_csv(RANKED_PATH)
    print(f"✅ Loaded {len(ranked_df)} pre-ranked results")
except FileNotFoundError:
    ranked_df = pd.DataFrame()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend
    Body (JSON): { "job_description": "...", "top_n": 5 }
    Returns ranked candidates as JSON.
    """
    data = request.get_json()

    if not data or not data.get("job_description"):
        return jsonify({"error": "job_description is required"}), 400

    job_description = data["job_description"].strip()
    top_n = int(data.get("top_n", 5))

    if len(job_description) < 20:
        return jsonify({"error": "Please provide a more detailed job description."}), 400

    # Extract entities from job description (for display)
    job_entities = extract_entities(job_description)

    # Get ranked candidates using your real dataset
    results = recommend_candidates(job_description, candidates_df.head(100), top_n=top_n)

    return jsonify({
        "job_entities": job_entities,
        "candidates": results,
        "total_candidates_searched": len(candidates_df),
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "candidates_loaded": len(candidates_df)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)