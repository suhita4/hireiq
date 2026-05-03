"""
app.py
------
Flask web application for the AI-Based Hiring Recommendation System.
Integrates spaCy NER + sentence-transformer semantic scoring.

Run:
    pip install flask spacy pandas sentence-transformers
    python -m spacy download en_core_web_sm
    python hiring_app/precompute_embeddings.py   # once, before first run
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from ner_pipeline import recommend_candidates, extract_entities, batch_extract_entities

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"

# ─────────────────────────────────────────────
# Load enriched candidate dataset
# ─────────────────────────────────────────────
CANDIDATES_PATH = DATA_DIR / "enriched_candidates.csv"

try:
    candidates_df = pd.read_csv(CANDIDATES_PATH)
    print(f"Loaded {len(candidates_df):,} candidates from enriched dataset")
except FileNotFoundError:
    print("enriched_candidates.csv not found — run precompute_embeddings.py first")
    candidates_df = pd.DataFrame(columns=["id", "cleaned_text", "category"])

# ─────────────────────────────────────────────
# Load sentence-transformer model + embeddings
# ─────────────────────────────────────────────
EMB_PATH = DATA_DIR / "candidate_embeddings.npy"
IDS_PATH = DATA_DIR / "candidate_ids.npy"

st_model      = None
embeddings    = None
embedding_ids = None

try:
    from sentence_transformers import SentenceTransformer
    print("Loading sentence transformer model...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings    = np.load(EMB_PATH)
    embedding_ids = np.load(IDS_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")
except FileNotFoundError:
    print("Embeddings not found — run precompute_embeddings.py to enable semantic scoring")
except Exception as e:
    print(f"Semantic scoring unavailable: {e}")

# ─────────────────────────────────────────────
# Load or build candidate NER entity cache
# Saved to disk so restarts skip the ~30s recomputation
# ─────────────────────────────────────────────
ENTITY_CACHE_PATH = DATA_DIR / "candidate_entities_cache.pkl"

if ENTITY_CACHE_PATH.exists():
    print("Loading entity cache from disk...")
    with open(ENTITY_CACHE_PATH, "rb") as _f:
        candidate_entities_cache = pickle.load(_f)
    print(f"Entity cache loaded: {len(candidate_entities_cache)} candidates")
else:
    print("Building entity cache (one-time cost — will be saved to disk)...")
    texts = [str(row["cleaned_text"]) for _, row in candidates_df.iterrows()]
    candidate_entities_cache = batch_extract_entities(texts)
    with open(ENTITY_CACHE_PATH, "wb") as _f:
        pickle.dump(candidate_entities_cache, _f)
    print(f"Entity cache built and saved: {len(candidate_entities_cache)} candidates")


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

    # Extract entities once — reused for both display and scoring
    job_entities = extract_entities(job_description)

    results = recommend_candidates(
        job_description,
        candidates_df,
        top_n=top_n,
        model=st_model,
        embeddings=embeddings,
        embedding_ids=embedding_ids,
        precomputed_entities=candidate_entities_cache,
        job_entities=job_entities,
    )

    return jsonify({
        "job_entities": job_entities,
        "candidates": results,
        "total_candidates_searched": len(candidates_df),  # full 2,814
        "semantic_scoring_enabled": st_model is not None,
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "candidates_loaded": len(candidates_df),
        "semantic_scoring": st_model is not None,
        "embeddings_shape": list(embeddings.shape) if embeddings is not None else None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)