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
from ner_pipeline import recommend_candidates, extract_entities, batch_extract_entities, compute_skill_gap

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

    # Optional weight overrides from client
    ner_weight = float(data.get("ner_weight", 0.5))
    sem_weight = float(data.get("sem_weight", 0.5))
    skill_w    = float(data.get("skill_w", 0.6))
    title_w    = float(data.get("title_w", 0.25))
    exp_w      = float(data.get("exp_w", 0.15))

    results = recommend_candidates(
        job_description,
        candidates_df,
        top_n=top_n,
        model=st_model,
        embeddings=embeddings,
        embedding_ids=embedding_ids,
        precomputed_entities=candidate_entities_cache,
        job_entities=job_entities,
        ner_weight=ner_weight,
        sem_weight=sem_weight,
        skill_w=skill_w,
        title_w=title_w,
        exp_w=exp_w,
    )

    return jsonify({
        "job_entities": job_entities,
        "candidates": results,
        "total_candidates_searched": len(candidates_df),
        "semantic_scoring_enabled": st_model is not None,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Body (JSON): { "job_description": "..." }
    Returns skill gap data across the full candidate pool.
    """
    data = request.get_json()
    if not data or not data.get("job_description"):
        return jsonify({"error": "job_description is required"}), 400

    job_description = data["job_description"].strip()
    if len(job_description) < 20:
        return jsonify({"error": "Please provide a more detailed job description."}), 400

    job_entities = extract_entities(job_description)
    skill_gap = compute_skill_gap(job_entities, candidate_entities_cache)

    avg_coverage = (
        round(sum(v["pct_have"] for v in skill_gap.values()) / len(skill_gap), 1)
        if skill_gap else 0
    )

    return jsonify({
        "skill_gap": skill_gap,
        "avg_coverage": avg_coverage,
        "total_candidates": len(candidates_df),
    })


FUNNEL_PATH = DATA_DIR / "funnel.json"
FUNNEL_STAGES = ["Screened", "Interview Scheduled", "Offered", "Hired", "Rejected"]


def _load_funnel() -> dict:
    if FUNNEL_PATH.exists():
        with open(FUNNEL_PATH) as f:
            return json.load(f)
    return {}


def _save_funnel(data: dict) -> None:
    with open(FUNNEL_PATH, "w") as f:
        json.dump(data, f)


@app.route("/funnel", methods=["GET"])
def funnel_get():
    """GET /funnel — returns { candidate_id: stage } mapping."""
    return jsonify(_load_funnel())


@app.route("/funnel/update", methods=["POST"])
def funnel_update():
    """POST /funnel/update — body: { "id": 42, "stage": "Offered" }"""
    data = request.get_json()
    if not data or "id" not in data or "stage" not in data:
        return jsonify({"error": "id and stage are required"}), 400

    stage = data["stage"]
    if stage not in FUNNEL_STAGES and stage != "":
        return jsonify({"error": f"Invalid stage. Must be one of: {FUNNEL_STAGES}"}), 400

    funnel = _load_funnel()
    cid = str(data["id"])
    if stage == "":
        funnel.pop(cid, None)
    else:
        funnel[cid] = stage
    _save_funnel(funnel)
    return jsonify({"ok": True})


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