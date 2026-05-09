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
import json
import uuid
from datetime import datetime, timezone
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
# Synthetic display names: stable adjective + animal per candidate id.
# The dataset is anonymized (no real names), so we derive a memorable handle
# from the id so the same candidate always shows up as e.g. "Swift Otter".
# ─────────────────────────────────────────────
_ADJECTIVES = [
    "Swift", "Brave", "Quiet", "Clever", "Bright", "Fierce", "Gentle", "Bold",
    "Curious", "Eager", "Humble", "Lucky", "Mellow", "Nimble", "Patient",
    "Sharp", "Steady", "Sunny", "Witty", "Zealous", "Calm", "Daring",
    "Earnest", "Fearless", "Graceful", "Honest", "Jolly", "Keen", "Lively",
    "Merry", "Noble", "Plucky", "Quick", "Radiant", "Sly", "Tidy", "Upbeat",
    "Vivid", "Warm", "Zesty", "Cosmic", "Crimson", "Golden", "Silver",
    "Velvet", "Wild", "Stoic", "Tame", "Rare", "Loyal",
]
_ANIMALS = [
    "Otter", "Falcon", "Tiger", "Owl", "Fox", "Wolf", "Heron", "Lynx",
    "Panda", "Koala", "Hawk", "Bear", "Stag", "Mole", "Crane", "Robin",
    "Badger", "Beaver", "Dolphin", "Eagle", "Gecko", "Ibis", "Jaguar",
    "Kestrel", "Lemur", "Marten", "Newt", "Osprey", "Puffin", "Quokka",
    "Raven", "Seal", "Toad", "Urial", "Vole", "Walrus", "Yak", "Zebra",
    "Cobra", "Falcon", "Mantis", "Orca", "Panther", "Rhino", "Sparrow",
    "Tapir", "Viper", "Whale", "Bison", "Cougar",
]


def _synthetic_name(cid) -> str:
    n = int(cid)
    return f"{_ADJECTIVES[n % len(_ADJECTIVES)]} {_ANIMALS[(n // len(_ADJECTIVES)) % len(_ANIMALS)]}"


if "id" in candidates_df.columns and len(candidates_df) > 0:
    candidates_df["name"] = candidates_df["id"].map(_synthetic_name)

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

def _static_mtime(filename: str) -> int:
    """Mtime-based cache buster for static files. Bumps automatically on edit."""
    p = Path(__file__).parent / "static" / filename
    return int(p.stat().st_mtime) if p.exists() else 0


@app.route("/")
def index():
    """Serve the main UI."""
    return render_template(
        "index.html",
        css_v=_static_mtime("style.css"),
        js_v=_static_mtime("app.js"),
    )


def _resolve_jd(data: dict) -> tuple[str, dict | None]:
    """Resolve a job description from request body.

    Accepts either job_description directly, or role_id pointing at a saved role.
    Returns (jd_text, error_response). error_response is a (jsonify, status) tuple
    if validation failed, else None.
    """
    role_id = data.get("role_id")
    job_description = (data.get("job_description") or "").strip()

    if role_id and not job_description:
        role = _load_roles().get(role_id)
        if not role:
            return "", (jsonify({"error": "role not found"}), 404)
        job_description = (role.get("jd_text") or "").strip()

    if not job_description:
        return "", (jsonify({"error": "job_description is required"}), 400)
    if len(job_description) < 20:
        return "", (jsonify({"error": "Please provide a more detailed job description."}), 400)
    return job_description, None


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend
    Body (JSON): { "job_description": "...", "top_n": 20, "role_id": "..." }
    Either job_description or role_id is required. Returns ranked candidates with
    a short resume excerpt for the detail pane.
    """
    data = request.get_json() or {}
    job_description, err = _resolve_jd(data)
    if err is not None:
        return err

    top_n = int(data.get("top_n", 20))
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

    # Merge a short resume excerpt for the detail pane (no second round-trip needed)
    text_by_id = dict(zip(
        candidates_df["id"].astype(int),
        candidates_df["cleaned_text"].astype(str),
    ))
    for r in results:
        full = text_by_id.get(int(r["id"]), "")
        r["resume_excerpt"] = (full[:600] + "…") if len(full) > 600 else full

    return jsonify({
        "job_entities": job_entities,
        "candidates": results,
        "total_candidates_searched": len(candidates_df),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Body (JSON): { "job_description": "..." } or { "role_id": "..." }
    Returns skill gap data across the full candidate pool.
    """
    data = request.get_json() or {}
    job_description, err = _resolve_jd(data)
    if err is not None:
        return err

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


ROLES_PATH    = DATA_DIR / "roles.json"
FUNNEL_PATH   = DATA_DIR / "funnel.json"
FUNNEL_STAGES = ["Screened", "Interview Scheduled", "Offered", "Hired", "Rejected"]
LEGACY_ROLE_ID = "_legacy"


# ─── Roles store ────────────────────────────────────────────────────
def _load_roles() -> dict:
    if ROLES_PATH.exists():
        with open(ROLES_PATH) as f:
            return json.load(f)
    return {}


def _save_roles(data: dict) -> None:
    with open(ROLES_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ─── Funnel store (per-role scoped) ─────────────────────────────────
def _load_funnel() -> dict:
    """Return {role_id: {candidate_id: stage}}.

    One-time migration: if funnel.json is the legacy flat shape
    {candidate_id: stage}, move it under LEGACY_ROLE_ID and rewrite the file.
    """
    if not FUNNEL_PATH.exists():
        return {}
    with open(FUNNEL_PATH) as f:
        data = json.load(f)
    if data and any(isinstance(v, str) for v in data.values()):
        data = {LEGACY_ROLE_ID: data}
        with open(FUNNEL_PATH, "w") as f:
            json.dump(data, f, indent=2)
    return data


def _save_funnel(data: dict) -> None:
    with open(FUNNEL_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ─── Roles routes ───────────────────────────────────────────────────
@app.route("/roles", methods=["GET"])
def roles_list():
    return jsonify(_load_roles())


@app.route("/roles", methods=["POST"])
def roles_create():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip() or "Untitled Role"
    jd_text = (data.get("jd_text") or "").strip()
    role_id = uuid.uuid4().hex
    roles = _load_roles()
    roles[role_id] = {
        "name": name,
        "jd_text": jd_text,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_roles(roles)
    return jsonify({"id": role_id, **roles[role_id]})


@app.route("/roles/<role_id>", methods=["PATCH"])
def roles_update(role_id):
    data = request.get_json() or {}
    roles = _load_roles()
    if role_id not in roles:
        return jsonify({"error": "role not found"}), 404
    if "name" in data:
        new_name = (data["name"] or "").strip()
        if new_name:
            roles[role_id]["name"] = new_name
    if "jd_text" in data:
        roles[role_id]["jd_text"] = data["jd_text"]
    _save_roles(roles)
    return jsonify({"id": role_id, **roles[role_id]})


@app.route("/roles/<role_id>", methods=["DELETE"])
def roles_delete(role_id):
    roles = _load_roles()
    roles.pop(role_id, None)
    _save_roles(roles)
    funnel = _load_funnel()
    funnel.pop(role_id, None)
    _save_funnel(funnel)
    return jsonify({"ok": True})


# ─── Funnel routes (role-scoped) ────────────────────────────────────
@app.route("/funnel", methods=["GET"])
def funnel_get():
    """GET /funnel?role_id=<id> → { candidate_id: stage } for that role.
    GET /funnel (no arg) → full {role_id: {…}} dict (admin/debug)."""
    role_id = request.args.get("role_id")
    funnel = _load_funnel()
    if role_id:
        return jsonify(funnel.get(role_id, {}))
    return jsonify(funnel)


@app.route("/funnel/update", methods=["POST"])
def funnel_update():
    """POST /funnel/update — body: { role_id, id, stage }."""
    data = request.get_json() or {}
    if "role_id" not in data or "id" not in data or "stage" not in data:
        return jsonify({"error": "role_id, id, and stage are required"}), 400

    stage = data["stage"]
    if stage not in FUNNEL_STAGES and stage != "":
        return jsonify({"error": f"Invalid stage. Must be one of: {FUNNEL_STAGES}"}), 400

    role_id = data["role_id"]
    cid = str(data["id"])
    funnel = _load_funnel()
    role_funnel = funnel.setdefault(role_id, {})
    if stage == "":
        role_funnel.pop(cid, None)
        if not role_funnel:
            funnel.pop(role_id, None)
    else:
        role_funnel[cid] = stage
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