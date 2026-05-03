"""
precompute_embeddings.py
------------------------
Run once to encode all candidate resumes into sentence embeddings.
Saves two .npy files used by the app at query time.

Usage:
    python hiring_app/precompute_embeddings.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "enriched_candidates.csv"
EMB_PATH = DATA_DIR / "candidate_embeddings.npy"
IDS_PATH = DATA_DIR / "candidate_ids.npy"

print("Loading enriched candidates...")
df = pd.read_csv(CSV_PATH)
texts = df["cleaned_text"].fillna("").tolist()
ids   = df["id"].values
print(f"  {len(texts):,} resumes loaded")

print("Loading sentence transformer model (downloads on first run ~22MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding resumes — this takes ~30s on CPU...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # unit vectors → dot product == cosine similarity
)

np.save(EMB_PATH, embeddings)
np.save(IDS_PATH, ids)

print(f"\nSaved embeddings : {EMB_PATH}  shape={embeddings.shape}")
print(f"Saved ID index   : {IDS_PATH}   shape={ids.shape}")
print("Done. Restart app.py to load the new embeddings.")
