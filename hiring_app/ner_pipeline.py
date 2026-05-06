"""
ner_pipeline.py
---------------
spaCy-Based NER pipeline for extracting entities from resumes and job descriptions,
then matching candidates to job postings using entity overlap scoring.

Entities extracted:
    SKILL       - Technical or soft skills (e.g., Python, TensorFlow, communication)
    JOB_TITLE   - Job roles (e.g., Data Scientist, Software Engineer)
    EXPERIENCE  - Years or level of experience (e.g., 5 years, senior level)

To swap your own dataset: replace data/candidates.csv with your file,
keeping columns: id, name, resume_text

To add new entity labels: add them to ENTITY_PATTERNS below.
"""

import spacy
from spacy.matcher import Matcher
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. Load spaCy model (NER-only — parser/tagger/lemmatizer disabled for speed)
# ─────────────────────────────────────────────
_DISABLE_PIPES = ["tagger", "parser", "senter", "attribute_ruler", "lemmatizer"]

try:
    nlp = spacy.load("en_core_web_sm", disable=_DISABLE_PIPES)
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# ─────────────────────────────────────────────
# 2. Rule-based entity patterns
#    TO CUSTOMISE: add/remove patterns here
# ─────────────────────────────────────────────
SKILL_KEYWORDS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "php", "swift",
    "kotlin", "ruby", "scala", "matlab", "perl", "bash", "shell", "vba",
    # ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "tensorflow", "pytorch", "keras",
    "scikit-learn", "xgboost", "lightgbm", "spacy", "nltk", "bert", "gpt", "llm",
    "random forest", "regression", "classification", "clustering",
    "neural network", "hugging face", "opencv",
    # Data science libraries
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis",
    "cassandra", "elasticsearch", "snowflake", "bigquery", "nosql",
    # Big data
    "spark", "hadoop", "hive", "kafka", "airflow", "databricks",
    # Visualisation / BI
    "tableau", "power bi", "excel", "looker",
    # Web / Cloud
    "html", "css", "react", "angular", "vue", "bootstrap", "node.js",
    "flask", "django", "fastapi", "spring boot", ".net",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "ci/cd", "git", "rest api", "graphql", "microservices",
    "xml", "json", "sharepoint",
    # Testing / QA
    "selenium", "appium", "junit", "pytest", "testng", "qa",
    "quality assurance", "automation testing", "manual testing",
    "jira", "postman",
    # Infra / Security / Networking
    "linux", "unix", "windows server", "networking", "firewall", "dns",
    # ERP / Accounting tools
    "sap", "erp", "tally", "quickbooks",
    # Finance domain
    "accounting", "financial reporting", "auditing", "budgeting",
    "forecasting", "financial modeling", "investment", "taxation",
    # HR / Business
    "recruitment", "talent acquisition", "payroll", "onboarding",
    "performance management", "hris", "crm", "salesforce",
    # Design
    "figma", "adobe xd", "photoshop", "illustrator", "indesign",
    # Soft / cross-cutting
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "agile", "scrum", "data analysis",
    "business analysis", "microsoft office", "presentation",
    # Blockchain
    "blockchain", "ethereum",
]

JOB_TITLE_KEYWORDS = [
    # From dataset frequency analysis
    "teacher", "professor", "educator",
    "chef",
    "public relations",
    "accountant",
    "project manager", "project coordinator",
    "advocate", "lawyer", "legal advisor",
    "sales manager",
    "java developer",
    "operations manager",
    "mechanical engineer", "civil engineer", "electrical engineer",
    "business analyst",
    "software engineer", "software developer",
    "recruiter",
    "graphic designer", "web designer",
    "construction manager",
    "etl developer",
    "marketing manager",
    "devops engineer",
    "financial analyst",
    "database administrator",
    "python developer",
    "hr manager", "hr specialist",
    "network engineer", "network security engineer",
    "hadoop developer",
    "product manager",
    "test engineer", "qa engineer", "automation tester",
    "blockchain developer",
    "fitness trainer",
    "data analyst", "data scientist", "data engineer",
    "web developer", "full stack developer", "backend developer", "frontend developer",
    "scrum master",
    "product designer",
    "copywriter", "content writer",
    "sap consultant", "sap developer",
    "solutions architect", "cloud architect",
    "research scientist",
    "machine learning engineer",
    "ui/ux designer",
    ".net developer",
    "mobile developer", "android developer", "ios developer",
    "site reliability engineer",
]

EXPERIENCE_PATTERNS = [
    r"\d+\+?\s+years?\s+of\s+experience",
    r"\d+\+?\s+years?\s+experience",
    r"senior|junior|mid-level|entry.level|principal|lead",
    r"phd|master'?s|bachelor'?s|mba",
]

# ─────────────────────────────────────────────
# 3. Entity extraction functions
# ─────────────────────────────────────────────
def _keyword_entities(text_lower: str) -> defaultdict:
    """Pure-Python keyword pass shared by single and batch extraction."""
    entities = defaultdict(set)
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            entities["SKILL"].add(skill)
    for title in JOB_TITLE_KEYWORDS:
        if title in text_lower:
            entities["JOB_TITLE"].add(title)
    for pattern in EXPERIENCE_PATTERNS:
        for match in re.findall(pattern, text_lower):
            entities["EXPERIENCE"].add(match.strip())
    return entities


def extract_entities(text: str) -> dict:
    """Extract SKILL, JOB_TITLE, and EXPERIENCE entities from text."""
    entities = _keyword_entities(text.lower())
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE" and "year" in ent.text.lower():
            entities["EXPERIENCE"].add(ent.text.lower().strip())
    return {k: list(v) for k, v in entities.items()}


def batch_extract_entities(texts: list) -> list:
    """Batch extraction using nlp.pipe() — ~5x faster than calling extract_entities() in a loop."""
    keyword_results = [_keyword_entities(t.lower()) for t in texts]
    for doc, entities in zip(nlp.pipe(texts, batch_size=64), keyword_results):
        for ent in doc.ents:
            if ent.label_ == "DATE" and "year" in ent.text.lower():
                entities["EXPERIENCE"].add(ent.text.lower().strip())
    return [{k: list(v) for k, v in e.items()} for e in keyword_results]


# ─────────────────────────────────────────────
# 4. Candidate matching function
# ─────────────────────────────────────────────
def compute_match_score(
    job_entities: dict,
    candidate_entities: dict,
    skill_w: float = 0.6,
    title_w: float = 0.25,
    exp_w: float = 0.15,
) -> dict:
    """
    Compute a match score between a job description and a candidate profile.
    Returns score (0-100) and breakdown by entity type.
    Weights are auto-normalized so they always sum to 1.0.
    """
    total_w = skill_w + title_w + exp_w
    if total_w > 0:
        skill_w, title_w, exp_w = skill_w / total_w, title_w / total_w, exp_w / total_w

    weights = {"SKILL": skill_w, "JOB_TITLE": title_w, "EXPERIENCE": exp_w}
    scores = {}

    for entity_type, weight in weights.items():
        job_set = set(job_entities.get(entity_type, []))
        candidate_set = set(candidate_entities.get(entity_type, []))

        if not job_set:
            scores[entity_type] = 0.0
            continue

        overlap = len(job_set & candidate_set)
        entity_score = overlap / len(job_set)
        scores[entity_type] = round(entity_score * weight * 100, 2)

    total_score = round(sum(scores.values()), 1)
    return {
        "total_score": total_score,
        "skill_score": scores.get("SKILL", 0),
        "title_score": scores.get("JOB_TITLE", 0),
        "experience_score": scores.get("EXPERIENCE", 0),
    }


# ─────────────────────────────────────────────
# 5. Semantic similarity (sentence embeddings)
# ─────────────────────────────────────────────
def compute_semantic_scores(
    jd_text: str,
    model,
    all_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Encode the JD and compute cosine similarity against all pre-computed
    resume embeddings. Returns a 1-D array of scores (0–1) aligned with
    the rows of all_embeddings.

    Because embeddings are unit-normalised, dot product == cosine similarity.
    """
    jd_vec = model.encode([jd_text], normalize_embeddings=True)
    return (all_embeddings @ jd_vec.T).flatten()


# ─────────────────────────────────────────────
# 6. Main recommendation function
# ─────────────────────────────────────────────
def generate_explanation(matched: list, missing: list, score: float) -> str:
    matched_str = ", ".join(matched[:3]) if matched else None
    missing_str = ", ".join(missing[:2]) if missing else None
    if matched_str and missing_str:
        return f"Strong match due to {matched_str}; lacks {missing_str}"
    elif matched_str:
        return f"Strong match due to {matched_str}"
    elif missing_str:
        return f"Weak skill overlap; lacks {missing_str}"
    else:
        return "Match based on title and experience alignment"


def recommend_candidates(
    job_description: str,
    candidates_df: pd.DataFrame,
    top_n: int = 5,
    model=None,
    embeddings: np.ndarray = None,
    embedding_ids: np.ndarray = None,
    precomputed_entities: list = None,
    job_entities: dict = None,
    ner_weight: float = 0.5,
    sem_weight: float = 0.5,
    skill_w: float = 0.6,
    title_w: float = 0.25,
    exp_w: float = 0.15,
) -> list:
    """
    Rank candidates against a job description.

    candidates_df columns: id, cleaned_text, category
    model / embeddings / embedding_ids: sentence-transformer model and
        pre-computed embeddings from precompute_embeddings.py (optional —
        falls back to NER-only scoring if not provided).
    precomputed_entities: list of entity dicts aligned with candidates_df rows,
        pre-computed at startup to avoid per-query spaCy overhead.
    job_entities: pre-extracted entities for the JD; extracted here if not provided.
    ner_weight / sem_weight: blending weights (auto-normalized).
    skill_w / title_w / exp_w: NER sub-weights (auto-normalized).

    Returns list of dicts sorted by total_score descending.
    """
    if job_entities is None:
        job_entities = extract_entities(job_description)

    # Normalize blend weights
    blend_total = ner_weight + sem_weight
    if blend_total > 0:
        ner_weight, sem_weight = ner_weight / blend_total, sem_weight / blend_total

    # Pre-compute semantic scores for all candidates in one batch
    if model is not None and embeddings is not None and embedding_ids is not None:
        sem_scores_all = compute_semantic_scores(job_description, model, embeddings)
        sem_lookup = {int(eid): float(s) for eid, s in zip(embedding_ids, sem_scores_all)}
    else:
        sem_lookup = {}

    results = []
    for i, (_, row) in enumerate(candidates_df.iterrows()):
        candidate_entities = precomputed_entities[i] if precomputed_entities is not None else extract_entities(str(row["cleaned_text"]))
        scores = compute_match_score(job_entities, candidate_entities, skill_w, title_w, exp_w)

        job_skills = set(s.lower() for s in job_entities.get("SKILL", []))
        candidate_skills = set(s.lower() for s in candidate_entities.get("SKILL", []))
        matched_skills = list(job_skills & candidate_skills)
        missing_skills = list(job_skills - candidate_skills)

        sem_score = sem_lookup.get(int(row["id"]), 0.0) * 100
        blended_score = round((scores["total_score"] * ner_weight) + (sem_score * sem_weight), 1)

        explanation = generate_explanation(matched_skills, missing_skills, blended_score)

        results.append({
            "id": row["id"],
            "name": row.get("name", "Candidate"),
            "category": row.get("category", "Unknown"),
            "total_score": blended_score,
            "skill_score": scores["skill_score"],
            "title_score": scores["title_score"],
            "experience_score": scores["experience_score"],
            "semantic_score": round(sem_score, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "explanation": explanation,
            "extracted_skills": candidate_entities.get("SKILL", []),
            "extracted_titles": candidate_entities.get("JOB_TITLE", []),
            "extracted_experience": candidate_entities.get("EXPERIENCE", []),
        })

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results[:top_n]


def compute_skill_gap(job_entities: dict, candidate_entities_cache: list) -> dict:
    """
    For each required skill in the JD, count how many candidates in the full pool
    have it vs. are missing it.
    Returns: { skill: { "have": int, "missing": int, "pct_have": float } }
    """
    job_skills = [s.lower() for s in job_entities.get("SKILL", [])]
    total = len(candidate_entities_cache)
    if not job_skills or total == 0:
        return {}

    result = {}
    for skill in job_skills:
        have = sum(
            1 for ents in candidate_entities_cache
            if skill in [s.lower() for s in ents.get("SKILL", [])]
        )
        result[skill] = {
            "have": have,
            "missing": total - have,
            "pct_have": round((have / total) * 100, 1),
        }
    return result