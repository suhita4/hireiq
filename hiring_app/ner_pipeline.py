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
import re
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. Load spaCy model
# ─────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
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
# 3. Entity extraction function
# ─────────────────────────────────────────────
def extract_entities(text: str) -> dict:
    """
    Extract SKILL, JOB_TITLE, and EXPERIENCE entities from text.
    Returns a dict: { 'SKILL': set(), 'JOB_TITLE': set(), 'EXPERIENCE': set() }
    """
    text_lower = text.lower()
    entities = defaultdict(set)

    # Extract skills
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            entities["SKILL"].add(skill)

    # Extract job titles
    for title in JOB_TITLE_KEYWORDS:
        if title in text_lower:
            entities["JOB_TITLE"].add(title)

    # Extract experience patterns
    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entities["EXPERIENCE"].add(match.strip())

    # Also run spaCy NER for additional entity hints (ORG, PERSON etc. ignored,
    # but GPE/DATE can hint at experience context)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE" and "year" in ent.text.lower():
            entities["EXPERIENCE"].add(ent.text.lower().strip())

    return {k: list(v) for k, v in entities.items()}


# ─────────────────────────────────────────────
# 4. Candidate matching function
# ─────────────────────────────────────────────
def compute_match_score(job_entities: dict, candidate_entities: dict) -> dict:
    """
    Compute a match score between a job description and a candidate profile.
    Returns score (0-100) and breakdown by entity type.
    """
    scores = {}
    weights = {"SKILL": 0.6, "JOB_TITLE": 0.25, "EXPERIENCE": 0.15}

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
# 5. Main recommendation function
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


def recommend_candidates(job_description: str, candidates_df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Given a job description and candidate dataframe, return top N ranked candidates.

    candidates_df must have columns: ID, clean_resume, Category, similarity_score
    Returns list of dicts sorted by match score descending.
    """
    job_entities = extract_entities(job_description)
    results = []

    for _, row in candidates_df.iterrows():
        candidate_entities = extract_entities(str(row["clean_resume"]))
        scores = compute_match_score(job_entities, candidate_entities)

        # Matched and missing skills for display
        job_skills = set(s.lower() for s in job_entities.get("SKILL", []))
        candidate_skills = set(s.lower() for s in candidate_entities.get("SKILL", []))
        matched_skills = list(job_skills & candidate_skills)
        missing_skills = list(job_skills - candidate_skills)

        # Blend NER score with existing similarity score from notebook (50/50)
        notebook_score = float(row.get("similarity_score", 0)) * 100
        blended_score = round((scores["total_score"] + notebook_score) / 2, 1)

        explanation = generate_explanation(matched_skills, missing_skills, blended_score)

        results.append({
            "id": row["ID"],
            "name": row.get('name', 'Unknown'),
            "category": row.get("Category", "Unknown"),
            "total_score": blended_score,
            "skill_score": scores["skill_score"],
            "title_score": scores["title_score"],
            "experience_score": scores["experience_score"],
            "similarity_score": round(notebook_score, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "explanation": explanation,
            "extracted_skills": candidate_entities.get("SKILL", []),
            "extracted_titles": candidate_entities.get("JOB_TITLE", []),
            "extracted_experience": candidate_entities.get("EXPERIENCE", []),
        })

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results[:top_n]