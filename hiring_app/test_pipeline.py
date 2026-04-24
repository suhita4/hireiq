import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from ner_pipeline import extract_entities, recommend_candidates


def _make_df(resumes: list[dict]) -> pd.DataFrame:
    """Build a minimal mock candidates DataFrame."""
    rows = []
    for i, r in enumerate(resumes):
        rows.append({
            "ID": str(i + 1),
            "clean_resume": r["resume"],
            "Category": r.get("category", "Engineering"),
            "similarity_score": r.get("similarity_score", 0.5),
        })
    return pd.DataFrame(rows)


def test_standard_case_top_candidate_has_expected_skills():
    """Top candidate for a Python/SQL/ML JD should match those skills."""
    jd = "We need a data scientist with expertise in python, sql, and machine learning."

    df = _make_df([
        {"resume": "Experienced with python, sql, machine learning, pandas, and numpy."},
        {"resume": "Background in project management, leadership, and communication."},
    ])

    results = recommend_candidates(jd, df, top_n=1)

    assert len(results) == 1
    top = results[0]
    assert "python" in top["matched_skills"]
    assert "sql" in top["matched_skills"]


def test_long_jd_entity_extraction_catches_all_skills():
    """extract_entities on a 10+ skill JD should return at least 8 skills."""
    long_jd = """
    We are looking for a senior machine learning engineer with strong experience
    in python and pytorch. The ideal candidate will have hands-on expertise with
    tensorflow, scikit-learn, and nlp pipelines. Cloud infrastructure skills
    including aws and docker are required. You should be proficient in sql for
    data querying, git for version control, and kubernetes for deployment.
    Familiarity with spark and hadoop is a plus. Excellent communication skills
    and experience working in agile teams are expected. The role involves
    computer vision projects and reinforcement learning research.
    """

    result = extract_entities(long_jd)

    assert "SKILL" in result
    assert len(result["SKILL"]) >= 8


def test_missing_skill_aws_not_in_candidate_resume():
    """Candidate without AWS in resume should have 'aws' in missing_skills."""
    jd = "Looking for a cloud engineer proficient in aws, python, and docker."

    df = _make_df([
        {"resume": "Skilled in python and docker. No cloud experience."},
    ])

    results = recommend_candidates(jd, df, top_n=1)

    assert len(results) == 1
    assert "aws" in results[0]["missing_skills"]
