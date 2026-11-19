"""Weighted TF-IDF and structured-feature candidate-job matching."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

WEIGHTS = {
    "text_similarity": 0.30,
    "required_skill_coverage": 0.45,
    "experience": 0.15,
    "education": 0.10,
}


def _skills(value):
    if pd.isna(value):
        return set()
    return {part.strip().casefold() for part in str(value).split("|") if part.strip()}


def _education_score(candidate_education, requirement):
    candidate = str(candidate_education).casefold()
    required = str(requirement).casefold()
    if "master" in required:
        if "master" in candidate or "doctor" in candidate:
            return 1.0
        if "bachelor" in candidate:
            return 0.5
        return 0.0
    if "bachelor" in required:
        return 1.0 if ("bachelor" in candidate or "master" in candidate or "doctor" in candidate) else 0.0
    return 1.0


def score_match(
    resume_text,
    job_description,
    candidate_skills,
    required_skills,
    experience_years,
    minimum_experience,
    education,
    education_requirement,
):
    required = _skills(required_skills)
    candidate = _skills(candidate_skills)
    skill_score = len(required & candidate) / len(required) if required else 1.0
    minimum = max(float(minimum_experience), 0.0)
    experience_score = 1.0 if minimum == 0 else min(max(float(experience_years), 0.0) / minimum, 1.0)
    education_score = _education_score(education, education_requirement)

    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(
        [f"{resume_text}", f"{job_description}"]
    )
    text_score = float(cosine_similarity(text_features[0:1], text_features[1:2])[0, 0])
    weighted_score = (
        WEIGHTS["text_similarity"] * text_score
        + WEIGHTS["required_skill_coverage"] * skill_score
        + WEIGHTS["experience"] * experience_score
        + WEIGHTS["education"] * education_score
    )
    return {
        "score": round(max(0.0, min(100.0, weighted_score * 100)), 2),
        "text_similarity": round(text_score, 4),
        "skill_coverage": round(skill_score, 4),
        "experience_score": round(experience_score, 4),
        "education_score": round(education_score, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True, help="Candidate ID, e.g. C0001")
    parser.add_argument("--job-id", required=True, help="Job ID, e.g. J0001")
    args = parser.parse_args()

    resumes = pd.read_csv(PROCESSED / "cleaned_resumes.csv")
    jobs = pd.read_csv(PROCESSED / "processed_jobs.csv")
    candidate = resumes[resumes["resume_id"] == args.candidate_id.replace("C", "R", 1)]
    job = jobs[jobs["job_id"] == args.job_id]
    if candidate.empty:
        parser.error(f"Candidate not found: {args.candidate_id}")
    if job.empty:
        parser.error(f"Job not found: {args.job_id}")

    resume = candidate.iloc[0]
    posting = job.iloc[0]
    result = score_match(
        resume["resume_text"],
        posting["job_description"],
        resume["skills"],
        posting["required_skills"],
        resume["experience_years"],
        posting["minimum_experience"],
        resume["education"],
        posting["education_requirement"],
    )
    print(
        f"Candidate {args.candidate_id} / job {args.job_id}: "
        f"{result['score']:.2f}/100 "
        f"(text={result['text_similarity']:.3f}, "
        f"skills={result['skill_coverage']:.3f}, "
        f"experience={result['experience_score']:.3f}, "
        f"education={result['education_score']:.3f})"
    )


if __name__ == "__main__":
    main()
