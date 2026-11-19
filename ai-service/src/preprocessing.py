"""Clean generated data, encode categories, and create leakage-safe splits."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_catalog import CATEGORIES
from src.data_cleaning import clean_frame

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"


def _split_resumes(resumes):
    train, remaining = train_test_split(
        resumes,
        test_size=0.30,
        random_state=42,
        stratify=resumes["job_category"],
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=42,
        stratify=remaining["job_category"],
    )
    return {
        "train": train.sort_values("resume_id"),
        "validation": validation.sort_values("resume_id"),
        "test": test.sort_values("resume_id"),
    }


def preprocess():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)

    resumes = clean_frame(
        pd.read_csv(RAW / "resume_dataset.csv"),
        "resume_id",
        ["resume_text", "education", "degree", "certifications", "projects"],
        ["skills"],
    )
    resumes["experience_years"] = pd.to_numeric(
        resumes["experience_years"], errors="coerce"
    ).fillna(0)
    resumes["job_category"] = resumes["job_category"].fillna("Unknown")
    resumes.to_csv(PROCESSED / "cleaned_resumes.csv", index=False)

    jobs = clean_frame(
        pd.read_csv(RAW / "job_dataset.csv"),
        "job_id",
        ["job_title", "job_description", "education_requirement"],
        ["required_skills", "preferred_skills"],
    )
    jobs["minimum_experience"] = pd.to_numeric(
        jobs["minimum_experience"], errors="coerce"
    ).fillna(0)
    jobs.to_csv(PROCESSED / "processed_jobs.csv", index=False)

    category_codes = {category: code for code, category in enumerate(CATEGORIES)}
    unknown = sorted(set(resumes["job_category"]) - set(category_codes))
    if unknown:
        raise ValueError(f"Unknown resume categories cannot be encoded: {unknown}")
    encoded = resumes[["resume_id", "job_category"]].copy()
    encoded["job_category_encoded"] = encoded["job_category"].map(category_codes).astype(int)
    encoded["experience_years"] = resumes["experience_years"].astype(int)
    encoded.to_csv(PROCESSED / "encoded_dataset.csv", index=False)

    split_frames = _split_resumes(resumes)
    candidate_split = {}
    for split_name, split_frame in split_frames.items():
        split_frame.to_csv(SPLITS / f"{split_name}.csv", index=False)
        candidate_split.update(
            {resume_id.replace("R", "C", 1): split_name for resume_id in split_frame["resume_id"]}
        )

    matches = pd.read_csv(RAW / "matching_dataset.csv")
    matches["data_split"] = matches["candidate_id"].map(candidate_split)
    if matches["data_split"].isna().any():
        raise ValueError("A matching example refers to a candidate missing from resume splits.")
    for split_name in ("train", "validation", "test"):
        matches[matches["data_split"] == split_name].drop(columns="data_split").to_csv(
            SPLITS / f"matching_{split_name}.csv", index=False
        )

    print(
        "Preprocessed "
        f"{len(resumes)} resumes into train/validation/test sizes "
        f"{len(split_frames['train'])}/{len(split_frames['validation'])}/"
        f"{len(split_frames['test'])}; matching examples were grouped by candidate."
    )


if __name__ == "__main__":
    preprocess()
