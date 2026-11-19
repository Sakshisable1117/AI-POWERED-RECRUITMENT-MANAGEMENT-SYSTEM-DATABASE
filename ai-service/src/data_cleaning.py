"""Reusable cleaning helpers for synthetic resume and job data."""

import re

import pandas as pd


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    substitutions = {
        "c++": "cpp",
        "c#": "csharp",
        ".net": "dotnet",
        "node.js": "nodejs",
        "scikit-learn": "scikitlearn",
    }
    for source, replacement in substitutions.items():
        text = re.sub(re.escape(source), replacement, text, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


def normalize_skills(value):
    if pd.isna(value):
        return ""
    items = []
    seen = set()
    for item in str(value).split("|"):
        normalized = " ".join(item.strip().casefold().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            items.append(normalized)
    return "|".join(items)


def clean_frame(frame, id_column, text_columns, skill_columns=()):
    cleaned = frame.drop_duplicates(subset=[id_column], keep="first").copy()
    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            median = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(0 if pd.isna(median) else median)
        else:
            cleaned[column] = cleaned[column].fillna("")
    for column in text_columns:
        cleaned[column] = cleaned[column].map(clean_text)
    for column in skill_columns:
        cleaned[column] = cleaned[column].map(normalize_skills)
    return cleaned
