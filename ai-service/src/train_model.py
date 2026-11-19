"""Train the resume category classifier using training data only."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"


def train():
    train_data = pd.read_csv(SPLITS / "train.csv")
    if train_data["job_category"].nunique() < 2:
        raise ValueError("Training data must contain at least two job categories.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_data["resume_text"].fillna(""))
    classifier = LogisticRegression(max_iter=2000, random_state=42)
    classifier.fit(train_features, train_data["job_category"])

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODELS / "resume_classifier.joblib")
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.joblib")
    print(
        f"Trained on {len(train_data)} resumes; saved classifier and vectorizer in {MODELS}"
    )


if __name__ == "__main__":
    train()
