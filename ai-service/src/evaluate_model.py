"""Evaluate classifier and matching labels; save reports and visualizations."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_catalog import CATEGORIES
from src.matching_engine import score_match

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MATCH_THRESHOLD = 50.0


def _classifier_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            labels, predictions, labels=CATEGORIES, output_dict=True, zero_division=0
        ),
    }


def _matching_evaluation(resumes, jobs, examples):
    resume_lookup = resumes.set_index("resume_id")
    job_lookup = jobs.set_index("job_id")
    scores = []
    for _, example in examples.iterrows():
        resume_id = example["candidate_id"].replace("C", "R", 1)
        if resume_id not in resume_lookup.index or example["job_id"] not in job_lookup.index:
            raise ValueError(f"Unknown candidate-job pair: {example['candidate_id']}, {example['job_id']}")
        candidate = resume_lookup.loc[resume_id]
        job = job_lookup.loc[example["job_id"]]
        result = score_match(
            candidate["resume_text"],
            job["job_description"],
            candidate["skills"],
            job["required_skills"],
            candidate["experience_years"],
            job["minimum_experience"],
            candidate["education"],
            job["education_requirement"],
        )
        scores.append(result["score"])

    labels = examples["match_label"].astype(int)
    predictions = [int(score >= MATCH_THRESHOLD) for score in scores]
    return {
        "examples": len(examples),
        "threshold": MATCH_THRESHOLD,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def _plot_category_counts(resumes, jobs):
    resume_counts = resumes["job_category"].value_counts().reindex(CATEGORIES, fill_value=0)
    fig, axis = plt.subplots(figsize=(12, 6))
    resume_counts.plot(kind="bar", ax=axis, color="#4472C4")
    axis.set(title="Synthetic resumes per job category", xlabel="Job category", ylabel="Resume count")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "resume_category_distribution.png", dpi=150)
    plt.close(fig)

    job_counts = jobs["job_category"].value_counts().reindex(CATEGORIES, fill_value=0)
    fig, axis = plt.subplots(figsize=(12, 6))
    job_counts.plot(kind="bar", ax=axis, color="#70AD47")
    axis.set(title="Synthetic job category distribution", xlabel="Job category", ylabel="Job count")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "job_category_distribution.png", dpi=150)
    plt.close(fig)


def _plot_skills(resumes):
    counts = resumes["skills"].fillna("").str.split("|").explode()
    counts = counts[counts.ne("")].value_counts().head(15).sort_values()
    fig, axis = plt.subplots(figsize=(10, 7))
    counts.plot(kind="barh", ax=axis, color="#ED7D31")
    axis.set(title="Most common skills in synthetic resumes", xlabel="Resume count", ylabel="Skill")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "technical_skills_frequency.png", dpi=150)
    plt.close(fig)


def _plot_split_counts():
    frames = {name: pd.read_csv(SPLITS / f"{name}.csv") for name in ("train", "validation", "test")}
    distribution = pd.DataFrame(
        {name: frame["job_category"].value_counts().reindex(CATEGORIES, fill_value=0) for name, frame in frames.items()}
    )
    axis = distribution.plot(kind="bar", figsize=(13, 6))
    axis.set(title="Resume category distribution by split", xlabel="Job category", ylabel="Resume count")
    axis.tick_params(axis="x", rotation=35)
    axis.figure.tight_layout()
    axis.figure.savefig(OUTPUTS / "split_distribution.png", dpi=150)
    plt.close(axis.figure)
    return frames


def _plot_model_metrics(metrics, matrix):
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    values = [metrics[name] for name in metric_names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(["Accuracy", "Precision", "Recall", "F1"], values, color="#4472C4")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Resume classifier test metrics")
    axes[0].set_ylabel("Score")
    image = axes[1].imshow(matrix, interpolation="nearest", cmap="Blues")
    axes[1].set(title="Test confusion matrix", xlabel="Predicted category", ylabel="True category")
    axes[1].set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=90)
    axes[1].set_yticks(range(len(CATEGORIES)), CATEGORIES)
    for row in range(len(CATEGORIES)):
        for column in range(len(CATEGORIES)):
            axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "model_evaluation.png", dpi=150)
    plt.close(fig)


def _plot_matching(metrics):
    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics[name.lower()] for name in names]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(names, values, color="#5B9BD5")
    axis.set(title="Candidate-job matching test metrics", ylabel="Score", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(OUTPUTS / "matching_evaluation.png", dpi=150)
    plt.close(fig)


def evaluate():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    classifier = joblib.load(MODELS / "resume_classifier.joblib")
    vectorizer = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    frames = _plot_split_counts()

    classifier_reports = {}
    test_matrix = None
    for split_name in ("validation", "test"):
        split_data = frames[split_name]
        predictions = classifier.predict(vectorizer.transform(split_data["resume_text"].fillna("")))
        classifier_reports[split_name] = _classifier_metrics(
            split_data["job_category"], predictions
        )
        if split_name == "test":
            test_matrix = confusion_matrix(
                split_data["job_category"], predictions, labels=CATEGORIES
            )
            pd.DataFrame(test_matrix, index=CATEGORIES, columns=CATEGORIES).to_csv(
                OUTPUTS / "confusion_matrix.csv", index_label="true_category"
            )
    with (OUTPUTS / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(classifier_reports, file, indent=2)

    resumes = pd.read_csv(PROCESSED / "cleaned_resumes.csv")
    jobs = pd.read_csv(PROCESSED / "processed_jobs.csv")
    match_test = pd.read_csv(SPLITS / "matching_test.csv")
    matching_report = _matching_evaluation(resumes, jobs, match_test)
    with (OUTPUTS / "matching_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(matching_report, file, indent=2)

    _plot_category_counts(resumes, jobs)
    _plot_skills(resumes)
    _plot_model_metrics(classifier_reports["test"], test_matrix)
    _plot_matching(matching_report)
    print("Classifier validation metrics:", {
        key: round(classifier_reports["validation"][key], 4)
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    })
    print("Classifier test metrics:", {
        key: round(classifier_reports["test"][key], 4)
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    })
    print("Matching test metrics:", {
        key: round(matching_report[key], 4)
        for key in ("accuracy", "precision", "recall", "f1")
    })
    print(f"Reports and charts saved in {OUTPUTS}")


if __name__ == "__main__":
    evaluate()
