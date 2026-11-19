# AI Recruitment Dataset and Model

This module provides reproducible, **entirely synthetic** data for the AI/ML portion of an AI-powered recruitment management system. Candidate names, resumes, job descriptions, and match labels are generated examples. They do not represent real people or hiring decisions.

## Dataset contents

Run the generator with seed `42` to create:

| File | Records | Description |
|---|---:|---|
| `data/raw/resume_dataset.csv` | 300 | 30 synthetic resumes for each of 10 job categories |
| `data/raw/job_dataset.csv` | 60 | 6 synthetic job descriptions per category |
| `data/raw/skills_dataset.csv` | 61 | Technical, AI/ML, design, database, and soft skills |
| `data/raw/matching_dataset.csv` | 600 | Two labeled candidate-job examples for each synthetic candidate |

Resume categories are Software Developer, Web Developer, Data Analyst, Data Scientist, Python Developer, Java Developer, UI/UX Designer, Machine Learning Engineer, Database Administrator, and Business Analyst.

### Column descriptions

- **Resume data:** `resume_id` (synthetic record ID), `candidate_name` (fictional name with a unique suffix), `education`, `degree`, `skills` (pipe-separated), `experience_years`, `certifications` (pipe-separated), `projects`, `resume_text`, and `job_category` (classification label).
- **Job data:** `job_id`, `job_title`, `job_category`, `required_skills` and `preferred_skills` (pipe-separated), `minimum_experience`, `education_requirement`, and `job_description`.
- **Skills data:** `skill_id`, `skill_name`, `skill_category`, `related_job_roles` (pipe-separated), and `skill_type`.
- **Matching data:** `candidate_id` (maps to a resume ID by replacing `C` with `R`), `job_id`, `candidate_skills`, `required_skills`, `matched_skills`, `missing_skills` (pipe-separated), `experience_match`, `education_match`, and binary `match_label` (`1` relevant, `0` not relevant).

Matching labels are generated from the synthetic role/qualification scenario: each candidate receives one same-category job whose required skills and experience threshold are satisfied, and one different-category comparison job. These are illustrative labels, not human-annotated ground truth.

## Processing and model

- `src/data_cleaning.py` removes duplicate IDs, fills missing values, normalizes resume/job text, and standardizes pipe-separated skills.
- `src/preprocessing.py` writes `data/processed/cleaned_resumes.csv`, `processed_jobs.csv`, and `encoded_dataset.csv`. It creates stratified 70%/15%/15% resume train/validation/test splits. Matching examples are split by candidate ID using the same resume assignment; no candidate occurs in more than one matching split. Matching split CSVs are also saved in `data/splits/`.
- `src/train_model.py` fits a word unigram/bigram TF-IDF vectorizer and multinomial Logistic Regression on the training resumes only. It saves `models/resume_classifier.joblib` and `models/tfidf_vectorizer.joblib`.
- `src/evaluate_model.py` measures validation and held-out test classification performance (accuracy, macro precision, macro recall, macro F1, and confusion matrix). It separately evaluates matching predictions on held-out matching examples using a fixed 50/100 score threshold. Metrics are calculated from the generated data at runtime; no performance values are assumed in advance.
- `src/matching_engine.py` returns a score from 0 to 100. Configurable weights in `WEIGHTS` are text TF-IDF cosine similarity 30%, required-skill coverage 45%, experience 15%, and education 10%. Experience is scaled from zero to the required minimum (or full credit if no minimum); education uses degree-level compatibility.

Generated evaluation reports and figures are placed in `outputs/`: JSON classification and matching reports, `confusion_matrix.csv`, resume/job category distributions, common skills, split distribution, and evaluation charts.

Because category-specific role language is deliberately present in these synthetic resumes, classifier metrics may be optimistic. They measure performance on this generated demonstration dataset only and should not be interpreted as evidence of real-world hiring or production accuracy.

## Windows PowerShell instructions

Run these commands from the module directory:

```powershell
Set-Location C:\Users\SAKSHI\Downloads\Database_Code_Folder\ai-service
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.dataset_generator
python -m src.preprocessing
python -m src.train_model
python -m src.evaluate_model
python -m src.matching_engine --candidate-id C0001 --job-id J0001
```

Re-running the generator with the same seed produces the same synthetic records. Use `python -m src.dataset_generator --seed 123` to choose another reproducible seed; rerun preprocessing and training/evaluation afterward to refresh downstream artifacts. The test split is reserved for final evaluation and is not used to fit the vectorizer or classifier.
