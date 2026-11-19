"""Generate deterministic, fictional resumes, jobs, skills, and match labels."""

import argparse
import random
from pathlib import Path

import pandas as pd

from src.data_catalog import (
    CATEGORIES,
    CERTIFICATIONS,
    FIRST_NAMES,
    LAST_NAMES,
    PROFILES,
    SOFT_SKILLS,
)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def _serialize(items):
    return "|".join(items)


def generate_resumes(rng):
    rows = []
    category_order = [category for category in CATEGORIES for _ in range(30)]
    rng.shuffle(category_order)

    for index, category in enumerate(category_order, start=1):
        profile = PROFILES[category]
        core_skills = profile["skills"][:3]
        additional = profile["skills"][3:]
        skills = core_skills + rng.sample(additional, k=min(len(additional), rng.randint(2, 4)))
        skills += rng.sample(SOFT_SKILLS, k=2)
        experience = rng.randint(0, 12)
        degree = rng.choice(profile["degrees"])
        education = "Master's degree" if degree.startswith("M.") else "Bachelor's degree"
        certifications = rng.sample(CERTIFICATIONS, k=rng.randint(1, 2))
        project = rng.choice(profile["projects"])
        candidate_name = (
            f"{FIRST_NAMES[(index - 1) % len(FIRST_NAMES)]} "
            f"{LAST_NAMES[((index - 1) // len(FIRST_NAMES)) % len(LAST_NAMES)]} "
            f"{index:03d}"
        )
        resume_text = (
            f"Fictional candidate seeking a {category} role. Holds a {degree} and has "
            f"{experience} years of project and professional experience. Skills include "
            f"{', '.join(skills)}. Completed {', '.join(certifications)}. "
            f"Project work includes a {project}."
        )
        rows.append(
            {
                "resume_id": f"R{index:04d}",
                "candidate_name": candidate_name,
                "education": education,
                "degree": degree,
                "skills": _serialize(skills),
                "experience_years": experience,
                "certifications": _serialize(certifications),
                "projects": project,
                "resume_text": resume_text,
                "job_category": category,
            }
        )
    return pd.DataFrame(rows)


def generate_jobs():
    rows = []
    for category in CATEGORIES:
        profile = PROFILES[category]
        role_skills = profile["skills"]
        for variant in range(6):
            minimum_experience = variant
            required = role_skills[:3] if variant == 0 else role_skills[:4]
            preferred = role_skills[4:7] or role_skills[3:]
            title = category if variant == 0 else f"{category} - Level {variant + 1}"
            description = (
                f"Join a collaborative team as a {category}. Build and improve reliable "
                f"solutions, document decisions, and work with stakeholders. The role uses "
                f"{', '.join(required)} and values {', '.join(preferred)}. "
                f"Contribute to code or design reviews, testing, and iterative delivery."
            )
            rows.append(
                {
                    "job_id": f"J{len(rows) + 1:04d}",
                    "job_title": title,
                    "job_category": category,
                    "required_skills": _serialize(required),
                    "preferred_skills": _serialize(preferred),
                    "minimum_experience": minimum_experience,
                    "education_requirement": "Bachelor's degree",
                    "job_description": description,
                }
            )
    return pd.DataFrame(rows)


def generate_skills():
    roles_by_skill = {}
    for category, profile in PROFILES.items():
        for skill in profile["skills"]:
            roles_by_skill.setdefault(skill, set()).add(category)
    for skill in SOFT_SKILLS:
        roles_by_skill[skill] = set(CATEGORIES)

    ai_skills = {
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
        "Scikit-learn", "MLOps", "Statistics", "Pandas", "NumPy",
    }
    design_skills = {
        "Figma", "Wireframing", "Prototyping", "User Research",
        "Usability Testing", "Adobe XD", "Design Systems", "Accessibility",
    }
    database_skills = {
        "SQL", "PostgreSQL", "MySQL", "Oracle", "Database Design",
        "Backup and Recovery", "Performance Tuning",
    }
    rows = []
    for index, skill in enumerate(sorted(roles_by_skill, key=str.casefold), start=1):
        if skill in SOFT_SKILLS:
            category, skill_type = "Soft Skills", "soft"
        elif skill in ai_skills:
            category, skill_type = "AI/ML", "technical"
        elif skill in design_skills:
            category, skill_type = "Design", "technical"
        elif skill in database_skills:
            category, skill_type = "Database", "technical"
        elif skill in {"HTML", "CSS", "JavaScript", "React", "Node.js", "Express", "Responsive Design"}:
            category, skill_type = "Web Development", "technical"
        elif skill in {"Excel", "Power BI", "Tableau", "Data Cleaning", "Data Visualization", "Data Analysis"}:
            category, skill_type = "Analytics", "technical"
        elif skill in {"Django", "Flask", "FastAPI", "Spring Boot", "Hibernate", "Maven", "Pytest"}:
            category, skill_type = "Frameworks and Tools", "technical"
        elif skill in {"Docker", "Git", "Linux", "REST APIs"}:
            category, skill_type = "Tools and Platforms", "technical"
        else:
            category, skill_type = "Programming and Methods", "technical"
        rows.append(
            {
                "skill_id": f"S{index:04d}",
                "skill_name": skill,
                "skill_category": category,
                "related_job_roles": _serialize(sorted(roles_by_skill[skill])),
                "skill_type": skill_type,
            }
        )
    return pd.DataFrame(rows)


def generate_matches(resumes, jobs, rng):
    jobs_by_category = {
        category: group.sort_values("minimum_experience")
        for category, group in jobs.groupby("job_category")
    }
    non_core_categories = {
        category: [other for other in CATEGORIES if other != category]
        for category in CATEGORIES
    }
    rows = []
    for _, candidate in resumes.iterrows():
        candidate_skills = candidate["skills"].split("|")
        eligible = jobs_by_category[candidate["job_category"]]
        eligible = eligible[eligible["minimum_experience"] <= candidate["experience_years"]]
        positive_job = eligible.iloc[0]
        negative_category = rng.choice(non_core_categories[candidate["job_category"]])
        negative_job = jobs_by_category[negative_category].iloc[-1]

        for job, label in ((positive_job, 1), (negative_job, 0)):
            required = job["required_skills"].split("|")
            matched = [skill for skill in required if skill in candidate_skills]
            missing = [skill for skill in required if skill not in candidate_skills]
            rows.append(
                {
                    "candidate_id": candidate["resume_id"].replace("R", "C", 1),
                    "job_id": job["job_id"],
                    "candidate_skills": _serialize(candidate_skills),
                    "required_skills": _serialize(required),
                    "matched_skills": _serialize(matched),
                    "missing_skills": _serialize(missing),
                    "experience_match": candidate["experience_years"] >= job["minimum_experience"],
                    "education_match": True,
                    "match_label": label,
                }
            )
    return pd.DataFrame(rows)


def generate(output_dir=ROOT / "data" / "raw", seed=SEED):
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resumes = generate_resumes(rng)
    jobs = generate_jobs()
    skills = generate_skills()
    matches = generate_matches(resumes, jobs, rng)

    for filename, frame in (
        ("resume_dataset.csv", resumes),
        ("job_dataset.csv", jobs),
        ("skills_dataset.csv", skills),
        ("matching_dataset.csv", matches),
    ):
        frame.to_csv(output_dir / filename, index=False)
    print(
        f"Generated {len(resumes)} resumes, {len(jobs)} jobs, "
        f"{len(skills)} skills, and {len(matches)} candidate-job pairs "
        f"with seed {seed} in {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
