"""Shared definitions for reproducible synthetic recruitment data."""

CATEGORIES = [
    "Software Developer",
    "Web Developer",
    "Data Analyst",
    "Data Scientist",
    "Python Developer",
    "Java Developer",
    "UI/UX Designer",
    "Machine Learning Engineer",
    "Database Administrator",
    "Business Analyst",
]

PROFILES = {
    "Software Developer": {
        "skills": ["Python", "Java", "C++", "Git", "REST APIs", "SQL", "OOP", "Docker", "Linux"],
        "degrees": ["B.Sc. Computer Science", "B.E. Software Engineering"],
        "projects": ["inventory service", "task scheduling application", "payment API prototype"],
    },
    "Web Developer": {
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express", "REST APIs", "Git", "Responsive Design"],
        "degrees": ["B.Sc. Information Technology", "B.E. Computer Engineering"],
        "projects": ["accessible event website", "online learning portal", "responsive store interface"],
    },
    "Data Analyst": {
        "skills": ["SQL", "Python", "Excel", "Power BI", "Tableau", "Statistics", "Data Cleaning", "Data Visualization"],
        "degrees": ["B.Sc. Statistics", "B.Com. Business Analytics"],
        "projects": ["retail sales dashboard", "customer churn analysis", "public transport performance report"],
    },
    "Data Scientist": {
        "skills": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas", "NumPy", "Scikit-learn", "Data Visualization"],
        "degrees": ["M.Sc. Data Science", "B.Sc. Mathematics"],
        "projects": ["demand forecasting study", "customer segmentation experiment", "anomaly detection prototype"],
    },
    "Python Developer": {
        "skills": ["Python", "Django", "Flask", "FastAPI", "REST APIs", "PostgreSQL", "Git", "Pytest"],
        "degrees": ["B.Sc. Computer Science", "B.E. Information Technology"],
        "projects": ["FastAPI booking service", "Django knowledge portal", "automated reporting utility"],
    },
    "Java Developer": {
        "skills": ["Java", "Spring Boot", "SQL", "Hibernate", "REST APIs", "Maven", "Git", "OOP"],
        "degrees": ["B.E. Computer Engineering", "B.Sc. Computer Science"],
        "projects": ["Spring Boot order service", "library management application", "Java workflow scheduler"],
    },
    "UI/UX Designer": {
        "skills": ["Figma", "Wireframing", "Prototyping", "User Research", "Usability Testing", "Adobe XD", "Design Systems", "Accessibility"],
        "degrees": ["B.Des. Interaction Design", "B.A. Visual Communication"],
        "projects": ["mobile banking redesign", "inclusive travel app prototype", "design system for a learning platform"],
    },
    "Machine Learning Engineer": {
        "skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "Docker", "MLOps", "SQL"],
        "degrees": ["M.Sc. Artificial Intelligence", "B.E. Computer Science"],
        "projects": ["model serving API", "image classification pipeline", "automated model monitoring prototype"],
    },
    "Database Administrator": {
        "skills": ["SQL", "PostgreSQL", "MySQL", "Oracle", "Database Design", "Backup and Recovery", "Performance Tuning", "Linux"],
        "degrees": ["B.Sc. Information Technology", "B.E. Computer Engineering"],
        "projects": ["database backup validation tool", "query performance review", "replica monitoring dashboard"],
    },
    "Business Analyst": {
        "skills": ["Requirements Gathering", "SQL", "Excel", "Power BI", "Process Mapping", "User Stories", "Stakeholder Management", "Data Analysis"],
        "degrees": ["B.B.A. Business Analytics", "B.Com. Information Systems"],
        "projects": ["order fulfillment process review", "requirements for a service portal", "monthly operations dashboard"],
    },
}

SOFT_SKILLS = [
    "Communication",
    "Problem Solving",
    "Teamwork",
    "Documentation",
    "Time Management",
]

CERTIFICATIONS = [
    "Foundations of Cloud Computing",
    "Agile Project Practices",
    "Data Literacy Essentials",
    "Secure Software Fundamentals",
    "User-Centered Design Basics",
    "Database Concepts Certificate",
    "Applied Python Certificate",
    "Business Analysis Foundations",
]

FIRST_NAMES = [
    "Aarav", "Aditi", "Anaya", "Arjun", "Dev", "Diya", "Ira", "Kabir", "Kavya", "Meera",
    "Neel", "Nisha", "Rohan", "Sana", "Vihaan", "Zoya", "Ishaan", "Mira", "Reyansh", "Tara",
    "Ayaan", "Kiara", "Milan", "Riya", "Samar", "Veda", "Yash", "Amara", "Nikhil", "Leela",
]

LAST_NAMES = [
    "Mehta", "Shah", "Rao", "Kapoor", "Iyer", "Joshi", "Nair", "Desai", "Malik", "Sen",
    "Kulkarni", "Patel", "Menon", "Das", "Bose", "Sethi", "Pillai", "Khan", "Bhat", "Roy",
]
