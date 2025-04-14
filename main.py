from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----- Step 1: Add Resume Text -----
resume_text = """
I am a data science enthusiast with skills in Python, Machine Learning, data analysis, Excel, and pandas.
I have experience building ML models and creating dashboards with matplotlib and seaborn.
"""

# ----- Step 2: Add Multiple Job Descriptions -----
job_descriptions = [
    "Looking for a Python developer experienced in Flask, SQL, and REST APIs.",
    "Hiring a data analyst skilled in Excel, Power BI, SQL, and Python. ML is a plus.",
    "Seeking a machine learning engineer with deployment skills (Flask/FastAPI), NLP, and model evaluation experience."
]

# ----- Step 3: Define Weighted Skill Keywords -----
weighted_skills = {
    "python": 3,
    "machine learning": 3,
    "data analysis": 2,
    "flask": 2,
    "sql": 2,
    "excel": 1,
    "nlp": 2,
    "model deployment": 2,
    "visualization": 1
}

# ----- Step 4: Score & Skill Checker Functions -----
def score_against_job(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_description.lower(), resume_text.lower()])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

def check_missing_skills(resume_text, weighted_skills):
    resume_text = resume_text.lower()
    missing = []
    for skill, weight in weighted_skills.items():
        if skill not in resume_text:
            missing.append((skill, weight))
    missing.sort(key=lambda x: x[1], reverse=True)
    return missing

# ----- Step 5: Run the Matcher -----
print("\n🔍 Resume Matching Results:\n")
for idx, job_desc in enumerate(job_descriptions, 1):
    score = score_against_job(resume_text, job_desc)
    missing_skills = check_missing_skills(resume_text, weighted_skills)

    print(f"--- Job {idx} ---")
    print(f"Score: {score} / 100")
    print(f"Description: {job_desc[:80]}...")

    if missing_skills:
        print("Missing Important Skills:")
        for skill, weight in missing_skills:
            print(f"  - {skill} (weight: {weight})")
    else:
        print("✅ All key skills present!")
    print("\n")
