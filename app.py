import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import docx
from datetime import datetime
from fpdf import FPDF
import re
import os

# Optional OpenAI
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="AI Resume Analyzer", layout="wide", page_icon="📄")

st.title("📄 AI-Powered Resume Analyzer & Screener")
st.markdown("**Portfolio Project 5** — Parses resumes, matches against job descriptions, and gives smart feedback.")

# Sidebar
st.sidebar.header("Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

job_desc = st.sidebar.text_area("Job Description (optional)",
                                value="Python Developer with experience in pandas, Flask, automation, and data analysis. Skills: Python, SQL, Git, APIs.",
                                height=150)

use_ai = st.sidebar.checkbox("Use AI Analysis (OpenAI)", value=False) if OPENAI_AVAILABLE else False
if use_ai:
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")


# Text extraction functions
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])


# Simple keyword-based parsing (works without AI)
def extract_sections(text):
    skills_pattern = re.compile(r'(?i)(skills|technologies|tools)[:\s]*(.*)')
    exp_pattern = re.compile(r'(?i)(experience|work history)[:\s]*(.*?)(\n\n|\Z)', re.DOTALL)

    skills_match = skills_pattern.search(text)
    skills = skills_match.group(2).strip() if skills_match else "Not found"

    # Simple skill list extraction (common tech keywords)
    common_skills = ["python", "pandas", "flask", "fastapi", "selenium", "sql", "git", "api", "automation", "django",
                     "streamlit", "openai"]
    found_skills = [skill for skill in common_skills if skill.lower() in text.lower()]

    return {
        "raw_text": text[:2000],  # truncate for display
        "skills": found_skills,
        "experience_snippet": exp_pattern.search(text).group(2).strip()[:300] if exp_pattern.search(
            text) else "Not found"
    }


# AI-enhanced analysis (if enabled)
def ai_analyze(resume_text, job_desc, api_key):
    if not OPENAI_AVAILABLE or not api_key:
        return "AI analysis not available. Using keyword matching instead."
    client = OpenAI(api_key=api_key)
    prompt = f"""
    You are an expert HR recruiter. Analyze this resume against the job description.
    Resume: {resume_text[:3000]}
    Job Description: {job_desc}

    Provide:
    1. Match score (0-100%)
    2. Key matching skills
    3. Missing skills/gaps
    4. Actionable improvement suggestions (3-5 bullet points)
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI error: {str(e)}. Falling back to basic analysis."


# Main logic
if resume_file:
    if resume_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = extract_text_from_docx(resume_file)

    parsed = extract_sections(resume_text)

    st.subheader("Extracted Resume Information")
    st.write("**Skills Detected:**", ", ".join(parsed["skills"]) if parsed["skills"] else "None detected")
    st.write("**Experience Snippet:**", parsed["experience_snippet"])

    # Match calculation (simple + AI)
    job_keywords = job_desc.lower().split()
    match_count = sum(1 for skill in parsed["skills"] if skill.lower() in job_desc.lower())
    base_score = min(100, int((match_count / max(1, len(job_keywords))) * 100)) if job_keywords else 50

    if use_ai and openai_key:
        ai_feedback = ai_analyze(resume_text, job_desc, openai_key)
        st.subheader("🧠 AI Analysis")
        st.write(ai_feedback)
        # Extract score from AI if possible (simple regex)
        score_match = re.search(r'(\d+)%', ai_feedback)
        final_score = int(score_match.group(1)) if score_match else base_score
    else:
        final_score = base_score
        st.subheader("📊 Basic Match Score")
        st.progress(final_score / 100)
        st.write(f"**Match Score: {final_score}%**")

    # Export report
    if st.button("Generate & Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Resume Analysis Report", ln=True, align='C')
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(0, 10, f"Match Score: {final_score}%", ln=True)
        pdf.ln(5)
        pdf.multi_cell(0, 8, f"Detected Skills: {', '.join(parsed['skills'])}")
        pdf.ln(5)
        if use_ai and openai_key:
            pdf.multi_cell(0, 8, ai_feedback)
        else:
            pdf.multi_cell(0, 8, "Basic keyword matching used.")

        report_path = f"output/resume_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        os.makedirs("output", exist_ok=True)
        pdf.output(report_path)
        with open(report_path, "rb") as f:
            st.download_button("📥 Download PDF Report", f, report_path, "application/pdf")

else:
    st.info("Upload a resume PDF or DOCX to start analysis.")

st.success("✅ Resume Analyzer ready! Extend it with more advanced parsing or full LangChain if desired.")