"""
AI Resume & Portfolio Builder
Generates tailored resumes, cover letters, and portfolios from student data.
Deployable on Hugging Face Spaces.
"""

import os
import gradio as gr
import tempfile
from pathlib import Path

# Optional: Hugging Face Inference for AI generation
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Optional: markdown -> html (for portfolio export)
try:
    import markdown as md_lib
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Optional: PDF generation for downloads
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Model for text generation (ungated, small, fast)
HF_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
FALLBACK_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

def get_client():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token or not HF_AVAILABLE:
        return None
    try:
        return InferenceClient(token=token)
    except Exception:
        return None


def generate_with_ai(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Generate text using Hugging Face Inference API with fallback."""
    client = get_client()
    if client is None:
        return ""
    try:
        out = client.text_generation(
            prompt,
            model=HF_MODEL,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            return_full_text=False,
        )
        return (out or "").strip()
    except Exception:
        try:
            out = client.text_generation(
                prompt,
                model=FALLBACK_MODEL,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                return_full_text=False,
            )
            return (out or "").strip()
        except Exception:
            return ""


def _safe_filename(s: str) -> str:
    s = (s or "").strip() or "student"
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_", " ")).strip()
    s = "_".join(s.split())
    return s[:60] or "student"


def _save_text(text: str, suffix: str, filename_base: str) -> str:
    out_dir = Path(tempfile.gettempdir()) / "ai_resume_builder"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_filename(filename_base)}{suffix}"
    path.write_text(text or "", encoding="utf-8")
    return str(path)


def _save_pdf(text: str, suffix: str, filename_base: str, title: str) -> str:
    """Save plain text content as a simple PDF; fall back safely if PDF lib missing or write fails."""
    out_dir = Path(tempfile.gettempdir()) / "ai_resume_builder"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        out_dir = Path(".") / "ai_resume_builder"
        out_dir.mkdir(parents=True, exist_ok=True)
    base_name = _safe_filename(filename_base)
    txt_path = out_dir / f"{base_name}.txt"
    txt_path.write_text(text or "", encoding="utf-8")

    # If PDF library is not available, return the .txt file
    if not PDF_AVAILABLE:
        return str(txt_path)

    path = out_dir / f"{base_name}{suffix}"
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        try:
            pdf.set_title(title)
        except Exception:
            pass
        pdf.set_font("Helvetica", size=11)
        for line in (text or "").splitlines():
            pdf.multi_cell(0, 8, line)
        pdf.output(str(path))
        return str(path)
    except Exception:
        return str(txt_path)


def _portfolio_html_from_markdown(markdown_text: str, title: str) -> str:
    body_html = ""
    if MARKDOWN_AVAILABLE:
        body_html = md_lib.markdown(markdown_text or "", extensions=["extra", "sane_lists"])
    else:
        # Minimal fallback if markdown lib isn't available
        escaped = (markdown_text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body_html = f"<pre style='white-space:pre-wrap;line-height:1.5'>{escaped}</pre>"

    css = """
    :root { --bg:#0b1220; --card:#121a2b; --text:#e7eefc; --muted:#a9b7d4; --accent:#63b3ff; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background:var(--bg); color:var(--text); }
    .wrap { max-width: 960px; margin: 0 auto; padding: 40px 18px; }
    .card { background: var(--card); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px; }
    h1, h2, h3 { margin-top: 0.2rem; }
    h1 { font-size: 2.0rem; }
    h2 { font-size: 1.25rem; margin-top: 1.4rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.08); }
    a { color: var(--accent); text-decoration: none; }
    p, li { color: var(--muted); line-height: 1.65; }
    ul { padding-left: 1.1rem; }
    code, pre { background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 8px; }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      {body_html}
    </div>
  </div>
</body>
</html>"""


# --------------- Template-based fallbacks (no API needed) ---------------


def _to_bullets_from_text_block(text: str, default: str = "") -> str:
    """Convert a multi-line block into markdown bullets."""
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("- "):
            line = f"- {line}"
        lines.append(line)
    if not lines and default:
        return f"- {default}"
    return "\n".join(lines)


def _to_bullets_from_separated(text: str, default: str = "") -> str:
    """Convert comma/semicolon/newline separated items into bullets."""
    raw = (text or "").replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts and default:
        return f"- {default}"
    return "\n".join(f"- {p}" for p in parts)


def build_resume_template(data: dict) -> str:
    """Professional resume from structured data (no AI)."""
    # Build contact line with optional GitHub and LinkedIn
    contact_parts = [
        part
        for part in [
            data.get("email", ""),
            data.get("phone", ""),
            data.get("location", ""),
        ]
        if part
    ]
    if data.get("github"):
        contact_parts.append(f"GitHub: {data['github']}")
    if data.get("linkedin"):
        contact_parts.append(f"LinkedIn: {data['linkedin']}")
    contact_line = " | ".join(contact_parts)

    # Sections as bullets
    education_block = _to_bullets_from_text_block(
        data.get("education", ""),
        default="Add your high school, undergraduate, and masters details.",
    )
    skills_block = _to_bullets_from_separated(
        data.get("skills", ""),
        default="List your key skills here.",
    )
    projects_block = _to_bullets_from_text_block(
        data.get("projects", ""),
        default="Project name - Brief description and technologies used.",
    )
    additional_block = _to_bullets_from_text_block(
        data.get("additional", ""),
        default="Certifications, languages, interests.",
    )

    # Experience section: optional for freshers, also in bullets
    experience_section = ""
    experience_text = (data.get("experience") or "").strip()
    is_fresher = bool(data.get("is_fresher"))
    if experience_text:
        exp_block = _to_bullets_from_text_block(experience_text)
        experience_section = f"## Experience\n{exp_block}\n\n"
    elif is_fresher:
        experience_section = (
            "## Experience\n"
            "- Fresher with no formal full-time work experience yet. "
            "Highlighting academic projects, internships, and skills.\n\n"
        )

    s = (
        f"# {data.get('name', 'Your Name')}\n"
        f"{contact_line}\n\n"
        "## Professional Summary\n"
        f"{data.get('summary', 'Motivated professional with strong technical and interpersonal skills.')}\n\n"
        "## Education\n"
        f"{education_block}\n\n"
        "## Skills\n"
        f"{skills_block}\n\n"
        "## Projects\n"
        f"{projects_block}\n\n"
        f"{experience_section}"
        "## Additional\n"
        f"{additional_block}\n"
    )
    return s


def generate_summary(data: dict) -> str:
    """Generate a detailed first-person professional summary from other fields using the LLM, with safe fallback."""
    name = data.get("name", "The candidate")
    fresher = bool(data.get("is_fresher"))
    edu = data.get("education", "")
    skills = data.get("skills", "")
    projects = data.get("projects", "")
    experience = data.get("experience", "")

    base_prompt = f"""Write a detailed, first-person professional summary for a resume.
- Length: 3–4 sentences.
- Point of view: first person ("I ..."), not third person.
- Content: briefly mention education, key skills/tech stack, 1–2 important projects, any internships or work experience, and a clear career goal.
- Style: concise, confident, and suitable for ATS-friendly resumes (no emojis, no slang).
Use only the information provided. Do NOT invent degrees, companies, or tools that are not listed.

Name: {name}
Education: {edu}
Skills: {skills}
Projects: {projects}
Experience: {experience or ('None (fresher)' if fresher else 'None')}
Fresher: {fresher}
"""
    ai_text = generate_with_ai(base_prompt, max_tokens=220, temperature=0.45)
    if ai_text:
        # Strip markdown bullets or headings if the model added them
        summary = ai_text.strip().lstrip("-#* ").strip()
        return summary or (
            "I am a motivated candidate with strong technical and interpersonal skills, "
            "eager to apply my knowledge to real-world projects and grow in a fast-paced environment."
        )

    # Fallback heuristic summary (3–4 sentences, first person)
    skills_list = [s.strip() for s in skills.replace(";", ",").split(",") if s.strip()]
    top_skills = ", ".join(skills_list[:4]) if skills_list else "relevant technical and analytical skills"
    has_projects = bool((projects or "").strip())
    has_exp = bool((experience or "").strip())

    if fresher:
        base = (
            f"I am an entry-level candidate with a solid academic background and hands-on exposure to projects "
            f"using {top_skills}. "
        )
        if has_projects:
            base += "Through my academic and personal projects, I have learned how to turn ideas into working solutions and document my work clearly. "
        if has_exp:
            base += "Internship experience has given me a taste of real-world data and collaboration with teams. "
        base += "I am eager to start my career, keep learning quickly, and contribute to impactful engineering or data teams."
        return base

    base = (
        f"I am a results-driven professional with experience applying {top_skills} to solve real-world problems for stakeholders. "
    )
    if has_projects:
        base += "I have led or contributed to projects where I designed, built, and refined solutions from data exploration to final delivery. "
    if has_exp:
        base += "My work experience has strengthened my ability to communicate insights, work with cross-functional teams, and ship improvements iteratively. "
    base += "I am looking to grow further in a role where I can take on more responsibility, mentor others over time, and deliver measurable impact."
    return base


# def evaluate_ats(resume_text: str) -> str:
#     """Very simple ATS-friendliness heuristic returning a markdown report."""
#     text = (resume_text or "").strip()
#     if not text:
#         return "**ATS Score: 0/100**\n\nNo resume content generated."

#     score = 50  # base
#     comments = []
#     lower = text.lower()

#     # Headings / sections
#     for section, weight in [
#         ("professional summary", 6),
#         ("education", 6),
#         ("skills", 6),
#         ("projects", 6),
#         ("experience", 6),
#     ]:
#         if section in lower:
#             score += weight
#         else:
#             comments.append(f"- Add or clearly label a **{section.title()}** section.")

#     # Bullet usage
#     bullet_count = text.count("- ")
#     if bullet_count >= 15:
#         score += 15
#         comments.append("- Good use of bullet points; easy for ATS to parse.")
#     elif bullet_count >= 8:
#         score += 10
#         comments.append("- Decent use of bullet points. You could add a few more for dense sections.")
#     else:
#         comments.append("- Use more bullet points in Skills, Projects, and Experience instead of long paragraphs.")

#     # Length
#     words = len(text.split())
#     if 250 <= words <= 700:
#         score += 10
#     elif words < 200:
#         comments.append("- Resume is quite short; consider adding more detail to projects or experience.")
#     else:
#         comments.append("- Resume is quite long; consider tightening wording for better ATS scanning.")

#     # Basic keyword richness (technical + soft)
#     keyword_hits = 0
#     for kw in ["python", "java", "sql", "machine learning", "deep learning", "communication", "team", "project"]:
#         if kw in lower:
#             keyword_hits += 1
#     if keyword_hits >= 5:
#         score += 7
#         comments.append("- Good mix of technical and soft-skill keywords detected.")
#     elif keyword_hits >= 2:
#         score += 4
#         comments.append("- Some useful keywords detected; consider adding more role-specific skills.")
#     else:
#         comments.append("- Very few recognizable skills/keywords; add more concrete technologies and tools.")

#     score = max(0, min(100, score))

#     detail = "\n".join(comments) if comments else "Looks strong from an ATS perspective."
#     return f"**ATS Score: {score}/100**\n\n{detail}"


def build_cover_template(data: dict, target_role: str, target_company: str) -> str:
    """Cover letter from template."""
    # Precompute bullet skills to avoid complex expressions inside f-string
    raw_skills = data.get("skills", "Relevant skills")
    bullet_skills = raw_skills.replace(",", "\n- ").replace(";", "\n- ")[:400]
    return f"""Dear Hiring Manager,

I am writing to express my interest in the {target_role or 'position'} role at {target_company or 'your company'}.

{data.get('summary', 'I am a motivated professional with relevant skills and experience.')}

My background includes:
- {bullet_skills}

I would welcome the opportunity to discuss how I can contribute to your team. Thank you for considering my application.

Sincerely,
{data.get('name', 'Your Name')}
{data.get('email', '')}
{data.get('phone', '')}
"""


def build_portfolio_template(data: dict) -> str:
    """Portfolio page content (HTML-friendly)."""
    contact_lines = [
        f"- Email: {data.get('email', '')}",
        f"- Phone: {data.get('phone', '')}",
    ]
    if data.get("github"):
        contact_lines.append(f"- GitHub: {data['github']}")
    if data.get("linkedin"):
        contact_lines.append(f"- LinkedIn: {data['linkedin']}")
    contact_block = "\n".join(contact_lines)

    education_block = _to_bullets_from_text_block(data.get("education", ""))
    skills_block = _to_bullets_from_separated(data.get("skills", ""))
    projects_block = _to_bullets_from_text_block(data.get("projects", ""))
    experience_block = _to_bullets_from_text_block(data.get("experience", ""))

    return f"""# {data.get('name', 'Portfolio')}

## About
{data.get('summary', 'Brief bio and career focus.')}

## Education
{education_block}

## Skills & Technologies
{skills_block}

## Projects
{projects_block}

## Experience
{experience_block}

## Contact
{contact_block}
"""


# --------------- Data extraction from Gradio inputs ---------------

def parse_inputs(
    name,
    email,
    phone,
    location,
    summary,
    high_school,
    ug,
    masters,
    skills,
    projects,
    experience,
    additional,
    github,
    linkedin,
    is_fresher,
):
    # Build combined education string with levels
    hs = (high_school or "").strip()
    ug_str = (ug or "").strip()
    ms = (masters or "").strip()
    edu_parts = []
    if hs:
        edu_parts.append(f"High School: {hs}")
    if ug_str:
        edu_parts.append(f"Undergraduate: {ug_str}")
    if ms:
        edu_parts.append(f"Masters: {ms}")
    education = "\n".join(edu_parts)

    return {
        "name": name or "Your Name",
        "email": email or "",
        "phone": phone or "",
        "location": location or "",
        "summary": summary
        or "Motivated professional with strong technical and interpersonal skills.",
        "education": education,
        "education_high_school": hs,
        "education_ug": ug_str,
        "education_masters": ms,
        "skills": skills or "",
        "projects": projects or "",
        "experience": experience or "",
        "additional": additional or "",
        "github": github or "",
        "linkedin": linkedin or "",
        "is_fresher": bool(is_fresher),
    }


# --------------- AI-enhanced generators ---------------

def generate_resume(
    name,
    email,
    phone,
    location,
    summary,
    high_school,
    ug,
    masters,
    skills,
    projects,
    experience,
    additional,
    github,
    linkedin,
    is_fresher,
):
    data = parse_inputs(
        name,
        email,
        phone,
        location,
        summary,
        high_school,
        ug,
        masters,
        skills,
        projects,
        experience,
        additional,
        github,
        linkedin,
        is_fresher,
    )
    # Always generate the professional summary from structured data (ignore any user input)
    data["summary"] = generate_summary(data)
    prompt = f"""You are a professional resume writer. Write a concise, highly ATS-friendly resume in markdown using ONLY the following information.
Use clear headings: Professional Summary, Education, Skills, Projects, Experience.
For Education, Skills, Projects and Experience, use compact markdown bullet lists (each item starting with \"- \").
No extra sections. No placeholders. Output only the resume text.

Name: {data['name']}
Email: {data['email']} | Phone: {data['phone']} | Location: {data['location']}
Summary: {data['summary']}
Education: {data['education']}
Skills: {data['skills']}
Projects: {data['projects']}
Experience: {data['experience']}
Additional: {data['additional']}
GitHub: {data['github']}
LinkedIn: {data['linkedin']}
Fresher: {data['is_fresher']}
"""
    ai_text = generate_with_ai(prompt, max_tokens=1200, temperature=0.5)
    if ai_text:
        return ai_text
    return build_resume_template(data)

def _resume_bundle_error(msg: str, detail: str = "") -> tuple:
    body = f"**Something went wrong**\n\n{msg}"
    if detail:
        body += f"\n\n*Details: {detail}*"
    return body, None


def generate_resume_bundle(
    name,
    email,
    phone,
    location,
    summary,
    high_school,
    ug,
    masters,
    skills,
    projects,
    experience,
    additional,
    github,
    linkedin,
    is_fresher,
):
    try:
        text = generate_resume(
            name,
            email,
            phone,
            location,
            summary,
            high_school,
            ug,
            masters,
            skills,
            projects,
            experience,
            additional,
            github,
            linkedin,
            is_fresher,
        )
        file_path = _save_pdf(text, suffix="_resume.pdf", filename_base=name or "resume", title="Resume")
        # ats_report = evaluate_ats(text)
        return text, file_path#, ats_report
    except Exception as e:
        err_msg = str(e).strip() or type(e).__name__
        return _resume_bundle_error(
            "Resume generation failed. Try again with at least Name and Email filled. "
            "If you use AI generation, add your HF token in Space settings → Repository secrets as `HF_TOKEN`.",
            err_msg,
        )


def generate_cover_letter(
    name,
    email,
    phone,
    location,
    summary,
    high_school,
    ug,
    masters,
    skills,
    projects,
    experience,
    additional,
    github,
    linkedin,
    is_fresher,
    target_role,
    target_company,
    extra_notes,
):
    data = parse_inputs(
        name,
        email,
        phone,
        location,
        summary,
        high_school,
        ug,
        masters,
        skills,
        projects,
        experience,
        additional,
        github,
        linkedin,
        is_fresher,
    )
    data["summary"] = generate_summary(data)
    data["name"] = name or data["name"]
    prompt = f"""Write a professional, concise cover letter (3 short paragraphs) for this candidate applying for {target_role or 'the position'} at {target_company or 'the company'}. Use only this info. Sign off with the candidate name. No placeholders.

Candidate: {data['name']}, {data['email']}, {data['phone']}
Summary: {data['summary']}
Education: {data['education']}
Skills: {data['skills']}
Projects: {data['projects']}
Experience: {data['experience']}
Extra notes for this application: {extra_notes or 'None'}
GitHub: {data['github']}
LinkedIn: {data['linkedin']}
Fresher: {data['is_fresher']}
"""
    ai_text = generate_with_ai(prompt, max_tokens=600, temperature=0.6)
    if ai_text:
        return ai_text
    return build_cover_template(data, target_role or "", target_company or "")

def generate_cover_bundle(
    name,
    email,
    phone,
    location,
    summary,
    high_school,
    ug,
    masters,
    skills,
    projects,
    experience,
    additional,
    github,
    linkedin,
    is_fresher,
    target_role,
    target_company,
    extra_notes,
):
    try:
        text = generate_cover_letter(
            name,
            email,
            phone,
            location,
            summary,
            high_school,
            ug,
            masters,
            skills,
            projects,
            experience,
            additional,
            github,
            linkedin,
            is_fresher,
            target_role,
            target_company,
            extra_notes,
        )
        file_path = _save_pdf(text, suffix="_cover_letter.pdf", filename_base=name or "cover_letter", title="Cover Letter")
        return text, file_path
    except Exception as e:
        err_msg = str(e).strip() or type(e).__name__
        msg = f"**Something went wrong**\n\nCover letter generation failed. Try again with at least Name and Email. If using AI, set `HF_TOKEN` in Space secrets.\n\n*Details: {err_msg}*"
        return msg, None


def generate_portfolio(
    name,
    email,
    phone,
    location,
    summary,
    high_school,
    ug,
    masters,
    skills,
    projects,
    experience,
    additional,
    github,
    linkedin,
    is_fresher,
):
    data = parse_inputs(
        name,
        email,
        phone,
        location,
        summary,
        high_school,
        ug,
        masters,
        skills,
        projects,
        experience,
        additional,
        github,
        linkedin,
        is_fresher,
    )
    data["summary"] = generate_summary(data)
    prompt = f"""Write a short portfolio "About" section (2–3 sentences) and keep Skills, Projects, and Experience as clear lists. Output in markdown with headings: About, Skills & Technologies, Projects, Experience, Contact. Use only this info. No placeholders.

Name: {data['name']}
Summary: {data['summary']}
Education: {data['education']}
Skills: {data['skills']}
Projects: {data['projects']}
Experience: {data['experience']}
Contact: {data['email']}, {data['phone']}, {data['location']}
GitHub: {data['github']}
LinkedIn: {data['linkedin']}
Fresher: {data['is_fresher']}
"""
    ai_text = generate_with_ai(prompt, max_tokens=800, temperature=0.5)
    if ai_text:
        return ai_text
    return build_portfolio_template(data)

def generate_portfolio_bundle(
    name,
    email,
    phone,
    location,
    summary,
    high_school,
    ug,
    masters,
    skills,
    projects,
    experience,
    additional,
    github,
    linkedin,
    is_fresher,
):
    try:
        md_text = generate_portfolio(
            name,
            email,
            phone,
            location,
            summary,
            high_school,
            ug,
            masters,
            skills,
            projects,
            experience,
            additional,
            github,
            linkedin,
            is_fresher,
        )
        html = _portfolio_html_from_markdown(md_text, title=f"{(name or 'Student')} - Portfolio")
        pdf_path = _save_pdf(md_text, suffix="_portfolio.pdf", filename_base=name or "portfolio", title="Portfolio")
        html_path = _save_text(html, suffix="_portfolio.html", filename_base=name or "portfolio")
        return md_text, html, pdf_path, html_path
    except Exception as e:
        err_msg = str(e).strip() or type(e).__name__
        msg = f"**Something went wrong**\n\nPortfolio generation failed. Try again with at least Name and Email. If using AI, set `HF_TOKEN` in Space secrets.\n\n*Details: {err_msg}*"
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return msg, f"<div class=\"card\"><p style='white-space:pre-wrap'>{safe}</p></div>", None, None


# --------------- Gradio UI ---------------

def create_ui():
    with gr.Blocks(title="AI Resume & Portfolio Builder") as app:
        gr.Markdown("""
        # 🎯 AI Resume & Portfolio Builder
        **Present your skills and projects in a professional format.**  
        Enter your details below, then generate a **tailored resume**, **cover letter**, or **portfolio**.
        """)

        with gr.Tabs():
            # ----- Tab 1: Your Data -----
            with gr.Tab("📝 Your Data"):
                gr.Markdown("Fill in your information. Fields marked with **(mandatory)** are strongly recommended for a good resume; others are **(optional)**.")
                with gr.Row():
                    with gr.Column(scale=1):
                        name = gr.Textbox(label="Full Name (mandatory)", placeholder="e.g. Jane Doe")
                        email = gr.Textbox(label="Email (mandatory)", placeholder="jane@example.com")
                        phone = gr.Textbox(
                            label="Phone (mandatory)",
                            placeholder="+919876543210",
                        )
                        location = gr.Textbox(label="Location (mandatory)", placeholder="City, Country")
                    with gr.Column(scale=1):
                        summary = gr.Textbox(
                            label="Professional Summary (auto-generated, no input needed)",
                            value="This will be generated automatically when you create your resume.",
                            lines=3,
                            interactive=False,
                        )
                with gr.Row():
                    high_school = gr.Textbox(
                        label="High School (optional)",
                        placeholder="School name, board, year",
                        lines=2,
                    )
                    ug = gr.Textbox(
                        label="Undergraduate (UG) (mandatory)",
                        placeholder="Degree, college, year",
                        lines=2,
                    )
                    masters = gr.Textbox(
                        label="Masters (PG) (optional)",
                        placeholder="Degree, university, year (optional)",
                        lines=2,
                    )
                skills = gr.Textbox(
                    label="Skills (mandatory)",
                    placeholder="e.g. Python, ML, React, Communication (comma or line separated)",
                    lines=2,
                )
                projects = gr.Textbox(
                    label="Projects (mandatory)",
                    placeholder="Project name - Short description and tech stack. One per line.",
                    lines=4,
                )
                experience = gr.Textbox(
                    label="Experience (optional)",
                    placeholder="Role @ Company (Duration). Key points. Leave empty if you are a fresher.",
                    lines=4,
                )
                is_fresher = gr.Checkbox(
                    label="I am a fresher (no full-time work experience yet) (optional)",
                    value=False,
                )
                with gr.Row():
                    github = gr.Textbox(
                        label="GitHub (optional)",
                        placeholder="https://github.com/username",
                    )
                    linkedin = gr.Textbox(
                        label="LinkedIn (optional)",
                        placeholder="https://www.linkedin.com/in/username",
                    )
                additional = gr.Textbox(
                    label="Additional (Certifications, Languages, etc.) (optional)",
                    placeholder="e.g. certifications, languages, hackathons (optional)",
                    lines=2,
                )

            # ----- Tab 2: Resume -----
            with gr.Tab("📄 Resume"):
                gr.Markdown("Generate an ATS-friendly resume from your data.")
                with gr.Row():
                    resume_btn = gr.Button("Generate Resume", variant="primary")
                resume_out = gr.Markdown(
                    value="*Click 'Generate Resume' to create your tailored resume.*",
                    elem_classes=["resume-preview"],
                )
                resume_file = gr.File(label="Download resume (.txt)")
                # ats_out = gr.Markdown(
                #     value="*ATS score and feedback will appear here after you generate a resume.*"
                # )
                resume_btn.click(
                    fn=generate_resume_bundle,
                    inputs=[
                        name,
                        email,
                        phone,
                        location,
                        summary,
                        high_school,
                        ug,
                        masters,
                        skills,
                        projects,
                        experience,
                        additional,
                        github,
                        linkedin,
                        is_fresher,
                    ],
                    outputs=[resume_out, resume_file],#ats_out],
                )

            # ----- Tab 3: Cover Letter -----
            with gr.Tab("✉️ Cover Letter"):
                gr.Markdown("Generate a cover letter for a specific role and company.")
                with gr.Row():
                    target_role = gr.Textbox(label="Target Role", placeholder="e.g. Software Engineer Intern")
                    target_company = gr.Textbox(label="Target Company", placeholder="e.g. Google")
                extra_notes = gr.Textbox(
                    label="Extra notes for this application",
                    placeholder="Optional: why this company, specific requirements, etc.",
                    lines=2,
                )
                cover_btn = gr.Button("Generate Cover Letter", variant="primary")
                cover_out = gr.Markdown(
                    value="*Click 'Generate Cover Letter' to generate.*",
                    elem_classes=["cover-preview"],
                )
                cover_file = gr.File(label="Download cover letter (.txt)")
                cover_btn.click(
                    fn=generate_cover_bundle,
                    inputs=[
                        name,
                        email,
                        phone,
                        location,
                        summary,
                        high_school,
                        ug,
                        masters,
                        skills,
                        projects,
                        experience,
                        additional,
                        github,
                        linkedin,
                        is_fresher,
                        target_role,
                        target_company,
                        extra_notes,
                    ],
                    outputs=[cover_out, cover_file],
                )

            # ----- Tab 4: Portfolio -----
            with gr.Tab("🌐 Portfolio"):
                gr.Markdown("Generate portfolio content you can use on a personal website or PDF.")
                port_btn = gr.Button("Generate Portfolio", variant="primary")
                port_out = gr.Markdown(
                    value="*Click 'Generate Portfolio' to generate.*",
                    elem_classes=["port-preview"],
                )
                port_html = gr.HTML(value="")
                with gr.Row():
                    port_md_file = gr.File(label="Download portfolio (.txt)")
                    port_html_file = gr.File(label="Download portfolio webpage (.html)")
                port_btn.click(
                    fn=generate_portfolio_bundle,
                    inputs=[
                        name,
                        email,
                        phone,
                        location,
                        summary,
                        high_school,
                        ug,
                        masters,
                        skills,
                        projects,
                        experience,
                        additional,
                        github,
                        linkedin,
                        is_fresher,
                    ],
                    outputs=[port_out, port_html, port_md_file, port_html_file],
                )

        return app


app = create_ui()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="slate", secondary_hue="blue"),
        css="""
        .main .wrap { max-width: 900px; margin: auto; }
        .resume-preview, .cover-preview, .port-preview { min-height: 320px; }
        """,
    )
