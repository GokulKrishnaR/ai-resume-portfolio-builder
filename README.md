---
title: AI Resume & Portfolio Builder
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# 📄 AI Resume & Portfolio Builder

An AI-powered Resume, Cover Letter, and Portfolio Generator built using **Gradio** and the **Hugging Face Inference API**.

This project helps students and freshers instantly generate:

- ✅ ATS-friendly resumes  
- ✅ Role-specific cover letters  
- ✅ Personal portfolio content  
- ✅ Downloadable TXT files  
- ✅ Portfolio webpage (.html)

---

## 🚀 Live Demo

🔗 Hugging Face Space:  
https://huggingface.co/spaces/GokulKrishnaR04/ai-resume-portfolio-builder

---

## ✨ Features

- 🎯 Structured professional resume generation  
- ✉️ AI-powered cover letter customization  
- 🌐 HTML portfolio webpage export  
- 📄 Downloadable `.txt` files (fully editable)  
- 🤖 AI generation with HF token  
- 🧩 Smart template fallback (works without token)  
- 🔐 Secure token handling via environment variables  

---

## 📂 Downloadable Outputs

The application generates:

- Resume → `.txt` file  
- Cover Letter → `.txt` file  
- Portfolio content → `.txt` file  
- Portfolio webpage → `.html` file  

All text outputs are intentionally provided in editable plain-text format for flexibility and easy modification.

---

## 🧠 AI Model Used

Primary Model:  
HuggingFaceTB/SmolLM2-1.7B-Instruct  

Fallback Model:  
HuggingFaceTB/SmolLM2-135M-Instruct  

If no Hugging Face token is provided, the system automatically switches to structured template-based generation.

---

## 🖥 Run Locally

### 1️⃣ Clone the repository


```bash
git clone https://github.com/GokulKrishnaR/ai-resume-portfolio-builder.git
cd ai-resume-portfolio-builder
```

### 2️⃣ Create Virtual Environment (Recommended)

### Windows

```
python -m venv venv  
venv\Scripts\activate  
```

### Mac/Linux

```
python3 -m venv venv  
source venv/bin/activate  
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt  
```

### 4️⃣ Run the App

```
python app.py  
```

Open in browser:  
http://localhost:7860  

---

## 🔐 Enable AI Generation (Optional)

To enable AI-powered generation:

1. Go to: https://huggingface.co/settings/tokens  
2. Generate a token  
3. Set environment variable  

### Windows

set HF_TOKEN=your_token_here  

### Mac/Linux

export HF_TOKEN=your_token_here  

If no token is provided, the system still works using structured templates.

---

## ☁ Deploy on Hugging Face Spaces

1. Create a new **Gradio Space**  
2. Upload:  
   - app.py  
   - requirements.txt  
   - README.md  
3. (Optional) Add `HF_TOKEN` in:  
   Space Settings → Repository Secrets  

To push updates:

```
git add .
git commit -m "Update app"
git push origin main      # GitHub
git push hf main          # Hugging Face Space
```

---

## 🏗 Tech Stack

- Python  
- Gradio  
- Hugging Face Inference API  
- Markdown  
- HTML + CSS  

---

## 🎯 Designed For

- Students  
- Freshers  
- Internship applicants  
- Entry-level job seekers  

---

## 👨‍💻 Author

Gokul Krishna R  
AI & Machine Learning Enthusiast  

⭐ If you found this helpful, consider starring the repository or liking the Space!
