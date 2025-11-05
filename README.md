# Dental-AI---Detection-Depth-Decision

🦷 Tooth Image Analyzer

A tiny full-stack demo that analyzes dental X-ray images.
Frontend (index.html) lets you upload an image and shows the result.
Backend (app.py) is a Flask API that:

sends the image to OpenAI Vision (GPT-4.1) to extract a structured dental report, and

optionally runs a simple manual baseline model (Logistic Regression on grayscale pixels) when the AI flags the tooth as unhealthy.

⚠️ Disclaimer: This tool is for research/education only and must not be used for clinical diagnosis or treatment decisions.

✨ Features

Clean, single-file HTML frontend with preview and result panel.

Flask API with CORS enabled.

Vision analysis via OpenAI Responses API.

Optional classical ML fallback (LogReg) trained on two folders:

/project/dataset/extraction/ → label 0

/project/dataset/rootcanal/ → label 1

Human-readable, consistent output template (Number of teeth, FDI tooth no., caries depth, treatment, etc.).

📸 Screenshot

<img width="1512" height="982" alt="image" src="https://github.com/user-attachments/assets/72bbf637-3100-4ed6-a60d-fe08364c3781" />


🗂️ Repo Structure
.
├─ index.html           # Frontend (file upload + preview + fetch to /upload)
├─ app.py               # Flask API (OpenAI + manual model)
├─ uploads/             # Runtime: uploaded files land here (auto-created)
└─ README.md
