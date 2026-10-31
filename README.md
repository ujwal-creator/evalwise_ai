# EvalWise AI — RAG Answer Sheet Evaluation

A college/final-year project prototype with a welcome page, new evaluation workflow, search for saved evaluations, uploaded file storage, question-wise scores, marks-lost explanations, personalised improvement feedback, and teacher review.

## Features

- Welcome UI with **New Evaluation** and **Search Evaluation**
- Student ID, subject and assessment name
- Multiple question paper images or PDF files
- Multiple answer sheet images or PDF files
- Required evaluation reference: scheme of evaluation, reference notes, or both (PDF/TXT/images)
- Chroma vector index scoped to each evaluation
- LLM-generated question-wise marks and feedback in structured JSON
- SQLite metadata/results store
- Uploaded question papers, answer sheets, and evaluation reference files stored on local disk and linked to saved evaluations
- Search by student ID or evaluation ID
- Review status and total score override
- JSON report download

## Requirements

Python 3.11 recommended. Install Tesseract and Poppler separately if using the optional OCR fallback for image PDFs.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

```bash
streamlit run app.py
```

The first RAG use downloads a Sentence Transformers embedding model. Data is stored in `evalwise.db`, `stored_uploads/`, and `chroma_db/`.

## OCR caveat

The included OCR path uses Tesseract, which is primarily intended for printed text and may be inaccurate on handwriting, maths, diagrams, faint scans, or rotated photos. The app prompts for manual text confirmation when it cannot extract question or answer text. For a stronger implementation, integrate a handwriting-capable OCR/vision model and provide a page-by-page transcription review UI.

## Prototype limitations

- The current scoring prompt processes extracted question paper and answer text as a combined assessment; complex answer layouts may be mismatched. Verify question-to-answer alignment before approval.
- The current interface provides a total-score override, but not editing of every question's individual score in the report UI yet.
- A production version should add authentication, access control, encryption, retention/deletion policies, upload limits, backups, and audit logs.
- Uploaded files contain potentially sensitive student data. Do not deploy publicly or send student documents to third-party APIs without appropriate consent, permissions, and data protection measures.
- AI-proposed marks are not final until reviewed by a teacher.

## Suggested project validation

Prepare a human-scored test set across at least 3 subjects. Record:
- MAE between AI and human marks
- exact agreement and agreement within ±0.5 / ±1 mark
- weighted Cohen's kappa when appropriate
- OCR word error rate on a manually transcribed subset
- teacher rating of feedback usefulness
- rate of unsupported claims and question/answer mismatches

Use separate development and held-out test sets.
