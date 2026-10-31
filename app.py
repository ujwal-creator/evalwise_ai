import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from document_utils import extract_uploaded_file
from evaluator import evaluate_assessment
from rag_store import build_index, retrieve_context

load_dotenv(override=True)
DB_PATH = os.getenv("EVALWISE_DB_PATH", "evalwise.db")
UPLOAD_ROOT = Path(os.getenv("EVALWISE_UPLOAD_DIR", "stored_uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="EvalWise AI", page_icon="🎓", layout="wide")

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            evaluation_id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            subject TEXT,
            assessment_name TEXT,
            created_at TEXT NOT NULL,
            question_paper_json TEXT NOT NULL,
            answer_sheet_json TEXT NOT NULL,
            reference_files_json TEXT NOT NULL DEFAULT '[]',
            rubric_text TEXT,
            reference_text TEXT,
            question_results_json TEXT NOT NULL,
            total_score REAL,
            total_max REAL,
            overall_feedback TEXT,
            teacher_approved INTEGER DEFAULT 0
        )
        """)
        columns = {row["name"] for row in con.execute("PRAGMA table_info(evaluations)").fetchall()}
        if "reference_files_json" not in columns:
            con.execute("ALTER TABLE evaluations ADD COLUMN reference_files_json TEXT NOT NULL DEFAULT '[]'")
init_db()

def save_uploads(files, evaluation_id, kind):
    saved = []
    folder = UPLOAD_ROOT / evaluation_id / kind
    folder.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        safe_name = Path(f.name).name.replace(" ", "_")
        target = folder / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        target.write_bytes(f.getvalue())
        saved.append({
            "original_name": f.name,
            "stored_path": str(target),
            "mime_type": getattr(f, "type", "application/octet-stream"),
            "size_bytes": len(f.getvalue()),
        })
    return saved

def get_evaluation(evaluation_id):
    with db() as con:
        return con.execute("SELECT * FROM evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone()

def media_preview(items, title):
    st.markdown(f"**{title}**")
    if not items:
        st.caption("No files attached.")
        return
    for item in items:
        path = Path(item["stored_path"])
        if not path.exists():
            st.warning(f"File missing: {item['original_name']}")
            continue
        suffix = path.suffix.lower()
        st.caption(item["original_name"])
        if suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            st.image(str(path), use_container_width=True)
        elif suffix == ".pdf":
            # Provide a reliable PDF view/download without embedding an unsafe local path.
            st.download_button(
                f"Open/download {item['original_name']}",
                data=path.read_bytes(),
                file_name=item["original_name"],
                mime="application/pdf",
                key=f"pdf_{path}",
            )
        else:
            st.download_button(
                f"Download {item['original_name']}",
                data=path.read_bytes(),
                file_name=item["original_name"],
                key=f"file_{path}",
            )

def render_report(row):
    qp = json.loads(row["question_paper_json"] or "[]")
    answers = json.loads(row["answer_sheet_json"] or "[]")
    references = json.loads(row["reference_files_json"] or "[]")
    results = json.loads(row["question_results_json"] or "[]")
    st.title("Evaluation Report")
    st.caption(f"Evaluation ID: {row['evaluation_id']} · Created: {row['created_at']}")
    a, b, c = st.columns(3)
    a.metric("Student ID", row["student_id"])
    b.metric("Total score", f"{row['total_score'] or 0:g} / {row['total_max'] or 0:g}")
    c.metric("Status", "Teacher approved" if row["teacher_approved"] else "Awaiting review")
    with st.expander("Question paper files", expanded=True):
        media_preview(qp, "Uploaded question paper")
    with st.expander("Answer sheet files", expanded=True):
        media_preview(answers, "Uploaded answer sheets")
    with st.expander("Evaluation reference files", expanded=True):
        media_preview(references, "Uploaded scheme of evaluation / reference notes")
    st.subheader("Question-wise marks and feedback")
    for idx, item in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"### {item.get('question_label', f'Question {idx}')}")
            st.metric("Marks", f"{item.get('score', 0)} / {item.get('max_marks', 0)}")
            st.markdown("**Why marks were lost**")
            lost = item.get("marks_lost_reason", "")
            st.write(lost if lost else "No marks lost according to the proposed evaluation.")
            st.markdown("**What was done well**")
            for x in item.get("strengths", []):
                st.write(f"- {x}")
            st.markdown("**What to improve**")
            for x in item.get("improvements", []):
                st.write(f"- {x}")
            st.markdown("**Criteria breakdown**")
            if item.get("criterion_scores"):
                st.dataframe(pd.DataFrame(item["criterion_scores"]), use_container_width=True, hide_index=True)
            if item.get("evidence"):
                with st.expander("Evidence and evaluation rationale"):
                    st.write(item.get("rationale", ""))
                    for ev in item["evidence"]:
                        st.write(f"- {ev}")
    st.subheader("Overall personalised feedback")
    st.write(row["overall_feedback"] or "No overall feedback saved.")
    with st.expander("Teacher review / score correction"):
        st.caption("Review OCR transcription and AI-proposed marks before approving.")
        approved = st.checkbox("I have reviewed this evaluation", value=bool(row["teacher_approved"]), key=f"approve_{row['evaluation_id']}")
        override_total = st.number_input(
            "Teacher-approved total marks",
            min_value=0.0,
            max_value=float(row["total_max"] or 0),
            value=float(row["total_score"] or 0),
            step=0.5,
            key=f"score_{row['evaluation_id']}",
        )
        if st.button("Save review status", key=f"save_review_{row['evaluation_id']}"):
            with db() as con:
                con.execute(
                    "UPDATE evaluations SET teacher_approved=?, total_score=? WHERE evaluation_id=?",
                    (1 if approved else 0, override_total, row["evaluation_id"]),
                )
            st.success("Review status saved.")
            st.rerun()
    export_obj = {
        "evaluation_id": row["evaluation_id"],
        "student_id": row["student_id"],
        "subject": row["subject"],
        "assessment_name": row["assessment_name"],
        "created_at": row["created_at"],
        "total_score": row["total_score"],
        "total_max": row["total_max"],
        "overall_feedback": row["overall_feedback"],
        "question_paper_files": qp,
        "answer_sheet_files": answers,
        "evaluation_reference_files": references,
        "questions": results,
    }
    st.download_button(
        "Download report JSON",
        data=json.dumps(export_obj, indent=2, ensure_ascii=False),
        file_name=f"evaluation_{row['student_id']}_{row['evaluation_id'][:8]}.json",
        mime="application/json",
        key=f"export_{row['evaluation_id']}",
    )

def home():
    st.markdown("<div style='text-align:center;padding:4rem 1rem 2rem'>", unsafe_allow_html=True)
    st.markdown("# 🎓 EvalWise AI")
    st.markdown("### Welcome to smarter answer sheet evaluation")
    st.markdown("Rubric-grounded grading, transparent marks, and personalised feedback.")
    st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("📝 New Evaluation")
            st.write("Upload a question paper and student answer sheets to create a new evaluation.")
            if st.button("Start new evaluation", type="primary", use_container_width=True):
                st.session_state["page"] = "new"
                st.rerun()
    with right:
        with st.container(border=True):
            st.subheader("🔎 Search Evaluation")
            st.write("Find previous evaluations by student ID or evaluation ID and reopen the full report.")
            if st.button("Search previous evaluations", use_container_width=True):
                st.session_state["page"] = "search"
                st.rerun()
    st.caption("AI marks are proposals. A teacher should review every result before it is treated as final.")

def new_evaluation():
    st.title("New Evaluation")
    st.caption("Upload files, provide a marking scheme, then evaluate and save the report.")
    with st.form("new_eval_form", clear_on_submit=False):
        student_id = st.text_input("Student ID *", placeholder="e.g. STU-2026-001")
        subject = st.text_input("Subject / course *", placeholder="e.g. Biology, Economics, Computer Science")
        assessment_name = st.text_input("Assessment name", placeholder="e.g. Unit Test 1")
        qp_files = st.file_uploader(
            "Question paper — upload one PDF or multiple images",
            type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="new_qp",
        )
        answer_files = st.file_uploader(
            "Answer sheet — upload one PDF or multiple images",
            type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="new_answers",
        )
        st.markdown("#### Evaluation reference — upload at least one option *")
        st.caption("Upload a scheme of evaluation/marking scheme, reference notes, or both. PDF, TXT, and image files are supported.")
        rubric_files = st.file_uploader(
            "Scheme of evaluation / marking scheme (optional if notes are provided)",
            type=["pdf", "txt", "png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True, key="new_rubric")
        reference_files = st.file_uploader(
            "Reference notes (optional if a scheme is provided)",
            type=["pdf", "txt", "png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True, key="new_refs")
        # Read the model from .env instead of a stale Streamlit text-input value.
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
        if model.startswith("models/"):
            model = model[len("models/"):]
        st.caption(f"Gemini model: {model}")
        submitted = st.form_submit_button("Create evaluation", type="primary")
    if submitted:
        if not student_id.strip() or not subject.strip():
            st.error("Student ID and subject are required.")
            return
        if not qp_files or not answer_files:
            st.error("Upload at least one question paper file and one answer sheet file.")
            return
        if not rubric_files and not reference_files:
            st.error("Upload at least one evaluation reference: scheme of evaluation, reference notes, or both.")
            return
        with st.spinner("Extracting documents, indexing references, and evaluating answers..."):
            evaluation_id = str(uuid.uuid4())
            try:
                qp_saved = save_uploads(qp_files, evaluation_id, "question_paper")
                ans_saved = save_uploads(answer_files, evaluation_id, "answer_sheet")
                rubric_files = rubric_files or []
                reference_files = reference_files or []
                reference_saved = save_uploads(rubric_files + reference_files, evaluation_id, "evaluation_reference")
                rubric_docs = [
                    {"source": f.name, "kind": "scheme_of_evaluation", "text": extract_uploaded_file(f.name, f.getvalue())}
                    for f in rubric_files
                ]
                ref_texts = [
                    {"source": f.name, "kind": "reference_notes", "text": extract_uploaded_file(f.name, f.getvalue())}
                    for f in reference_files
                ]
                rubric_docs = [d for d in rubric_docs if d["text"].strip()]
                ref_texts = [d for d in ref_texts if d["text"].strip()]
                if not rubric_docs and not ref_texts:
                    st.error("Could not extract text from the uploaded notes/scheme. Try clearer scans or upload text-based PDFs/TXT files.")
                    return
                rubric_text = "\n\n".join(f"[Scheme: {d['source']}]\n{d['text']}" for d in rubric_docs)
                documents = rubric_docs + ref_texts
                build_index(documents, namespace=evaluation_id)
                # Uploaded PDFs/images are transcribed locally by PaddleOCR via document_utils.
                qp_texts = [
                    extract_uploaded_file(f.name, f.getvalue())
                    for f in qp_files
                ]
                answer_texts = [
                    extract_uploaded_file(f.name, f.getvalue())
                    for f in answer_files
                ]
                qp_text = "\n\n".join(
                    f"[{f.name}]\n{t}" for f, t in zip(qp_files, qp_texts) if t.strip()
                )
                answer_text = "\n\n".join(
                    f"[{f.name}]\n{t}" for f, t in zip(answer_files, answer_texts) if t.strip()
                )

                # Keep OCR output editable so a teacher can correct mistakes before marking.
                st.session_state["pending_eval"] = {
                    "evaluation_id": evaluation_id,
                    "student_id": student_id,
                    "subject": subject,
                    "assessment_name": assessment_name,
                    "qp_saved": qp_saved,
                    "ans_saved": ans_saved,
                    "reference_saved": reference_saved,
                    "rubric_text": rubric_text,
                    "ref_texts": ref_texts,
                    "qp_text": qp_text,
                    "answer_text": answer_text,
                    "model": model,
                    "qp_files": [f.name for f in qp_files],
                }
                st.session_state["page"] = "manual_text"
                st.rerun()
            except Exception as e:
                st.exception(e)

def manual_text_page():
    st.title("Confirm extracted text")
    pending = st.session_state.get("pending_eval")
    if not pending:
        st.info("No pending evaluation. Start a new evaluation.")
        return
    st.warning("Review the extracted text before scoring. Correct recognition mistakes, especially question numbers, names, formulas, and numbers. If any text is missing or unreadable, type the correction here.")
    st.caption("The extracted text is a draft. Evaluation will use the text you confirm below.")
    qp_text = st.text_area(
        "Question paper text — review and correct",
        value=pending.get("qp_text", ""),
        height=260,
        key=f"qp_ocr_{pending['evaluation_id']}",
    )
    answer_text = st.text_area(
        "Student answer sheet text — review and correct",
        value=pending.get("answer_text", ""),
        height=340,
        key=f"answer_ocr_{pending['evaluation_id']}",
    )
    if st.button("Evaluate and save", type="primary"):
        if not qp_text.strip() or not answer_text.strip():
            st.error("Question text and answer text are both required.")
            return
        try:
            context = retrieve_context(qp_text + "\n" + pending["rubric_text"], namespace=pending["evaluation_id"], top_k=8)
            # Use the current .env model, not the model cached when this pending evaluation was created.
            model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
            if model.startswith("models/"):
                model = model[len("models/"):]
            result = evaluate_assessment(pending["subject"], qp_text, answer_text, pending["rubric_text"], context, model=model)
            total = sum(float(q.get("score", 0)) for q in result.get("questions", []))
            max_total = sum(float(q.get("max_marks", 0)) for q in result.get("questions", []))
            with db() as con:
                con.execute("""INSERT INTO evaluations
                (evaluation_id, student_id, subject, assessment_name, created_at,
                 question_paper_json, answer_sheet_json, reference_files_json, rubric_text, reference_text,
                 question_results_json, total_score, total_max, overall_feedback, teacher_approved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (pending["evaluation_id"], pending["student_id"], pending["subject"], pending["assessment_name"],
                 datetime.now().isoformat(timespec="seconds"), json.dumps(pending["qp_saved"]), json.dumps(pending["ans_saved"]),
                 json.dumps(pending.get("reference_saved", [])), pending["rubric_text"], "\n\n".join(x["text"] for x in pending["ref_texts"]),
                 json.dumps(result.get("questions", []), ensure_ascii=False), total, max_total,
                 result.get("overall_feedback", "")))
            st.session_state["current_evaluation_id"] = pending["evaluation_id"]
            st.session_state.pop("pending_eval", None)
            st.session_state["page"] = "report"
            st.rerun()
        except Exception as e:
            st.exception(e)

def search_page():
    st.title("Search Evaluation")
    query = st.text_input("Search by Student ID or Evaluation ID", placeholder="Enter full or partial ID")
    if query.strip():
        with db() as con:
            rows = con.execute("""
                SELECT evaluation_id, student_id, subject, assessment_name, created_at,
                       total_score, total_max, teacher_approved
                FROM evaluations
                WHERE student_id LIKE ? OR evaluation_id LIKE ?
                ORDER BY created_at DESC LIMIT 100
            """, (f"%{query.strip()}%", f"%{query.strip()}%")).fetchall()
    else:
        with db() as con:
            rows = con.execute("""
                SELECT evaluation_id, student_id, subject, assessment_name, created_at,
                       total_score, total_max, teacher_approved
                FROM evaluations ORDER BY created_at DESC LIMIT 20
            """).fetchall()
    if not rows:
        st.info("No saved evaluations found. Try another search or create a new evaluation.")
    for row in rows:
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.markdown(f"**Student:** {row['student_id']}")
            c1.caption(f"Assessment: {row['assessment_name'] or 'Untitled'}")
            c2.write(f"**Subject:** {row['subject'] or '—'}")
            c2.write(f"**Marks:** {row['total_score'] or 0:g} / {row['total_max'] or 0:g}")
            c2.caption(row["created_at"])
            c3.write("Reviewed" if row["teacher_approved"] else "Needs review")
            if st.button("Open evaluation", key=f"open_{row['evaluation_id']}"):
                st.session_state["current_evaluation_id"] = row["evaluation_id"]
                st.session_state["page"] = "report"
                st.rerun()

if "page" not in st.session_state:
    st.session_state["page"] = "home"

with st.sidebar:
    st.markdown("## EvalWise AI")
    if st.button("Home", use_container_width=True):
        st.session_state["page"] = "home"
        st.rerun()
    if st.button("New Evaluation", use_container_width=True):
        st.session_state["page"] = "new"
        st.rerun()
    if st.button("Search Evaluation", use_container_width=True):
        st.session_state["page"] = "search"
        st.rerun()

page = st.session_state["page"]
if page == "home":
    home()
elif page == "new":
    new_evaluation()
elif page == "search":
    search_page()
elif page == "manual_text":
    manual_text_page()
elif page == "report":
    evaluation_id = st.session_state.get("current_evaluation_id")
    row = get_evaluation(evaluation_id) if evaluation_id else None
    if row:
        render_report(row)
    else:
        st.error("Evaluation record not found.")
        if st.button("Return home"):
            st.session_state["page"] = "home"
            st.rerun()
