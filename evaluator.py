
import json
import os

from google import genai
from google.genai import types



SYSTEM = """
You are a strict, fair educational assessment and answer-sheet grading assistant.

Your task is to evaluate EVERY question in the question paper against the
student's submitted answer sheet and the supplied marking scheme.

MANDATORY QUESTION-COVERAGE RULES:
1. Read the entire question paper first and identify every question and
   sub-question, preserving its original number and order.
2. Return one result for EVERY question and sub-question that is separately
   assigned marks in the question paper.
3. Never omit a question just because the student did not answer it.
4. Match answers using question numbers, sub-question labels, and content.
   Do not assume that an answer to Q2 answers Q1 or Q3.
5. If a question has no identifiable answer in the submitted answer sheet,
   award 0 marks for that question.
6. For an unanswered question, set score to 0, explain that the answer is
   missing in marks_lost_reason, and set review_required to false only when
   the absence is clear. If answer alignment is uncertain, require teacher
   review.
7. Do not confuse an answer that is present but incorrect with an answer
   that is absent. Evaluate present answers using the marking scheme.
8. Award partial credit only for relevant, correct work actually present
   in the student's answer.
9. Use the marking scheme as the scoring authority. Retrieved reference
   passages are supporting evidence, not a replacement for the scheme.
10. Preserve the maximum marks for every question, including unanswered
    questions. Never reduce total possible marks because a question is blank.
11. Do not invent questions, answers, marks, or evidence. If question text
    or marks are genuinely unreadable or unspecified, flag the uncertainty
    for teacher review instead of guessing.
12. Check your final output against the question paper: every numbered
    question and separately marked sub-question must appear exactly once.

Return one valid JSON object with these top-level keys:
- questions
- overall_feedback

Each item in questions must contain:
- question_label
- question_text
- max_marks
- score
- marks_lost_reason
- rationale
- criterion_scores
- strengths
- improvements
- evidence
- review_required

For unanswered questions, include the question in the output with score 0
and the correct max_marks. State that no answer was found.

Ensure 0 <= score <= max_marks for every question.
The sum of question scores must reflect the marks awarded question by question.
Return JSON only, with no Markdown fences or extra commentary.
"""


def _as_text(value):
    return value if isinstance(value, str) else str(value or "")


def _normalise_result(result):
    if not isinstance(result, dict) or not isinstance(
        result.get("questions"), list
    ):
        raise RuntimeError(
            "Gemini returned invalid JSON: expected a questions list."
        )

    cleaned_questions = []

    for index, question in enumerate(result["questions"], start=1):
        if not isinstance(question, dict):
            continue

        try:
            max_marks = max(0.0, float(question.get("max_marks", 0)))
            score = float(question.get("score", 0))
        except (TypeError, ValueError):
            max_marks, score = 0.0, 0.0

        score = min(max(score, 0.0), max_marks)

        criteria = question.get("criterion_scores", [])
        if not isinstance(criteria, list):
            criteria = []

        cleaned_criteria = []

        for criterion in criteria:
            if not isinstance(criterion, dict):
                continue

            try:
                available = max(
                    0.0, float(criterion.get("marks_available", 0))
                )
                awarded = min(
                    max(float(criterion.get("marks_awarded", 0)), 0.0),
                    available,
                )
            except (TypeError, ValueError):
                available, awarded = 0.0, 0.0

            cleaned_criteria.append({
                "criterion": _as_text(criterion.get("criterion")),
                "marks_awarded": awarded,
                "marks_available": available,
                "evidence_from_answer": _as_text(
                    criterion.get("evidence_from_answer")
                ),
                "reason": _as_text(criterion.get("reason")),
            })

        def string_list(key):
            value = question.get(key, [])
            if not isinstance(value, list):
                return []
            return [_as_text(item) for item in value]

        cleaned_questions.append({
            "question_label": _as_text(
                question.get("question_label") or f"Question {index}"
            ),
            "question_text": _as_text(question.get("question_text")),
            "max_marks": max_marks,
            "score": score,
            "marks_lost_reason": _as_text(
                question.get("marks_lost_reason")
            ),
            "rationale": _as_text(question.get("rationale")),
            "criterion_scores": cleaned_criteria,
            "strengths": string_list("strengths"),
            "improvements": string_list("improvements"),
            "evidence": string_list("evidence"),
            "review_required": bool(
                question.get("review_required", True)
            ),
        })

    return {
        "questions": cleaned_questions,
        "overall_feedback": _as_text(result.get("overall_feedback")),
    }


def evaluate_assessment(
    subject,
    question_paper_text,
    answer_sheet_text,
    rubric_text,
    retrieved_context,
    model=None,
):
    """Evaluate an assessment using Google's Gemini API."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    model = model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    client = genai.Client(api_key=api_key)

    context_parts = []
    for item in retrieved_context or []:
        if isinstance(item, dict):
            source = _as_text(item.get("source", "Unknown source"))
            kind = _as_text(item.get("kind", "reference"))
            text = _as_text(item.get("text", ""))
            if text.strip():
                context_parts.append(
                    f"[Source: {source} | Type: {kind}]\n{text}"
                )

    context = "\n\n".join(context_parts) or (
        "No reference passages were retrieved."
    )

    payload = {
        "subject": _as_text(subject),
        "question_paper_text": _as_text(question_paper_text),
        "student_answer_sheet_transcription": _as_text(answer_sheet_text),
        "marking_scheme": _as_text(rubric_text),
        "retrieved_context": context,
        "instructions": [
            "Evaluate each identifiable question separately.",
            "Use marks and criterion weights from the marking scheme.",
            "Do not invent maximum marks.",
            "If marks are unclear, require teacher review.",
            "Keep every score between zero and the question maximum.",
            "Explain specifically why marks were lost.",
            "Flag uncertain transcription and question-answer alignment.",
            "Do not claim incomplete source text was fully understood.",
            "Each question must include question_label, question_text, "
            "max_marks, score, marks_lost_reason, rationale, "
            "criterion_scores, strengths, improvements, evidence, "
            "and review_required.",
            "Each criterion_scores item must include criterion, "
            "marks_awarded, marks_available, evidence_from_answer, reason.",
            "Return JSON with questions (array) and "
            "overall_feedback (string).",
        ],
    }

    try:
        response = client.models.generate_content(
            model=model,
            contents=SYSTEM + "\n\nAssessment data:\n" + json.dumps(
                payload, ensure_ascii=False
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        result = json.loads(response.text)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON. Please retry."
        ) from exc
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"Gemini API request failed: {exc}"
        ) from exc

    return _normalise_result(result)
