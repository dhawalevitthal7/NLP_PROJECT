import re

from app.models.schemas import QuestionItem


def normalize_sub_no(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", value).lower()
    return cleaned


def make_question_id(question_no: int, sub_question_no: str | None = None) -> str:
    sub = normalize_sub_no(sub_question_no)
    return f"q{question_no}_{sub}" if sub else f"q{question_no}"


def flatten_scheme_questions(questions: list[QuestionItem]) -> list[dict]:
    units: list[dict] = []
    for q in questions:
        if q.sub_questions:
            default_marks = (q.total_marks or 0.0) / max(len(q.sub_questions), 1)
            for sub in q.sub_questions:
                answer_text = (sub.answer or sub.whole_answer or "").strip()
                units.append(
                    {
                        "question_id": make_question_id(q.question_no, sub.sub_question_no),
                        "question_no": q.question_no,
                        "sub_question_no": sub.sub_question_no,
                        "answer_text": answer_text,
                        "max_marks": float(sub.marks if sub.marks is not None else default_marks),
                        "key_points": q.key_points,
                        "mcq_option": q.mcq_option,
                        "source_ocr_confidence": q.ocr_confidence,
                    }
                )
        else:
            units.append(
                {
                    "question_id": make_question_id(q.question_no),
                    "question_no": q.question_no,
                    "sub_question_no": None,
                    "answer_text": q.whole_answer.strip(),
                    "max_marks": float(q.total_marks or 0.0),
                    "key_points": q.key_points,
                    "mcq_option": q.mcq_option,
                    "source_ocr_confidence": q.ocr_confidence,
                }
            )
    return units


def flatten_student_questions(questions: list[QuestionItem]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for q in questions:
        if q.sub_questions:
            for sub in q.sub_questions:
                text = (sub.whole_answer or sub.answer or "").strip()
                question_id = make_question_id(q.question_no, sub.sub_question_no)
                lookup[question_id] = {
                    "question_no": q.question_no,
                    "sub_question_no": sub.sub_question_no,
                    "answer_text": text,
                    "mcq_option": q.mcq_option,
                    "ocr_confidence": float(sub.ocr_confidence or q.ocr_confidence or 0.0),
                }
        else:
            lookup[make_question_id(q.question_no)] = {
                "question_no": q.question_no,
                "sub_question_no": None,
                "answer_text": q.whole_answer.strip(),
                "mcq_option": q.mcq_option,
                "ocr_confidence": float(q.ocr_confidence or 0.0),
            }
    return lookup

