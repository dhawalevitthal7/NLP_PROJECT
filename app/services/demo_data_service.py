import json
from pathlib import Path

from app.models.schemas import ExtractedPaper, QuestionItem


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_extracted_paper_from_json(json_path: str, paper_type: str) -> ExtractedPaper:
    resolved_path = Path(json_path)
    if not resolved_path.is_absolute():
        resolved_path = _resolve_repo_root() / json_path

    with resolved_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    raw_questions = payload.get("questions", [])
    questions = [QuestionItem.model_validate(question) for question in raw_questions]
    return ExtractedPaper(
        paper_type=paper_type,
        source=str(resolved_path),
        questions=questions,
        extraction_confidence=1.0,
        warnings=[],
    )
