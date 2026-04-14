from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvaluationMode(str, Enum):
    fast_semantic = "fast_semantic"
    balanced = "balanced"
    llm_heavy = "llm_heavy"


class SubQuestion(BaseModel):
    sub_question_no: str = ""
    answer: str | None = None
    whole_answer: str | None = None
    marks: float | None = None
    ocr_confidence: float | None = None


class QuestionItem(BaseModel):
    question_no: int
    section: str = ""
    mcq_option: str | None = None
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    whole_answer: str = ""
    total_marks: float | None = None
    key_points: list[str] = Field(default_factory=list)
    ocr_confidence: float | None = None


class ExtractedPaper(BaseModel):
    paper_type: str
    source: str
    questions: list[QuestionItem]
    extraction_confidence: float
    warnings: list[str] = Field(default_factory=list)


class GradeBreakdown(BaseModel):
    embedding_similarity: float
    nli_entailment: float
    keypoint_coverage: float
    llm_semantic_score: float
    weighted_score: float


class QuestionEvaluation(BaseModel):
    question_id: str
    question_no: int
    sub_question_no: str | None = None
    obtained_marks: float
    max_marks: float
    feedback: str
    confidence: float
    needs_manual_review: bool
    breakdown: GradeBreakdown
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    mode: EvaluationMode
    total_obtained: float
    total_max_marks: float
    percentage: float
    overall_confidence: float
    auto_graded_ratio: float
    questions: list[QuestionEvaluation]
    meta: dict[str, Any] = Field(default_factory=dict)


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    progress_message: str = ""
    result: EvaluationResult | None = None
    error: str | None = None

