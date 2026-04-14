import asyncio
from statistics import mean
from uuid import uuid4

from app.config import settings
from app.models.schemas import EvaluationMode, EvaluationResult, GradeBreakdown, QuestionEvaluation, QuestionItem
from app.services.chroma_service import ChromaQuestionRetriever
from app.services.demo_data_service import load_extracted_paper_from_json
from app.services.extraction_service import ExtractionService
from app.services.scoring_service import SemanticScoringService


class EvaluationOrchestrator:
    def __init__(self, extraction: ExtractionService, scoring: SemanticScoringService) -> None:
        self.extraction = extraction
        self.scoring = scoring
        self.retriever = ChromaQuestionRetriever()

    async def evaluate(
        self,
        marking_scheme_path: str,
        student_sheet_path: str,
        mode: EvaluationMode = EvaluationMode.balanced,
    ) -> EvaluationResult:
        if settings.use_demo_json_shortcut:
            scheme_paper = load_extracted_paper_from_json(
                settings.demo_marking_scheme_json,
                paper_type="marking_scheme",
            )
            student_paper = load_extracted_paper_from_json(
                settings.demo_student_answers_json,
                paper_type="student_answers",
            )
        else:
            scheme_task = self.extraction.extract_marking_scheme(marking_scheme_path)
            student_task = self.extraction.extract_student_answers(student_sheet_path)
            scheme_paper, student_paper = await asyncio.gather(scheme_task, student_task)

        collection_name = f"marking_scheme_{uuid4().hex}"
        scheme_questions = [self._build_scheme_question_record(question) for question in scheme_paper.questions]
        scheme_texts = [question["answer_text"] for question in scheme_questions]
        scheme_embeddings = self.scoring.encode_texts(scheme_texts)
        self.retriever.index_scheme_questions(collection_name, scheme_questions, scheme_embeddings)

        evaluations: list[QuestionEvaluation] = []
        confidence_scores: list[float] = []
        auto_graded_count = 0
        skipped_questions: list[int] = []

        for student_question in student_paper.questions:
            question_no = int(student_question.question_no)
            retrieved_question = self.retriever.get_question_by_number(collection_name, question_no)
            if not retrieved_question:
                skipped_questions.append(question_no)
                continue

            scheme_answer = str(retrieved_question["document"] or "").strip()
            scheme_metadata = retrieved_question.get("metadata", {})
            max_marks = float(scheme_metadata.get("total_marks", 0.0) or 0.0)
            if max_marks <= 0:
                skipped_questions.append(question_no)
                continue

            student_answer, answer_source = self._build_student_effective_answer(student_question)
            student_ocr_conf = self._student_confidence(student_question)
            student_mcq_option = (student_question.mcq_option or "").strip().upper()
            correct_mcq_option = str(scheme_metadata.get("mcq_option", "") or "").strip().upper()
            is_mcq = bool(student_mcq_option) and bool(correct_mcq_option)

            if is_mcq:
                is_correct = student_mcq_option == correct_mcq_option
                obtained = max_marks if is_correct else 0.0
                mcq_score = 1.0 if is_correct else 0.0
                confidence = 1.0 if is_correct else 0.95
                evaluation = QuestionEvaluation(
                    question_id=str(retrieved_question["question_id"]),
                    question_no=question_no,
                    sub_question_no=None,
                    obtained_marks=float(obtained),
                    max_marks=max_marks,
                    feedback="Correct MCQ option." if is_correct else "Incorrect MCQ option.",
                    confidence=confidence,
                    needs_manual_review=False,
                    breakdown=GradeBreakdown(
                        embedding_similarity=mcq_score,
                        nli_entailment=mcq_score,
                        keypoint_coverage=mcq_score,
                        llm_semantic_score=mcq_score,
                        weighted_score=mcq_score,
                    ),
                    diagnostics={
                        "student_mcq_option": student_mcq_option,
                        "correct_mcq_option": correct_mcq_option,
                        "answer_source": answer_source,
                        "student_ocr_confidence": round(student_ocr_conf, 4),
                        "scheme_answer_preview": scheme_answer[:200],
                    },
                )
            else:
                scored = await self.scoring.score(
                    scheme_answer=scheme_answer,
                    student_answer=student_answer,
                    key_points=[],
                    max_marks=max_marks,
                    mode=mode,
                )
                merged_confidence = float(np_clip(0.7 * scored.confidence + 0.3 * student_ocr_conf, 0.0, 1.0))
                evaluation = QuestionEvaluation(
                    question_id=str(retrieved_question["question_id"]),
                    question_no=question_no,
                    sub_question_no=None,
                    obtained_marks=scored.obtained_marks,
                    max_marks=max_marks,
                    feedback=scored.feedback,
                    confidence=round(merged_confidence, 4),
                    needs_manual_review=merged_confidence < 0.6,
                    breakdown=scored.breakdown,
                    diagnostics={
                        **scored.diagnostics,
                        "answer_source": answer_source,
                        "student_ocr_confidence": round(student_ocr_conf, 4),
                        "scheme_answer_preview": scheme_answer[:200],
                        "student_answer_preview": student_answer[:200],
                    },
                )

            if not evaluation.needs_manual_review:
                auto_graded_count += 1
            confidence_scores.append(evaluation.confidence)
            evaluations.append(evaluation)

        total_obtained = round(sum(item.obtained_marks for item in evaluations), 2)
        total_max_marks = round(sum(item.max_marks for item in evaluations), 2)
        percentage = round((total_obtained / total_max_marks * 100), 2) if total_max_marks > 0 else 0.0
        overall_confidence = round(float(mean(confidence_scores)) if confidence_scores else 0.0, 4)
        auto_graded_ratio = round(auto_graded_count / max(len(evaluations), 1), 4)

        return EvaluationResult(
            mode=mode,
            total_obtained=total_obtained,
            total_max_marks=total_max_marks,
            percentage=percentage,
            overall_confidence=overall_confidence,
            auto_graded_ratio=auto_graded_ratio,
            questions=evaluations,
            meta={
                "marking_scheme_extraction_confidence": scheme_paper.extraction_confidence,
                "student_extraction_confidence": student_paper.extraction_confidence,
                "marking_scheme_warnings": scheme_paper.warnings,
                "student_warnings": student_paper.warnings,
                "total_questions_evaluated": len(evaluations),
                "skipped_questions": skipped_questions,
                "used_demo_json_shortcut": settings.use_demo_json_shortcut,
                "chroma_collection_name": collection_name,
            },
        )

    @staticmethod
    def _build_scheme_question_record(question: QuestionItem) -> dict[str, str | float | int | None]:
        whole_answer = (question.whole_answer or "").strip()
        if whole_answer:
            answer_text = whole_answer
        else:
            parts: list[str] = []
            for sub_question in question.sub_questions:
                sub_answer = (sub_question.answer or sub_question.whole_answer or "").strip()
                if sub_answer:
                    parts.append(f"({sub_question.sub_question_no}) {sub_answer}")
            answer_text = " | ".join(parts).strip()

        return {
            "question_id": f"q{question.question_no}",
            "question_no": question.question_no,
            "answer_text": answer_text,
            "mcq_option": question.mcq_option,
            "total_marks": float(question.total_marks or 0.0),
        }

    @staticmethod
    def _build_student_effective_answer(question: QuestionItem) -> tuple[str, str]:
        whole_answer = (question.whole_answer or "").strip()
        if whole_answer:
            return whole_answer, "whole_answer"

        parts: list[str] = []
        for sub_question in question.sub_questions:
            sub_answer = (sub_question.whole_answer or sub_question.answer or "").strip()
            sub_no = sub_question.sub_question_no or ""
            if sub_answer:
                parts.append(f"({sub_no}) {sub_answer}")

        combined = " | ".join(parts).strip()
        return combined, "sub_questions" if combined else "empty"

    @staticmethod
    def _student_confidence(question: QuestionItem) -> float:
        confidences: list[float] = []
        if question.ocr_confidence is not None:
            confidences.append(float(question.ocr_confidence))
        for sub_question in question.sub_questions:
            if sub_question.ocr_confidence is not None:
                confidences.append(float(sub_question.ocr_confidence))
        return float(mean(confidences)) if confidences else 1.0


def np_clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

