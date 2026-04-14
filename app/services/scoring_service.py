import asyncio
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.models.schemas import EvaluationMode, GradeBreakdown
from app.services.azure_client import AzureOpenAIService

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional import guard
    SentenceTransformer = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional import guard
    pipeline = None


@dataclass
class ScoreResult:
    obtained_marks: float
    feedback: str
    confidence: float
    breakdown: GradeBreakdown
    diagnostics: dict[str, Any]


class SemanticScoringService:
    def __init__(self, azure_service: AzureOpenAIService) -> None:
        self.azure = azure_service
        self._embedder = None
        self._nli = None

    def _get_embedder(self):
        if self._embedder is None and SentenceTransformer is not None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        embedder = self._get_embedder()
        if embedder is None:
            # Keeps pipeline functional even when sentence-transformers is unavailable.
            return [[0.0] * 384 for _ in texts]
        vectors = embedder.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def _get_nli(self):
        if self._nli is None and pipeline is not None:
            self._nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
        return self._nli

    async def score(
        self,
        scheme_answer: str,
        student_answer: str,
        key_points: list[str],
        max_marks: float,
        mode: EvaluationMode,
    ) -> ScoreResult:
        if max_marks <= 0:
            empty = GradeBreakdown(
                embedding_similarity=0.0,
                nli_entailment=0.0,
                keypoint_coverage=0.0,
                llm_semantic_score=0.0,
                weighted_score=0.0,
            )
            return ScoreResult(0.0, "No marks configured for this question.", 0.0, empty, {})

        if not student_answer.strip():
            empty = GradeBreakdown(
                embedding_similarity=0.0,
                nli_entailment=0.0,
                keypoint_coverage=0.0,
                llm_semantic_score=0.0,
                weighted_score=0.0,
            )
            return ScoreResult(0.0, "No answer detected for this question.", 0.85, empty, {"empty_answer": True})

        embedding_task = asyncio.to_thread(self._embedding_similarity, scheme_answer, student_answer)
        nli_task = asyncio.to_thread(self._nli_entailment, scheme_answer, student_answer)
        keypoint_task = asyncio.to_thread(self._keypoint_coverage, key_points, student_answer)

        emb, nli_score, key_cov = await asyncio.gather(embedding_task, nli_task, keypoint_task)
        llm_score, llm_feedback, llm_marks = await asyncio.to_thread(
            self._llm_semantic_grade,
            scheme_answer,
            student_answer,
            max_marks,
        )

        weights = self._weights_for_mode(mode)
        weighted_score = (
            weights["embedding"] * emb
            + weights["nli"] * nli_score
            + weights["keypoint"] * key_cov
            + weights["llm"] * llm_score
        )
        weighted_score = float(np.clip(weighted_score, 0.0, 1.0))
        if llm_marks is not None:
            # Mirror Colab: ceil and cap at max_marks
            obtained = float(min(max_marks, float(np.ceil(float(llm_marks)))))
        else:
            obtained = round(weighted_score * max_marks, 2)

        confidence = self._confidence(emb, nli_score, key_cov, llm_score, mode)
        breakdown = GradeBreakdown(
            embedding_similarity=round(emb, 4),
            nli_entailment=round(nli_score, 4),
            keypoint_coverage=round(key_cov, 4),
            llm_semantic_score=round(llm_score, 4),
            weighted_score=round(weighted_score, 4),
        )
        diagnostics = {"weights": weights}
        return ScoreResult(
            obtained_marks=obtained,
            feedback=llm_feedback,
            confidence=round(confidence, 4),
            breakdown=breakdown,
            diagnostics=diagnostics,
        )

    def _weights_for_mode(self, mode: EvaluationMode) -> dict[str, float]:
        if mode == EvaluationMode.fast_semantic:
            # Favor OpenAI semantic judgement while keeping light local signals
            return {"embedding": 0.25, "nli": 0.20, "keypoint": 0.15, "llm": 0.40}
        if mode == EvaluationMode.llm_heavy:
            return {"embedding": 0.20, "nli": 0.20, "keypoint": 0.15, "llm": 0.45}
        return {
            "embedding": settings.weight_embedding,
            "nli": settings.weight_nli,
            "keypoint": settings.weight_keypoint,
            "llm": settings.weight_llm,
        }

    def _embedding_similarity(self, scheme_answer: str, student_answer: str) -> float:
        embedder = self._get_embedder()
        if embedder is None:
            return 0.0
        vectors = embedder.encode([scheme_answer, student_answer], normalize_embeddings=True)
        sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def _nli_entailment(self, scheme_answer: str, student_answer: str) -> float:
        model = self._get_nli()
        if model is None:
            return 0.0
        result = model(f"{student_answer} [SEP] {scheme_answer}")[0]
        label = (result.get("label") or "").upper()
        score = float(result.get("score", 0.0))
        if label == "ENTAILMENT":
            return score
        if label == "NEUTRAL":
            return score * 0.5
        return 0.0

    def _keypoint_coverage(self, key_points: list[str], student_answer: str) -> float:
        if not key_points:
            return 0.5
        normalized_answer = self._normalize_text(student_answer)
        matched = 0
        for point in key_points:
            tokens = [tok for tok in self._normalize_text(point).split() if len(tok) > 2]
            if not tokens:
                continue
            overlap = sum(1 for tok in tokens if tok in normalized_answer)
            ratio = overlap / max(len(tokens), 1)
            if ratio >= 0.4:
                matched += 1
        return matched / max(len(key_points), 1)

    def _llm_semantic_grade(self, scheme_answer: str, student_answer: str, max_marks: float) -> tuple[float, str, float | None]:
        prompt = f"""
You are evaluating a student's exam answer.

Correct Answer:
{scheme_answer}

Student Answer:
{student_answer}

Maximum Marks: {max_marks}

Tasks:
1. Compute semantic similarity between the two answers (0.0 to 1.0).
2. Assign marks out of {max_marks} proportional to conceptual correctness.

Rules:
- Accept synonyms, paraphrasing, and equivalent scientific terms.
- Ignore spelling and grammar mistakes.
- Award partial marks for partially correct answers.
- Similarity must reflect conceptual overlap, not just word matching.
- If student answer covers all key points → marks close to {max_marks}.
- If completely wrong or irrelevant → marks = 0.

Return ONLY this JSON (no markdown, no explanation):
{{
  "similarity": <float 0.0 to 1.0>,
  "marks": <float 0.0 to {max_marks}>,
  "feedback": "<short explanation>"
}}
"""
        try:
            data = self.azure.chat_json(
                system_prompt="You are a fair semantic examiner. Output valid JSON only.",
                user_content=prompt,
                max_tokens=700,
            )
            score = float(data.get("similarity", data.get("score_0_to_1", 0.0)))
            feedback = str(data.get("feedback", "Semantic grading completed by LLM."))
            marks = data.get("marks")
            marks_val: float | None = float(marks) if marks is not None else None
            return float(np.clip(score, 0.0, 1.0)), feedback, marks_val
        except Exception as exc:
            return 0.0, f"LLM grading unavailable: {exc}", None

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _confidence(emb: float, nli: float, key_cov: float, llm_score: float, mode: EvaluationMode) -> float:
        llm_weight = 0.25 if mode != EvaluationMode.fast_semantic else 0.0
        base = 0.35 * emb + 0.30 * nli + 0.25 * key_cov + llm_weight * llm_score
        if mode == EvaluationMode.fast_semantic:
            return float(np.clip(base + 0.1, 0.0, 1.0))
        return float(np.clip(base, 0.0, 1.0))

