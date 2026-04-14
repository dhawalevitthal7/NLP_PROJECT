"""
Extraction service: PDF → structured QuestionItem list via Azure GPT-4o Vision.

Architecture:
  1. PDF is rendered to JPEG images (in pdf_service.py, low DPI for small payloads).
  2. Images are split into batches (2 pages per batch by default).
  3. Each batch is sent to Azure GPT-4o via an async-level retry wrapper.
  4. A semaphore limits simultaneous Azure calls (avoids rate-limit pressure).
  5. Results are merged by question_no, combining split answers across pages.

Retry strategy:
  - Uses asyncio.sleep between retry attempts so the event loop is NOT blocked.
  - 3 attempts per batch with delays [5s, 15s, 30s].
"""

import asyncio
import logging
from statistics import mean
from typing import Any

from openai import APITimeoutError, RateLimitError

from app.config import settings
from app.models.schemas import ExtractedPaper, QuestionItem
from app.services.azure_client import AzureOpenAIService
from app.services.pdf_service import chunk_pages, pdf_to_base64_images

logger = logging.getLogger(__name__)

# Retry config — async-level so event loop is not blocked
_MAX_RETRIES = 3
_RETRY_DELAYS = [5, 15, 30]  # seconds; uses asyncio.sleep

# Semaphore caps simultaneous Azure calls across BOTH extraction tasks
_azure_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _azure_semaphore
    if _azure_semaphore is None:
        limit = settings.max_concurrent_azure_calls
        logger.info("Creating Azure concurrency semaphore | max_concurrent=%d", limit)
        _azure_semaphore = asyncio.Semaphore(limit)
    return _azure_semaphore


# ─────────────────────────────────────────────────────────────────────────────
# System prompts  (returned JSON must include batch_confidence + warnings)
# ─────────────────────────────────────────────────────────────────────────────

SCHEME_PROMPT = """You are an expert at extracting marking scheme data from printed/scanned exam pages.

Return ONLY valid JSON with NO markdown fences, NO extra text:
{
  "questions": [
    {
      "question_no": <integer>,
      "section": "<A/B/C/D or empty string>",
      "mcq_option": "<correct option letter A/B/C/D, or null>",
      "sub_questions": [
        {
          "sub_question_no": "<a/b/c/i/ii/iii>",
          "answer": "<full answer text for this sub-question>",
          "marks": <number or null>
        }
      ],
      "whole_answer": "<full answer when no sub-questions, else empty string>",
      "total_marks": <number or null>,
      "key_points": ["<rubric point 1>", "<rubric point 2>", "..."],
      "ocr_confidence": <float 0.0 to 1.0>
    }
  ],
  "batch_confidence": <float 0.0 to 1.0>,
  "warnings": ["<any issues noticed>"]
}

Strict rules:
1. Extract EVERY question visible on the pages — do not skip any.
2. question_no MUST be an integer.
3. For descriptive answers, extract 3–7 concise rubric key_points (main concepts/terms the student must mention).
4. If sub-question marks are not printed, set marks to null (do NOT guess).
5. For MCQ questions, set mcq_option to the correct option letter and whole_answer to "".
6. Set ocr_confidence based on print quality (1.0 = perfectly clear, 0.5 = some blur/damage).
7. Set batch_confidence based on your overall confidence extracting this batch.
"""

STUDENT_PROMPT = """You are an expert at reading handwritten student exam answer sheets.

Return ONLY valid JSON with NO markdown fences, NO extra text:
{
  "questions": [
    {
      "question_no": <integer>,
      "section": "<A/B/C/D or empty string>",
      "mcq_option": "<option the student chose: A/B/C/D, or null if not MCQ>",
      "sub_questions": [
        {
          "sub_question_no": "<a/b/c/i/ii/iii>",
          "whole_answer": "<transcribed handwritten text for this sub-question>",
          "ocr_confidence": <float 0.0 to 1.0>
        }
      ],
      "whole_answer": "<full answer text if no sub-parts, else empty string>",
      "total_marks": null,
      "ocr_confidence": <float 0.0 to 1.0>
    }
  ],
  "batch_confidence": <float 0.0 to 1.0>,
  "warnings": ["<any handwriting issues, illegible sections, etc.>"]
}

Strict rules:
1. Detect ALL question numbers — Q1, Q.1, 1., 1, Question 1 are all treated as Q1.
2. Keep sub-questions SEPARATE in the sub_questions array.
3. Transcribe handwriting FAITHFULLY — do NOT correct, improve, or paraphrase.
4. Use [illegible] where a word cannot be read.
5. For MCQ: set mcq_option to the letter the student circled/wrote; whole_answer = "".
6. total_marks is always null — never fill it.
7. question_no MUST be an integer.
8. Do NOT skip any question you can see.
9. If a question is attempted but left blank, set whole_answer = "".
"""


# ─────────────────────────────────────────────────────────────────────────────
# Extraction service
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionService:
    """Orchestrates PDF → base64 images → batched Azure GPT-4o calls → merged list."""

    def __init__(self, azure_service: AzureOpenAIService) -> None:
        self.azure = azure_service

    async def extract_marking_scheme(self, pdf_path: str) -> ExtractedPaper:
        return await self._extract(pdf_path, paper_type="marking_scheme", prompt=SCHEME_PROMPT)

    async def extract_student_answers(self, pdf_path: str) -> ExtractedPaper:
        return await self._extract(pdf_path, paper_type="student_answers", prompt=STUDENT_PROMPT)

    # ── Internal orchestration ──────────────────────────────────────────────

    async def _extract(self, pdf_path: str, paper_type: str, prompt: str) -> ExtractedPaper:
        logger.info("═══ [%s] Starting extraction ══════════════════ path=%s", paper_type, pdf_path)

        images = pdf_to_base64_images(pdf_path)
        batches = chunk_pages(images)
        sem = _get_semaphore()

        logger.info(
            "[%s] total_pages=%d | batches=%d | pages_per_batch=%d | concurrent_limit=%d",
            paper_type, len(images), len(batches),
            settings.page_batch_size, settings.max_concurrent_azure_calls,
        )

        async def _bounded_batch(batch: list[str], idx: int) -> Any:
            async with sem:
                logger.info(
                    "[%s] Batch %d/%d — acquired semaphore slot, sending %d page(s) to Azure…",
                    paper_type, idx, len(batches), len(batch),
                )
                return await self._call_batch_with_retry(batch, prompt, idx, paper_type, len(batches))

        tasks = [_bounded_batch(batch, idx) for idx, batch in enumerate(batches, start=1)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_questions: list[QuestionItem] = []
        confidences: list[float] = []
        warnings: list[str] = []

        for idx, result in enumerate(batch_results, start=1):
            if isinstance(result, Exception):
                msg = f"Batch {idx} failed after all retries: {type(result).__name__}: {result}"
                warnings.append(msg)
                logger.error("[%s] ✗ %s", paper_type, msg)
                continue

            batch_conf = float(result.get("batch_confidence", 0.8))
            batch_warns = result.get("warnings", [])
            questions_raw = result.get("questions", [])
            confidences.append(batch_conf)
            if batch_warns:
                warnings.extend(batch_warns)

            logger.info(
                "[%s] Batch %d ✓ | confidence=%.2f | questions_extracted=%d | warnings=%d",
                paper_type, idx, batch_conf, len(questions_raw), len(batch_warns),
            )
            for warn in batch_warns:
                logger.warning("[%s]   ⚠ Batch %d warning: %s", paper_type, idx, warn)

            for q_raw in questions_raw:
                try:
                    item = QuestionItem.model_validate(q_raw)
                    all_questions.append(item)
                    logger.info(
                        "[%s]   Q%-3s section=%-2s | mcq=%-4s | sub_qs=%d | marks=%s | "
                        "ocr_conf=%s | answer_preview=%.50s",
                        paper_type,
                        item.question_no,
                        item.section or "—",
                        item.mcq_option or "—",
                        len(item.sub_questions),
                        item.total_marks,
                        f"{item.ocr_confidence:.2f}" if item.ocr_confidence is not None else "—",
                        item.whole_answer or (
                            " | ".join(
                                (sub.answer or sub.whole_answer or "") for sub in item.sub_questions
                            )
                        ),
                    )
                except Exception as exc:
                    msg = f"Batch {idx} – invalid question payload: {exc} | raw={q_raw}"
                    warnings.append(msg)
                    logger.warning("[%s] %s", paper_type, msg)

        merged = self._merge_by_question(all_questions)
        overall_conf = float(mean(confidences)) if confidences else 0.0

        logger.info(
            "═══ [%s] Extraction COMPLETE ═════════════════ "
            "unique_questions=%d | overall_confidence=%.2f | warnings=%d",
            paper_type, len(merged), overall_conf, len(warnings),
        )
        for q in merged:
            logger.info(
                "[%s]   MERGED Q%d | marks=%s | sub_qs=%d | key_points=%d",
                paper_type, q.question_no, q.total_marks,
                len(q.sub_questions), len(q.key_points),
            )

        return ExtractedPaper(
            paper_type=paper_type,
            source=pdf_path,
            questions=merged,
            extraction_confidence=overall_conf,
            warnings=warnings,
        )

    # ── Async-level retry wrapper ───────────────────────────────────────────

    async def _call_batch_with_retry(
        self,
        page_images: list[str],
        prompt: str,
        batch_index: int,
        paper_type: str,
        total_batches: int,
    ) -> dict[str, Any]:
        """Call Azure GPT-4o for a batch with async retry on timeout / rate-limit."""
        label = f"{paper_type}-b{batch_index}/{total_batches}"

        user_content: list[dict] = []
        for img_b64 in page_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "high",
                    },
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Extract ALL questions and answers from these {len(page_images)} page(s). "
                    "Return only valid JSON."
                ),
            }
        )

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "[%s] Attempt %d/%d — %d page(s) | ~%.0f KB base64 each",
                    label, attempt, _MAX_RETRIES, len(page_images),
                    sum(len(img) for img in page_images) / len(page_images) / 1024,
                )
                result = await asyncio.to_thread(
                    self.azure.chat_json,
                    prompt,
                    user_content,
                    4096,
                    label,
                )
                logger.info("[%s] Attempt %d SUCCESS", label, attempt)
                return result

            except (APITimeoutError, RateLimitError) as exc:
                last_exc = exc
                delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
                logger.warning(
                    "[%s] Attempt %d/%d RETRIABLE ERROR: %s. Retrying in %ds…",
                    label, attempt, _MAX_RETRIES, type(exc).__name__, delay,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(delay)  # non-blocking — frees event loop

            except Exception as exc:
                last_exc = exc
                logger.error(
                    "[%s] Attempt %d/%d NON-RETRIABLE ERROR: %s: %s",
                    label, attempt, _MAX_RETRIES, type(exc).__name__, exc,
                )
                break  # don't retry non-timeout errors

        logger.error("[%s] All %d attempts exhausted. Last error: %s", label, _MAX_RETRIES, last_exc)
        raise last_exc  # type: ignore[misc]

    # ── Merge question fragments split across page batches ─────────────────

    @staticmethod
    def _merge_by_question(questions: list[QuestionItem]) -> list[QuestionItem]:
        merged: dict[int, QuestionItem] = {}
        for q in questions:
            if q.question_no not in merged:
                merged[q.question_no] = q
                continue

            existing = merged[q.question_no]

            # Append continuation of whole_answer
            if q.whole_answer:
                existing.whole_answer = (
                    f"{existing.whole_answer}\n{q.whole_answer}".strip()
                    if existing.whole_answer
                    else q.whole_answer
                )

            # Merge sub_questions (de-duplicate by sub_question_no)
            existing_subs = {s.sub_question_no: s for s in existing.sub_questions}
            for sub in q.sub_questions:
                if sub.sub_question_no not in existing_subs:
                    existing_subs[sub.sub_question_no] = sub
                else:
                    esub = existing_subs[sub.sub_question_no]
                    addition = sub.answer or sub.whole_answer or ""
                    if addition:
                        current = esub.answer or esub.whole_answer or ""
                        if addition not in current:   # avoid duplicating same text
                            if esub.answer is not None:
                                esub.answer = f"{current} {addition}".strip()
                            else:
                                esub.whole_answer = f"{current} {addition}".strip()
            existing.sub_questions = list(existing_subs.values())

            # Keep first non-null total_marks
            if existing.total_marks is None and q.total_marks is not None:
                existing.total_marks = q.total_marks

            # Union key_points (preserve insertion order, no duplicates)
            existing.key_points = list(dict.fromkeys(existing.key_points + q.key_points))

            # Keep first non-null ocr_confidence
            if existing.ocr_confidence is None and q.ocr_confidence is not None:
                existing.ocr_confidence = q.ocr_confidence

        return [merged[k] for k in sorted(merged.keys())]
