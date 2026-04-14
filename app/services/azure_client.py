"""
Azure OpenAI client wrapper.

Design decisions:
- Retries are intentionally NOT done inside this synchronous function.
  This is because `chat_json` is called via `asyncio.to_thread`, and
  `time.sleep` inside that thread blocks the thread pool but NOT the event
  loop.  However, async-level retries in `extraction_service.py` (using
  `asyncio.sleep`) are cleaner and release the event loop between attempts.
- This function performs ONE call and raises on failure so that the caller
  can decide retry/abort strategy.
- `response_format={"type": "json_object"}` is used so Azure GPT-4o always
  returns valid JSON (no markdown fences needed in most cases, but we still
  strip them as a safety measure).
"""

import json
import logging
import time
from typing import Any

from openai import AzureOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class AzureOpenAIService:
    """Synchronous Azure OpenAI wrapper intended to be called via asyncio.to_thread."""

    def __init__(self) -> None:
        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retries handled at async level in extraction_service
        )
        self._model = settings.azure_openai_deployment
        logger.info(
            "AzureOpenAI client ready | deployment=%s | endpoint=%s | timeout=%ds",
            self._model,
            settings.azure_openai_endpoint,
            settings.llm_timeout_seconds,
        )

    @property
    def model(self) -> str:
        return self._model

    def chat_json(
        self,
        system_prompt: str,
        user_content: Any,
        max_tokens: int = 4000,
        label: str = "",
    ) -> dict[str, Any]:
        """
        Single Azure OpenAI chat completion call → parsed JSON dict.

        Raises on any error so that the async retry wrapper in
        ExtractionService can handle retry with asyncio.sleep.

        Args:
            system_prompt: System instruction for the model.
            user_content:  String or list-of-content-parts (for vision).
            max_tokens:    Maximum output tokens.
            label:         Short identifier shown in logs (e.g. 'scheme-batch3').
        """
        tag = f"[{label}] " if label else ""
        logger.info("%sSending request to Azure GPT-4o | model=%s | max_tokens=%d", tag, self._model, max_tokens)
        t_start = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
            elapsed = round(time.time() - t_start, 1)
            raw = (response.choices[0].message.content or "").strip()
            tokens_used = response.usage.total_tokens if response.usage else "?"
            logger.info(
                "%sAzure GPT-4o SUCCESS in %.1fs | tokens=%s | response_chars=%d",
                tag, elapsed, tokens_used, len(raw),
            )
            return json.loads(self._strip_json_fence(raw))

        except json.JSONDecodeError as exc:
            elapsed = round(time.time() - t_start, 1)
            logger.error("%sJSON parse error after %.1fs: %s", tag, elapsed, exc)
            raise

        except Exception as exc:
            elapsed = round(time.time() - t_start, 1)
            logger.error(
                "%s%s after %.1fs: %s",
                tag, type(exc).__name__, elapsed, exc,
            )
            raise

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        """Remove ```json ... ``` fences that the model sometimes adds."""
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                payload = parts[1]
                if payload.lstrip().startswith("json"):
                    payload = payload.lstrip()[4:]
                return payload.strip()
        return text
