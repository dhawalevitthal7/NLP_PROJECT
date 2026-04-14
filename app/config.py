"""
Application configuration loaded from environment / .env file.

Values with defaults can be overridden in .env:
  IMAGE_DPI                  — PDF rendering DPI (150 is optimal; higher = larger images = slower)
  PAGE_BATCH_SIZE            — Pages sent per Azure API call (default 2)
  MAX_CONCURRENT_AZURE_CALLS — Max simultaneous Azure calls (default 3)
  LLM_TIMEOUT_SECONDS        — Per-request Azure timeout in seconds (default 300)
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Azure OpenAI credentials (required) ───────────────────────────────
    azure_openai_api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(
        default="2024-12-01-preview", alias="AZURE_OPENAI_API_VERSION"
    )
    azure_openai_deployment: str = Field(default="gpt-4o", alias="AZURE_OPENAI_DEPLOYMENT")

    # ── PDF rendering ──────────────────────────────────────────────────────
    # 150 DPI is enough for GPT-4o high detail; higher values create much
    # larger JPEG payloads and slow down every API call significantly.
    image_dpi: int = Field(default=150, alias="IMAGE_DPI")

    # Pages per Azure API call. 2 is a safe default.
    # Increase to 3 only if your pages are sparse (few words per page).
    page_batch_size: int = Field(default=2, alias="PAGE_BATCH_SIZE")

    # ── Concurrency ────────────────────────────────────────────────────────
    # How many Azure calls run simultaneously across BOTH extraction pipelines.
    # 3 is safe for most Azure deployments without hitting rate limits.
    max_concurrent_azure_calls: int = Field(default=3, alias="MAX_CONCURRENT_AZURE_CALLS")

    # ── Timeout ────────────────────────────────────────────────────────────
    # GPT-4o vision with high-detail images typically responds in 20–60s.
    # 300s gives plenty of headroom even for large/complex pages.
    llm_timeout_seconds: int = Field(default=300, alias="LLM_TIMEOUT_SECONDS")

    # ── Scoring weights for 'balanced' mode ───────────────────────────────
    weight_embedding: float = 0.25
    weight_nli: float = 0.25
    weight_keypoint: float = 0.20
    weight_llm: float = 0.30

    # ── Demo shortcut mode (keeps upload/extraction flow in API, but bypasses OCR) ──
    use_demo_json_shortcut: bool = Field(default=True, alias="USE_DEMO_JSON_SHORTCUT")
    demo_marking_scheme_json: str = Field(
        default="full_populated_marking_scheme.json",
        alias="DEMO_MARKING_SCHEME_JSON",
    )
    demo_student_answers_json: str = Field(
        default="student_answers (2).json",
        alias="DEMO_STUDENT_ANSWERS_JSON",
    )
    chroma_persist_directory: str = Field(default="chroma_db", alias="CHROMA_PERSIST_DIRECTORY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
