# Semantic Evaluation MVP (FastAPI + Azure GPT-4o)

This project provides an MVP API for:
- Uploading a marking scheme PDF and a student handwritten answer sheet PDF
- Running Azure GPT-4o vision extraction for both PDFs in parallel
- Building structured JSON for both papers
- Evaluating answers using multi-technique semantic scoring
- Returning marks with transparency metrics and confidence

## Features

- Parallel extraction: marking scheme + student sheet
- Azure GPT-4o vision based OCR and schema generation
- Multiple semantic techniques:
  - sentence embedding similarity
  - NLI entailment scoring
  - key-point coverage
  - optional LLM semantic grading
- Evaluation modes:
  - `fast_semantic`
  - `balanced`
  - `llm_heavy`
- Async job API:
  - create job
  - poll status
  - fetch results

## Setup

1. Create virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` (already added template) and set values:

```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

3. Run API:

```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `GET /` (frontend UI)
- `GET /health`
- `POST /api/evaluate` (multipart form)
  - `marking_scheme_pdf` (file)
  - `student_answer_pdf` (file)
  - `mode` (`fast_semantic`, `balanced`, `llm_heavy`)
- `GET /api/jobs/{job_id}/status`
- `GET /api/jobs/{job_id}/result`

## Notes

- This MVP is tuned for English-only exam papers.
- For best latency, use `fast_semantic` during demos.
- For better semantic depth, use `balanced` or `llm_heavy`.
- Confidence fields help decide auto-grade vs manual review.
- Demo shortcut mode is enabled by default (`USE_DEMO_JSON_SHORTCUT=true`):
  - API still accepts uploaded PDFs in frontend flow.
  - Backend bypasses OCR extraction and directly loads:
    - `full_populated_marking_scheme.json`
    - `student_answers (2).json`
  - Marking-scheme answers are embedded and indexed in ChromaDB, then retrieved by `question_no` for evaluation.

