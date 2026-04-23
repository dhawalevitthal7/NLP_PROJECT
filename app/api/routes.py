
import asyncio
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import EvaluationMode, JobStatus
from app.services.azure_client import AzureOpenAIService
from app.services.evaluation_service import EvaluationOrchestrator
from app.services.extraction_service import ExtractionService
from app.services.scoring_service import SemanticScoringService
from app.storage.job_store import InMemoryJobStore

router = APIRouter(prefix="/api", tags=["evaluation"])

job_store = InMemoryJobStore()
azure_service = AzureOpenAIService()
extraction_service = ExtractionService(azure_service)
scoring_service = SemanticScoringService(azure_service)
orchestrator = EvaluationOrchestrator(extraction_service, scoring_service)


@router.post("/evaluate")
async def evaluate(
    marking_scheme_pdf: UploadFile = File(...),
    student_answer_pdf: UploadFile = File(...),
    mode: EvaluationMode = EvaluationMode.balanced,
):
    if not marking_scheme_pdf.filename or not marking_scheme_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="marking_scheme_pdf must be a PDF")
    if not student_answer_pdf.filename or not student_answer_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="student_answer_pdf must be a PDF")

    job_id = str(uuid.uuid4())
    await job_store.create(job_id)
    await job_store.update(job_id, status=JobStatus.processing, progress_message="Uploading input files")

    tmp_dir = tempfile.mkdtemp(prefix=f"eval_{job_id}_")
    scheme_path = str(Path(tmp_dir) / "marking_scheme.pdf")
    student_path = str(Path(tmp_dir) / "student_answer_sheet.pdf")

    with open(scheme_path, "wb") as f:
        f.write(await marking_scheme_pdf.read())
    with open(student_path, "wb") as f:
        f.write(await student_answer_pdf.read())

    async def _runner() -> None:
        try:
            await job_store.update(job_id, progress_message="Running OCR and extraction in parallel")
            result = await orchestrator.evaluate(scheme_path, student_path, mode=mode)
            await job_store.update(
                job_id,
                status=JobStatus.completed,
                progress_message="Completed",
                result=result,
            )
        except Exception as exc:
            await job_store.update(job_id, status=JobStatus.failed, error=str(exc), progress_message="Failed")
        finally:
            for path in [scheme_path, student_path]:
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    asyncio.create_task(_runner())
    return {"job_id": job_id, "status": JobStatus.processing, "message": "Evaluation job started"}


@router.get("/jobs/{job_id}/status")
async def get_status(job_id: str):
    state = await job_store.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": state.job_id,
        "status": state.status,
        "progress_message": state.progress_message,
        "error": state.error,
    }


@router.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    state = await job_store.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.status != JobStatus.completed:
        return {
            "job_id": state.job_id,
            "status": state.status,
            "progress_message": state.progress_message,
            "error": state.error,
        }
    return state.result

