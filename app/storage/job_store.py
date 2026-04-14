import asyncio
from typing import Dict

from app.models.schemas import JobState, JobStatus


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._lock = asyncio.Lock()

    async def create(self, job_id: str) -> JobState:
        async with self._lock:
            state = JobState(job_id=job_id, status=JobStatus.queued, progress_message="Queued")
            self._jobs[job_id] = state
            return state

    async def update(self, job_id: str, **kwargs) -> JobState | None:
        async with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return None
            data = state.model_dump()
            data.update(kwargs)
            updated = JobState.model_validate(data)
            self._jobs[job_id] = updated
            return updated

    async def get(self, job_id: str) -> JobState | None:
        async with self._lock:
            return self._jobs.get(job_id)

