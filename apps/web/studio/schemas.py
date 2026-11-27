"""Request and response schemas of the API (also documented at /docs)."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_INSTRUCTION = 500


class RouteRequest(BaseModel):
    instruction: str = Field("", max_length=MAX_INSTRUCTION)


class ProcessRequest(BaseModel):
    image_id: str | None = Field(None, description="ID returned by POST /api/upload")
    sample_id: str | None = Field(None, description="ID of a dev_split sample (GET /api/samples)")
    instruction: str = Field(..., max_length=MAX_INSTRUCTION, description="The edit request, English or Vietnamese")
    models: list[str] | None = Field(None, description="Run these model IDs instead of the routed ones",
                                     max_length=40)

    @field_validator("instruction")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("the instruction is empty")
        return value

    @model_validator(mode="after")
    def one_image(self):
        if bool(self.image_id) == bool(self.sample_id):
            raise ValueError("give exactly one of image_id (an upload) or sample_id (a dev_split sample)")
        return self


class UploadResponse(BaseModel):
    image_id: str
    url: str
    width: int
    height: int
    filename: str
    bytes: int


class Score(BaseModel):
    gain: float | None = Field(None, description="Score of the edit minus the score of the unedited input")
    raw: float | None = None
    baseline: float | None = None
    scorer: str


class Candidate(BaseModel):
    model_id: str
    name: str
    date: str
    categories: list[str]
    backend: Literal["real", "mock"]
    status: Literal["ok", "failed"]
    rank: int | None = None
    elapsed_ms: float
    vram_gb_estimate: float | None = None
    image_url: str | None = None
    thumb_url: str | None = None
    width: int | None = None
    height: int | None = None
    transparent: bool = False
    note: str | None = None
    error: str | None = None
    score: Score | None = None
    proxy: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None


class ProcessResponse(BaseModel):
    run_id: str
    instruction: str
    route: dict[str, Any]
    original: dict[str, Any]
    scorer: dict[str, Any]
    best: Candidate | None
    candidates: list[Candidate]
    counts: dict[str, int]
    timings_ms: dict[str, float]
    created_at: str


class JobCreated(BaseModel):
    job_id: str
    models: list[str]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "scoring", "done", "error"]
    progress: dict[str, str]
    done: int
    total: int
    error: str | None = None
    result: ProcessResponse | None = None
