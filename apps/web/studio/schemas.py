"""Request and response schemas of the API (also documented at /docs)."""

from pydantic import BaseModel, Field, field_validator, model_validator

from .catalog import TASKS

MAX_INSTRUCTION = 100      # the request box allows 100 characters


class JobRequest(BaseModel):
    image_id: str | None = Field(None, description="ID returned by POST /api/upload")
    sample_id: str | None = Field(None, description="ID of a sample photo (GET /api/samples)")
    instruction: str = Field(..., max_length=MAX_INSTRUCTION, description="The edit request, English or Vietnamese")
    task: str | None = Field(None, description="Use this task instead of the one read from the request: "
                                               + ", ".join(TASKS))

    @field_validator("instruction")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("the instruction is empty")
        return value

    @field_validator("task")
    @classmethod
    def known_task(cls, value: str | None) -> str | None:
        if value is not None and value not in TASKS:
            raise ValueError(f"unknown task {value!r}; one of {', '.join(TASKS)}")
        return value

    @model_validator(mode="after")
    def one_image(self):
        if bool(self.image_id) == bool(self.sample_id):
            raise ValueError("give exactly one of image_id (an upload) or sample_id (a sample photo)")
        return self


class UploadResponse(BaseModel):
    image_id: str
    url: str
    width: int
    height: int
    filename: str
    bytes: int
