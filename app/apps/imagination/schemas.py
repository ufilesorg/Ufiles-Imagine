from datetime import datetime
from typing import Any, Generator

from apps.ai.schemas import ImaginationEngines, ImaginationStatus
from fastapi_mongo_base.schemas import OwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel, field_validator, model_validator


class ImagineCreateSchema(BaseModel):
    # prompt: str | None = None
    engine: ImaginationEngines = ImaginationEngines.midjourney
    aspect_ratio: str | None = "1:1"
    # sync: bool | None = False
    delineation: str | None = None
    context: list[dict[str, Any]] | None = None
    enhance_prompt: bool = False
    # number: int = 1

    @model_validator(mode="after")
    def validate_data(cls, values: "ImagineCreateSchema"):
        engine = values.engine
        validated, message = engine.get_class(None).validate(values)

        if not validated:
            raise ValueError(message)
        return values


class ImagineResponse(BaseModel):
    url: str
    width: int
    height: int


class ImagineSchema(TaskMixin, OwnedEntitySchema):
    prompt: str | None = None
    delineation: str | None = None
    context: list[dict[str, Any]] | None = None
    error: Any | None = None

    engine: ImaginationEngines = ImaginationEngines.midjourney
    aspect_ratio: str | None = "1:1"
    enhance_prompt: bool = False

    bulk: str | None = None

    status: ImaginationStatus = ImaginationStatus.init
    results: list[ImagineResponse] | None = None


class ImagineWebhookData(BaseModel):
    prompt: str
    status: ImaginationStatus
    percentage: int
    result: dict[str, Any] | None = None
    error: Any | None = None

    @field_validator("percentage", mode="before")
    def validate_percentage(cls, value):
        if value is None:
            return -1
        if isinstance(value, str):
            return int(value.replace("%", ""))
        if value < -1:
            return -1
        if value > 100:
            return 100
        return value


class ImagineBulkResponse(BaseModel):
    url: str
    width: int
    height: int
    engine: ImaginationEngines


class ImagineBulkError(BaseModel):
    engine: ImaginationEngines
    message: str


class ImagineBulkSchema(TaskMixin, OwnedEntitySchema):
    prompt: str | None = None
    delineation: str | None = None
    context: list[dict[str, Any]] | None = None
    enhance_prompt: bool = False

    completed_at: datetime | None = None
    total_tasks: int = 0
    total_completed: int = 0
    total_failed: int = 0
    results: list[ImagineBulkResponse] = []
    errors: list[ImagineBulkError] = []
    aspect_ratios: list[str]
    engines: list[ImaginationEngines] = ImaginationEngines.bulk_engines

    def get_combinations(
        self,
    ) -> Generator[tuple[str, ImaginationEngines], None, None]:
        for ar, e in zip(self.aspect_ratios, self.engines):
            yield ar, e


class ImagineCreateBulkSchema(BaseModel):
    prompt: str | None = None
    delineation: str | None = None
    context: list[dict[str, Any]] | None = None
    enhance_prompt: bool = False
    sync: bool = False
    aspect_ratios: list[str] = []
    engines: list[ImaginationEngines] = ImaginationEngines.bulk_engines
    webhook_url: str | None = None

    @model_validator(mode="after")
    def validate_data(cls, values: "ImagineCreateBulkSchema"):
        values.aspect_ratios = (
            values.aspect_ratios
            if len(values.aspect_ratios) == len(values.engines)
            else ["1:1" for _ in values.engines]
        )
        for ar, engine in zip(values.aspect_ratios, values.engines):
            data = values.model_dump()
            data["aspect_ratio"] = ar
            data = ImagineCreateSchema(**data, engine=engine)

        return values

    def get_combinations(
        self,
    ) -> Generator[tuple[str, ImaginationEngines], None, None]:
        for ar, e in zip(self.aspect_ratios, self.engines):
            yield ar, e
