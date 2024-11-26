from typing import Any, Literal, Generator

from apps.ai.schemas import ImaginationEngines, ImaginationStatus
from fastapi_mongo_base.schemas import OwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel, field_validator, model_validator


class ImagineCreateSchema(BaseModel):
    prompt: str | None = None
    engine: ImaginationEngines = ImaginationEngines.midjourney
    aspect_ratio: str | None = "1:1"
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
    aspect_ratio: str | None = "1:1"
    context: list[dict[str, Any]] | None = None
    engine: ImaginationEngines = ImaginationEngines.midjourney
    mode: Literal["imagine"] = "imagine"
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


class ImagineBulkSchema(TaskMixin, OwnedEntitySchema):
    prompt: str | None = None
    delineation: str | None = None
    context: list[dict[str, Any]] | None = None

    aspect_ratios: list[str] = ["1:1"]
    engines: list[ImaginationEngines] = ImaginationEngines.midjourney
    number: int = 1

    imaginations: list[ImagineSchema] | None = None

    results: list[ImagineResponse] | None = None

    def get_combinations(self) -> Generator[tuple[int, str, ImaginationEngines]]:
        from itertools import product

        for i, ar, e in product(range(self.number), self.aspect_ratios, self.engines):
            yield i, ar, e


class ImagineCreateBulkSchema(BaseModel):
    prompt: str | None = None
    aspect_ratio: str | None = "1:1"
    enhance_prompt: bool = False
    engines: list[ImaginationEngines] = ImaginationEngines.bulk_engines()
    number: int = 1
    webhook_url: str | None = None
