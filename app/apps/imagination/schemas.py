import itertools
import uuid
from datetime import datetime
from typing import Any, Generator

from apps.ai.engine import ImaginationEngines
from apps.ai.schemas import ImaginationStatus
from fastapi_mongo_base.schemas import OwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel, field_validator, model_validator


class ImagineCreateSchema(BaseModel):
    engine: ImaginationEngines = ImaginationEngines.midjourney
    aspect_ratio: str | None = "1:1"

    delineation: str | None = None
    context: list[dict[str, Any]] | None = None
    enhance_prompt: bool = False

    @model_validator(mode="after")
    def validate_data(cls, values: "ImagineCreateSchema"):
        engine = values.engine
        validated, message = engine.get_class().validate(values)

        if not validated:
            raise ValueError(message)
        return values


class ImagineResponse(BaseModel):
    url: str
    width: int
    height: int


class ImagineSchema(ImagineCreateSchema, TaskMixin, OwnedEntitySchema):
    prompt: str | None = None
    error: Any | None = None

    bulk: str | None = None
    status: ImaginationStatus = ImaginationStatus.init
    results: list[ImagineResponse] | None = None
    usage_id: uuid.UUID | None = None


class MidjourneyWebhookData(BaseModel):
    prompt: str
    status: ImaginationStatus
    percentage: int
    result: dict[str, Any] | None = None
    error: Any | None = None

    @field_validator("status", mode="before")
    def validate_status(cls, value):
        return ImaginationStatus.from_midjourney(value)

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


class ImagineCreateBulkSchema(BaseModel):
    delineation: str | None = None
    context: list[dict[str, Any]] | None = None
    enhance_prompt: bool = False

    aspect_ratios: list[str] = ["1:1"]
    engines: list[ImaginationEngines] | None = None
    webhook_url: str | None = None
    number_of_tasks: int = 1

    @model_validator(mode="after")
    def validate_engines(cls, item: "ImagineCreateBulkSchema"):
        if item.engines is None:
            all_aspect_ratios = [ar for ar in item.aspect_ratios if ar is not None]
            item.engines = ImaginationEngines.bulk_engines(all_aspect_ratios)
        return item

    @field_validator("aspect_ratios", mode="before")
    def validate_aspect_ratios(cls, value):
        if isinstance(value, str):
            return [value]
        return value

    def get_combinations(
        self,
    ) -> Generator[tuple[str, ImaginationEngines], None, None]:
        for ar, e in itertools.product(self.aspect_ratios, self.engines):
            if ar not in e.supported_aspect_ratios:
                continue
            yield ar, e

    @property
    def total_price(self):
        return sum(e.price for _, e in self.get_combinations())


class ImagineBulkSchema(ImagineCreateBulkSchema, TaskMixin, OwnedEntitySchema):
    prompt: str | None = None

    completed_at: datetime | None = None
    total_tasks: int = 0
    total_completed: int = 0
    total_failed: int = 0
    results: list[ImagineBulkResponse] | None = []
    errors: list[ImagineBulkError] | None = []
