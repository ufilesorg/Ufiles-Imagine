from datetime import datetime

from pydantic import BaseModel, field_validator, model_validator

from .schemas import ImaginationStatus


class PredictionModelWebhookData(BaseModel):
    completed_at: datetime | None = None
    created_at: datetime
    data_removed: bool = False
    error: str | None = None
    id: str
    input: dict | None = None
    logs: str | None = None
    metrics: dict | None = None
    model: str
    output: str | None = None
    started_at: datetime | None = None
    status: ImaginationStatus
    percentage: int = 0
    urls: dict[str, str] | None = None
    version: str
    webhook: str | None = None
    webhook_events_filter: list[str] | None = None

    @field_validator("status", mode="before")
    def validate_status(cls, value):
        return ImaginationStatus.from_replicate(value)

    @model_validator(mode="after")
    def validate_percentage(cls, item: "PredictionModelWebhookData"):
        item.percentage = item.status.progress
        return item
