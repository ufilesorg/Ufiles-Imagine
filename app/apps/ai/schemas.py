from enum import Enum

from fastapi_mongo_base.tasks import TaskStatusEnum
from pydantic import BaseModel, field_validator


class ImaginationStatus(str, Enum):
    none = "none"
    draft = "draft"
    init = "init"
    queue = "queue"
    waiting = "waiting"
    running = "running"
    processing = "processing"
    done = "done"
    completed = "completed"
    error = "error"
    cancelled = "cancelled"

    @property
    def progress(self):
        return {
            ImaginationStatus.processing: 50,
            ImaginationStatus.done: 100,
        }.get(self, 0)

    @classmethod
    def from_midjourney(cls, status: str):
        return {
            "initialized": ImaginationStatus.init,
            "queue": ImaginationStatus.queue,
            "waiting": ImaginationStatus.waiting,
            "running": ImaginationStatus.processing,
            "completed": ImaginationStatus.completed,
            "error": ImaginationStatus.error,
        }.get(status, ImaginationStatus.error)

    @classmethod
    def from_replicate(cls, status: str):
        return {
            "processing": ImaginationStatus.processing,
            "succeeded": ImaginationStatus.completed,
            "completed": ImaginationStatus.completed,
            "error": ImaginationStatus.error,
        }.get(status, ImaginationStatus.error)

    @classmethod
    def done_statuses(cls):
        return [
            status.value
            for status in [
                ImaginationStatus.done,
                ImaginationStatus.completed,
                ImaginationStatus.cancelled,
                ImaginationStatus.error,
            ]
        ]

    @property
    def task_status(self):
        return {
            ImaginationStatus.none: TaskStatusEnum.none,
            ImaginationStatus.draft: TaskStatusEnum.draft,
            ImaginationStatus.init: TaskStatusEnum.init,
            ImaginationStatus.queue: TaskStatusEnum.processing,
            ImaginationStatus.waiting: TaskStatusEnum.processing,
            ImaginationStatus.running: TaskStatusEnum.processing,
            ImaginationStatus.processing: TaskStatusEnum.processing,
            ImaginationStatus.done: TaskStatusEnum.completed,
            ImaginationStatus.completed: TaskStatusEnum.completed,
            ImaginationStatus.error: TaskStatusEnum.error,
            ImaginationStatus.cancelled: TaskStatusEnum.completed,
        }[self]

    @property
    def is_done(self):
        return self in (
            ImaginationStatus.done,
            ImaginationStatus.completed,
            ImaginationStatus.error,
            ImaginationStatus.cancelled,
        )


class EnginesDetails(BaseModel):
    id: str | None
    error: str | None = None
    prompt: str
    status: ImaginationStatus
    percentage: int | None = None
    result: dict | None = None

    @field_validator("status", mode="before")
    def validate_status(cls, value):
        mid = ImaginationStatus.from_midjourney(value)
        rep = ImaginationStatus.from_replicate(value)
        return mid if mid != ImaginationStatus.error else rep

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
