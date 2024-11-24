from enum import Enum
from typing import Any

from fastapi_mongo_base.tasks import TaskStatusEnum
from pydantic import BaseModel


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


class ImaginationEngines(str, Enum):
    midjourney = "midjourney"
    ideogram = "ideogram"
    flux_schnell = "flux_schnell"
    stability = "stability"
    flux_1_1 = "flux_1.1"
    # flux = "flux"
    dalle = "dalle"
    # leonardo = "leonardo"

    @property
    def metis_bot_id(self):
        return {
            # ImaginationEngines.midjourney: "1fff8298-4f56-4912-89b4-3529106c5a0a",
            # ImaginationEngines.flux: "68ec6038-6701-4f9b-a3f5-0c674b106f0e",
            ImaginationEngines.dalle: "59631f67-3199-4e47-af7f-18eb44f69ea2",
            # ImaginationEngines.leonardo: "4b69d78d-e454-4f4b-93ed-427b46368977",
        }[self]

    def get_class(self, imagination: Any):
        from .dalle import Dalle
        from .midjourney import Midjourney
        from .replicate_engine import Replicate

        return {
            ImaginationEngines.dalle: lambda: Dalle(imagination),
            ImaginationEngines.midjourney: lambda: Midjourney(imagination),
            ImaginationEngines.ideogram: lambda: Replicate(imagination, self.value),
            ImaginationEngines.flux_schnell: lambda: Replicate(imagination, self.value),
            ImaginationEngines.stability: lambda: Replicate(imagination, self.value),
            ImaginationEngines.flux_1_1: lambda: Replicate(imagination, self.value),
        }[self]()

    @property
    def thumbnail_url(self):
        return {
            ImaginationEngines.dalle: "",
            ImaginationEngines.midjourney: "",
            ImaginationEngines.ideogram: "",
            ImaginationEngines.flux_schnell: "",
            ImaginationEngines.stability: "",
            ImaginationEngines.flux_1_1: "",
        }[self]

    @property
    def price(self):
        return 0.1


class ImaginationEnginesSchema(BaseModel):
    engine: ImaginationEngines = ImaginationEngines.midjourney
    thumbnail_url: str
    price: float

    @classmethod
    def from_model(cls, model: ImaginationEngines):
        return cls(engine=model, thumbnail_url=model.thumbnail_url, price=model.price)
