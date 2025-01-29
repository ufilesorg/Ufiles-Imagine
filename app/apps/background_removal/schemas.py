from enum import Enum
from typing import Any

from fastapi_mongo_base.schemas import OwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel

from apps.ai.replicate_schemas import PredictionModelWebhookData
from apps.imagination.schemas import ImaginationStatus, ImagineResponse


class BackgroundRemovalEngines(str, Enum):
    cjwbw = "cjwbw"
    lucataco = "lucataco"
    pollinations = "pollinations"

    def get_class(self, background_removal: Any):
        from .ai import (
            CjwbwReplicateBackgroundRemoval,
            LucatacoReplicateBackgroundRemoval,
            PollinationsReplicateBackgroundRemoval,
        )

        return {
            BackgroundRemovalEngines.cjwbw: lambda: CjwbwReplicateBackgroundRemoval(
                background_removal
            ),
            BackgroundRemovalEngines.lucataco: lambda: LucatacoReplicateBackgroundRemoval(
                background_removal
            ),
            BackgroundRemovalEngines.pollinations: lambda: PollinationsReplicateBackgroundRemoval(
                background_removal
            ),
        }[self]()

    @property
    def thumbnail_url(self):
        return {
            BackgroundRemovalEngines.cjwbw: "https://media.pixiee.io/v1/f/c3044ffe-8fb5-410c-b5e2-3939a9140266/cjwbw-icon.png",
            BackgroundRemovalEngines.lucataco: "https://media.pixiee.io/v1/f/2a6a6b6d-45ac-486b-861d-e00477d52ac4/lucataco-icon.png",
            BackgroundRemovalEngines.pollinations: "https://media.pixiee.io/v1/f/aa6f73c6-ab73-48bf-8b96-1a83aa238a1c/pollinations-icon.png",
        }[self]

    @property
    def price(self):
        return 0.1


class BackgroundRemovalEnginesSchema(BaseModel):
    engine: BackgroundRemovalEngines = BackgroundRemovalEngines.lucataco
    thumbnail_url: str
    price: float

    @classmethod
    def from_model(cls, model: BackgroundRemovalEngines):
        return cls(engine=model, thumbnail_url=model.thumbnail_url, price=model.price)


class BackgroundRemovalCreateSchema(BaseModel):
    engine: BackgroundRemovalEngines = BackgroundRemovalEngines.lucataco
    image_url: str


class BackgroundRemovalSchema(
    BackgroundRemovalCreateSchema, TaskMixin, OwnedEntitySchema
):
    status: ImaginationStatus = ImaginationStatus.draft
    result: ImagineResponse | None = None


class BackgroundRemovalWebhookData(PredictionModelWebhookData):
    pass
