from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator
from server.config import Settings
from singleton import Singleton

from .schemas import ImaginationStatus


# Required data
class EnginesResponse(BaseModel):
    id: str | None
    error: str | None = None
    prompt: str
    status: ImaginationStatus
    percentage: int | None = None
    result: dict | None = None

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


class BaseEngine(metaclass=Singleton):
    def __init__(self) -> None:
        self.name = self.__class__.__name__.lower()

    @property
    def supported_aspect_ratios(self):
        return {"1:1"}

    @property
    def thumbnail_url(self):
        raise NotImplementedError("thumbnail_url is not implemented")

    @property
    def price(self):
        raise NotImplementedError("price is not implemented")

    # Get Result from service(client / API)
    async def result(self, item, **kwargs) -> EnginesResponse:
        pass

    # Validate schema
    def validate(self, item) -> tuple[bool, str | None]:
        aspect_ratio_valid = item.aspect_ratio in self.supported_aspect_ratios
        message = (
            f"aspect_ratio must be one of them {self.supported_aspect_ratios}"
            if not aspect_ratio_valid
            else None
        )
        return aspect_ratio_valid, message

    # Send request to service(client / API)
    async def _request(self, **kwargs) -> EnginesResponse:
        pass

    # Get current request service(Convert service status to ImaginationStatus)
    def _status(self, status: str) -> ImaginationStatus:
        return {
            "initialized": ImaginationStatus.init,
            "queue": ImaginationStatus.queue,
            "waiting": ImaginationStatus.waiting,
            "running": ImaginationStatus.processing,
            "completed": ImaginationStatus.completed,
            "error": ImaginationStatus.error,
        }.get(status, ImaginationStatus.error)

    async def imagine(self, item, **kwargs) -> EnginesResponse:
        response = await self._request(item, **kwargs)
        return response

    # Convert service response to EnginesDetails
    async def _result_to_details(self, res) -> EnginesResponse:
        pass


class Engine:  # (metaclass=Singleton):
    def __init__(self, item, *, name=None, engine=None) -> None:
        self.item = item
        self.name = name or engine
        self.engine = name or engine
        # self.name = self.__class__.__name__.lower()

    @property
    def supported_aspect_ratios(self):
        return {"1:1"}

    @property
    def thumbnail_url(self):
        raise NotImplementedError("thumbnail_url is not implemented")

    @property
    def price(self):
        raise NotImplementedError("price is not implemented")

    # Get Result from service(client / API)
    async def result(self, **kwargs) -> EnginesResponse:
        pass

    # Validate schema
    def validate(self, data: BaseModel) -> tuple[bool, str | None]:
        return True, None

    # Send request to service(client / API)
    async def _request(self, **kwargs) -> EnginesResponse:
        pass

    # Get property from item meta_data
    # item.meta_data: It is a response sent from the service
    def _get_data(self, name: str, **kwargs):
        value = (self.item.meta_data or {}).get(name, None)
        if value is None:
            raise ValueError(f"Missing value {name}")
        return value

    # Get current request service(Convert service status to ImaginationStatus)
    def _status(self, status: str) -> ImaginationStatus:
        pass

    async def imagine(self, **kwargs):
        response = await self._request(**kwargs)
        return response

    # Convert service response to EnginesDetails
    async def _result_to_details(self, res) -> EnginesResponse:
        pass


class ImaginationEngines(str, Enum):
    midjourney = "midjourney"
    ideogram = "ideogram"
    flux_schnell = "flux_schnell"
    stability = "stability"
    flux_1_1 = "flux_1.1"
    dalle = "dalle"
    flux = "flux"
    photon = "photon"
    photon_flash = "photon_flash"
    # leonardo = "leonardo"

    @classmethod
    def bulk_engines(
        cls, aspect_ratio: list[str] = ["1:1"]
    ) -> list["ImaginationEngines"]:
        target_engines = [
            ImaginationEngines.midjourney,
            ImaginationEngines.dalle,
            ImaginationEngines.flux_1_1,
            ImaginationEngines.ideogram,
            ImaginationEngines.photon,
        ]
        return [
            engine
            for engine in target_engines
            if all(ar in engine.supported_aspect_ratios for ar in aspect_ratio)
        ]

    @property
    def metis_bot_id(self):
        return {
            ImaginationEngines.dalle: Settings.METIS_DALLE_BOT_ID,
        }[self]

    def get_class(self, imagination=None) -> BaseEngine:
        from .dalle import BaseDalle
        from .midjourney import BaseMidjourney
        from .replicate_engine import (
            BaseReplicate,
            Photon,
            PhotonFlash,
            Flux11,
            FluxSchnell,
            Ideogram,
            StableDiffusion3,
        )

        return {
            ImaginationEngines.dalle: BaseDalle(),
            ImaginationEngines.midjourney: BaseMidjourney(),
            ImaginationEngines.ideogram: Ideogram(),
            ImaginationEngines.flux_schnell: FluxSchnell(),
            ImaginationEngines.stability: StableDiffusion3(),
            ImaginationEngines.flux_1_1: Flux11(),
            ImaginationEngines.flux: Flux11(),
            ImaginationEngines.photon: Photon(),
            ImaginationEngines.photon_flash: PhotonFlash(),
        }[self]

    @property
    def thumbnail_url(self):
        return {
            ImaginationEngines.dalle: "https://media.pixiee.io/v1/f/41af8b03-b4df-4b2f-ba52-ea638d10b5f3/dalle-icon.png?width=100",
            ImaginationEngines.midjourney: "https://media.pixiee.io/v1/f/4a0980aa-8d97-4493-bdb1-fb3d67d891e3/midjourney-icon.png?width=100",
            ImaginationEngines.ideogram: "https://media.pixiee.io/v1/f/19d4df43-ea1e-4562-a8e1-8ee301bd0a88/ideogram-icon.png?width=100",
            ImaginationEngines.flux_schnell: "https://media.pixiee.io/v1/f/cf21c500-6e84-4915-a5d1-19b8f325a382/flux-icon.png?width=100",
            ImaginationEngines.flux: "https://media.pixiee.io/v1/f/cf21c500-6e84-4915-a5d1-19b8f325a382/flux-icon.png?width=100",
            ImaginationEngines.stability: "https://media.pixiee.io/v1/f/6d0a2e82-7667-46ec-af33-0e557f16e356/stability-icon.png?width=100",
            ImaginationEngines.flux_1_1: "https://media.pixiee.io/v1/f/cf21c500-6e84-4915-a5d1-19b8f325a382/flux-icon.png?width=100",
            ImaginationEngines.photon: "https://media.pixiee.io/v1/f/4701330c-aa98-4d86-91d4-982ff94d30f3/photon.png?width=100",
            ImaginationEngines.photon_flash: "https://media.pixiee.io/v1/f/4701330c-aa98-4d86-91d4-982ff94d30f3/photon.png?width=100",
        }[self]

    @property
    def supported_aspect_ratios(self):
        return {
            ImaginationEngines.ideogram: {
                "1:1",
                "16:9",
                "9:16",
                "4:3",
                "3:4",
                "3:2",
                "2:3",
                "16:10",
                "10:16",
                "3:1",
                "1:3",
            },
            ImaginationEngines.flux_schnell: {
                "1:1",
                "16:9",
                "21:9",
                "3:2",
                "2:3",
                "4:5",
                "5:4",
                "3:4",
                "4:3",
                "9:16",
                "9:21",
            },
            ImaginationEngines.flux_1_1: {
                "1:1",
                "16:9",
                "2:3",
                "3:2",
                "4:5",
                "5:4",
                "9:16",
                "3:4",
                "4:3",
            },
            ImaginationEngines.stability: {
                "1:1",
                "16:9",
                "21:9",
                "3:2",
                "2:3",
                "4:5",
                "5:4",
                "9:16",
                "9:21",
            },
            ImaginationEngines.dalle: {"1:1", "7:4", "4:7"},
            ImaginationEngines.midjourney: {
                "10:16",
                "16:10",
                "16:9",
                "1:1",
                "1:3",
                "21:9",
                "2:3",
                "3:1",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "4:7",
                "5:4",
                "7:4",
                "9:16",
                "9:21",
            },
            ImaginationEngines.flux: {
                "1:1",
                "16:9",
                "21:9",
                "3:2",
                "2:3",
                "4:5",
                "5:4",
                "3:4",
                "4:3",
                "9:16",
                "9:21",
            },
            ImaginationEngines.photon: {
                "1:1",
                "16:9",
                "21:9",
                "4:3",
                "3:4",
                "9:16",
                "9:21",
            },
            ImaginationEngines.photon_flash: {
                "1:1",
                "16:9",
                "21:9",
                "4:3",
                "3:4",
                "9:16",
                "9:21",
            },
        }[self]

    @property
    def price(self):
        return 10_000


class ImaginationEnginesSchema(BaseModel):
    engine: ImaginationEngines = ImaginationEngines.midjourney
    thumbnail_url: str
    price: float
    supported_aspect_ratios: set

    @classmethod
    def from_model(cls, model: ImaginationEngines):
        return cls(
            engine=model,
            thumbnail_url=model.thumbnail_url,
            price=model.price,
            supported_aspect_ratios=model.supported_aspect_ratios,
        )
