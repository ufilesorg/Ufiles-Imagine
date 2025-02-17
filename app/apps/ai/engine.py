from enum import Enum

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

    @property
    def core(self):
        raise NotImplementedError("price is not implemented")

    # Get Result from service(client / API)
    async def result(self, item, **kwargs) -> EnginesResponse:
        return item

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
            "draft": ImaginationStatus.init,
            "init": ImaginationStatus.init,
            "initialized": ImaginationStatus.init,
            "queue": ImaginationStatus.queue,
            "pending": ImaginationStatus.queue,
            "waiting": ImaginationStatus.waiting,
            "running": ImaginationStatus.processing,
            "completed": ImaginationStatus.completed,
            "succeeded": ImaginationStatus.completed,
            "error": ImaginationStatus.error,
        }.get(status, ImaginationStatus.error)

    async def imagine(self, item, **kwargs) -> EnginesResponse:
        response = await self._request(item, **kwargs)
        return response

    # Convert service response to EnginesDetails
    async def _result_to_details(self, res) -> EnginesResponse:
        pass


class ImaginationEngines(str, Enum):
    flux_schnell = "flux_schnell"
    flux_1_1 = "flux_1.1"
    imagen = "imagen"
    photon_flash = "photon_flash"
    photon = "photon"
    dalle = "dalle"
    ideogram_turbo = "ideogram_turbo"
    ideogram = "ideogram"
    midjourney = "midjourney"
    stability = "stability"
    # flux = "flux"
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
            ImaginationEngines.ideogram_turbo,
            ImaginationEngines.photon,
            ImaginationEngines.imagen,
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

    def get_class(self) -> BaseEngine:
        from .dalle import Dalle
        from .midjourney import Midjourney
        from .replicate_engine import (
            Flux11,
            FluxSchnell,
            Ideogram,
            IdeogramTurbo,
            Imagen,
            Photon,
            PhotonFlash,
            StableDiffusion3,
        )

        return {
            ImaginationEngines.dalle: Dalle(),
            ImaginationEngines.midjourney: Midjourney(),
            ImaginationEngines.ideogram: Ideogram(),
            ImaginationEngines.ideogram_turbo: IdeogramTurbo(),
            ImaginationEngines.flux_schnell: FluxSchnell(),
            ImaginationEngines.stability: StableDiffusion3(),
            ImaginationEngines.flux_1_1: Flux11(),
            # ImaginationEngines.flux: Flux11(),
            ImaginationEngines.photon: Photon(),
            ImaginationEngines.photon_flash: PhotonFlash(),
            ImaginationEngines.imagen: Imagen(),
        }[self]

    @property
    def thumbnail_url(self):
        return self.get_class().thumbnail_url

    @property
    def supported_aspect_ratios(self):
        return self.get_class().supported_aspect_ratios

    @property
    def price(self):
        return self.get_class().price

    @property
    def core(self):
        return self.get_class().core


class ImaginationEnginesSchema(BaseModel):
    engine: ImaginationEngines = ImaginationEngines.midjourney
    thumbnail_url: str
    supported_aspect_ratios: set
    coin: float

    @classmethod
    def from_model(cls, model: ImaginationEngines):
        return cls(
            engine=model,
            thumbnail_url=model.thumbnail_url,
            supported_aspect_ratios=model.supported_aspect_ratios,
            coin=model.price,
        )
