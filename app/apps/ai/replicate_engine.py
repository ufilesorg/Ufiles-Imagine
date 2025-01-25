from typing import Any, Literal

import replicate.prediction
from apps.imagination.schemas import ImagineSchema
from server.config import Settings

from .engine import BaseEngine, EnginesResponse
from .replicate_schemas import PredictionModelWebhookData
from .schemas import ImaginationStatus


class ReplicateDetails(EnginesResponse):
    input: dict[str, Any]
    model: Literal[
        "ideogram-ai/ideogram-v2-turbo",
        "ideogram-ai/ideogram-v2",
        "black-forest-labs/flux-schnell",
        "black-forest-labs/flux-1.1-pro",
        "stability-ai/stable-diffusion-3",
        "cjwbw/rembg",
        "lucataco/remove-bg",
        "pollinations/modnet",
        "luma/photon",
        "luma/photon-flash",
    ] = "ideogram-ai/ideogram-v2-turbo"


class Replicate(BaseEngine):
    application_name: Literal["luma/photon-flash"] = "luma/photon-flash"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _status(
        self,
        status: Literal["starting", "processing", "succeeded", "failed", "canceled"],
    ):
        return {
            "starting": ImaginationStatus.init,
            "canceled": ImaginationStatus.cancelled,
            "processing": ImaginationStatus.processing,
            "succeeded": ImaginationStatus.completed,
            "failed": ImaginationStatus.error,
        }.get(status, ImaginationStatus.error)

    async def result(
        self, imagination: ImagineSchema, **kwargs
    ) -> PredictionModelWebhookData:
        id = imagination.meta_data.get("id")
        prediction = await replicate.predictions.async_get(id)
        return await self._result_to_details(prediction, imagination)

    async def imagine(self, imagination: ImagineSchema, **kwargs) -> ReplicateDetails:
        prediction = replicate.predictions.create(
            model=self.application_name,
            input={
                "prompt": imagination.prompt,
                "aspect_ratio": imagination.aspect_ratio,
            },
            webhook=imagination.item_webhook_url,
            webhook_events_filter=["start", "completed"],
        )
        imagination.meta_data = (
            imagination.meta_data or {}
        ) | prediction.__dict__.copy()
        return await self._result_to_details(prediction, imagination)

    async def _result_to_details(
        self, prediction: replicate.prediction.Prediction, imagination: ImagineSchema
    ):
        prediction_data = prediction.__dict__.copy()
        prediction_data.pop("status", None)
        prediction_data.pop("model", None)
        return PredictionModelWebhookData(
            **prediction_data,
            prompt=(
                prediction.input["prompt"] if prediction.input else imagination.prompt
            ),
            status=self._status(prediction.status),
            model=self.application_name,
            result=(
                {
                    "uri": (
                        prediction.output
                        if isinstance(prediction.output, str)
                        else prediction.output[0]
                    )
                }
                if prediction.output
                else None
            ),
            percentage=100,
        )

    @property
    def core(self):
        return "replicate"


class Ideogram(Replicate):
    application_name: Literal["ideogram-ai/ideogram-v2"] = (
        "ideogram-ai/ideogram-v2-turbo"
    )

    @property
    def supported_aspect_ratios(self):
        return {
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
        }

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/19d4df43-ea1e-4562-a8e1-8ee301bd0a88/ideogram-icon.png?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 24


class IdeogramTurbo(Replicate):
    application_name: Literal["ideogram-ai/ideogram-v2-turbo"] = (
        "ideogram-ai/ideogram-v2-turbo"
    )

    @property
    def supported_aspect_ratios(self):
        return {
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
        }

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/19d4df43-ea1e-4562-a8e1-8ee301bd0a88/ideogram-icon.png?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 15


class FluxSchnell(Replicate):
    application_name: Literal["black-forest-labs/flux-schnell"] = (
        "black-forest-labs/flux-schnell"
    )

    @property
    def supported_aspect_ratios(self):
        return {
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
        }

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/cf21c500-6e84-4915-a5d1-19b8f325a382/flux-icon.png?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 1


class Flux11(Replicate):
    application_name: Literal["black-forest-labs/flux-1.1-pro"] = (
        "black-forest-labs/flux-1.1-pro"
    )

    @property
    def supported_aspect_ratios(self):
        return {
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
        }

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/cf21c500-6e84-4915-a5d1-19b8f325a382/flux-icon.png?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 12


class Photon(Replicate):
    application_name: Literal["luma/photon"] = "luma/photon"

    @property
    def supported_aspect_ratios(self):
        return {
            "1:1",
            "16:9",
            "21:9",
            "4:3",
            "3:4",
            "9:16",
            "9:21",
        }

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/4701330c-aa98-4d86-91d4-982ff94d30f3/photon.png?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 9


class PhotonFlash(Replicate):
    application_name: Literal["luma/photon-flash"] = "luma/photon-flash"

    @property
    def supported_aspect_ratios(self):
        return {
            "1:1",
            "16:9",
            "21:9",
            "4:3",
            "3:4",
            "9:16",
            "9:21",
        }

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/4701330c-aa98-4d86-91d4-982ff94d30f3/photon.png?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 3


class StableDiffusion3(Replicate):
    application_name: Literal["stability-ai/stable-diffusion-3"] = (
        "stability-ai/stable-diffusion-3"
    )

    @property
    def supported_aspect_ratios(self):
        return {
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
        }

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/cf21c500-6e84-4915-a5d1-19b8f325a382/flux-icon.png?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 2
