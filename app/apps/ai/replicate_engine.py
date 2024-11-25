from typing import Any, Literal

import replicate.prediction

from apps.ai.schemas import ImaginationEngines

from .engine import Engine, EnginesDetails
from .schemas import ImaginationStatus


class ReplicateDetails(EnginesDetails):
    input: dict[str, Any]
    model: Literal[
        "ideogram-ai/ideogram-v2-turbo",
        "black-forest-labs/flux-schnell",
        "black-forest-labs/flux-1.1-pro",
        "stability-ai/stable-diffusion-3",
        "cjwbw/rembg",
        "lucataco/remove-bg",
        "pollinations/modnet",
    ] = "ideogram-ai/ideogram-v2-turbo"


class Replicate(Engine):
    def __init__(self, item, engine, **kwargs) -> None:
        super().__init__(item, engine, **kwargs)
        self.application_name = {
            ImaginationEngines.ideogram: "ideogram-ai/ideogram-v2-turbo",
            ImaginationEngines.flux_schnell: "black-forest-labs/flux-schnell",
            ImaginationEngines.flux_1_1: "black-forest-labs/flux-1.1-pro",
            ImaginationEngines.stability: "stability-ai/stable-diffusion-3",
        }[engine]

    async def result(self, **kwargs) -> ReplicateDetails:
        id = self._get_data("id")
        prediction = await replicate.predictions.async_get(id)
        return await self._result_to_details(prediction)

    async def _request(self, **kwargs) -> ReplicateDetails:
        prediction = replicate.predictions.create(
            model=self.application_name,
            input={"prompt": self.item.prompt, "aspect_ratio": self.item.aspect_ratio},
            # webhook=self.item.item_webhook_url,
            # webhook_events_filter=["start","completed"],
        )
        return await self._result_to_details(prediction)

    def validate(self, data):
        aspect_ratio_valid = data.aspect_ratio in self.engine.supported_aspect_ratios
        message = (
            f"aspect_ratio must be one of them {self.engine.supported_aspect_ratios}"
            if not aspect_ratio_valid
            else None
        )
        return aspect_ratio_valid, message

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

    async def _result_to_details(self, prediction: replicate.prediction.Prediction):
        prediction_data = prediction.__dict__.copy()
        prediction_data.pop("status", None)
        prediction_data.pop("model", None)
        return ReplicateDetails(
            **prediction_data,
            prompt=prediction.input["prompt"],
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
