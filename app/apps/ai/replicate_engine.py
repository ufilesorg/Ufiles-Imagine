from typing import Any, Literal

import replicate.prediction

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
    def __init__(self, item, name) -> None:
        super().__init__(item)
        self.application_name = {
            "ideogram": "ideogram-ai/ideogram-v2-turbo",
            "flux_schnell": "black-forest-labs/flux-schnell",
            "flux_1.1": "black-forest-labs/flux-1.1-pro",
            "stability": "stability-ai/stable-diffusion-3",
        }[name]

    async def result(self, **kwargs) -> ReplicateDetails:
        id = self._get_data("id")
        prediction = await replicate.predictions.async_get(id)
        return await self._result_to_details(prediction)

    async def _request(self, **kwargs) -> ReplicateDetails:
        prediction = replicate.predictions.create(
            model=self.application_name,
            input={"prompt": self.item.prompt, "aspect_ratio": self.item.aspect_ratio},
            # webhook=self.item.item_webhook_url,
            # webhook_events_filter=["completed"],
        )
        return await self._result_to_details(prediction)

    def validate(self, data):
        aspect_ratios = {
            "ideogram-ai/ideogram-v2-turbo": {
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
            "black-forest-labs/flux-schnell": {
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
            "black-forest-labs/flux-1.1-pro": {
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
            "stability-ai/stable-diffusion-3": {
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
        }[self.application_name]

        aspect_ratio_valid = data.aspect_ratio in aspect_ratios
        message = (
            f"aspect_ratio must be one of them {aspect_ratios}"
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
