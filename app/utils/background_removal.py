import replicate
from replicate.identifier import ModelVersionIdentifier

from .ai import Replicate, ReplicateDetails


class ReplicateBackgroundRemoval(Replicate):
    def __init__(self, item, name) -> None:
        self.item = item
        self.application_name = {
            "cjwbw": ModelVersionIdentifier(
                "cjwbw",
                "rembg",
                "fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            ),
            "lucataco": ModelVersionIdentifier(
                "lucataco",
                "remove",
                "bg:95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
            ),
            "pollinations": ModelVersionIdentifier(
                "pollinations",
                "modnet",
                "da7d45f3b836795f945f221fc0b01a6d3ab7f5e163f13208948ad436001e2255",
            ),
        }[name]

    async def _request(self, **kwargs) -> ReplicateDetails:
        prediction = replicate.predictions.create(
            version=self.application_name.version,
            input={"image": self.item.image},
            # webhook=self.item.item_webhook_url,
            # webhook_events_filter=["start", "completed"],
        )
        return await self._result_to_details(prediction)

    async def _result_to_details(self, prediction: replicate.prediction.Prediction):
        prediction_data = prediction.__dict__.copy()
        prediction_data.pop("status", None)
        prediction_data.pop("model", None)
        return ReplicateDetails(
            **prediction_data,
            prompt="",
            status=self._status(prediction.status),
            model=f"{self.application_name.owner}/{self.application_name.name}",
            result=({"uri": prediction.output} if prediction.output else None),
            percentage=100,
        )
