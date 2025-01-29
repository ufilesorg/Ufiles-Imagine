import replicate
import replicate.prediction

from apps.ai.replicate_engine import Replicate, ReplicateDetails


class ReplicateBackgroundRemoval(Replicate):
    application_name = "cjwbw/rembg"
    version = "fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003"

    def __init__(self, item) -> None:
        self.item = item

    async def _request(self, **kwargs) -> ReplicateDetails:
        prediction = replicate.predictions.create(
            # model=self.application_name,
            version=self.version,
            input={"image": self.item.image_url},
            webhook=self.item.item_webhook_url,
            # webhook_events_filter=["start", "completed"],
        )
        return await self._result_to_details(prediction)

    async def _result_to_details(
        self, prediction: replicate.prediction.Prediction, *args, **kwargs
    ):
        prediction_data = prediction.__dict__.copy()
        prediction_data.pop("status", None)
        prediction_data.pop("model", None)
        return ReplicateDetails(
            **prediction_data,
            prompt="",
            status=self._status(prediction.status),
            model=self.application_name,
            result=({"uri": prediction.output} if prediction.output else None),
            percentage=100,
        )


class CjwbwReplicateBackgroundRemoval(ReplicateBackgroundRemoval):
    application_name = "cjwbw/rembg"
    version = "fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003"


class LucatacoReplicateBackgroundRemoval(ReplicateBackgroundRemoval):
    application_name = "lucataco/remove"
    version = "95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1"


class PollinationsReplicateBackgroundRemoval(ReplicateBackgroundRemoval):
    application_name = "pollinations/modnet"
    version = "da7d45f3b836795f945f221fc0b01a6d3ab7f5e163f13208948ad436001e2255"
