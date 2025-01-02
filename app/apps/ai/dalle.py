import openai
from apps.imagination.schemas import ImagineSchema
from openai.types import ImagesResponse as OpenAiImagesResponse
from server.config import Settings

from .engine import BaseEngine, EnginesResponse
from .schemas import ImaginationStatus


class Dalle(BaseEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = openai.AsyncOpenAI(
            api_key=Settings.METIS_API_KEY,
            base_url="https://api.metisai.ir/openai/v1",
        )

    @property
    def supported_aspect_ratios(self):
        return {"1:1", "7:4", "4:7"}

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/41af8b03-b4df-4b2f-ba52-ea638d10b5f3/dalle-icon.png"  # + "?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 2

    @property
    def core(self):
        return "dalle"

    async def result(self, imagination: ImagineSchema, **kwargs) -> EnginesResponse:
        return imagination.results

    async def imagine(self, imagination: ImagineSchema, **kwargs) -> EnginesResponse:
        request_body = dict(
            prompt=imagination.prompt,
            model="dall-e-3",
            quality="standard",
            size=self._get_size(imagination.aspect_ratio),
            n=1,
        )
        response = await self.client.images.generate(**request_body)
        return self._result_to_details(response, imagination)

    def _status(self, response: OpenAiImagesResponse):
        return (
            ImaginationStatus.completed
            if len(response.data) > 0
            else ImaginationStatus.error
        )

    @classmethod
    def _get_size(cls, aspect_ratio: str):
        return {
            "1:1": "1024x1024",
            "7:4": "1792x1024",
            "4:7": "1024x1792",
        }.get(aspect_ratio)

    def _result_to_details(
        self, response: OpenAiImagesResponse, imagination: ImagineSchema
    ):
        return EnginesResponse(
            id=None,
            prompt=imagination.prompt,
            error=str(response.error) if response.error else None,
            status=self._status(response),
            result=({"uri": response.data[0].url} if len(response.data) > 0 else None),
        )
