import json
import os
from datetime import datetime
from typing import Any

import httpx
from apps.imagination.schemas import ImagineSchema
from server.config import Settings

from .engine import BaseEngine, EnginesResponse


class MidjourneyDetails(EnginesResponse):
    deleted: bool = False
    active: bool = True
    createdBy: str | None = None
    user: str | None = None
    command: str
    callback_url: str | None = None
    free: bool = False
    temp_uri: list[str] = []
    createdAt: datetime
    updatedAt: datetime
    turn: int = 0
    account: str | None = None
    uri: str | None = None

    message: str | None = None
    sender_data: dict | None = None


class Midjourney(BaseEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.api_url = "https://mid.aision.io/task"
        self.token = os.getenv("MIDAPI_TOKEN")
        self.headers = {
            "Authorization": self.token,
            "Content-Type": "application/json",
        }

    @property
    def supported_aspect_ratios(self):
        return {
            "1:1",
            "16:10",
            "10:16",
            "16:9",
            "9:16",
            "21:9",
            "9:21",
            "3:1",
            "1:3",
            "3:2",
            "2:3",
            "4:3",
            "3:4",
            "5:4",
            "4:5",
            "7:4",
            "4:7",
        }

    @property
    def thumbnail_url(self):
        return "https://media.pixiee.io/v1/f/4a0980aa-8d97-4493-bdb1-fb3d67d891e3/midjourney-icon.png?width=100"

    @property
    def price(self):
        return Settings.base_image_price * 10

    @property
    def core(self):
        return "midjourney"

    async def result(self, imagination: ImagineSchema, **kwargs) -> MidjourneyDetails:
        id = imagination.meta_data.get("id")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_url}/{id}", headers=self.headers)
            response.raise_for_status()
            result = response.json()
            return await self._result_to_details(result)

    async def imagine(self, imagination: ImagineSchema, **kwargs) -> MidjourneyDetails:
        prompt = imagination.prompt.strip(".").strip(",").strip()
        if imagination.aspect_ratio != "1:1":
            prompt += f" --ar {imagination.aspect_ratio}"

        payload = json.dumps(
            {
                "prompt": prompt,
                "command": "imagine",
                "callback": imagination.item_webhook_url,
            }
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=self.api_url, headers=self.headers, data=payload
            )
            response.raise_for_status()
            result = response.json()
            imagination.meta_data = (
                (imagination.meta_data or {}) | result | {"id": result.get("uuid")}
            )

            return await self._result_to_details(result)

    async def _result_to_details(self, result: dict[str, Any], **kwargs):
        status = self._status(result["status"])
        result.pop("status", None)
        result.pop("error", None)
        return MidjourneyDetails(
            **result,
            id=result.get("uuid"),
            status=status,
            error=(
                result["error"]["message"]
                if result.get("error") and result["error"]["message"]
                else None
            ),
            result={"uri": result.get("uri")},
        )
