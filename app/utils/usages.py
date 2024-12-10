import uuid
import os
import aiohttp
import singleton
import json

from enum import Enum
from usso.async_session import AsyncUssoSession
from pydantic import BaseModel


class UsageInput(BaseModel):
    user_id: str
    asset: str = "images"
    amount: int = 1
    meta_data: dict | None = None


class Usages(metaclass=singleton.Singleton):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.refresh_token = os.getenv("USSO_REFRESH_TOKEN")
        self.refresh_url = os.getenv("USSO_REFRESH_URL")
        self.base_url = os.getenv("WALLET_URL")
        self.upload_url = f"{self.base_url}/usages/"

    async def create_usage(self, data: UsageInput):
        async with AsyncUssoSession(self.refresh_url, self.refresh_token) as client:
            return await self.create_usage_session(client, data)

    async def create_usage_session(
        self, client: aiohttp.ClientSession, data: UsageInput
    ):
        payload = aiohttp.FormData()
        for key, value in data.model_dump().items():
            if value is not None:
                if isinstance(value, dict) or isinstance(value, list):
                    value = json.dumps(value)
                payload.add_field(key, value)
        async with client.post(self.upload_url, json=data.model_dump()) as response:
            print(await response.json())
            print(data.model_dump())
            response.raise_for_status()
