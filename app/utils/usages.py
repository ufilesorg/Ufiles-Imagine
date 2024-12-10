import os
import aiohttp
import singleton

from pydantic import BaseModel
from usso.async_session import AsyncUssoSession
from core.exceptions import BaseHTTPException
from fastapi_mongo_base.schemas import OwnedEntitySchema


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

    async def create(self, item: OwnedEntitySchema, amount: int = 1):
        async with AsyncUssoSession(self.refresh_url, self.refresh_token) as client:
            return await self.create_ussage_session(
                client,
                UsageInput(
                    user_id=str(item.user_id),
                    meta_data={"uid": str(item.uid)},
                    amount=amount,
                ),
            )

    async def create_ussage_session(
        self, client: aiohttp.ClientSession, data: UsageInput
    ):
        async with client.post(self.upload_url, json=data.model_dump()) as response:
            result = await response.json()
            try:
                response.raise_for_status()
            except Exception:
                raise BaseHTTPException(
                    status_code=response.status,
                    error=result["error"] or "",
                    message=result["message"] or "",
                )
