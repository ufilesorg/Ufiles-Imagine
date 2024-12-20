import os
import uuid

import singleton
from fastapi_mongo_base.core.exceptions import BaseHTTPException
from fastapi_mongo_base.schemas import OwnedEntitySchema
from pydantic import BaseModel
from usso.session import AsyncUssoSession


class UsageInput(BaseModel):
    uid: str | None = None
    user_id: str | None = None
    asset: str = "images"
    amount: int = 1
    meta_data: dict | None = None


class Usages(metaclass=singleton.Singleton):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.item = None
        self.refresh_token = os.getenv("USSO_REFRESH_TOKEN")
        self.refresh_url = os.getenv("USSO_REFRESH_URL")
        self.base_url = os.getenv("WALLET_URL")
        self.upload_url = f"{self.base_url}/usages/"

    async def create(self, user_id: str | uuid.UUID, amount: int = 1):
        return
        async with AsyncUssoSession(self.refresh_url, self.refresh_token) as client:
            return await self.create_usage_session(
                client, UsageInput(user_id=str(user_id), amount=amount)
            )

    async def update(self, item: OwnedEntitySchema):
        return
        self.item.meta_data = {"uid": item.uid}
        async with AsyncUssoSession(self.refresh_url, self.refresh_token) as client:
            return await self.update_usage_session(client)

    async def create_usage_session(self, client: AsyncUssoSession, data: UsageInput):
        try:
            response = await client.post(
                self.upload_url, json=data.model_dump(mode="json")
            )
            response.raise_for_status()
            result = response.json()
            self.item = UsageInput(**result)
        except Exception:
            raise BaseHTTPException(
                status_code=response.status,
                error=result["error"] or "",
                message=result["message"] or "",
            )

    async def update_usage_session(self, client: AsyncUssoSession):
        response = await client.patch(
            f"{self.upload_url}{self.item.uid}", json=self.item.model_dump(mode="json")
        )
        response.raise_for_status()
        result = response.json()
        self.item = UsageInput(**result)
