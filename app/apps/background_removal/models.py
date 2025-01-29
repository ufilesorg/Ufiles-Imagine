import asyncio
import logging

from fastapi_mongo_base.models import OwnedEntity
from server.config import Settings

from .schemas import BackgroundRemovalSchema


class BackgroundRemoval(BackgroundRemovalSchema, OwnedEntity):
    class Settings:
        indexes = OwnedEntity.Settings.indexes

    @property
    def item_url(self):
        return (
            f"https://{Settings.root_url}/v1/apps/imagine/background-removal/{self.uid}"
        )

    async def start_processing(self):
        from .services import background_removal_request

        return await background_removal_request(self)

    async def retry(self, message: str, max_retries: int = 5):
        self.meta_data = self.meta_data or {}
        retry_count = self.meta_data.get("retry_count", 0)

        if retry_count < max_retries:
            self.meta_data["retry_count"] = retry_count + 1
            await self.save_report(
                f"Retry {self.uid} {self.meta_data.get('retry_count')}", emit=False
            )
            await self.save_and_emit()
            asyncio.create_task(self.start_processing())
            logging.info(f"Retry {retry_count} {self.uid}")
            return retry_count + 1

        await self.fail(message)
        return -1

    async def fail(self, message: str):
        self.task_status = "error"
        self.status = "error"
        await self.save_report(f"Image failed after retries, {message}", emit=False)
        await self.save_and_emit()

    @classmethod
    async def get_item(cls, uid, user_id, *args, **kwargs) -> "BackgroundRemoval":
        return await super(OwnedEntity, cls).get_item(
            uid, user_id=user_id, *args, **kwargs
        )
