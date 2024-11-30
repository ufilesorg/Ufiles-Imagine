import asyncio
import logging

from fastapi_mongo_base._utils.basic import try_except_wrapper
from fastapi_mongo_base.models import OwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum
from server.config import Settings

from .schemas import (
    ImaginationStatus,
    ImagineBulkError,
    ImagineBulkSchema,
    ImagineSchema,
)


class Imagination(ImagineSchema, OwnedEntity):
    class Settings:
        indexes = OwnedEntity.Settings.indexes

    @property
    def item_url(self):
        super().item_url
        # TODO: Change to use the business url
        return f"https://{Settings.root_url}{Settings.base_path}/imagination/{self.uid}"

    async def start_processing(self):
        from .services import imagine_request

        await imagine_request(self)

    async def end_processing(self):
        if self.bulk and self.status.is_done:
            main_task = await ImaginationBulk.get(self.bulk)
            await main_task.end_processing(self)

    async def retry(self, message: str, max_retries: int = 0):
        self.meta_data = self.meta_data or {}
        retry_count = self.meta_data.get("retry_count", 0)
        print(max_retries)
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
        self.task_status = TaskStatusEnum.error
        self.status = ImaginationStatus.error
        await self.save_report(f"Image failed after retries, {message}", emit=False)
        await self.save_and_emit()
        if self.bulk:
            main_task = await ImaginationBulk.get(self.bulk)
            print("failed")
            await main_task.fail()

    @classmethod
    async def get_item(cls, uid, user_id, *args, **kwargs) -> "Imagination":
        # if user_id == None:
        #     raise ValueError("user_id is required")
        return await super(OwnedEntity, cls).get_item(
            uid, user_id=user_id, *args, **kwargs
        )


class ImaginationBulk(ImagineBulkSchema, OwnedEntity):
    class Settings:
        indexes = OwnedEntity.Settings.indexes

    @property
    def item_url(self):
        return f"https://{Settings.root_url}{Settings.base_path}/imagination/bulk/{self.uid}"

    async def start_processing(self):
        from .services import imagine_bulk_request

        await imagine_bulk_request(self)

    @try_except_wrapper
    async def fail(self):
        self.total_failed += 1
        print(f"self.total_failed:{self.total_failed}")
        data: list[Imagination] = await Imagination.find(
            {
                "bulk": {"$eq": str(self.id)},
                "status": {"$eq": ImaginationStatus.completed},
            }
        ).to_list()
        print(data)
        self.error = []
        for item in data:
            self.error.append(
                ImagineBulkError(task=str(item.id), message=item.error or "")
            )
        await self.save()
        from .services import imagine_bulk_process

        await imagine_bulk_process(self)

    async def end_processing(self, imagination: Imagination):
        from .services import imagine_bulk_result

        await imagine_bulk_result(self, imagination)
