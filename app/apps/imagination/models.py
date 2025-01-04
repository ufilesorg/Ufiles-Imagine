import asyncio
import logging

from fastapi_mongo_base.models import OwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum
from fastapi_mongo_base.utils.basic import try_except_wrapper
from server.config import Settings

from .schemas import (
    ImaginationStatus,
    ImagineBulkError,
    ImagineBulkResponse,
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

    async def retry(self, message: str, max_retries: int = 5):
        self.meta_data = self.meta_data or {}
        retry_count = self.meta_data.get("retry_count", 0)

        if retry_count < max_retries:
            self.meta_data["retry_count"] = retry_count + 1
            logging.warning(f"Retry {retry_count} {self.uid}")
            await self.save_report(
                f"Retry {self.uid} {self.meta_data.get('retry_count')}", emit=False
            )
            await self.save_and_emit()
            asyncio.create_task(self.start_processing())
            logging.info(f"Retry {retry_count} {self.uid}")
            return retry_count + 1
        await self.fail(message)
        return -1

    async def fail(self, message: str, log_type: str = "error"):
        from .services import cancel_usage

        self.task_status = TaskStatusEnum.error
        self.status = ImaginationStatus.error
        logging.error(f"Failed {self.uid} {message}")
        await self.save_report(
            f"Imagine failed, {message}", emit=False, log_type=log_type
        )
        await self.save_and_emit()
        await cancel_usage(self)
        if self.bulk:
            main_task = await ImaginationBulk.get(self.bulk)
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

        return await imagine_bulk_request(self)

    @try_except_wrapper
    async def fail(self):
        self.total_failed += 1
        data: list[Imagination] = await Imagination.find(
            {
                "bulk": {"$eq": str(self.id)},
                "status": {"$eq": ImaginationStatus.error},
            }
        ).to_list()
        self.errors = []
        for item in data:
            self.errors.append(
                ImagineBulkError(engine=item.engine, message=item.error or f"Error")
            )
        await self.save()
        from .services import imagine_bulk_process

        await imagine_bulk_process(self)

    async def end_processing(self, imagination: Imagination):
        from .services import imagine_bulk_result

        await imagine_bulk_result(self, imagination)

    async def completed_tasks(self) -> list[Imagination]:
        return await Imagination.find(
            {
                "bulk": {"$eq": str(self.id)},
                "status": {"$eq": ImaginationStatus.completed},
                "results": {"$ne": None},
            }
        ).to_list()

    async def failed_tasks(self) -> list[Imagination]:
        return await Imagination.find(
            {
                "bulk": {"$eq": str(self.id)},
                "status": {"$eq": ImaginationStatus.error},
            }
        ).to_list()

    async def collect_results(self):
        try:
            completed_tasks = await self.completed_tasks()
            results = []
            for item in completed_tasks:
                for result in item.results:
                    results.append(
                        ImagineBulkResponse(engine=item.engine, **result.model_dump())
                    )
            return results
        except Exception as e:
            import traceback

            traceback_str = "".join(traceback.format_tb(e.__traceback__))
            logging.error(
                "\n".join(
                    [
                        f"{item.uid}: {item.engine} {item.results}"
                        for item in completed_tasks
                    ]
                )
            )
            logging.error(
                f"An error occurred in collect_results:\n{traceback_str}\n{e}"
            )
            return None
