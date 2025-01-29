import asyncio
import logging

from bson import UUID_SUBTYPE, Binary
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

        return await imagine_request(self)

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
        return
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

    @classmethod
    async def get_item(cls, uid, user_id, *args, **kwargs) -> "ImaginationBulk":
        uid = Binary.from_uuid(uid, UUID_SUBTYPE)
        user_id = Binary.from_uuid(user_id, UUID_SUBTYPE)
        # oid = ObjectId("6753f48dfd1306dbcb91604a")

        result = await ImaginationBulk.aggregate(
            [
                {
                    "$match": {"uid": uid, "user_id": user_id}
                },  # Match the specific ImagineBulkSchema by its ID
                {
                    "$lookup": {
                        "from": "Imagination",  # The collection name of ImagineSchema
                        "localField": "uid",  # The field in ImagineBulkSchema
                        "foreignField": "bulk",  # The field in ImagineSchema that references the bulk ID
                        "as": "child",  # The resulting array of related ImagineSchema entries
                    }
                },
            ]
        ).to_list()
        bulk = ImaginationBulk(**result[0])
        bulk.results = []
        for imagine_dict in result[0]["child"]:
            if imagine_dict is None:
                continue
            for imagine_result in imagine_dict.get("results") or []:
                if imagine_result is None:
                    continue

                updated_at, created_at = imagine_dict.get(
                    "updated_at"
                ), imagine_dict.get("created_at")
                if updated_at and created_at:
                    execution_time = (updated_at - created_at).total_seconds()
                else:
                    execution_time = None

                bulk.results.append(
                    ImagineBulkResponse(
                        **imagine_result,
                        engine=imagine_dict.get("engine"),
                        execution_time=execution_time,
                    )
                )

        return bulk

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
