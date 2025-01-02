import logging
import uuid

import fastapi
from apps.ai.engine import ImaginationEngines, ImaginationEnginesSchema
from apps.ai.replicate_schemas import PredictionModelWebhookData
from fastapi import BackgroundTasks
from fastapi_mongo_base.core.exceptions import BaseHTTPException
from fastapi_mongo_base.routes import AbstractBaseRouter
from fastapi_mongo_base.tasks import TaskStatusEnum
from usso.fastapi import jwt_access_security

from .models import Imagination, ImaginationBulk
from .schemas import (
    ImagineBulkSchema,
    ImagineCreateBulkSchema,
    ImagineCreateSchema,
    ImagineSchema,
    MidjourneyWebhookData,
)
from .services import process_imagine_webhook


class ImaginationRouter(AbstractBaseRouter[Imagination, ImagineSchema]):
    def __init__(self):
        super().__init__(
            model=Imagination,
            schema=ImagineSchema,
            user_dependency=jwt_access_security,
            tags=["Imagination"],
            prefix="",
        )

    def config_routes(self, **kwargs):
        self.router.add_api_route(
            "/imagination/",
            self.list_items,
            methods=["GET"],
            response_model=self.list_response_schema,
            status_code=200,
        )
        self.router.add_api_route(
            "/imagination/",
            self.create_item,
            methods=["POST"],
            response_model=self.create_response_schema,
            status_code=201,
        )
        self.router.add_api_route(
            "/imagination/{uid:uuid}",
            self.retrieve_item,
            methods=["GET"],
            response_model=self.retrieve_response_schema,
            status_code=200,
        )
        self.router.add_api_route(
            "/imagination/{uid:uuid}",
            self.delete_item,
            methods=["DELETE"],
            # status_code=204,
            response_model=self.delete_response_schema,
        )
        self.router.add_api_route(
            "/imagination/{uid:uuid}/webhook",
            self.webhook,
            methods=["POST"],
            status_code=200,
        )

    async def create_item(
        self,
        request: fastapi.Request,
        data: ImagineCreateSchema,
        background_tasks: BackgroundTasks,
        sync: bool = False,
    ):
        item: Imagination = await super().create_item(request, data.model_dump())
        item.task_status = "init"
        if sync:
            await item.start_processing()
        else:
            background_tasks.add_task(item.start_processing)
        return item

    async def webhook(
        self,
        request: fastapi.Request,
        uid: uuid.UUID,
        data: dict,
    ):
        item: Imagination = await self.get_item(uid, user_id=None)
        if item.engine.core == "midjourney":
            data = MidjourneyWebhookData(**data)
        elif item.engine.core == "replicate":
            data = PredictionModelWebhookData(**data)
        else:
            logging.info(f"{type(data)} {item.engine.value} {data}")
            return {}
        if item.status == "cancelled":
            return {"message": "Imagination has been cancelled."}
        await process_imagine_webhook(item, data)
        return {}


class ImaginationBulkRouter(AbstractBaseRouter[ImaginationBulk, ImagineBulkSchema]):
    def __init__(self):
        super().__init__(
            model=ImaginationBulk,
            schema=ImagineBulkSchema,
            user_dependency=jwt_access_security,
            prefix="/imagination/bulk",
            tags=["Imagination"],
        )

    def config_schemas(self, schema, **kwargs):
        super().config_schemas(schema, **kwargs)
        # self.create_request_schema = ImagineCreateBulkSchema

    def config_routes(self, **kwargs):
        self.router.add_api_route(
            "/",
            self.list_items,
            methods=["GET"],
            response_model=self.list_response_schema,
            status_code=200,
        )
        self.router.add_api_route(
            "/",
            self.create_bulk_item,
            methods=["POST"],
            response_model=self.create_response_schema,
            status_code=201,
        )
        self.router.add_api_route(
            "/{uid:uuid}",
            self.retrieve_bulk_item,
            methods=["GET"],
            response_model=self.retrieve_response_schema,
        )

    async def create_bulk_item(
        self,
        request: fastapi.Request,
        data: ImagineCreateBulkSchema,
        background_tasks: BackgroundTasks,
        sync: bool = False,
    ):
        user_id = await self.get_user_id(request)
        item: ImaginationBulk = await ImaginationBulk.create_item(
            {
                "user_id": user_id,
                "task_status": TaskStatusEnum.init,
                **data.model_dump(),
                "total_tasks": len([d for d in data.get_combinations()]),
            }
        )
        if sync:
            item = await item.start_processing()
        else:
            background_tasks.add_task(item.start_processing)
        return item

    async def retrieve_bulk_item(
        self,
        request: fastapi.Request,
        uid: uuid.UUID,
        background_tasks: BackgroundTasks,
    ):
        user_id = await self.get_user_id(request)
        item = await ImaginationBulk.get_item(uid, user_id=user_id)
        if item is None:
            raise BaseHTTPException(
                status_code=404,
                error="item_not_found",
                message=f"{ImaginationBulk.__name__.capitalize()} not found",
            )
        return item

    async def webhook_bulk(self, request: fastapi.Request, uid: uuid.UUID, data: dict):
        logging.info(f"Webhook received: {await request.json()}")
        return {}
        item: Imagination = await self.get_item(uid, user_id=None)
        if item.status == "cancelled":
            return {"message": "Imagination has been cancelled."}
        await process_imagine_webhook(item, data)
        return {}


router = ImaginationRouter().router
bulk_router = ImaginationBulkRouter().router


@router.get("/engines/", response_model=list[ImaginationEnginesSchema])
async def engines(aspect_ratio: str = None):
    engines = [
        ImaginationEnginesSchema.from_model(engine) for engine in ImaginationEngines
    ]
    if aspect_ratio:
        engines = [e for e in engines if aspect_ratio in e.supported_aspect_ratios]
    return engines
