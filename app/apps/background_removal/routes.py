import uuid
import fastapi

from fastapi import BackgroundTasks
from fastapi_mongo_base.routes import AbstractBaseRouter
from usso.fastapi import jwt_access_security

from apps.imagination.schemas import ImaginationEnginesSchema
from .services import process_background_removal_webhook
from .models import BackgroundRemoval
from .schemas import (
    BackgroundRemovalEngines,
    BackgroundRemovalCreateSchema,
    BackgroundRemovalSchema,
    BackgroundRemovalWebhookData,
)


class BackgroundRemovalRouter(
    AbstractBaseRouter[BackgroundRemoval, BackgroundRemovalSchema]
):
    def __init__(self):
        super().__init__(
            model=BackgroundRemoval,
            schema=BackgroundRemovalSchema,
            user_dependency=jwt_access_security,
            tags=["BackgroundRemoval"],
            prefix="/background-removal",
        )

    def config_routes(self, **kwargs):
        self.router.add_api_route(
            "",
            self.list_items,
            methods=["GET"],
            response_model=self.list_response_schema,
            status_code=200,
        )
        self.router.add_api_route(
            "/",
            self.create_item,
            methods=["POST"],
            response_model=self.create_response_schema,
            status_code=201,
        )
        self.router.add_api_route(
            "/{uid:uuid}",
            self.retrieve_item,
            methods=["GET"],
            response_model=self.retrieve_response_schema,
            status_code=200,
        )
        self.router.add_api_route(
            "/{uid:uuid}",
            self.delete_item,
            methods=["DELETE"],
            # status_code=204,
            response_model=self.delete_response_schema,
        )
        self.router.add_api_route(
            "/{uid:uuid}/webhook",
            self.webhook,
            methods=["POST"],
            status_code=200,
        )

    async def create_item(
        self,
        request: fastapi.Request,
        data: BackgroundRemovalCreateSchema,
        background_tasks: BackgroundTasks,
    ):
        item: BackgroundRemoval = await super().create_item(request, data.model_dump())
        background_tasks.add_task(item.start_processing)
        return item

    async def webhook(
        self,
        request: fastapi.Request,
        uid: uuid.UUID,
        data: BackgroundRemovalWebhookData,
    ):
        item: BackgroundRemoval = await self.get_item(uid, user_id=None)
        if item.status == "cancelled":
            return {"message": "BackgroundRemoval has been cancelled."}
        await process_background_removal_webhook(item, data)
        return {}


router = BackgroundRemovalRouter().router


@router.get("/background-removal/engines")
async def engines():
    engines = [
        ImaginationEnginesSchema.from_model(engine)
        for engine in BackgroundRemovalEngines
    ]
    return engines
