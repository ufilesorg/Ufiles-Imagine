import uuid

import fastapi
from fastapi import BackgroundTasks
from fastapi_mongo_base.routes import AbstractBaseRouter
from usso.fastapi import jwt_access_security

from apps.ai.schemas import ImaginationEngines, ImaginationEnginesSchema
from utils.ai import build_prompt

from .models import Imagination
from .schemas import (
    ImagineCreateSchema,
    ImagineSchema,
    ImagineWebhookData,
    PromptBuilderData,
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
            "/imagination",
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
    ):
        item: Imagination = await super().create_item(request, data.model_dump())
        background_tasks.add_task(item.start_processing)
        return item

    async def webhook(
        self, request: fastapi.Request, uid: uuid.UUID, data: ImagineWebhookData
    ):
        # logging.info(f"Webhook received: {await request.json()}")
        item: Imagination = await self.get_item(uid, user_id=None)
        if item.status == "cancelled":
            return {"message": "Imagination has been cancelled."}
        await process_imagine_webhook(item, data)
        return {}


router = ImaginationRouter().router


@router.get("/engines")
async def engines():
    engines = [
        ImaginationEnginesSchema.from_model(engine) for engine in ImaginationEngines
    ]
    return engines


@router.post("/prompt-builder")
async def prompt_builder(data: PromptBuilderData):
    prompt = await build_prompt(data.idea, data.engine)
    return {"prompt": prompt, "engine": data.engine}
