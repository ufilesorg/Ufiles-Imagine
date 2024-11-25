import uuid

import fastapi
from fastapi import BackgroundTasks
from fastapi_mongo_base.tasks import TaskReferenceList, TaskReference
from fastapi_mongo_base.routes import AbstractBaseRouter
from usso.fastapi import jwt_access_security

from apps.ai.schemas import ImaginationEngines, ImaginationEnginesSchema

from .models import Imagination, MultiEngine
from .schemas import (
    ImagineCreateSchema,
    ImagineSchema,
    ImagineWebhookData,
    MultiEngineSchema,
    MultiEngineCreateSchema,
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


class ImaginationMultiEngineRouter(AbstractBaseRouter[MultiEngine, MultiEngineSchema]):
    def __init__(self):
        super().__init__(
            model=MultiEngine,
            schema=MultiEngineSchema,
            user_dependency=jwt_access_security,
            tags=["MultiEngine"],
            prefix="/imagination/multi-engine",
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
            response_model=self.delete_response_schema,
        )

    async def create_item(
        self,
        request: fastapi.Request,
        data: MultiEngineCreateSchema,
        background_tasks: BackgroundTasks,
    ):
        user_id = await self.get_user_id(request)
        item = await self.model.create_item(
            self.schema(
                user_id=user_id,
                tasks_count=len(data.engines),
                engines=data.engines,
                aspect_ratio=data.aspect_ratio,
                prompt=data.prompt,
            ).model_dump()
        )
        item.task_references = TaskReferenceList(
            tasks=[],
            mode="parallel",
        )
        for index, engine in enumerate(item.engines):
            imagine = await Imagination.create_item(
                ImagineSchema(
                    user_id=user_id,
                    manager=str(item.id),
                    engine=engine,
                    **data.model_dump()
                ).model_dump()
            )
            item.task_references.tasks.append(
                TaskReference(task_id=imagine.uid, task_type="Imagination")
            )

        await item.save()
        background_tasks.add_task(item.start_processing)
        return item


router = ImaginationRouter().router
multi_engine_router = ImaginationMultiEngineRouter().router


@router.get("/engines")
async def engines():
    engines = [
        ImaginationEnginesSchema.from_model(engine) for engine in ImaginationEngines
    ]
    return engines
