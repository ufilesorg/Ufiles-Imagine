import asyncio
import json
import logging
import uuid
from io import BytesIO

import ufiles
from apps.imagination.schemas import ImagineResponse
from fastapi_mongo_base.utils import aionetwork, basic, imagetools
from PIL import Image
from server.config import Settings

from .models import BackgroundRemoval
from .schemas import BackgroundRemovalEngines, BackgroundRemovalWebhookData

_background_removal_conditions: dict[uuid.UUID, asyncio.Condition] = {}


def get_condition(imagination_id: uuid.UUID) -> asyncio.Condition:
    """Get or create condition for an imagination"""
    if imagination_id not in _background_removal_conditions:
        _background_removal_conditions[imagination_id] = asyncio.Condition()
    return _background_removal_conditions[imagination_id]


def cleanup_condition(imagination_id: uuid.UUID):
    """Remove condition when imagination is complete"""
    if imagination_id in _background_removal_conditions:
        del _background_removal_conditions[imagination_id]


async def release_condition(imagination_id: uuid.UUID):
    condition = get_condition(imagination_id)
    async with condition:
        condition.notify_all()
    cleanup_condition(imagination_id)


async def upload_image(
    image: Image.Image,
    image_name: str,
    user_id: uuid.UUID,
    engine: BackgroundRemovalEngines = BackgroundRemovalEngines.cjwbw,
    file_upload_dir: str = "imaginations",
):
    ufiles_client = ufiles.AsyncUFiles(
        ufiles_base_url=Settings.UFILES_BASE_URL,
        usso_base_url=Settings.USSO_BASE_URL,
        api_key=Settings.UFILES_API_KEY,
    )
    image_bytes: BytesIO = imagetools.convert_image_bytes(image, format="webp")
    image_bytes.name = f"{image_name}.webp"
    return await ufiles_client.upload_bytes(
        image_bytes,
        filename=f"{file_upload_dir}/{image_bytes.name}",
        public_permission=json.dumps({"permission": ufiles.PermissionEnum.READ}),
        user_id=str(user_id),
        meta_data={
            "engine": engine.value,
            "width": image.width,
            "height": image.height,
        },
    )


@basic.try_except_wrapper
async def process_result(background_removal: BackgroundRemoval, generated_url: str):
    # Download the image
    image_bytes = await aionetwork.aio_request_binary(url=generated_url)
    image = Image.open(image_bytes)
    uploaded_item = await upload_image(
        image,
        image_name=f"bg_{image.filename}",
        user_id=background_removal.user_id,
        engine=background_removal.engine,
        file_upload_dir="backgrounds_removal",
    )

    return ImagineResponse(
        url=uploaded_item.url,
        width=image.width,
        height=image.height,
    )


async def process_background_removal_webhook(
    background_removal: BackgroundRemoval, data: BackgroundRemovalWebhookData
):
    if data.status == "error":
        await background_removal.retry(data.error)
        return

    if data.status == "completed":
        result_url = data.output
        background_removal.result = await process_result(background_removal, result_url)
        await background_removal.save()

    background_removal.task_progress = data.percentage
    background_removal.task_status = background_removal.status.task_status

    report = (
        f"Replicate completed."
        if data.status == "completed"
        else f"Replicate update. {background_removal.status}"
    )

    await background_removal.save_report(report)

    logging.info(f"Background removal finished: {background_removal.uid} {data.status} {background_removal.result} {report}")

    if data.status == "completed":
        await release_condition(background_removal.uid)


@basic.try_except_wrapper
async def background_removal_request(background_removal: BackgroundRemoval):
    # Get Engine class and validate it
    BGEngineClass = background_removal.engine.get_class(background_removal)
    if BGEngineClass is None:
        raise NotImplementedError(
            "The supported engines are Replicate, Replicate and Dalle."
        )
    mid_request = await BGEngineClass._request(
        callback=background_removal.item_webhook_url
    )

    # Store Engine response
    background_removal.meta_data = (
        background_removal.meta_data or {}
    ) | mid_request.model_dump()
    await background_removal.save_report(f"Replicate has been requested.")

    condition = get_condition(background_removal.uid)
    async with condition:
        await condition.wait()
    cleanup_condition(background_removal.uid)

    background_removal = await BackgroundRemoval.get_item(background_removal.uid, background_removal.user_id)

    logging.info(f"Background removal finished: {background_removal.uid} {background_removal.result}")
    return background_removal
