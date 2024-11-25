import asyncio
import json
import logging
import uuid

import aiohttp
from fastapi_mongo_base._utils.basic import delay_execution, try_except_wrapper
from PIL import Image
from usso.async_session import AsyncUssoSession

from apps.imagination.schemas import ImagineResponse
from server.config import Settings
from utils import aionetwork, imagetools, ufiles

from .models import BackgroundRemoval
from .schemas import BackgroundRemovalEngines, BackgroundRemovalWebhookData


async def upload_image(
    client: AsyncUssoSession | aiohttp.ClientSession,
    image: Image.Image,
    image_name: str,
    user_id: uuid.UUID,
    engine: BackgroundRemovalEngines = BackgroundRemovalEngines.cjwbw,
    file_upload_dir: str = "imaginations",
):
    image_bytes = imagetools.convert_to_webp_bytes(image)
    image_bytes.name = f"{image_name}.webp"
    return await ufiles.AsyncUFiles().upload_bytes_session(
        client,
        image_bytes,
        filename=f"{file_upload_dir}/{image_bytes.name}",
        public_permission=json.dumps({"permission": ufiles.PermissionEnum.READ}),
        user_id=str(user_id),
        meta_data={"engine": engine.value},
    )


async def process_result(background_removal: BackgroundRemoval, generated_url: str):
    try:
        # Download the image
        image_bytes = await aionetwork.aio_request_binary(url=generated_url)
        image = Image.open(image_bytes)
        async with AsyncUssoSession(
            ufiles.AsyncUFiles().refresh_url, ufiles.AsyncUFiles().refresh_token
        ) as client:
            # Upload result images on ufiles
            uploaded_item = await upload_image(
                client,
                image,
                image_name=f"{uuid.uuid4()}_{image.filename}",
                user_id=background_removal.user_id,
                engine=background_removal.engine,
                file_upload_dir="backgrounds_removal",
            )

        background_removal.result = ImagineResponse(
            url=uploaded_item.url,
            width=uploaded_item.width,
            height=uploaded_item.height,
        )

    except Exception as e:
        import traceback

        traceback_str = "".join(traceback.format_tb(e.__traceback__))
        logging.error(f"Error processing image: {e}\n{traceback_str}")

    return


async def process_background_removal_webhook(
    background_removal: BackgroundRemoval, data: BackgroundRemovalWebhookData
):
    if data.status == "error":
        await background_removal.retry(data.error)
        return

    if data.status == "completed":
        result_url = (data.result or {}).get("uri")
        await process_result(background_removal, result_url)

    background_removal.task_progress = data.percentage
    background_removal.task_status = background_removal.status.task_status

    report = (
        f"Replicate completed."
        if data.status == "completed"
        else f"Replicate update. {background_removal.status}"
    )

    await background_removal.save_report(report)


@try_except_wrapper
async def background_removal_request(background_removal: BackgroundRemoval):
    # Get Engine class and validate it
    Item = background_removal.engine.get_class(background_removal)
    if Item is None:
        raise NotImplementedError(
            "The supported engines are Replicate, Replicate and Dalle."
        )
    mid_request = await Item._request(callback=background_removal.item_webhook_url)

    # Store Engine response
    background_removal.meta_data = (
        background_removal.meta_data or {}
    ) | mid_request.model_dump()
    await background_removal.save_report(f"Replicate has been requested.")

    # Create Short Polling process know the status of the request
    new_task = asyncio.create_task(background_removal_update(background_removal))
    return new_task


@try_except_wrapper
@delay_execution(Settings.update_time)
async def background_removal_update(background_removal: BackgroundRemoval, i=0):
    # Stop Short polling when the request is finished
    if background_removal.status.is_done:
        return

    Item = background_removal.engine.get_class(background_removal)

    # Get Result from service by engine class
    # And Update background_removal status
    result = await Item.result()
    background_removal.status = result.status

    # Process Result
    await process_background_removal_webhook(
        background_removal, BackgroundRemovalWebhookData(**result.model_dump())
    )
    return asyncio.create_task(background_removal_update(background_removal, i + 1))
