import json
import logging
import uuid

import ufiles
from apps.imagination.schemas import ImagineResponse
from fastapi_mongo_base.utils import aionetwork, basic, imagetools
from PIL import Image
from server.config import Settings

from .models import BackgroundRemoval
from .schemas import BackgroundRemovalEngines, BackgroundRemovalWebhookData


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
    image_bytes = imagetools.convert_format_bytes(image, convert_format="webp")
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


async def process_result(background_removal: BackgroundRemoval, generated_url: str):
    try:
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

        background_removal.result = ImagineResponse(
            url=uploaded_item.url,
            width=image.width,
            height=image.height,
        )

    except Exception as e:
        import traceback

        traceback_str = "".join(traceback.format_tb(e.__traceback__))
        logging.error(f"Error processing image: {e}\n{traceback_str}")


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


@basic.try_except_wrapper
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
    return await background_removal_update(background_removal)


@basic.try_except_wrapper
@basic.delay_execution(Settings.update_time)
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
    return await background_removal_update(background_removal, i + 1)
