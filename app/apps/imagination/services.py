import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta

import ufiles
from apps.imagination.models import Imagination, ImaginationBulk
from apps.imagination.schemas import (
    ImaginationEngines,
    ImaginationStatus,
    ImagineResponse,
    ImagineSchema,
    ImagineWebhookData,
)
from fastapi_mongo_base.tasks import TaskReference, TaskReferenceList, TaskStatusEnum
from fastapi_mongo_base.utils import aionetwork, basic, texttools
from metisai.async_metis import AsyncMetisBot
from PIL import Image
from server.config import Settings
from utils import ai, imagetools


async def upload_image(
    image: Image.Image,
    image_name: str,
    user_id: uuid.UUID,
    prompt: str,
    engine: ImaginationEngines = ImaginationEngines.midjourney,
    file_upload_dir: str = "imaginations",
):
    ufiles_client = ufiles.AsyncUFiles(
        ufiles_base_url=Settings.UFILES_BASE_URL,
        usso_base_url=Settings.USSO_BASE_URL,
        api_key=Settings.UFILES_API_KEY,
    )
    image_bytes = imagetools.convert_to_jpg_bytes(image)
    image_bytes.name = f"{image_name}.jpg"
    return await ufiles_client.upload_bytes(
        image_bytes,
        filename=f"{file_upload_dir}/{image_bytes.name}",
        public_permission=json.dumps({"permission": ufiles.PermissionEnum.READ}),
        user_id=str(user_id),
        meta_data={"prompt": prompt, "engine": engine.value},
    )


async def upload_images(
    images: list[Image.Image],
    user_id: uuid.UUID,
    prompt: str,
    engine: ImaginationEngines = ImaginationEngines.midjourney,
    file_upload_dir="imaginations",
):
    image_name = texttools.sanitize_filename(prompt)

    uploaded_items = [
        await upload_image(
            images[0],
            image_name=f"{image_name}_{1}",
            user_id=user_id,
            prompt=prompt,
            engine=engine,
            file_upload_dir=file_upload_dir,
        )
    ]
    uploaded_items += await asyncio.gather(
        *[
            upload_image(
                image,
                image_name=f"{image_name}_{i+2}",
                user_id=user_id,
                prompt=prompt,
                engine=engine,
                file_upload_dir=file_upload_dir,
            )
            for i, image in enumerate(images[1:])
        ]
    )
    return uploaded_items


async def process_result(imagination: Imagination, generated_url: str):
    try:
        # Download the image
        image_bytes = await aionetwork.aio_request_binary(url=generated_url)
        images = [Image.open(image_bytes)]
        file_upload_dir = "imaginations"

        # Crop the image into 4 sections for midjourney engine
        if imagination.engine == ImaginationEngines.midjourney:
            images = imagetools.crop_image(images[0], sections=(2, 2))

        # Upload result images on ufiles
        uploaded_items = await upload_images(
            images=images,
            user_id=imagination.user_id,
            prompt=imagination.prompt,
            engine=imagination.engine,
            file_upload_dir=file_upload_dir,
        )

        imagination.results = [
            ImagineResponse(url=uploaded.url, width=image.width, height=image.height)
            for uploaded, image in zip(uploaded_items, images)
        ]

    except Exception as e:
        import traceback

        traceback_str = "".join(traceback.format_tb(e.__traceback__))
        logging.error(f"Error processing image: \n{traceback_str}\n{e}")

    return


async def process_imagine_webhook(imagination: Imagination, data: ImagineWebhookData):
    import json_advanced as json

    if data.status == "error":
        logging.info(
            f"Error processing image: {json.dumps(data.model_dump(), indent=2, ensure_ascii=False)}"
        )
        await imagination.retry(data.error)
        return

    if data.status == "completed":
        result_url = (data.result or {}).get("uri")
        await process_result(imagination, result_url)
        await imagination.end_processing()

    imagination.task_progress = data.percentage
    imagination.task_status = data.status.task_status

    report = (
        f"{imagination.engine.value} completed."
        if data.status == "completed"
        else f"{imagination.engine.value} update. {imagination.status}"
    )

    await imagination.save_report(report)

    if data.status == "completed" and imagination.task_status != "completed":
        logging.info(json.dumps(imagination.model_dump(), indent=2, ensure_ascii=False))
        logging.info(json.dumps(data.model_dump(), indent=2, ensure_ascii=False))

    if not data.status.is_done and datetime.now() - imagination.created_at >= timedelta(
        minutes=10
    ):
        imagination.task_status = TaskStatusEnum.error
        imagination.status = ImaginationStatus.error
        imagination.error = "Service Timeout Error: The service did not provide a result within the expected time frame."
        await imagination.save_report(
            f"{imagination.engine.value} service didn't respond in time."
        )


async def request_imagine_metis(
    imagination: Imagination,
    engine: ImaginationEngines = ImaginationEngines.midjourney,
):
    bot = AsyncMetisBot(Settings.METIS_API_KEY, engine.metis_bot_id)
    session = await bot.create_session()
    await bot.send_message_async(session, f"/imagine {imagination.prompt}")


async def create_prompt(imagination: Imagination | ImaginationBulk):
    async def get_prompt_row(item: dict):
        return f'{item.get("topic", "")} {await ai.translate(item.get("value", ""))}'

    # Translate prompt using ai
    raw = imagination.prompt or imagination.delineation or ""
    if imagination.enhance_prompt:
        resp = await ai.answer_with_ai(key="prompt_builder", image_idea=raw)
        prompt = resp.get("image_prompt", raw)
    else:
        prompt = await ai.translate(raw)

    # Convert prompt ai properties to array
    context = await asyncio.gather(
        *[get_prompt_row(item) for item in imagination.context or []]
    )

    # Create final prompt using user prompt and prompt properties
    prompt += ", " + ", ".join(context)
    prompt = prompt.strip(",. ")

    return prompt


@basic.try_except_wrapper
async def imagine_request(imagination: Imagination):
    # Get Engine class and validate it
    imagine_engine = imagination.engine.get_class(imagination)
    if imagine_engine is None:
        raise NotImplementedError(
            "The supported engines are Midjourney, Replicate and Dalle."
        )

    # Create prompt using context attributes (ratio, style ...)
    imagination.prompt = await create_prompt(imagination)

    # Request to client or api using Engine classes
    mid_request = await imagine_engine.imagine(callback=imagination.item_webhook_url)

    # Store Engine response
    imagination.meta_data = (imagination.meta_data or {}) | mid_request.model_dump()
    imagination.error = mid_request.error
    imagination.status = mid_request.status
    imagination.task_status = mid_request.status.task_status
    await imagination.save_report(f"{imagination.engine.value} has been requested.")

    # Create Short Polling process know the status of the request
    return await imagine_update(imagination)


@basic.try_except_wrapper
@basic.delay_execution(Settings.update_time)
async def imagine_update(imagination: Imagination, i=0):
    imagine_engine = imagination.engine.get_class(imagination)

    # Get Result from service by engine class
    # And Update imagination status
    result = await imagine_engine.result()
    imagination.error = result.error
    imagination.status = result.status

    # Process Result
    await process_imagine_webhook(
        imagination, ImagineWebhookData(**result.model_dump())
    )

    # Stop Short polling when the request is finished
    if imagination.status.is_done:
        return

    return await imagine_update(imagination, i + 1)


@basic.try_except_wrapper
async def update_imagination_status(imagination: Imagination):
    try:
        if imagination.meta_data is None:
            raise ValueError("Imagination has no meta_data.")

        imagine_engine = imagination.engine.get_class(imagination)
        result = await imagine_engine.result()
        imagination.error = result.error
        imagination.status = result.status
        await process_imagine_webhook(
            imagination, ImagineWebhookData(**result.model_dump())
        )
    except Exception as e:
        imagination.status = ImaginationStatus.error
        imagination.task_status = ImaginationStatus.error
        imagination.error = str(e)
        await imagination.save()


@basic.try_except_wrapper
async def imagine_bulk_request(imagination_bulk: ImaginationBulk):
    imagination_bulk.task_references = TaskReferenceList(
        tasks=[],
        mode="parallel",
    )
    imagination_bulk.prompt = await create_prompt(imagination_bulk)
    for aspect_ratio, engine in imagination_bulk.get_combinations():
        imagine = await Imagination.create_item(
            ImagineSchema(
                user_id=imagination_bulk.user_id,
                bulk=str(imagination_bulk.id),
                engine=engine,
                prompt=imagination_bulk.prompt,
                # delineation=imagination_bulk.delineation,
                # context=imagination_bulk.context,
                aspect_ratio=aspect_ratio,
                mode="imagine",
            ).model_dump()
        )
        imagination_bulk.task_references.tasks.append(
            TaskReference(task_id=imagine.uid, task_type="Imagination")
        )

    imagination_bulk.task_status = TaskStatusEnum.processing
    await imagination_bulk.save_report(f"Bulk task was ordered.")
    task_items: list[Imagination] = [
        await task.get_task_item() for task in imagination_bulk.task_references.tasks
    ]
    await asyncio.gather(*[task.start_processing() for task in task_items])
    imagination_bulk = await ImaginationBulk.get_item(
        imagination_bulk.uid, imagination_bulk.user_id
    )
    return imagination_bulk


@basic.try_except_wrapper
async def imagine_bulk_result(
    imagination_bulk: ImaginationBulk, imagination: Imagination
):
    await imagination.save()
    completed_tasks = await imagination_bulk.completed_tasks()
    imagination_bulk.results = await imagination_bulk.collect_results()

    # imagination_bulk.total_completed += 1
    imagination_bulk.total_completed = len(completed_tasks)
    await imagination_bulk.save_report(f"Engine {imagination.engine.value} is ended.")
    await imagine_bulk_process(imagination_bulk)


@basic.try_except_wrapper
async def imagine_bulk_process(imagination_bulk: ImaginationBulk):
    if (
        len(await imagination_bulk.completed_tasks())
        + len(await imagination_bulk.failed_tasks())
        == imagination_bulk.total_tasks
    ):
        imagination_bulk.task_status = TaskStatusEnum.completed
        imagination_bulk.completed_at = datetime.now()
        await imagination_bulk.save_report(f"Bulk task is completed.")
