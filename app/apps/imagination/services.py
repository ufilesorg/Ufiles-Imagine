import asyncio
import logging
from datetime import datetime, timedelta
from io import BytesIO

import httpx
from apps.ai.engine import EnginesResponse
from apps.ai.replicate_schemas import PredictionModelWebhookData
from apps.imagination.models import Imagination, ImaginationBulk
from apps.imagination.schemas import (
    ImaginationEngines,
    ImaginationStatus,
    ImagineResponse,
    ImagineSchema,
    MidjourneyWebhookData,
)
from fastapi_mongo_base.tasks import TaskReference, TaskReferenceList, TaskStatusEnum
from fastapi_mongo_base.utils import basic, conditions, imagetools
from PIL import Image
from utils import ai, finance, media


@basic.try_except_wrapper
async def process_result(imagination: Imagination, generated_url: str):
    logging.info(f"process_result {generated_url=}")
    # Download the image
    async with httpx.AsyncClient() as client:
        response = await client.get(generated_url)
        image_bytes = BytesIO(response.content)
    images = [Image.open(image_bytes)]
    file_upload_dir = "imaginations"

    # Crop the image into 4 sections for midjourney engine
    if imagination.engine == ImaginationEngines.midjourney:
        images = imagetools.split_image(images[0], sections=(2, 2))

    # Upload result images on ufiles
    uploaded_items = await media.upload_images(
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


async def process_imagine_webhook(
    imagination: Imagination,
    data: MidjourneyWebhookData | PredictionModelWebhookData | EnginesResponse,
):
    import json_advanced as json

    if data.status == "error":
        logging.info(
            f"Error processing image: {json.dumps(data.model_dump(), indent=2, ensure_ascii=False)}"
        )
        await imagination.retry(data.error)
        return imagination

    if data.status == "completed":
        if isinstance(data, MidjourneyWebhookData | EnginesResponse):
            result_url = (data.result or {}).get("uri")
        elif isinstance(data, PredictionModelWebhookData):
            result_url = data.output
        elif isinstance(data, EnginesResponse):
            result_url = data.result
        else:
            raise ValueError(f"Invalid data type: {type(data)}")

        if isinstance(result_url, str):
            result_url = [result_url]
        for url in result_url:
            await process_result(imagination, url)

        # Add condition notification with logging
        await conditions.Conditions().release_condition(imagination.uid)

    imagination.task_progress = (
        getattr(data, "percentage", data.status.progress)
        if not data.status.is_done
        else 100
    )
    imagination.task_status = data.status.task_status
    # logging.info(
    #     f"{imagination.engine.value=} {imagination.task_progress=} {imagination.task_status=} {len(_conditions)=} {type(data).__name__=}"
    # )

    report = (
        f"{imagination.engine.value} completed."
        if data.status == "completed"
        else f"{imagination.engine.value} update. {imagination.status}"
    )

    await imagination.save_report(report)

    if data.status == "completed" and imagination.task_status != "completed":
        logging.info(f"task completed")
        logging.info(json.dumps(imagination.model_dump(), indent=2, ensure_ascii=False))

    if not data.status.is_done and datetime.now() - imagination.created_at >= timedelta(
        minutes=10
    ):
        imagination.task_status = TaskStatusEnum.error
        imagination.status = ImaginationStatus.error
        imagination.error = "Service Timeout Error: The service did not provide a result within the expected time frame."
        await imagination.fail(
            f"{imagination.engine.value} service didn't respond in time."
        )
        await conditions.Conditions().release_condition(imagination.uid)


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


async def register_cost(imagination: Imagination):
    usage = await finance.meter_cost(imagination.user_id, imagination.engine.price)
    if usage is None:
        logging.error(
            f"Insufficient balance. {imagination.user_id} {imagination.engine.value}"
        )
        await imagination.fail("Insufficient balance.")
        return

    imagination.usage_id = usage.uid
    return imagination


async def imagine_request(imagination: Imagination, **kwargs):
    try:
        # Get Engine class and validate it
        imagine_engine = imagination.engine.get_class()
        if imagine_engine is None:
            raise NotImplementedError(
                "The supported engines are Midjourney, Replicate and Dalle."
            )

        if imagination.usage_id is None:
            await register_cost(imagination)
        
        # Create prompt using context attributes (ratio, style ...)
        imagination.prompt = await create_prompt(imagination)

        # Request to client or api using Engine classes
        imagine_request = await imagine_engine.imagine(imagination)

        # Store Engine response
        imagination.meta_data = (
            imagination.meta_data or {}
        ) | imagine_request.model_dump()
        imagination.error = imagine_request.error
        imagination.status = imagine_request.status
        imagination.task_status = imagine_request.status.task_status
        await imagination.save_report(f"{imagination.engine.value} has been requested.")

        if imagination.engine.core == "dalle":
            return await process_imagine_webhook(imagination, imagine_request)

        if kwargs.get("sync", False):
            await conditions.Conditions().wait_condition(imagination.uid)

        imagination = await Imagination.get_item(imagination.uid, imagination.user_id)
        return imagination

    except Exception as e:
        import traceback

        traceback_str = "".join(traceback.format_tb(e.__traceback__))
        logging.error(f"Error updating imagination status: \n{traceback_str}\n{e}")

        imagination.status = ImaginationStatus.error
        imagination.task_status = ImaginationStatus.error
        imagination.error = str(e)
        await conditions.Conditions().release_condition(imagination.uid)

        await imagination.fail(str(e))
        return imagination


async def check_imagination_status(imagination: Imagination):
    imagine_engine = imagination.engine.get_class()

    # Get Result from service by engine class
    # And Update imagination status
    result = await imagine_engine.result(imagination)
    if result:
        imagination.error = result.error
        imagination.status = result.status

    if imagination.engine.core == "midjourney":
        data = MidjourneyWebhookData(**result.model_dump())
    elif imagination.engine.core == "replicate":
        data = PredictionModelWebhookData(**result.model_dump())
    else:
        # dalle
        pass

    # Process Result
    return await process_imagine_webhook(imagination, data)


async def update_imagination_status(imagination: Imagination):
    try:
        await check_imagination_status(imagination)

        # If status is done, notify with logging
        if imagination.status.is_done:
            await conditions.Conditions().release_condition(imagination.uid)

    except Exception as e:
        import traceback

        traceback_str = "".join(traceback.format_tb(e.__traceback__))
        logging.error(f"Error updating imagination status: \n{traceback_str}\n{e}")

        imagination.status = ImaginationStatus.error
        imagination.task_status = ImaginationStatus.error
        imagination.error = str(e)
        await conditions.Conditions().release_condition(imagination.uid)

        await imagination.fail(str(e))


# @basic.try_except_wrapper
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
                bulk=imagination_bulk.uid,
                engine=engine,
                prompt=imagination_bulk.prompt,
                delineation=imagination_bulk.delineation,
                context=imagination_bulk.context,
                aspect_ratio=aspect_ratio,
                mode="imagine",
                webhook_url=imagination_bulk.webhook_url,
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
    logging.info(f"Bulk task items: {len(task_items)}")
    await asyncio.gather(*[task.start_processing(sync=True) for task in task_items])
    imagination_bulk = await ImaginationBulk.get_item(
        imagination_bulk.uid, imagination_bulk.user_id
    )
    imagination_bulk.task_status = TaskStatusEnum.completed
    await imagination_bulk.save_report(f"Bulk Imagination completed.")
    return imagination_bulk
