import asyncio
import json
import uuid

import ufiles
from apps.imagination.schemas import ImaginationEngines
from fastapi_mongo_base.utils import imagetools, texttools
from PIL import Image
from server.config import Settings


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
    image_bytes = imagetools.convert_image_bytes(image, format="JPEG", quality=90)
    image_bytes.name = f"{engine.value}_{image_name}.jpg"
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
    image_name = texttools.sanitize_filename(prompt, 40)

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
