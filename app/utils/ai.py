import httpx
from fastapi_mongo_base.utils import basic
import os

from fastapi_mongo_base.utils.basic import retry_execution, try_except_wrapper
from usso.session import AsyncUssoSession


@try_except_wrapper
@retry_execution(attempts=3, delay=1)
async def answer_with_ai(key, **kwargs) -> dict:
    kwargs["source_language"] = kwargs.get("lang", "Persian")
    kwargs["target_language"] = kwargs.get("target_language", "English")
    async with AsyncUssoSession(
        usso_refresh_url=os.getenv("USSO_REFRESH_URL"),
        api_key=os.getenv("UFILES_API_KEY"),
    ) as session:
        response = await session.post(f'{os.getenv("PROMPTLY_URL")}/{key}', json=kwargs)
        response.raise_for_status()
        return response.json()


async def translate(text: str) -> str:
    resp: dict = await answer_with_ai("graphic_translate", text=text)
    return resp.get("translated_text")


class PromptlyClient(httpx.AsyncClient):
    PROMPTLY_URL = os.getenv("PROMPTLY_URL")
    UFILES_API_KEY = os.getenv("UFILES_API_KEY")

    def __init__(self):
        super().__init__(
            base_url=self.PROMPTLY_URL,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": self.UFILES_API_KEY,
            },
        )

    @basic.try_except_wrapper
    @basic.retry_execution(attempts=3, delay=1)
    async def ai_image(self, image_url: str, key: str, data: dict = {}) -> dict:
        r = await self.post(f"/image/{key}", json={**data, "image_url": image_url})
        r.raise_for_status()
        return r.json()

    @basic.try_except_wrapper
    @basic.retry_execution(attempts=3, delay=1)
    async def ai(self, key: str, data: dict = {}) -> dict:
        r = await self.post(f"/{key}", json=data)
        r.raise_for_status()
        return r.json()

    async def ai_search(self, key: str, data: dict = {}) -> dict:
        return await self.ai(f"/search/{key}", data)

    async def translate(self, text: str) -> str:
        resp: dict = await self.ai("graphic_translate", text=text)
        return resp.get("translated_text")
