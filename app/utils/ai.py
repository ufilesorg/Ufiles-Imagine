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
