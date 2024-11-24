import json
import logging
import os

import langdetect
from metisai.async_metis import AsyncMetisBot

from apps.imagination.schemas import ImaginationEngines
from utils.texttools import backtick_formatter

metis_client = AsyncMetisBot(
    api_key=os.getenv("METIS_API_KEY"), bot_id="a7a3c055-fd9c-4162-ab02-d5d3053c7b6f"
)


async def metis_chat(messages: dict, **kwargs):
    user_id = kwargs.get("user_id")
    session = await metis_client.create_session(user_id)
    prompt = "\n\n".join([message["content"] for message in messages])
    response = await metis_client.send_message(session, prompt)
    await metis_client.delete_session(session)
    resp_text = backtick_formatter(response.content)
    return resp_text


async def answer_messages(messages: dict, **kwargs):
    # resp_text = await openai_chat(messages, **kwargs)
    resp_text = await metis_chat(messages, **kwargs)
    try:
        return json.loads(resp_text)
    except json.JSONDecodeError:
        return {"answer": resp_text}


async def translate(query: str, to: str = "en"):
    try:
        lang = langdetect.detect(query)
    except:
        lang = "en"

    if lang == to:
        return query

    languages = {
        "en": "English",
        "fa": "Persian",
    }
    if not languages.get(to):
        to = "en"
    prompt = "\n".join(
        [
            f"You are perfect translator to {to} language.",
            f"Just reply the answer in json format like",
            f'`{{"answer": "Your translated text"}}`',
            f"",
            f"Translate the following text to '{to}': \"{query}\".",
        ]
    )

    messages = [{"content": prompt}]
    response = await answer_messages(messages)
    logging.info(f"process_task {query} {response}")
    return response["answer"]

    session = await metis_client.create_session()
    response = await metis_client.send_message(session, prompt)
    await metis_client.delete_session(session)
    resp_text = backtick_formatter(response.content)
    return resp_text


async def build_prompt(
    idea: str,
    engine: ImaginationEngines,
):
    prompt = "\n".join(
        [
            f"Translate my idea: [{idea}] into English. Then, using the AI image generator [{engine.value}], create or improve a optimized prompt  tailored to that generator. Include specific parameters or formatting required by the model.",
            f"Only provide the final text for the prompt, without quotation marks, symbols, or additional explanations.",
        ]
    )

    messages = [{"content": prompt}]
    response = await answer_messages(messages)
    return response["answer"]
