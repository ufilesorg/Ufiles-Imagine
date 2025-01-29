import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from io import BytesIO

import httpx
import ufiles
from aiocache import cached
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
from fastapi_mongo_base.utils import basic, imagetools, texttools
from PIL import Image
from server.config import Settings
from singleton import Singleton
from ufaas import AsyncUFaaS, exceptions
from ufaas.apps.saas.schemas import UsageCreateSchema
from utils import ai


async def meter_cost(imagination: Imagination):
    ufaas_client = AsyncUFaaS(
        ufaas_base_url=Settings.UFAAS_BASE_URL,
        usso_base_url=Settings.USSO_BASE_URL,
        # TODO: Change to UFAAS_API_KEY name
        api_key=Settings.UFILES_API_KEY,
    )
    usage_schema = UsageCreateSchema(
        user_id=imagination.user_id,
        asset="coin",
        amount=imagination.engine.price,
        variant="imagine",
    )
    usage = await ufaas_client.saas.usages.create_item(
        usage_schema.model_dump(mode="json"), timeout=30
    )
    imagination.usage_id = usage.uid
    await imagination.save()
    return usage


@basic.try_except_wrapper
@cached(ttl=5)
async def get_quota(user_id: uuid.UUID):
    ufaas_client = AsyncUFaaS(
        ufaas_base_url=Settings.UFAAS_BASE_URL,
        usso_base_url=Settings.USSO_BASE_URL,
        # TODO: Change to UFAAS_API_KEY name
        api_key=Settings.UFILES_API_KEY,
    )
    quotas = await ufaas_client.saas.enrollments.get_quotas(
        user_id=user_id,
        asset="coin",
        variant="imagine",
        timeout=30,
    )
    return quotas.quota


@basic.try_except_wrapper
async def cancel_usage(imagination: Imagination):
    if imagination.usage_id is None:
        return

    ufaas_client = AsyncUFaaS(
        ufaas_base_url=Settings.UFAAS_BASE_URL,
        usso_base_url=Settings.USSO_BASE_URL,
        api_key=Settings.UFILES_API_KEY,
    )
    await ufaas_client.saas.usages.cancel_item(imagination.usage_id)


async def check_quota(user_id: uuid.UUID, coin: float):
    quota = await get_quota(user_id)
    if quota is None or quota < coin:
        raise exceptions.InsufficientFunds(
            f"You have only {quota} coins, while you need {coin} coins."
        )
    return quota
