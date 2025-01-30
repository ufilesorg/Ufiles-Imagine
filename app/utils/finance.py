import uuid
from typing import AsyncGenerator
from decimal import Decimal
from contextlib import asynccontextmanager

from aiocache import cached
from fastapi_mongo_base.utils import basic
from server.config import Settings
from ufaas import AsyncUFaaS, exceptions
from ufaas.apps.saas.schemas import UsageCreateSchema, UsageSchema

resource_variant = getattr(Settings, "UFAAS_RESOURCE_VARIANT", "subtitle")


@asynccontextmanager
async def get_ufaas_client() -> AsyncGenerator[AsyncUFaaS, None]:
    client = AsyncUFaaS(
        ufaas_base_url=Settings.UFAAS_BASE_URL,
        usso_base_url=Settings.USSO_BASE_URL,
        api_key=Settings.UFILES_API_KEY,
    )
    try:
        yield client
    finally:
        # Add cleanup here if needed in the future
        pass


async def meter_cost(user_id: uuid.UUID, amount: float) -> UsageSchema:
    async with get_ufaas_client() as ufaas_client:
        usage_schema = UsageCreateSchema(
            user_id=user_id,
            asset="coin",
            amount=amount,
            variant=resource_variant,
        )
        usage = await ufaas_client.saas.usages.create_item(
            usage_schema.model_dump(mode="json"), timeout=30
        )
        return usage


@basic.try_except_wrapper
@cached(ttl=5)
async def get_quota(user_id: uuid.UUID) -> Decimal:
    async with get_ufaas_client() as ufaas_client:
        quotas = await ufaas_client.saas.enrollments.get_quotas(
            user_id=user_id,
            asset="coin",
            variant=resource_variant,
            timeout=30,
        )
    return quotas.quota


@basic.try_except_wrapper
async def cancel_usage(usage_id: uuid.UUID) -> None:
    async with get_ufaas_client() as ufaas_client:
        await ufaas_client.saas.usages.cancel_item(usage_id)


async def check_quota(user_id: uuid.UUID, coin: float) -> Decimal:
    quota = await get_quota(user_id)
    if quota is None or quota < coin:
        raise exceptions.InsufficientFunds(
            f"You have only {quota} coins, while you need {coin} coins."
        )
    return quota
