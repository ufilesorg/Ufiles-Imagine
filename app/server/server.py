import asyncio
import logging
from contextlib import asynccontextmanager

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mongo_base.core import db, exceptions
from ufaas_fastapi_business.core import middlewares
from usso.fastapi.integration import EXCEPTION_HANDLERS as USSO_EXCEPTION_HANDLERS

from . import config, worker


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):  # type: ignore
    """Initialize application services."""
    config.Settings().config_logger()
    await db.init_mongo_db()
    app.state.worker = asyncio.create_task(worker.worker())

    logging.info("Startup complete")
    yield
    app.state.worker.cancel()
    logging.info("Shutdown complete")


app = fastapi.FastAPI(
    title=config.Settings.project_name.replace("-", " ").title(),
    # description=DESCRIPTION,
    version="0.1.0",
    contact={
        "name": "Mahdi Kiani",
        "url": "https://github.com/mahdikiani/FastAPILaunchpad",
        "email": "mahdikiany@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/mahdikiani/FastAPILaunchpad/blob/main/LICENSE",
    },
    docs_url=f"{config.Settings.base_path}/docs",
    openapi_url=f"{config.Settings.base_path}/openapi.json",
    lifespan=lifespan,
)

for exc_class, handler in (
    exceptions.EXCEPTION_HANDLERS | USSO_EXCEPTION_HANDLERS
).items():
    app.exception_handler(exc_class)(handler)


origins = [
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(middlewares.OriginalHostMiddleware)

from apps.background_removal.routes import router as background_removal_router
from apps.imagination.routes import bulk_router
from apps.imagination.routes import router as imagination_router

app.include_router(imagination_router, prefix=f"{config.Settings.base_path}")
app.include_router(bulk_router, prefix=f"{config.Settings.base_path}")
app.include_router(background_removal_router, prefix=f"{config.Settings.base_path}")


@app.get(f"{config.Settings.base_path}/health")
async def health(request: fastapi.Request):
    return {
        "status": "up",
    }


@app.get(f"{config.Settings.base_path}/logs", include_in_schema=False)
async def logs():
    from collections import deque

    with open(config.Settings.base_dir / "logs" / "info.log", "rb") as f:
        last_100_lines = deque(f, maxlen=100)

    return [line.decode("utf-8") for line in last_100_lines]
