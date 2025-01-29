from apps.background_removal.routes import router as background_removal_router
from apps.imagination.routes import bulk_router
from apps.imagination.routes import router as imagination_router
from fastapi_mongo_base.core import app_factory

from . import config, worker

app = app_factory.create_app(
    settings=config.Settings(), worker=worker.worker, original_host_middleware=True
)
app.include_router(imagination_router, prefix=f"{config.Settings.base_path}")
app.include_router(bulk_router, prefix=f"{config.Settings.base_path}")
app.include_router(background_removal_router, prefix=f"{config.Settings.base_path}")
