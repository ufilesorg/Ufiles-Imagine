"""FastAPI server configuration."""

import dataclasses
import os
from pathlib import Path

import dotenv
from fastapi_mongo_base.core.config import Settings as BaseSettings

dotenv.load_dotenv()


@dataclasses.dataclass
class Settings(BaseSettings):
    """Server config settings."""

    base_dir: Path = Path(__file__).resolve().parent.parent
    base_path: str = "/v1/apps/imagine"
    update_time: int = 6

    METIS_DALLE_BOT_ID: str = os.getenv("METIS_DALLE_BOT_ID")
    METIS_BOT_ID: str = os.getenv("METIS_BOT_ID")
    METIS_API_KEY: str = os.getenv("METIS_API_KEY")

    UFILES_API_KEY: str = os.getenv("UFILES_API_KEY")
    UFILES_BASE_URL: str = os.getenv("UFILES_URL")
    UFAAS_BASE_URL: str = os.getenv("UFAAS_BASE_URL")
    USSO_BASE_URL: str = os.getenv("USSO_URL")

    base_image_price: int = 10_000
