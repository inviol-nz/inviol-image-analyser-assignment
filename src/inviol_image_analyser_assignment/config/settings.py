from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv


def _load_environment_files() -> str:
    load_dotenv(dotenv_path=".env", override=False)

    # Base
    env_name = os.getenv("APP_ENV", "dev")

    # Load environment-specific overrides
    env_file = f".env.{env_name}"
    load_dotenv(dotenv_path=env_file, override=True)

    return env_name


CURRENT_ENV = _load_environment_files()


@dataclass
class Settings:
    """
    Application settings, loaded from environment variables (with defaults).

    Environment-aware behaviour:
    - APP_ENV selects which .env.<APP_ENV> file to use for overrides.
    - Values: "dev", "prod".
    """

    environment: str = CURRENT_ENV

    api_key: str | None = os.getenv("API_KEY")

    max_file_size_bytes: int = int(os.getenv("MAX_FILE_SIZE_BYTES", "10485760"))

    allowed_content_types: Tuple[str, ...] = (
        "image/jpeg",
        "image/png",
    )

    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    cache_size: int = int(os.getenv("CACHE_SIZE", "128"))


settings = Settings()
