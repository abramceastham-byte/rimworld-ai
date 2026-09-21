"""Load settings from environment variables (and a .env file, if present).

Real environment variables win over values in .env, so you can override a
setting for a single run without editing the file.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    rimapi_url: str


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            f"or set {name} in your environment."
        )
    return value.rstrip("/")


def load_settings() -> Settings:
    load_dotenv()  # does not overwrite variables that are already set
    return Settings(
        ollama_host=_require("OLLAMA_HOST"),
        rimapi_url=_require("RIMAPI_URL"),
    )
