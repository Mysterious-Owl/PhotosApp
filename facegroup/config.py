"""Typed settings: defaults < %LOCALAPPDATA%\\FaceGroup\\config.toml < FACEGROUP_* environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


def default_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "FaceGroup"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FACEGROUP_", extra="ignore")

    data_dir: Path = Field(default_factory=default_data_dir)

    # Decoding
    working_max_side: int = 2048
    thumbnail_size: int = 320
    max_image_pixels: int = 250_000_000

    # Faces
    face_model: str = "buffalo_l"
    min_face_px: float = 40.0
    min_det_score: float = 0.5
    device: str = "auto"  # auto (DirectML GPU if available, else CPU), directml, cpu
    ort_threads: int = 0  # 0 = let ONNX Runtime choose

    # Pipeline
    batch_size: int = 16
    decode_workers: int = 3
    lease_seconds: int = 900
    max_attempts: int = 3
    backoff_seconds: int = 30
    keep_working_images: bool = False
    missing_grace_days: int = 30

    @property
    def db_path(self) -> Path:
        return self.data_dir / "facegroup.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings) -> tuple[PydanticBaseSettingsSource, ...]:
        data_dir = Path(os.environ.get("FACEGROUP_DATA_DIR") or default_data_dir())
        toml = TomlConfigSettingsSource(settings_cls, toml_file=data_dir / "config.toml")
        return init_settings, env_settings, toml


def load_settings(**overrides) -> Settings:
    return Settings(**overrides)
