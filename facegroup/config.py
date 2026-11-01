"""Typed settings: defaults < data\\app\\config.toml < FACEGROUP_* environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


def _data_root() -> Path:
    """data\\ in the project folder (git-ignored), holding app, eval and bench data.

    Only when running from the source checkout; an installed package would otherwise put the data inside the
    virtual environment, so it falls back to %LOCALAPPDATA%\\FaceGroup.
    """
    project = Path(__file__).resolve().parent.parent
    if (project / "pyproject.toml").exists():
        return project / "data"
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "FaceGroup"


DATA_ROOT = _data_root()


def default_data_dir() -> Path:
    return DATA_ROOT / "app"


def default_eval_dir() -> Path:
    return DATA_ROOT / "eval"


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

    # People (M2). Thresholds are cosine similarity; see docs/EVALUATION.md for how they were chosen.
    assign_k: int = 10
    assign_threshold: float = 0.42
    assign_min_votes: int = 2
    suggest_min_quality: float = 0.15
    suggest_member_similarity: float = 0.40
    suggest_merge_similarity: float = 0.55
    suggest_min_group: int = 3
    # Self-improvement: automatic matches must be right this often, measured on your own labels.
    learn_target_precision: float = 0.98

    # Text in images (OCR): on for screenshots, documents, receipts and memes. ocr_all_photos also reads ordinary
    # photos, off by default because it is slow (~2 s per text-heavy image on CPU).
    ocr_enabled: bool = True
    ocr_all_photos: bool = False

    # Search (M3). "none" turns image search off.
    search_model: str = "siglip2-base"

    # Web UI
    port: int = 8765

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
