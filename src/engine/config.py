"""Runtime configuration.

Every setting can be overridden with an environment variable prefixed ``SENTINEL_``,
e.g. ``SENTINEL_MODE=live`` or ``SENTINEL_REPLAY_SPEED=60``. No setting is a secret:
the engine uses keyless sources only.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class RunMode(StrEnum):
    REPLAY = "replay"  # every source reads its recorded twin; fully offline
    LIVE = "live"  # live connectors; a failing source falls back to its replay twin


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    mode: RunMode = RunMode.REPLAY
    replay_speed: float = Field(
        default=30.0, gt=0, description="Replay clock multiplier (30 = 30x real time)."
    )
    replay_max_gap_s: float = Field(
        default=2.0, gt=0, description="Longest real wait between two replayed docs."
    )
    replay_loop: bool = True

    data_dir: Path = REPO_ROOT / "data"
    sources_file: Path = REPO_ROOT / "data" / "reference" / "sources.yaml"

    user_agent: str = "Sentinel-RiskEngine/0.1 (research prototype; contact sentinel@example.org)"
    http_timeout_s: float = 20.0

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8000"]
    analyze_max_bytes: int = 20_000
    rate_limit_rps: float = 100.0

    pipeline_enabled: bool = True
    zero_shot: bool = True  # load the NLI fallback for low-confidence event predictions
    zero_shot_stream: bool = False  # also use it on the live stream (~0.45 s/doc on CPU); /analyze always may
    nlp_batch_size: int = 64
    onnx_threads: int | None = None  # ONNX Runtime intra-op threads per model (None = min(4, cores))
    signal_buffer: int = 20_000
    write_files: bool = True  # data/output/signals.jsonl + hourly Parquet

    bus_queue_size: int = 10_000
    recent_buffer: int = 2_000
    enabled_sources: list[str] | None = None  # None = everything in sources.yaml

    @property
    def replay_dir(self) -> Path:
        return self.data_dir / "replay"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
