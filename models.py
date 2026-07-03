import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    backend: str = "mlx"
    path: Path | None = None
    base_url: str | None = None
    model_id: str | None = None


MODELS = {
    "qwen35": ModelConfig(
        key="qwen35",
        name="Qwen3.5-27B Opus-Distilled 4bit",
        path=Path("/Users/mojoservo/.omlx/models/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"),
    ),
    "qwen36": ModelConfig(
        key="qwen36",
        name="Qwen3.6-27B oQ8 MTP",
        path=Path("/Users/mojoservo/.omlx/models/Jundot/Qwen3.6-27B-oQ8-mtp"),
    ),
    "ds4": ModelConfig(
        key="ds4",
        name="DeepSeek V4 Flash DS4",
        backend="openai",
        base_url=os.environ.get("DS4_BASE_URL", "http://127.0.0.1:8001/v1"),
        model_id=os.environ.get("DS4_MODEL", "deepseek-v4-flash"),
    ),
}


def get_selected_model():
    key = os.environ.get("MLX_MODEL", "qwen35").strip().lower()
    try:
        return MODELS[key]
    except KeyError:
        available = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown MLX_MODEL '{key}'. Available: {available}") from None
