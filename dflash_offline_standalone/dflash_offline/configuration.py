"""Draft configuration loading without AngelSlim's model factory."""

import json
from pathlib import Path

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


def load_draft_config(path: str | Path) -> Qwen3Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    architectures = raw.get("architectures", [])
    if architectures != ["QwenDFlashDraftModel"]:
        raise ValueError(
            "The standalone trainer only supports architectures="
            "['QwenDFlashDraftModel']; got " + repr(architectures)
        )

    config = Qwen3Config(**raw)
    config.architectures = architectures
    return config
