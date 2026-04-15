from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve_runtime(executable: str, local_candidates: list[str]) -> str | None:
    for candidate in local_candidates:
        path = ROOT / candidate
        if path.exists():
            return str(path)
    return shutil.which(executable)
