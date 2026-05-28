from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


RANDOM_STATE = 42


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Fixa fontes de aleatoriedade usadas no projeto."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(paths: list[Path]) -> None:
    """Cria diretórios se eles não existirem."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Salva um dicionário em um arquivo JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_text(text: str, path: Path) -> None:
    """Salva um texto em um arquivo de texto."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_serializable(value: Any) -> Any:
    """Converte valores para tipos serializáveis, como int, float, list e dict, para garantir que possam ser salvos em formatos como JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): as_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_serializable(v) for v in value]
    return value
