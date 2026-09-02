"""Módulo de proteção para fluxos de I/O padrão (sys.stdout / sys.stderr).

Previne erros de 'NoneType object has no attribute write' em ambientes de interface gráfica
(PySide6 / Windows GUI) e executáveis empacotados com PyInstaller (console=False).
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any


class SafeStream:
    """Stream nulo e seguro para ambientes GUI onde sys.stdout ou sys.stderr são None."""

    def write(self, text: str) -> int:
        return len(text) if text else 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def reconfigure(self, **kwargs: Any) -> None:
        pass

    def writelines(self, lines: Any) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def closed(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"


def ensure_safe_streams() -> None:
    """Garante que sys.stdout e sys.stderr nunca sejam None ou inválidos."""
    if sys.stdout is None or not hasattr(sys.stdout, "write"):
        sys.stdout = SafeStream()
    if sys.stderr is None or not hasattr(sys.stderr, "write"):
        sys.stderr = SafeStream()


# Executa automaticamente na importação
ensure_safe_streams()
