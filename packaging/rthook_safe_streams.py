"""PyInstaller Runtime Hook para inicializar streams I/O seguros antes de qualquer módulo."""

import sys


class SafeStream:
    def write(self, text: str) -> int:
        return len(text) if text else 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def reconfigure(self, **kwargs) -> None:
        pass

    def writelines(self, lines) -> None:
        pass


if sys.stdout is None or not hasattr(sys.stdout, "write"):
    sys.stdout = SafeStream()

if sys.stderr is None or not hasattr(sys.stderr, "write"):
    sys.stderr = SafeStream()
