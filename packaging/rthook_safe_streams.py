"""PyInstaller Runtime Hook para inicializar streams I/O seguros antes de qualquer módulo."""

import os
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

# ─── Carregamento do venv complementar (python_env/) ────────────────────────
# O KmellVox permite instalar dependências pesadas (PyTorch, F5-TTS) em um venv
# separado ao lado do .exe. Adicionamos seu site-packages ao sys.path aqui para
# que os imports funcionem normalmente.
_app_root = os.path.dirname(sys.executable)
_addon_sp = os.path.join(_app_root, "python_env", "Lib", "site-packages")
_DLL_HANDLES = []

if os.path.isdir(_addon_sp) and _addon_sp not in sys.path:
    sys.path.insert(0, _addon_sp)

    # Adiciona a biblioteca padrão do Python do sistema (Lib e DLLs) apontada pelo pyvenv.cfg
    _cfg_path = os.path.join(_app_root, "python_env", "pyvenv.cfg")
    if os.path.isfile(_cfg_path):
        try:
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    if _line.strip().startswith("home ="):
                        _home = _line.split("=", 1)[1].strip()
                        _lib = os.path.join(_home, "Lib")
                        _dlls = os.path.join(_home, "DLLs")
                        if os.path.isdir(_lib):
                            if _lib not in sys.path:
                                sys.path.append(_lib)

                            _dirs = [_addon_sp, _lib]

                            class _FallbackFinder:
                                def __init__(self, search_dirs):
                                    self.search_dirs = search_dirs

                                def find_spec(self, fullname, path=None, target=None):
                                    if path is not None:
                                        parts = fullname.split(".")
                                        for base in self.search_dirs:
                                            candidate = os.path.join(base, *parts[:-1])
                                            if os.path.isdir(candidate) and candidate not in path:
                                                path.append(candidate)
                                    return None

                            sys.meta_path.insert(0, _FallbackFinder(_dirs))

                        if os.path.isdir(_dlls) and _dlls not in sys.path:
                            sys.path.append(_dlls)
                        break
        except Exception:
            pass

    def _add_dll_dir(p: str) -> None:
        if os.path.isdir(p):
            # Prepend no PATH para compatibilidade de carregamento dinâmico
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            # os.add_dll_directory com referência retida (evita remoção por GC)
            if hasattr(os, "add_dll_directory"):
                try:
                    h = os.add_dll_directory(p)
                    _DLL_HANDLES.append(h)
                except OSError:
                    pass

    # 1. torch/lib/ contém torch_python.dll, c10.dll, etc.
    _add_dll_dir(os.path.join(_addon_sp, "torch", "lib"))

    # 2. Pacotes nvidia (cuda_runtime, cublas, cudnn, etc.)
    _nvidia_root = os.path.join(_addon_sp, "nvidia")
    if os.path.isdir(_nvidia_root):
        try:
            for _pkg in os.listdir(_nvidia_root):
                _add_dll_dir(os.path.join(_nvidia_root, _pkg, "bin"))
                _add_dll_dir(os.path.join(_nvidia_root, _pkg, "lib"))
        except OSError:
            pass

    # 3. torchaudio e torchvision
    _add_dll_dir(os.path.join(_addon_sp, "torchaudio", "lib"))
    _add_dll_dir(os.path.join(_addon_sp, "torchvision"))
