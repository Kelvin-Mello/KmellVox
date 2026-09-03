"""Módulo de verificação e instalação de dependências opcionais do KmellVox.

Responsável por:
- Verificar quais componentes de IA (PyTorch, F5-TTS, IndexTTS-2, etc.) estão disponíveis.
- Localizar o Python do sistema para instalação de pacotes.
- Gerenciar um venv complementar (python_env/) para dependências pesadas de GPU.
- Fornecer progresso e status para a UI durante a instalação.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("KmellVox.DependencyManager")


@dataclass
class DependencyStatus:
    """Status de um componente/dependência."""
    name: str
    display_name: str
    installed: bool
    version: str = ""
    detail: str = ""
    has_update: bool = False
    latest_version: str = ""


def _get_app_root() -> Path:
    """Retorna a raiz do app (funciona tanto dev quanto PyInstaller frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent.parent


def get_addon_env_path() -> Path:
    """Retorna o caminho do venv complementar para dependências de GPU."""
    return _get_app_root() / "python_env"


def get_addon_site_packages() -> Optional[Path]:
    """Retorna o site-packages do venv complementar, se existir."""
    env_dir = get_addon_env_path()
    # Windows: python_env/Lib/site-packages
    sp = env_dir / "Lib" / "site-packages"
    if sp.is_dir():
        return sp
    return None


def ensure_addon_in_sys_path() -> None:
    """Adiciona o site-packages do python_env ao sys.path e registra DLLs nativas."""
    sp = get_addon_site_packages()
    if sp and str(sp) not in sys.path:
        sys.path.insert(0, str(sp))
        logger.info("Addon site-packages adicionado ao sys.path: %s", sp)

    # Adiciona a biblioteca padrão do Python do sistema (Lib e DLLs) apontada pelo pyvenv.cfg
    cfg_path = get_addon_env_path() / "pyvenv.cfg"
    if cfg_path.is_file():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("home ="):
                        home = Path(line.split("=", 1)[1].strip())
                        lib = home / "Lib"
                        dlls = home / "DLLs"
                        if lib.is_dir():
                            if str(lib) not in sys.path:
                                sys.path.append(str(lib))

                            search_dirs = [str(sp), str(lib)]

                            class _FallbackFinder:
                                def __init__(self, dirs: list):
                                    self.dirs = dirs

                                def find_spec(self, fullname, path=None, target=None):
                                    if path is not None:
                                        parts = fullname.split(".")
                                        for base in self.dirs:
                                            candidate = os.path.join(base, *parts[:-1])
                                            if os.path.isdir(candidate) and candidate not in path:
                                                path.append(candidate)
                                    return None

                            sys.meta_path.insert(0, _FallbackFinder(search_dirs))

                        if dlls.is_dir() and str(dlls) not in sys.path:
                            sys.path.append(str(dlls))
                        break
        except Exception as e:
            logger.warning("Falha ao ler pyvenv.cfg: %s", e)

    # Registra diretórios de DLLs nativas (torch, CUDA) no Windows.
    # Necessário para que módulos .pyd encontrem suas DLLs companheiras.
    if sp and hasattr(os, "add_dll_directory"):
        _register_dll_directories(sp)


_DLL_HANDLES: List[Any] = []


def _register_dll_directories(sp: Path) -> None:
    """Registra DLL directories de torch e nvidia CUDA a partir de um site-packages com retenção de handle."""
    def _add_dir(d: Path) -> None:
        if d.is_dir():
            sd = str(d)
            if sd not in os.environ.get("PATH", ""):
                os.environ["PATH"] = sd + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    h = os.add_dll_directory(sd)
                    _DLL_HANDLES.append(h)
                except OSError:
                    pass

    _add_dir(sp / "torch" / "lib")

    nvidia_root = sp / "nvidia"
    if nvidia_root.is_dir():
        try:
            for pkg in nvidia_root.iterdir():
                if pkg.is_dir():
                    _add_dir(pkg / "bin")
                    _add_dir(pkg / "lib")
        except OSError:
            pass

    _add_dir(sp / "torchaudio" / "lib")
    _add_dir(sp / "torchvision")


# ─── Verificação de Dependências ────────────────────────────────────────────


def _check_torch() -> DependencyStatus:
    """Verifica se PyTorch está disponível e se tem suporte CUDA."""
    try:
        import torch
        cuda_info = ""
        if torch.cuda.is_available():
            cuda_info = f" — CUDA {torch.version.cuda}"
        return DependencyStatus(
            name="torch",
            display_name="PyTorch (GPU/CUDA)",
            installed=True,
            version=f"v{torch.__version__}{cuda_info}",
            detail="Aceleração de IA por GPU",
        )
    except Exception as e:
        logger.warning("PyTorch não disponível: %s", e, exc_info=True)
        return DependencyStatus(
            name="torch",
            display_name="PyTorch (GPU/CUDA)",
            installed=False,
            detail="Necessário para aceleração de IA por GPU",
        )


def _check_f5tts() -> DependencyStatus:
    """Verifica se o F5-TTS está disponível e funcional."""
    try:
        import f5_tts
        from f5_tts.model import CFM  # noqa: F401
        ver = getattr(f5_tts, "__version__", "1.1.22")
        return DependencyStatus(
            name="f5_tts",
            display_name="F5-TTS (Clonagem de Voz)",
            installed=True,
            version=f"v{ver}",
            detail="Motor padrão de clonagem e síntese de voz",
        )
    except Exception as e:
        logger.warning("F5-TTS não disponível: %s", e, exc_info=True)
        return DependencyStatus(
            name="f5_tts",
            display_name="F5-TTS (Clonagem de Voz)",
            installed=False,
            detail="Necessário para gerar áudio com clonagem de voz",
        )


def _check_indextts() -> DependencyStatus:
    """Verifica se o IndexTTS-2 está disponível."""
    try:
        ver = "v2.0.0"
        try:
            from index_tts import IndexTTS  # noqa: F401
        except ImportError:
            import indextts
            v = getattr(indextts, "__version__", "2.0.0")
            ver = f"v{v}" if not str(v).startswith("v") else str(v)
        return DependencyStatus(
            name="index_tts",
            display_name="IndexTTS-2 (Voz Avançada)",
            installed=True,
            version=ver,
            detail="Motor avançado de alta fidelidade (8GB+ VRAM)",
        )
    except Exception as e:
        logger.debug("IndexTTS-2 não disponível: %s", e)
        return DependencyStatus(
            name="index_tts",
            display_name="IndexTTS-2 (Voz Avançada)",
            installed=False,
            detail="Opcional — motor avançado de alta fidelidade",
        )


def _check_faster_whisper() -> DependencyStatus:
    """Verifica se o Faster-Whisper está disponível."""
    try:
        import faster_whisper  # noqa: F401
        ver = getattr(faster_whisper, "__version__", "?")
        return DependencyStatus(
            name="faster_whisper",
            display_name="Faster-Whisper (Transcrição)",
            installed=True,
            version=f"v{ver}",
            detail="Motor de transcrição de áudio para legendagem",
        )
    except Exception as e:
        logger.warning("Faster-Whisper não disponível: %s", e)
        return DependencyStatus(
            name="faster_whisper",
            display_name="Faster-Whisper (Transcrição)",
            installed=False,
            detail="Necessário para transcrição automática de áudio",
        )


def _check_llama_cpp() -> DependencyStatus:
    """Verifica se o llama-cpp-python está disponível."""
    try:
        import llama_cpp  # noqa: F401
        ver = getattr(llama_cpp, "__version__", "?")
        return DependencyStatus(
            name="llama_cpp",
            display_name="Llama-CPP (Tradução LLM)",
            installed=True,
            version=f"v{ver}",
            detail="Motor de tradução com modelos GGUF locais",
        )
    except Exception as e:
        logger.warning("Llama-CPP não disponível: %s", e, exc_info=True)
        return DependencyStatus(
            name="llama_cpp",
            display_name="Llama-CPP (Tradução LLM)",
            installed=False,
            detail="Necessário para tradução automática de texto",
        )


def _check_ffmpeg() -> DependencyStatus:
    """Verifica se FFmpeg está acessível."""
    try:
        from core.audio_extract import resolve_ffmpeg_binary
        ffmpeg_path = resolve_ffmpeg_binary()
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            return DependencyStatus(
                name="ffmpeg",
                display_name="FFmpeg (Processamento A/V)",
                installed=True,
                version="Encontrado",
                detail=f"Executável: {ffmpeg_path}",
            )
    except Exception as e:
        logger.warning("FFmpeg não disponível: %s", e)

    return DependencyStatus(
        name="ffmpeg",
        display_name="FFmpeg (Processamento A/V)",
        installed=False,
        detail="Necessário para processamento de áudio e vídeo",
    )


def _is_newer_version(installed_str: str, latest_str: str) -> bool:
    """Compara duas versões e retorna True se latest_str for estritamente mais recente."""
    if not installed_str or not latest_str:
        return False
    try:
        from packaging import version
        v_inst = version.parse(re.sub(r"^[^\d]*", "", installed_str.split("+")[0]))
        v_lat = version.parse(re.sub(r"^[^\d]*", "", latest_str.split("+")[0]))
        return v_lat > v_inst
    except Exception:
        def to_tuple(v: str):
            clean = re.sub(r"^[^\d]*", "", v.split("+")[0])
            nums = re.findall(r"\d+", clean)
            return tuple(int(x) for x in nums) if nums else (0,)
        return to_tuple(latest_str) > to_tuple(installed_str)


def check_dependency_updates(deps: List[DependencyStatus]) -> List[DependencyStatus]:
    """
    Verifica se há atualizações disponíveis para os componentes instalados em paralelo.
    Não lança exceções em caso de falha de conexão (offline-safe).
    """
    import concurrent.futures

    def _check_single(dep: DependencyStatus) -> None:
        if not dep.installed:
            return
        try:
            if dep.name == "f5_tts":
                req = urllib.request.Request("https://pypi.org/pypi/f5-tts/json", headers={"User-Agent": "KmellVox/1.0"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    latest = data.get("info", {}).get("version", "")
                    if latest and _is_newer_version(dep.version, latest):
                        dep.has_update = True
                        dep.latest_version = f"v{latest}" if not latest.startswith("v") else latest
            elif dep.name == "index_tts":
                req = urllib.request.Request("https://raw.githubusercontent.com/index-tts/index-tts/main/pyproject.toml", headers={"User-Agent": "KmellVox/1.0"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    content = resp.read().decode("utf-8")
                    m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                    if m:
                        latest = m.group(1)
                        if _is_newer_version(dep.version, latest):
                            dep.has_update = True
                            dep.latest_version = f"v{latest}" if not latest.startswith("v") else latest
            elif dep.name == "torch":
                # PyTorch CUDA é fixado em 2.6.0+cu124 para estabilidade e compatibilidade com CUDA 12.4
                pass
        except Exception as e:
            logger.debug("Verificação de atualização online ignorada para %s: %s", dep.name, e)

    target_deps = [d for d in deps if d.installed and d.name in ("f5_tts", "index_tts")]
    if target_deps:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(target_deps))) as executor:
            list(executor.map(_check_single, target_deps))

    return deps


def check_all_dependencies(check_updates: bool = False) -> List[DependencyStatus]:
    """Verifica todas as dependências e retorna lista de status."""
    # Garante que o addon env esteja no path antes de verificar
    ensure_addon_in_sys_path()

    deps = [
        _check_torch(),
        _check_faster_whisper(),
        _check_llama_cpp(),
        _check_f5tts(),
        _check_indextts(),
        _check_ffmpeg(),
    ]
    if check_updates:
        check_dependency_updates(deps)
    return deps


def is_tts_available() -> bool:
    """Verifica rapidamente se pelo menos um motor TTS está disponível."""
    ensure_addon_in_sys_path()
    try:
        import f5_tts  # noqa: F401
        return True
    except Exception:
        pass
    try:
        from index_tts import IndexTTS  # noqa: F401
        return True
    except Exception:
        pass
    return False


# ─── Instalação de Dependências ─────────────────────────────────────────────


def _get_python_version(python_path: Path) -> Optional[tuple]:
    """Retorna (major, minor) de um executável Python, ou None se falhar."""
    try:
        _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            [str(python_path), "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_NO_WINDOW,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return None


# Versões de Python que possuem wheels de PyTorch no índice cu124
_PYTORCH_COMPATIBLE_VERSIONS = {(3, 9), (3, 10), (3, 11), (3, 12)}


def find_system_python() -> Optional[Path]:
    """
    Localiza um Python compatível com PyTorch no sistema.
    
    Prioridade:
        1. Em modo dev: sys.executable (se compatível).
        2. Launcher 'py' do Windows com versões específicas: py -3.12, py -3.11, py -3.10.
        3. Busca direta em caminhos conhecidos (LOCALAPPDATA, C:/Python).
        4. 'python' no PATH (somente se versão compatível).
    
    Rejeita versões >= 3.13 e < 3.9 que não possuem wheels de PyTorch CUDA.
    
    Returns:
        Path do executável Python compatível, ou None.
    """
    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # Em modo desenvolvimento, usa o próprio Python se compatível
    if not getattr(sys, "frozen", False):
        ver = _get_python_version(Path(sys.executable))
        if ver and ver in _PYTORCH_COMPATIBLE_VERSIONS:
            return Path(sys.executable)

    # 1. Tenta o launcher 'py' do Windows com versão explícita (mais confiável)
    py_launcher = shutil.which("py")
    if py_launcher:
        for minor in [12, 11, 10, 9]:
            try:
                result = subprocess.run(
                    [py_launcher, f"-3.{minor}", "-c", "import sys; print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=_NO_WINDOW,
                )
                if result.returncode == 0:
                    exe_path = Path(result.stdout.strip())
                    if exe_path.is_file():
                        logger.info("Python 3.%d encontrado via launcher 'py': %s", minor, exe_path)
                        return exe_path
            except Exception:
                continue

    # 2. Busca direta em caminhos conhecidos no Windows
    for minor in [12, 11, 10, 9]:
        folder = f"Python3{minor}"
        for base in [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / folder,
            Path("C:/Python") / folder,
            Path(f"C:/Python3{minor}"),
        ]:
            exe = base / "python.exe"
            if exe.is_file():
                ver = _get_python_version(exe)
                if ver and ver in _PYTORCH_COMPATIBLE_VERSIONS:
                    logger.info("Python %d.%d encontrado em: %s", ver[0], ver[1], exe)
                    return exe

    # 3. Fallback: 'python' no PATH (valida versão antes de usar)
    for name in ["python", "python3"]:
        found = shutil.which(name)
        if found:
            found_path = Path(found)
            if found_path != Path(sys.executable):
                ver = _get_python_version(found_path)
                if ver and ver in _PYTORCH_COMPATIBLE_VERSIONS:
                    logger.info("Python %d.%d encontrado no PATH: %s", ver[0], ver[1], found_path)
                    return found_path
                elif ver:
                    logger.warning(
                        "Python %d.%d encontrado no PATH (%s) não é compatível com PyTorch. "
                        "Versões suportadas: 3.9–3.12.",
                        ver[0], ver[1], found_path,
                    )

    return None


def _ensure_addon_pip(
    notify_func: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Path, str]:
    """
    Garante que o venv complementar (python_env) exista e retorne (env_path, pip_exe).
    """
    def notify(pct: float, msg: str) -> None:
        if notify_func:
            notify_func(pct, msg)
        logger.info("[%.0f%%] %s", pct * 100, msg)

    notify(0.02, "Localizando Python compatível no sistema...")
    system_python = find_system_python()
    if not system_python:
        raise RuntimeError(
            "Python compatível não encontrado no sistema (requer Python 3.9 a 3.12).\n\n"
            "Para instalar as dependências de voz, o KmellVox precisa do Python instalado.\n"
            "Baixe em: https://www.python.org/downloads/\n\n"
            "Ao instalar, marque a opção 'Add Python to PATH'."
        )

    env_path = get_addon_env_path()
    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # Cria venv se ainda não existir
    if not (env_path / "Scripts" / "python.exe").is_file():
        notify(0.08, "Criando ambiente virtual complementar (python_env)...")
        result = subprocess.run(
            [str(system_python), "-m", "venv", str(env_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_NO_WINDOW,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Falha ao criar venv: {result.stderr}")
    else:
        notify(0.08, "Ambiente virtual localizado...")

    pip_exe = str(env_path / "Scripts" / "pip.exe")
    if not Path(pip_exe).is_file():
        raise RuntimeError(
            f"pip não encontrado em {pip_exe}. "
            "Tente deletar a pasta python_env/ e repetir a operação."
        )

    return env_path, pip_exe


def install_pytorch(
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Instala ou atualiza o PyTorch com suporte a CUDA 12.4."""
    def notify(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)
        logger.info("[%.0f%%] %s", pct * 100, msg)

    env_path, pip_exe = _ensure_addon_pip(progress_callback)
    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    notify(0.20, "Baixando e instalando PyTorch + CUDA 12.4 (~2.5 GB)...")
    result = subprocess.run(
        [
            pip_exe, "install", "--upgrade",
            "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao instalar PyTorch:\n{result.stderr[-500:]}")

    ensure_addon_in_sys_path()
    notify(1.0, "PyTorch (CUDA 12.4) instalado com sucesso!")
    return {"success": True, "message": "PyTorch (CUDA) instalado/atualizado com sucesso."}


def install_f5tts(
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Instala ou atualiza o motor F5-TTS."""
    def notify(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)
        logger.info("[%.0f%%] %s", pct * 100, msg)

    env_path, pip_exe = _ensure_addon_pip(progress_callback)
    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    notify(0.25, "Instalando / Atualizando F5-TTS e dependências de áudio...")
    result = subprocess.run(
        [pip_exe, "install", "--upgrade", "f5-tts"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao instalar F5-TTS:\n{result.stderr[-500:]}")

    ensure_addon_in_sys_path()
    notify(1.0, "F5-TTS instalado com sucesso!")
    return {"success": True, "message": "F5-TTS instalado/atualizado com sucesso."}


def install_indextts(
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Instala o motor avançado IndexTTS-2 e suas dependências."""
    def notify(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)
        logger.info("[%.0f%%] %s", pct * 100, msg)

    env_path, pip_exe = _ensure_addon_pip(progress_callback)
    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    notify(0.20, "Instalando bibliotecas auxiliares do IndexTTS-2...")
    result_deps = subprocess.run(
        [pip_exe, "install", "--upgrade", "scipy", "soundfile", "librosa"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
        timeout=600,
    )
    if result_deps.returncode != 0:
        raise RuntimeError(f"Falha ao instalar dependências do IndexTTS-2:\n{result_deps.stderr[-500:]}")

    notify(0.60, "Instalando o pacote IndexTTS-2...")
    result_pkg = subprocess.run(
        [pip_exe, "install", "--upgrade", "https://codeload.github.com/index-tts/index-tts/zip/refs/heads/main", "--no-deps"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
        timeout=600,
    )
    if result_pkg.returncode != 0:
        raise RuntimeError(f"Falha ao instalar pacote IndexTTS-2:\n{result_pkg.stderr[-500:]}")

    ensure_addon_in_sys_path()
    notify(1.0, "IndexTTS-2 instalado com sucesso!")
    return {"success": True, "message": "IndexTTS-2 instalado e pronto para uso."}


def install_smart_all(
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Verifica inteligentemente quais componentes faltam ou possuem atualização disponível.
    Se tudo já estiver instalado e na versão mais recente, não reinstala desnecessariamente.
    """
    def notify(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)
        logger.info("[%.0f%%] %s", pct * 100, msg)

    notify(0.05, "Analisando componentes e verificando versões...")
    deps = check_all_dependencies(check_updates=True)
    dep_dict = {d.name: d for d in deps}

    actions_taken = []

    # 1. PyTorch
    torch_dep = dep_dict.get("torch")
    if not torch_dep or not torch_dep.installed:
        notify(0.15, "PyTorch ausente. Instalando PyTorch (CUDA 12.4)...")
        install_pytorch(progress_callback=lambda p, m: notify(0.15 + p * 0.40, m))
        actions_taken.append("PyTorch CUDA instalado")
    else:
        notify(0.45, "PyTorch já instalado e operacional.")

    # 2. F5-TTS
    f5_dep = dep_dict.get("f5_tts")
    if not f5_dep or not f5_dep.installed:
        notify(0.50, "F5-TTS ausente. Instalando F5-TTS...")
        install_f5tts(progress_callback=lambda p, m: notify(0.50 + p * 0.25, m))
        actions_taken.append("F5-TTS instalado")
    elif getattr(f5_dep, "has_update", False):
        notify(0.50, f"Nova versão do F5-TTS detectada ({f5_dep.latest_version}). Atualizando...")
        install_f5tts(progress_callback=lambda p, m: notify(0.50 + p * 0.25, m))
        actions_taken.append(f"F5-TTS atualizado para {f5_dep.latest_version}")
    else:
        notify(0.75, "F5-TTS já está na versão mais recente.")

    # 3. IndexTTS-2
    idx_dep = dep_dict.get("index_tts")
    if not idx_dep or not idx_dep.installed:
        notify(0.78, "IndexTTS-2 ausente. Instalando IndexTTS-2...")
        install_indextts(progress_callback=lambda p, m: notify(0.78 + p * 0.20, m))
        actions_taken.append("IndexTTS-2 instalado")
    elif getattr(idx_dep, "has_update", False):
        notify(0.78, f"Nova versão do IndexTTS-2 detectada ({idx_dep.latest_version}). Atualizando...")
        install_indextts(progress_callback=lambda p, m: notify(0.78 + p * 0.20, m))
        actions_taken.append(f"IndexTTS-2 atualizado para {idx_dep.latest_version}")
    else:
        notify(0.95, "IndexTTS-2 já está na versão mais recente.")

    ensure_addon_in_sys_path()
    notify(1.0, "Processamento concluído com sucesso!")

    if actions_taken:
        summary = f"Concluído: {', '.join(actions_taken)}."
    else:
        summary = "Todas as dependências já estão instaladas e na versão mais recente."

    return {"success": True, "message": summary}


def install_tts_dependencies(
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Alias de compatibilidade para install_smart_all."""
    return install_smart_all(progress_callback=progress_callback)
