"""Módulo de extração e pré-processamento de áudio a partir de arquivos de vídeo usando ffmpeg-python."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import ffmpeg

logger = logging.getLogger("KmellVox.AudioExtract")


@dataclass
class AudioMetadata:
    """Metadados de áudio extraídos pelo FFprobe."""
    duration_seconds: float = 0.0
    sample_rate: int = 16000
    channels: int = 1
    codec_name: str = "pcm_s16le"
    bit_rate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "codec_name": self.codec_name,
            "bit_rate": self.bit_rate,
        }


def resolve_ffmpeg_binary(custom_path: Optional[str] = None) -> str:
    """Resolve o caminho executável do FFmpeg (customizado, pasta do executável congelado, raiz do projeto ou PATH)."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    # 1. Verifica na pasta do executável congelado (PyInstaller)
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        frozen_cand = exe_dir / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
        if frozen_cand.is_file():
            return str(frozen_cand.resolve())
        if hasattr(sys, "_MEIPASS"):
            meipass_cand = Path(sys._MEIPASS) / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
            if meipass_cand.is_file():
                return str(meipass_cand.resolve())

    # 2. Verifica a partir da localização deste arquivo no projeto
    project_root = Path(__file__).resolve().parent.parent
    root_cand = project_root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if root_cand.is_file():
        return str(root_cand.resolve())

    # 3. Verifica no diretório de trabalho atual
    cwd_cand = Path("tools/ffmpeg/bin/ffmpeg.exe").resolve()
    if cwd_cand.is_file():
        return str(cwd_cand.resolve())

    # 4. Verifica no PATH do sistema
    found = shutil.which("ffmpeg")
    if found:
        return found

    return "ffmpeg"


def resolve_ffprobe_binary(custom_path: Optional[str] = None) -> str:
    """Resolve o caminho executável do FFprobe (customizado, pasta do executável congelado, raiz do projeto ou PATH)."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    # 1. Verifica na pasta do executável congelado (PyInstaller)
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        frozen_cand = exe_dir / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
        if frozen_cand.is_file():
            return str(frozen_cand.resolve())
        if hasattr(sys, "_MEIPASS"):
            meipass_cand = Path(sys._MEIPASS) / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
            if meipass_cand.is_file():
                return str(meipass_cand.resolve())

    # 2. Verifica a partir da localização deste arquivo no projeto
    project_root = Path(__file__).resolve().parent.parent
    root_cand = project_root / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
    if root_cand.is_file():
        return str(root_cand.resolve())

    # 3. Verifica no diretório de trabalho atual
    cwd_cand = Path("tools/ffmpeg/bin/ffprobe.exe").resolve()
    if cwd_cand.is_file():
        return str(cwd_cand.resolve())

    # 4. Verifica no PATH do sistema
    found = shutil.which("ffprobe")
    if found:
        return found

    return "ffprobe"


def get_audio_info(input_file: str, ffprobe_bin: Optional[str] = None) -> AudioMetadata:
    """
    Obtém metadados de áudio/vídeo utilizando ffprobe.
    """
    bin_path = resolve_ffprobe_binary(ffprobe_bin)
    cmd = [
        bin_path,
        "-v", "error",
        "-show_entries", "stream=codec_name,channels,sample_rate,bit_rate,duration:format=duration",
        "-select_streams", "a:0",
        "-of", "json",
        str(input_file)
    ]
    
    try:
        _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            creationflags=_NO_WINDOW,
        )
        data = json.loads(result.stdout)
        
        streams = data.get("streams", [])
        meta = AudioMetadata()
        if streams:
            s = streams[0]
            meta.codec_name = s.get("codec_name", "unknown")
            meta.channels = int(s.get("channels", 1))
            meta.sample_rate = int(s.get("sample_rate", 16000))
            meta.bit_rate = int(s.get("bit_rate", 0)) if s.get("bit_rate") else 0
            if "duration" in s:
                meta.duration_seconds = float(s["duration"])

        if meta.duration_seconds == 0.0 and "format" in data and "duration" in data["format"]:
            meta.duration_seconds = float(data["format"]["duration"])

        return meta
    except Exception as e:
        logger.warning("Falha ao ler metadados com ffprobe: %s", e)
        return AudioMetadata()


def extract_audio(
    video_path: str,
    output_path: str,
    ffmpeg_bin: Optional[str] = None,
    overwrite: bool = True
) -> str:
    """
    Extrai o áudio do vídeo e converte em WAV mono 16kHz utilizando ffmpeg-python
    (formato ideal para transcrição com o Whisper).
    
    Args:
        video_path: Caminho do arquivo de vídeo de entrada.
        output_path: Caminho de destino para o arquivo WAV gerado.
        ffmpeg_bin: Caminho customizado para o executável do FFmpeg (opcional).
        overwrite: Se True, sobrescreve o arquivo de saída se já existir.
        
    Returns:
        str: Caminho absoluto do áudio WAV mono 16kHz gerado.
    """
    bin_path = resolve_ffmpeg_binary(ffmpeg_bin)
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Extraindo áudio via ffmpeg-python (16kHz mono WAV): %s -> %s", video_path, out_file.name)

    try:
        stream = (
            ffmpeg
            .input(str(video_path))
            .output(
                str(out_file),
                acodec="pcm_s16le",
                ac=1,
                ar=16000,
                vn=None
            )
        )
        stream.run(cmd=bin_path, overwrite_output=overwrite, capture_stdout=True, capture_stderr=True)
        logger.info("Áudio extraído com sucesso em: %s", out_file)
        return str(out_file)
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        logger.error("Erro do FFmpeg ao extrair áudio: %s", stderr_msg)
        raise RuntimeError(f"FFmpeg falhou ao extrair áudio mono 16kHz: {stderr_msg}") from e
