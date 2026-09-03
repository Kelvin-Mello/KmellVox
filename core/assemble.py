"""Módulo de montagem, muxing, estampa de legendas e renderização final via FFmpeg."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .audio_extract import resolve_ffmpeg_binary

logger = logging.getLogger("KmellVox.Assemble")


@dataclass
class AssemblyConfig:
    """Configurações de exportação do vídeo final."""
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    preset: str = "medium"
    crf: int = 20
    burn_subtitles: bool = False
    subtitle_file: Optional[str] = None
    export_raw_package: bool = False
    ffmpeg_bin: Optional[str] = None


def escape_ffmpeg_path(file_path: str) -> str:
    """Escapa caminhos de arquivos para filtros do FFmpeg (especialmente útil no Windows)."""
    p = Path(file_path).resolve().as_posix()
    return p.replace(":", "\\:").replace("'", "\\'")


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    ffmpeg_bin: Optional[str] = None,
    crf: int = 20,
    preset: str = "medium",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Usa o filtro 'subtitles' do FFmpeg para 'queimar' (renderizar) as legendas
    diretamente no vídeo final.
    
    Args:
        video_path: Caminho do vídeo de entrada.
        srt_path: Caminho do arquivo .srt de legendas sincronizadas.
        output_path: Caminho de destino do vídeo resultante.
        ffmpeg_bin: Caminho opcional do executável do FFmpeg.
        crf: Fator de qualidade constante do x264 (padrão 20).
        preset: Preset de codificação do x264 (padrão medium).
        progress_callback: Callback de progresso.
        
    Returns:
        str: Caminho do vídeo gerado com legendas estampadas.
    """
    bin_path = resolve_ffmpeg_binary(ffmpeg_bin)
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"Arquivo de legenda SRT não encontrado: {srt_path}")

    escaped_srt = escape_ffmpeg_path(srt_path)

    cmd = [
        bin_path,
        "-y",
        "-i", str(video_path),
        "-vf", f"subtitles='{escaped_srt}'",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "copy",
        str(out),
    ]

    logger.info("Estampando legendas no vídeo: %s -> %s (SRT: %s)", video_path, out.name, srt_path)

    if progress_callback:
        progress_callback(0.20, "Codificando vídeo com legendas estampadas...")

    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            creationflags=_NO_WINDOW,
        )
        if progress_callback:
            progress_callback(1.0, "Legendas estampadas com sucesso.")
        return str(out)
    except subprocess.CalledProcessError as e:
        logger.error("Erro no FFmpeg ao estampar legendas: %s", e.stderr)
        raise RuntimeError(f"FFmpeg falhou ao estampar legendas: {e.stderr}") from e


def mux_audio_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    ffmpeg_bin: Optional[str] = None,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Substitui a faixa de áudio original pelo novo áudio dublado, mantendo a trilha de vídeo
    intacta e sem recodificação desnecessária (-c:v copy) quando o lip sync estiver desativado.
    
    Args:
        video_path: Caminho do vídeo original de entrada.
        audio_path: Caminho do novo áudio dublado e alinhado (WAV).
        output_path: Caminho de destino do arquivo MP4 final.
        ffmpeg_bin: Caminho opcional do executável do FFmpeg.
        audio_codec: Codec de áudio de saída (padrão aac).
        audio_bitrate: Taxa de bits do áudio (padrão 192k).
        progress_callback: Callback de progresso.
        
    Returns:
        str: Caminho do arquivo de vídeo final montado.
    """
    bin_path = resolve_ffmpeg_binary(ffmpeg_bin)
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        bin_path,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",             # Mantém o fluxo de vídeo intacto, cópia direta sem perda
        "-c:a", audio_codec,        # Codifica o áudio dublado
        "-b:a", audio_bitrate,
        "-map", "0:v:0",             # Primeiro fluxo de vídeo do vídeo de entrada
        "-map", "1:a:0",             # Primeiro fluxo de áudio do áudio dublado
        "-shortest",                 # Encerra quando o menor fluxo terminar
        str(out),
    ]

    logger.info("Muxing áudio dublado com vídeo original: %s + %s -> %s", video_path, audio_path, out.name)

    if progress_callback:
        progress_callback(0.30, "Substituindo faixa de áudio (muxing direto)...")

    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            creationflags=_NO_WINDOW,
        )
        if progress_callback:
            progress_callback(1.0, "Muxing concluído com sucesso.")
        return str(out)
    except subprocess.CalledProcessError as e:
        logger.error("Erro no FFmpeg durante muxing de áudio e vídeo: %s", e.stderr)
        raise RuntimeError(f"FFmpeg falhou no muxing: {e.stderr}") from e


def export_raw_package(
    audio_path: str,
    srt_path: str,
    output_dir: str,
    base_name: Optional[str] = None,
    audio_bitrate: str = "192k",
    ffmpeg_bin: Optional[str] = None,
) -> Dict[str, str]:
    """
    Empacota o áudio dublado convertido em MP3 e a legenda SRT sincronizada
    para quem deseja apenas o pacote de áudio e texto, sem arquivo de vídeo.
    
    Args:
        audio_path: Caminho do arquivo de áudio dublado gerado (WAV).
        srt_path: Caminho do arquivo de legenda .srt correspondente.
        output_dir: Diretório de destino para o pacote exportado.
        base_name: Nome base opcional para os arquivos gerados.
        audio_bitrate: Taxa de bits do MP3 gerado (padrão 192k).
        ffmpeg_bin: Caminho opcional do executável do FFmpeg.
        
    Returns:
        Dict[str, str]: Caminhos gerados {'audio_mp3': '...', 'subtitles_srt': '...'}.
    """
    bin_path = resolve_ffmpeg_binary(ffmpeg_bin)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    name = base_name or Path(audio_path).stem
    out_mp3 = out_dir / f"{name}.mp3"
    out_srt = out_dir / f"{name}.srt"

    # 1. Converte áudio WAV para MP3
    cmd = [
        bin_path,
        "-y",
        "-i", str(audio_path),
        "-c:a", "libmp3lame",
        "-b:a", audio_bitrate,
        str(out_mp3),
    ]

    logger.info("Exportando pacote bruto: MP3 (%s) e SRT (%s)...", out_mp3.name, out_srt.name)

    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            creationflags=_NO_WINDOW,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Erro ao converter áudio para MP3 no pacote bruto: %s", e.stderr)
        raise RuntimeError(f"Falha ao gerar MP3 do pacote bruto: {e.stderr}") from e

    # 2. Copia legenda SRT
    if os.path.isfile(srt_path):
        shutil.copyfile(srt_path, str(out_srt))
    else:
        out_srt.write_text("", encoding="utf-8")

    return {
        "audio_mp3": str(out_mp3),
        "subtitles_srt": str(out_srt),
    }


def assemble_final_video(
    video_source: str,
    audio_source: str,
    output_video: str,
    config: Optional[AssemblyConfig] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Orquestra a montagem final do vídeo combinando muxing de áudio/vídeo
    e estampa de legendas quando habilitada.
    """
    cfg = config or AssemblyConfig()
    out_path = Path(output_video).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.burn_subtitles and cfg.subtitle_file and os.path.isfile(cfg.subtitle_file):
        # Cria arquivo intermediário com áudio muxado e depois queima legendas
        temp_mux = str(out_path.with_suffix(".mux_temp.mp4"))
        try:
            mux_audio_video(
                video_path=video_source,
                audio_path=audio_source,
                output_path=temp_mux,
                ffmpeg_bin=cfg.ffmpeg_bin,
                audio_codec=cfg.audio_codec,
                audio_bitrate=cfg.audio_bitrate,
            )
            burn_subtitles(
                video_path=temp_mux,
                srt_path=cfg.subtitle_file,
                output_path=str(out_path),
                ffmpeg_bin=cfg.ffmpeg_bin,
                crf=cfg.crf,
                preset=cfg.preset,
                progress_callback=progress_callback,
            )
            return str(out_path)
        finally:
            if os.path.isfile(temp_mux):
                try:
                    os.remove(temp_mux)
                except Exception:
                    pass
    else:
        return mux_audio_video(
            video_path=video_source,
            audio_path=audio_source,
            output_path=str(out_path),
            ffmpeg_bin=cfg.ffmpeg_bin,
            audio_codec=cfg.audio_codec,
            audio_bitrate=cfg.audio_bitrate,
            progress_callback=progress_callback,
        )
