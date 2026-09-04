"""Módulo de Geração de Narração em Áudio (Texto Puro e Legendas SRT).

Permite sintetizar áudio falado a partir de texto puro ou blocos de legendas SRT,
utilizando os motores TTS existentes (F5-TTS e IndexTTS-2), com suporte a:
- Exportação em arquivos separados por segmento SRT ou arquivo único concatenado com silêncio proporcional.
- Descoberta automática de vozes pré-definidas/presets.
- Resolução inteligente de pastas de destino (mesma pasta de origem ou Downloads / subpasta 'Áudio').
- Conversão e exportação final em formato MP3 de alta fidelidade.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import soundfile as sf

from core.audio_extract import resolve_ffmpeg_binary
from core.hardware import ModelProfile, detect_hardware
from core.transcribe import TranscriptionSegment
from core.voice_clone import BaseTTSEngine, get_audio_duration, get_tts_engine

logger = logging.getLogger("KmellVox.Narration")


def detect_text_format(text: str) -> str:
    """
    Identifica se o conteúdo de entrada segue o padrão SRT ou é texto puro.
    
    Verifica a presença de marcações de tempo padrão:
    HH:MM:SS,mmm --> HH:MM:SS,mmm ou H:MM:SS,mmm --> H:MM:SS,mmm
    """
    if not text or not text.strip():
        return "txt"

    # Regex para capturar timestamps do padrão SRT (ex: 00:00:01,500 --> 00:00:04,200)
    srt_pattern = re.compile(
        r"\d{1,2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{3}"
    )

    if srt_pattern.search(text):
        return "srt"
    return "txt"


def parse_srt(text: str) -> List[TranscriptionSegment]:
    """
    Extrai índice, tempo de início, tempo de fim e texto de cada bloco SRT.
    
    Retorna uma lista de TranscriptionSegment compatível com o restante do sistema.
    """
    if not text or not text.strip():
        return []

    segments: List[TranscriptionSegment] = []
    blocks = re.split(r"\n\s*\n", text.strip().replace("\r\n", "\n"))

    time_pattern = re.compile(
        r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{3})"
    )

    def parse_time_to_seconds(h: str, m: str, s: str, ms: str) -> float:
        return (int(h) * 3600) + (int(m) * 60) + int(s) + (int(ms) / 1000.0)

    seg_counter = 1
    for block in blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if not lines:
            continue

        start_time = 0.0
        end_time = 0.0
        text_lines = []
        found_time = False

        for i, line in enumerate(lines):
            match = time_pattern.search(line)
            if match:
                g = match.groups()
                start_time = parse_time_to_seconds(g[0], g[1], g[2], g[3])
                end_time = parse_time_to_seconds(g[4], g[5], g[6], g[7])
                found_time = True
                text_lines = lines[i + 1 :]
                break

        if found_time and text_lines:
            clean_text = " ".join(text_lines).strip()
            # Remove tags HTML de legendas como <i>, <b>, <font> se existirem
            clean_text = re.sub(r"<[^>]+>", "", clean_text)
            if clean_text:
                segments.append(
                    TranscriptionSegment(
                        id=seg_counter,
                        start=start_time,
                        end=max(end_time, start_time + 0.5),
                        text=clean_text,
                    )
                )
                seg_counter += 1

    return segments


def parse_block_ranges(range_str: str, max_blocks: int) -> List[int]:
    """
    Interpreta uma string de seleção de blocos de legenda (ex: "1-5", "1, 3, 5-8", "10")
    e retorna a lista ordenada de índices (1-indexed) válidos.
    Se range_str estiver vazio ou contiver '*', retorna todos os blocos de 1 a max_blocks.
    """
    clean = range_str.strip() if range_str else ""
    if not clean or clean == "*":
        return list(range(1, max_blocks + 1))

    selected = set()
    parts = clean.split(",")
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            sub = p.split("-")
            if len(sub) == 2:
                try:
                    start_i = max(1, int(sub[0].strip()))
                    end_i = min(max_blocks, int(sub[1].strip()))
                    if start_i <= end_i:
                        selected.update(range(start_i, end_i + 1))
                except ValueError:
                    pass
        else:
            try:
                val = int(p)
                if 1 <= val <= max_blocks:
                    selected.add(val)
            except ValueError:
                pass

    res = sorted(list(selected))
    return res if res else list(range(1, max_blocks + 1))


def slugify_text(text: str, max_words: int = 4, max_len: int = 30) -> str:
    """Gera um slug curto e limpo a partir das primeiras palavras do texto para nomes de arquivos."""
    # Remove acentos
    text_norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    words = re.findall(r"[a-zA-Z0-9]+", text_norm)
    selected_words = words[:max_words]
    slug = "_".join(selected_words).lower()
    return slug[:max_len] if slug else "trecho"


def list_preset_voices(
    model_profile: Optional[ModelProfile] = None,
    models_dir: str = "models",
) -> List[Dict[str, Any]]:
    """
    Varre os diretórios de vozes procurando referências de áudio válidas (.wav, .mp3, .ogg, .flac).
    Retorna metadados para popular a UI com vozes pré-definidas.
    """
    candidate_paths: List[Path] = [
        get_voices_directory(models_dir),
        Path(models_dir).resolve() / "tts" / "presets",
        Path(models_dir).resolve() / "tts" / "samples",
        Path(models_dir).resolve() / "presets",
        Path(models_dir).resolve() / "voices",
    ]
    discovered_voices: List[Dict[str, Any]] = []
    seen_paths = set()
    valid_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    for folder in candidate_paths:
        if folder.is_dir():
            for file_path in sorted(folder.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    abs_path = str(file_path.resolve())
                    if abs_path in seen_paths:
                        continue
                    seen_paths.add(abs_path)

                    clean_name = file_path.stem.replace("_", " ").replace("-", " ")
                    clean_name = re.sub(r"\s+", " ", clean_name).strip().title()

                    txt_path = file_path.with_suffix(".txt")
                    transcript = ""
                    if txt_path.is_file():
                        try:
                            transcript = sanitize_tts_text(txt_path.read_text(encoding="utf-8-sig"))
                        except Exception:
                            pass

                    discovered_voices.append({
                        "id": file_path.stem,
                        "name": clean_name,
                        "label": f"Voz Preset: {clean_name}",
                        "audio_path": abs_path,
                        "txt_path": str(txt_path.resolve()) if txt_path.is_file() else "",
                        "transcript": transcript,
                        "is_preset": True,
                    })

    return discovered_voices


@dataclass
class AudioMasteringConfig:
    """Configuração de masterização de voz e dinâmica de estúdio."""
    bass_gain_db: float = 3.0           # Realce de graves no peito (150Hz)
    treble_gain_db: float = 2.0         # Brilho e clareza vocal (3500Hz)
    presence_gain_db: float = 1.5       # Presença vocal (2800Hz)
    compressor_threshold: float = -18.0 # Nivelamento dinâmico em dB
    compressor_ratio: float = 2.5       # Razão de compressão
    target_lufs: float = -16.0          # Padrão de loudness da indústria (broadcast/streaming)
    speech_speed: float = 1.0           # Fator de velocidade selecionado na UI (1.0 = nativo)
    tempo_calibration: float = 1.00     # 1.00 = 100% velocidade original real (sem aceleração oculta)
    sentence_pause_seconds: float = 0.80 # Pausa natural e respiro entre frases completas (. ! ?)
    export_raw_wav: bool = True         # Salva o arquivo WAV puro antes do pós-processamento
    enabled: bool = True

    def build_ffmpeg_filter(self) -> str:
        """Monta a string de filtros de áudio do FFmpeg para processamento de estúdio."""
        if not self.enabled:
            return ""
        filters = []
        effective_speed = self.speech_speed * self.tempo_calibration
        if abs(effective_speed - 1.0) > 0.01:
            spd = max(0.5, min(2.0, effective_speed))
            filters.append(f"atempo={spd:.2f}")

        filters.extend([
            f"bass=g={self.bass_gain_db:.1f}:f=150:w=0.6",
            f"equalizer=f=2800:t=q:w=1.5:g={self.presence_gain_db:.1f}",
            f"treble={self.treble_gain_db:.1f}:f=3500",
            f"acompressor=threshold={self.compressor_threshold:.1f}dB:ratio={self.compressor_ratio:.1f}:attack=20:release=250",
            f"loudnorm=I={self.target_lufs:.1f}:TP=-1.5:LRA=11",
        ])
        return ",".join(filters)


@dataclass
class NarrationJob:
    """Representa uma tarefa na fila de geração de narração em áudio."""
    job_id: str
    source_text: str
    source_format: str = "txt"                    # "srt" | "txt"
    source_file_path: Optional[str] = None       # None se colado manualmente
    voice_mode: str = "clone"                     # "clone" | "preset"
    reference_audio_path: Optional[str] = None   # Usado se voice_mode == "clone"
    preset_voice_id: Optional[str] = None        # Usado se voice_mode == "preset"
    selected_engine: str = "f5-tts"               # Motor TTS selecionado: "f5-tts", "indextts-2", etc.
    split_mode: str = "unico"                     # "separado" | "unico" (para SRT)
    srt_range: Optional[str] = None               # Intervalo ou blocos específicos (ex: "1-10", "1,3,5" ou None para todos)
    speech_speed: float = 1.00                    # Velocidade nativa da fala (1.00 = original)
    sentence_pause_seconds: float = 0.80          # Pausa natural após ponto final em segundos
    mastering_config: Optional[AudioMasteringConfig] = None
    export_raw_wav: bool = False                  # Salva versão WAV pura sem masterização (controlado pela UI)
    destination_folder: str = str(Path.home() / "Downloads")
    save_to_source_folder: bool = True
    create_audio_subfolder: bool = False
    status: str = "Pendente"                      # "Pendente", "Processando", "Concluído", "Erro", "Cancelado"
    progress: float = 0.0
    status_message: str = "Aguardando"
    output_files: List[str] = field(default_factory=list)


class NarrationEngine:
    """
    Motor de síntese de narração que consome os motores TTS do KmellVox (F5-TTS / IndexTTS-2)
    e orquestra a geração de arquivos MP3 individuais ou concatenados no tempo.
    """

    def __init__(
        self,
        model_profile: Optional[ModelProfile] = None,
        models_dir: str = "models",
        ffmpeg_bin: Optional[str] = None,
    ) -> None:
        if model_profile is None:
            hw_info = detect_hardware()
            self.model_profile = hw_info.model_profile
        else:
            self.model_profile = model_profile

        if models_dir == "models" and getattr(sys, "frozen", False):
            self.models_dir = str((Path(sys.executable).parent / "models").resolve())
        else:
            self.models_dir = str(Path(models_dir).resolve())

        self.ffmpeg_bin = ffmpeg_bin or resolve_ffmpeg_binary()
        self.is_cancelled: bool = False

    def cancel(self) -> None:
        """Cancela a execução atual."""
        self.is_cancelled = True

    def resolve_destination_folder(self, job: NarrationJob) -> Path:
        """
        Resolve a pasta final de destino seguindo a regra de prioridade:
        1. Se save_to_source_folder=True e source_file_path não for None, usa a pasta de origem.
        2. Caso contrário, usa destination_folder (padrão Downloads).
        3. Se create_audio_subfolder=True, adiciona subpasta 'Áudio'.
        """
        if job.save_to_source_folder and job.source_file_path:
            src_parent = Path(job.source_file_path).parent
            if src_parent.is_dir():
                base_dir = src_parent
            else:
                base_dir = Path(job.destination_folder)
        else:
            base_dir = Path(job.destination_folder) if job.destination_folder else Path.home() / "Downloads"

        if job.create_audio_subfolder:
            final_dir = base_dir / "Áudio"
        else:
            final_dir = base_dir

        final_dir.mkdir(parents=True, exist_ok=True)
        return final_dir

    def _convert_to_mp3(
        self,
        input_audio: str,
        output_mp3: str,
        bitrate: str = "192k",
        mastering_config: Optional[AudioMasteringConfig] = None,
        apply_mastering: bool = True,
    ) -> str:
        """
        Converte qualquer áudio para MP3 utilizando o FFmpeg com masterização de voz.
        Aplica realce de graves profundos (warmth), brilho nos agudos, compressão dinâmica
        e normalização de loudness de estúdio (-16 LUFS) para dar potência e presença.
        """
        out_path = Path(output_mp3).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cfg = mastering_config or AudioMasteringConfig()
        vocal_filter = cfg.build_ffmpeg_filter() if apply_mastering else ""

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(Path(input_audio).resolve()),
        ]
        if vocal_filter:
            cmd.extend(["-af", vocal_filter])
        cmd.extend([
            "-codec:a", "libmp3lame",
            "-b:a", bitrate,
            str(out_path),
        ])

        logger.debug("Executando conversão MP3 com masterização: %s", " ".join(cmd))
        _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_NO_WINDOW,
        )
        if proc.returncode != 0:
            logger.warning("Falha no FFmpeg com filtros (%s). Tentando conversão simples...", proc.stderr)
            # Fallback sem filtros de masterização
            fallback_cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(Path(input_audio).resolve()),
                "-codec:a", "libmp3lame",
                "-b:a", bitrate,
                str(out_path),
            ]
            proc_fb = subprocess.run(
                fallback_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_NO_WINDOW,
            )
            if proc_fb.returncode != 0 and not out_path.exists():
                shutil.copyfile(input_audio, str(out_path))

        return str(out_path)

    def _create_silence_wav(self, duration_seconds: float, output_path: str, sample_rate: int = 24000) -> str:
        """Gera um arquivo WAV contendo apenas silêncio com duração precisa."""
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        num_samples = int(max(0.01, duration_seconds) * sample_rate)
        import numpy as np
        silence_data = np.zeros(num_samples, dtype=np.float32)
        sf.write(str(out_p), silence_data, sample_rate)
        return str(out_p)

    def _concat_audio_segments(
        self,
        audio_files: List[str],
        output_mp3: str,
        mastering_config: Optional[AudioMasteringConfig] = None,
        apply_mastering: bool = True,
    ) -> str:
        """Concatena múltiplos arquivos de áudio via demuxer concat do FFmpeg com masterização vocal."""
        out_path = Path(output_mp3).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not audio_files:
            raise ValueError("Nenhum arquivo de áudio para concatenar.")

        if len(audio_files) == 1:
            return self._convert_to_mp3(
                audio_files[0],
                str(out_path),
                mastering_config=mastering_config,
                apply_mastering=apply_mastering,
            )

        temp_list = out_path.parent / f"concat_list_{int(time.time() * 1000)}.txt"
        try:
            with open(temp_list, "w", encoding="utf-8") as f:
                for a_file in audio_files:
                    escaped_path = str(Path(a_file).resolve()).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            cfg = mastering_config or AudioMasteringConfig()
            vocal_filter = cfg.build_ffmpeg_filter() if apply_mastering else ""

            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(temp_list),
            ]
            if vocal_filter:
                cmd.extend(["-af", vocal_filter])

            if out_path.suffix.lower() == ".wav":
                cmd.extend(["-codec:a", "pcm_s16le", str(out_path)])
            else:
                cmd.extend([
                    "-codec:a", "libmp3lame",
                    "-b:a", "192k",
                    str(out_path),
                ])

            _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_NO_WINDOW,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Erro no FFmpeg concat: {proc.stderr}")

            return str(out_path)
        finally:
            if temp_list.exists():
                try:
                    os.remove(temp_list)
                except Exception:
                    pass

    def run(
        self,
        job: NarrationJob,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[str]:
        """
        Executa a síntese de narração conforme a configuração do NarrationJob.
        
        Returns:
            List[str]: Lista de arquivos MP3 gerados.
        """
        self.is_cancelled = False
        target_dir = self.resolve_destination_folder(job)

        # 1. Resolve o áudio de referência para clonagem ou preset
        reference_audio = None
        if job.voice_mode == "clone":
            reference_audio = job.reference_audio_path
            if not reference_audio or not os.path.isfile(reference_audio):
                raise FileNotFoundError(
                    f"Áudio de referência para clonagem de voz não encontrado: '{reference_audio}'"
                )
        elif job.voice_mode == "preset":
            presets = list_preset_voices(self.model_profile, self.models_dir)
            matched = [p["audio_path"] for p in presets if p["id"] == job.preset_voice_id]
            if matched and os.path.isfile(matched[0]):
                reference_audio = matched[0]
            elif presets:
                reference_audio = presets[0]["audio_path"]
            else:
                raise ValueError("Nenhuma voz preset encontrada no sistema para modo 'preset'.")

        # 2. Inicializa o motor TTS (falha = erro fatal, SEM fallback silencioso)
        engine_choice = getattr(job, "selected_engine", "f5-tts")
        tts_engine: BaseTTSEngine = get_tts_engine(
            model_profile=self.model_profile,
            engine_name=engine_choice,
            use_advanced=self.model_profile.enable_indextts_2,
            models_dir=self.models_dir,
        )

        actual_engine_name = engine_choice.upper()

        temp_work_dir = Path(tempfile.mkdtemp(prefix="kmellvox_narration_"))
        generated_outputs: List[str] = []

        def notify(pct: float, msg: str) -> None:
            if self.is_cancelled:
                raise RuntimeError("Operação de narração cancelada pelo usuário.")
            logger.info("[Narração %.0f%%] %s", pct * 100, msg)
            if progress_callback:
                progress_callback(pct, msg)

        try:
            # -------------------------------------------------------------
            # CENÁRIO A: Formato Texto Puro (.txt)
            # -------------------------------------------------------------
            if job.mastering_config is not None:
                job.mastering_config.speech_speed = job.speech_speed

            if job.source_format == "txt":
                notify(0.10, f"Iniciando síntese de texto puro com motor [{actual_engine_name}]...")

                base_stem = (
                    Path(job.source_file_path).stem
                    if job.source_file_path
                    else f"narracao_{int(time.time())}"
                )
                output_mp3_path = target_dir / f"{base_stem}.mp3"
                temp_wav = temp_work_dir / f"{base_stem}_raw.wav"

                pause_sec = (
                    job.mastering_config.sentence_pause_seconds
                    if job.mastering_config is not None
                    else job.sentence_pause_seconds
                )
                notify(0.30, f"Sintetizando fala em alta fidelidade acústica (1x difusão, pausas de {pause_sec:.2f}s)...")
                tts_engine.clone_and_synthesize(
                    text=job.source_text.strip(),
                    reference_audio_path=reference_audio,
                    output_path=str(temp_wav),
                    speed=1.0,
                    sentence_pause_seconds=pause_sec,
                )

                # Exporta cópia do áudio bruto (.wav sem efeitos) se solicitado
                if getattr(job, "export_raw_wav", False):
                    raw_dest_wav = target_dir / f"{base_stem}_raw.wav"
                    if temp_wav.is_file():
                        shutil.copyfile(str(temp_wav), str(raw_dest_wav))
                        generated_outputs.append(str(raw_dest_wav))
                        logger.info("Áudio bruto exportado para: %s", raw_dest_wav.name)

                notify(0.85, f"Masterizando áudio de estúdio ({job.speech_speed:.2f}x, graves e dinâmica)...")
                if job.mastering_config is not None:
                    final_mp3 = self._convert_to_mp3(
                        str(temp_wav),
                        str(output_mp3_path),
                        mastering_config=job.mastering_config,
                    )
                else:
                    final_mp3 = self._convert_to_mp3(str(temp_wav), str(output_mp3_path))
                generated_outputs.append(final_mp3)

                notify(1.0, f"Narração concluída com sucesso: {os.path.basename(final_mp3)}")

            # -------------------------------------------------------------
            # CENÁRIO B: Formato SRT (.srt) - Áudios Separados por Trecho
            # -------------------------------------------------------------
            elif job.source_format == "srt" and job.split_mode == "separado":
                notify(0.05, "Analisando blocos de legendas SRT...")
                segments = parse_srt(job.source_text)
                if not segments:
                    raise ValueError("Nenhum segmento válido encontrado no conteúdo SRT.")

                if job.srt_range:
                    selected_ids = set(parse_block_ranges(job.srt_range, len(segments)))
                    segments = [s for s in segments if s.id in selected_ids]
                    if not segments:
                        raise ValueError(f"Nenhum bloco SRT correspondeu ao intervalo selecionado: '{job.srt_range}'.")

                total = len(segments)
                notify(0.10, f"Processando {total} trecho(s) de áudio selecionados com motor [{actual_engine_name}] ({job.speech_speed:.2f}x)...")

                pause_sec = (
                    job.mastering_config.sentence_pause_seconds
                    if job.mastering_config is not None
                    else job.sentence_pause_seconds
                )

                for idx, seg in enumerate(segments, 1):
                    if self.is_cancelled:
                        raise RuntimeError("Operação cancelada pelo usuário.")

                    slug = slugify_text(seg.text, max_words=4)
                    file_name = f"{seg.id:03d}_{slug}.mp3"
                    out_mp3 = target_dir / file_name
                    temp_seg_wav = temp_work_dir / f"seg_{seg.id:03d}.wav"

                    sub_pct = 0.10 + ((idx / total) * 0.80)
                    notify(sub_pct, f"Sintetizando bloco {seg.id} ({idx}/{total}): '{seg.text[:30]}...'")

                    tts_engine.clone_and_synthesize(
                        text=seg.text,
                        reference_audio_path=reference_audio,
                        output_path=str(temp_seg_wav),
                        target_duration=None,
                        speed=1.0,
                        sentence_pause_seconds=pause_sec,
                    )

                    # Exporta cópia do áudio bruto (.wav sem efeitos) se solicitado
                    if getattr(job, "export_raw_wav", False):
                        raw_seg_wav = target_dir / f"{seg.id:03d}_{slug}_raw.wav"
                        if temp_seg_wav.is_file():
                            shutil.copyfile(str(temp_seg_wav), str(raw_seg_wav))
                            generated_outputs.append(str(raw_seg_wav))
                            logger.info("Áudio bruto exportado para: %s", raw_seg_wav.name)

                    if job.mastering_config is not None:
                        self._convert_to_mp3(
                            str(temp_seg_wav),
                            str(out_mp3),
                            mastering_config=job.mastering_config,
                        )
                    else:
                        self._convert_to_mp3(str(temp_seg_wav), str(out_mp3))
                    generated_outputs.append(str(out_mp3))

                notify(1.0, f"{len(generated_outputs)} arquivos de áudio gerados com sucesso na pasta de destino.")

            # -------------------------------------------------------------
            # CENÁRIO C: Formato SRT (.srt) - Arquivo Único Concatenado com Ritmo
            # -------------------------------------------------------------
            elif job.source_format == "srt" and job.split_mode == "unico":
                notify(0.05, "Analisando blocos de legendas SRT para áudio contínuo...")
                segments = parse_srt(job.source_text)
                if not segments:
                    raise ValueError("Nenhum segmento válido encontrado no conteúdo SRT.")

                if job.srt_range:
                    selected_ids = set(parse_block_ranges(job.srt_range, len(segments)))
                    segments = [s for s in segments if s.id in selected_ids]
                    if not segments:
                        raise ValueError(f"Nenhum bloco SRT correspondeu ao intervalo selecionado: '{job.srt_range}'.")

                total = len(segments)
                notify(0.10, f"Sintetizando e ajustando ritmo para {total} segmentos selecionados com motor [{actual_engine_name}] ({job.speech_speed:.2f}x)...")

                pieces_to_concat: List[str] = []
                current_timeline_time = 0.0
                pause_sec = (
                    job.mastering_config.sentence_pause_seconds
                    if job.mastering_config is not None
                    else job.sentence_pause_seconds
                )

                for idx, seg in enumerate(segments, 1):
                    if self.is_cancelled:
                        raise RuntimeError("Operação cancelada pelo usuário.")

                    # Inserção de silêncio proporcional entre os trechos para preservar o ritmo
                    if seg.start > current_timeline_time:
                        gap_duration = seg.start - current_timeline_time
                        if gap_duration >= 0.05:  # Inserir silêncio se >= 50ms
                            silence_file = temp_work_dir / f"silence_{idx:03d}.wav"
                            self._create_silence_wav(gap_duration, str(silence_file))
                            pieces_to_concat.append(str(silence_file))
                            current_timeline_time = seg.start

                    temp_speech_wav = temp_work_dir / f"speech_{idx:03d}.wav"
                    sub_pct = 0.10 + ((idx / total) * 0.70)
                    notify(sub_pct, f"Sintetizando fala {idx}/{total}...")

                    tts_engine.clone_and_synthesize(
                        text=seg.text,
                        reference_audio_path=reference_audio,
                        output_path=str(temp_speech_wav),
                        target_duration=None,
                        speed=1.0,
                        sentence_pause_seconds=pause_sec,
                    )

                    pieces_to_concat.append(str(temp_speech_wav))
                    actual_dur = get_audio_duration(str(temp_speech_wav))
                    current_timeline_time = max(current_timeline_time + actual_dur, seg.end)

                notify(0.85, "Concatenando trechos e silêncios no arquivo final (com masterização)...")

                base_stem = (
                    Path(job.source_file_path).stem
                    if job.source_file_path
                    else f"narracao_srt_completa_{int(time.time())}"
                )
                final_combined_mp3 = target_dir / f"{base_stem}.mp3"
                # Exporta cópia concatenada bruta (.wav sem efeitos) se solicitado
                if getattr(job, "export_raw_wav", False):
                    raw_combined_wav = target_dir / f"{base_stem}_raw.wav"
                    self._concat_audio_segments(pieces_to_concat, str(raw_combined_wav), mastering_config=None)
                    generated_outputs.append(str(raw_combined_wav))
                    logger.info("Áudio contínuo bruto exportado para: %s", raw_combined_wav.name)

                if job.mastering_config is not None:
                    self._concat_audio_segments(
                        pieces_to_concat,
                        str(final_combined_mp3),
                        mastering_config=job.mastering_config,
                    )
                else:
                    self._concat_audio_segments(pieces_to_concat, str(final_combined_mp3))
                generated_outputs.append(str(final_combined_mp3))

                notify(1.0, f"Áudio contínuo gerado com sucesso: {os.path.basename(final_combined_mp3)}")

            else:
                raise ValueError(f"Combinação não suportada: format={job.source_format}, split={job.split_mode}")

            job.output_files = generated_outputs
            job.status = "Concluído"
            job.progress = 1.0
            job.status_message = "Concluído"
            return generated_outputs

        finally:
            # Liberação rigorosa de VRAM
            tts_engine.unload_model()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # Limpa pasta temporária
            if temp_work_dir.exists():
                shutil.rmtree(temp_work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Funções de Gerenciamento e Clonagem de Vozes Salvas (voices/)
# ---------------------------------------------------------------------------

def get_voices_directory(models_dir: str = "models") -> Path:
    """Retorna o caminho canônico da pasta 'voices/' do KmellVox."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent.parent
    voices_dir = base / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    return voices_dir


def list_all_saved_voices(models_dir: str = "models") -> List[Dict[str, Any]]:
    """
    Retorna uma lista estruturada de todas as vozes disponíveis no sistema
    (tanto presets do aplicativo quanto vozes clonadas pelo usuário).
    """
    voices_dir = get_voices_directory(models_dir)
    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    voices_list: List[Dict[str, Any]] = []
    seen_stems = set()

    # Prioriza arquivos da pasta voices/
    all_files = sorted(voices_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in all_files:
        if not f.is_file() or f.suffix.lower() not in valid_exts:
            continue
        # Ignora backups e arquivos temporários
        if f.name.endswith(".bak") or f.name.startswith("_ref_trimmed_") or f.name.startswith("."):
            continue

        stem = f.stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        txt_file = f.with_suffix(".txt")
        has_txt = txt_file.is_file()
        transcript = txt_file.read_text(encoding="utf-8").strip() if has_txt else ""

        dur = get_audio_duration(str(f))
        size_kb = round(f.stat().st_size / 1024, 1)
        mtime_str = time.strftime("%d/%m/%Y %H:%M", time.localtime(f.stat().st_mtime))

        voices_list.append({
            "id": stem.lower(),
            "name": stem,
            "display_name": stem.replace("_", " ").replace("-", " ").title(),
            "audio_path": str(f.resolve()),
            "txt_path": str(txt_file.resolve()) if has_txt else "",
            "transcript": transcript,
            "duration": dur,
            "size_kb": size_kb,
            "date_str": mtime_str,
            "extension": f.suffix.lower(),
        })

    return voices_list


def smart_trim_audio_reference(
    input_audio_path: str,
    output_audio_path: str,
    min_duration: float = 8.0,
    max_duration: float = 12.0,
    ffmpeg_bin: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Recorta com precisão cirúrgica o áudio de referência para a faixa ideal (8s a 12s),
    garantindo que NÃO corte no meio de uma palavra, analisando energia acústica e pausas naturais
    de silêncio entre orações.
    
    Returns:
        Tuple[str, float]: (caminho do arquivo WAV recortado a 24kHz mono, duração real em segundos)
    """
    bin_path = ffmpeg_bin or resolve_ffmpeg_binary()
    in_p = Path(input_audio_path).resolve()
    out_p = Path(output_audio_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Converte áudio completo para WAV mono 24kHz temporário para inspeção precisa
    temp_conv = out_p.parent / f"_temp_smart_trim_{int(time.time() * 1000)}.wav"
    try:
        cmd = [
            bin_path, "-y",
            "-i", str(in_p),
            "-ar", "24000", "-ac", "1",
            str(temp_conv),
        ]
        _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            creationflags=_NO_WINDOW,
        )

        import numpy as np
        data, sr = sf.read(str(temp_conv), dtype="float32")
        dur = len(data) / sr

        # Se já estiver dentro do limite ideal (<= max_duration), mantém integral
        if dur <= max_duration:
            shutil.move(str(temp_conv), str(out_p))
            return str(out_p), dur

        # 2. Análise de energia por janelas (50ms hop, 100ms window) para achar pausas
        hop = int(0.05 * sr)
        win = int(0.10 * sr)
        n_frames = max(1, (len(data) - win) // hop)
        frame_rms = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            frame = data[i * hop : i * hop + win]
            frame_rms[i] = np.sqrt(np.mean(frame**2))

        # Considera silêncio frames com RMS < 0.015
        silence_threshold = 0.015
        is_silence = frame_rms < silence_threshold

        # Identifica vales de silêncio contínuo com pelo menos 150ms (3 frames)
        valleys = []
        in_v = False
        v_start = 0
        for i, s in enumerate(is_silence):
            if s and not in_v:
                in_v = True
                v_start = i
            elif not s and in_v:
                in_v = False
                if (i - v_start) >= 3:
                    center_frame = (v_start + i) // 2
                    valleys.append(center_frame * hop / sr)
        if in_v and (n_frames - v_start) >= 3:
            valleys.append(((v_start + n_frames) // 2) * hop / sr)

        if not valleys or valleys[0] > 0.5:
            valleys.insert(0, 0.0)

        # 3. Busca o melhor par de vales [start, end] com duração entre min_duration e max_duration
        best_candidate = None
        best_score = -1.0

        for i in range(len(valleys)):
            t_start = valleys[i]
            for j in range(i + 1, len(valleys)):
                t_end = valleys[j]
                candidate_dur = t_end - t_start

                if min_duration <= candidate_dur <= max_duration:
                    s_idx = int(t_start * sr)
                    e_idx = int(t_end * sr)
                    sub = data[s_idx:e_idx]
                    cand_rms = np.sqrt(np.mean(sub**2))
                    dur_score = 1.0 - abs(candidate_dur - 10.5) / 10.5
                    rms_score = 1.0 - abs(cand_rms - 0.06) / 0.06 if cand_rms > 0.02 else 0.1
                    score = dur_score * 0.5 + rms_score * 0.5

                    if score > best_score:
                        best_score = score
                        best_candidate = (t_start, t_end)
                elif candidate_dur > max_duration:
                    break

        if best_candidate is not None:
            t_start, t_end = best_candidate
            logger.info(
                "Recorte inteligente selecionou trecho de %.2fs a %.2fs (duração: %.2fs)",
                t_start, t_end, t_end - t_start,
            )
            cut_data = data[int(t_start * sr) : int(t_end * sr)]
        else:
            cut_end = 10.0
            for v in valleys:
                if 7.0 <= v <= max_duration:
                    cut_end = v
                    break
            cut_data = data[: int(cut_end * sr)]

        # Aplica micro-fade in/out de 10ms para evitar estalos nas bordas
        fade_len = int(0.010 * sr)
        if len(cut_data) >= 2 * fade_len:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            cut_data[:fade_len] *= fade_in
            cut_data[-fade_len:] *= fade_out

        sf.write(str(out_p), cut_data, sr)
        real_dur = len(cut_data) / sr
        return str(out_p), real_dur

    finally:
        if temp_conv.exists():
            try:
                os.remove(temp_conv)
            except Exception:
                pass


def save_cloned_voice(
    voice_name: str,
    audio_path: str,
    transcript: Optional[str] = None,
    models_dir: str = "models",
) -> Dict[str, Any]:
    """
    Processa e salva uma nova voz clonada na pasta voices/ com recorte inteligente e transcrição.
    """
    clean_name = re.sub(r'[\\/*?:"<>|]', "", voice_name).strip()
    if not clean_name:
        clean_name = f"Voz_Clonada_{int(time.time())}"

    voices_dir = get_voices_directory(models_dir)
    target_wav = voices_dir / f"{clean_name}.wav"
    target_txt = voices_dir / f"{clean_name}.txt"

    # Aplica recorte inteligente para o tempo ideal do modelo (8s a 12s)
    smart_trim_audio_reference(
        input_audio_path=audio_path,
        output_audio_path=str(target_wav),
        min_duration=8.0,
        max_duration=12.0,
    )

    # Salva transcrição de referência higienizada sem BOM
    if transcript and transcript.strip():
        clean_txt = sanitize_tts_text(transcript.strip())
        if not clean_txt.endswith((".", "!", "?")):
            clean_txt += "."
        target_txt.write_text(clean_txt, encoding="utf-8")
    else:
        # Se não forneceu nova transcrição ao atualizar a voz, limpa o arquivo para evitar
        # que uma transcrição antiga e incompatível com o novo áudio permaneça ativa
        target_txt.write_text("", encoding="utf-8")

    return {
        "name": clean_name,
        "audio_path": str(target_wav.resolve()),
        "txt_path": str(target_txt.resolve()) if target_txt.exists() else "",
    }


def rename_saved_voice(old_name: str, new_name: str, models_dir: str = "models") -> bool:
    """Renomeia uma voz salva e seus arquivos associados (.wav, .mp3, .txt)."""
    voices_dir = get_voices_directory(models_dir)
    clean_new = re.sub(r'[\\/*?:"<>|]', "", new_name).strip()
    if not clean_new or old_name == clean_new:
        return False

    success = False
    for ext in [".wav", ".mp3", ".txt", ".flac", ".ogg", ".m4a"]:
        old_f = voices_dir / f"{old_name}{ext}"
        new_f = voices_dir / f"{clean_new}{ext}"
        if old_f.is_file():
            old_f.rename(new_f)
            success = True

    return success


def delete_saved_voice(voice_name: str, models_dir: str = "models") -> bool:
    """Remove uma voz salva e todos os seus arquivos correspondentes da pasta voices/."""
    voices_dir = get_voices_directory(models_dir)
    deleted = False
    for ext in [".wav", ".mp3", ".txt", ".flac", ".ogg", ".m4a", ".mp3.bak"]:
        f = voices_dir / f"{voice_name}{ext}"
        if f.is_file():
            try:
                f.unlink()
                deleted = True
            except Exception as e:
                logger.warning("Erro ao excluir arquivo de voz '%s': %s", f.name, e)

    return deleted
