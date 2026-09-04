"""Módulo Orquestrador do Pipeline Completo do KmellVox (DubPipeline).

Orquestra todas as etapas de dublagem, tradução, clonagem de voz e lip sync,
gerenciando callbacks de progresso (0-100%), fila de múltiplos vídeos x múltiplos idiomas,
e garantindo liberação rigorosa de VRAM entre cada etapa.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .assemble import (
    AssemblyConfig,
    assemble_final_video,
    build_timeline_audio,
    burn_subtitles,
    export_raw_package,
    mux_audio_video,
)
from .audio_extract import extract_audio
from .hardware import HardwareInfo, ModelProfile, detect_hardware
from .lipsync import LipSyncEngine, LipSyncResult
from .transcribe import Transcriber, TranscriptionSegment
from .translate import TranslatedSegment, TranslationResult, Translator
from .voice_clone import ClonedAudioSegment, get_audio_duration, get_tts_engine

logger = logging.getLogger("KmellVox.Pipeline")


@dataclass
class PipelineProgress:
    """Estado e progresso granular da execução do pipeline."""
    stage: str = "idle"
    percentage: float = 0.0
    message: str = "Pronto"
    elapsed_time: float = 0.0
    current_job_index: int = 1
    total_jobs: int = 1

    @property
    def percent_100(self) -> int:
        return int(self.percentage * 100)


@dataclass
class PipelineConfig:
    """Configurações completas de execução para um trabalho de dublagem."""
    input_video: str
    output_video: str
    source_language: str = "auto"
    target_language: str = "pt"
    hardware_profile: str = "auto"
    temp_dir: str = "temp"
    models_dir: str = "models"
    keep_temp_files: bool = False
    enable_lipsync: bool = False
    use_indextts2: bool = False
    burn_subtitles: bool = False
    export_raw_package: bool = False
    ffmpeg_bin: Optional[str] = None
    whisper_model: Optional[str] = None
    llm_model_path: Optional[str] = None


class DubPipeline:
    """
    Orquestrador central do KmellVox que conecta todas as etapas do fluxo:
    1. Extração de áudio 16kHz mono (FFmpeg)
    2. Transcrição com timestamps (faster-whisper) + VRAM cleanup
    3. Tradução em lote com contexto de fala (Qwen3 GGUF via llama-cpp-python) + VRAM cleanup
    4. Clonagem e síntese de voz (F5-TTS com Controle de Ritmo ou IndexTTS-2 em FP16) + VRAM cleanup
    5. Sincronização labial opcional (MuseTalk 1.5 em FP16) + VRAM cleanup
    6. Muxing, estampa de legendas e pacote bruto (FFmpeg)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        hardware_profile: str = "auto",
        temp_dir: str = "temp",
        models_dir: str = "models",
        ffmpeg_bin: Optional[str] = None,
    ) -> None:
        self.config = config
        profile_to_detect = config.hardware_profile if config else hardware_profile
        self.hardware_info: HardwareInfo = detect_hardware(profile_to_detect)
        self.temp_dir = temp_dir if not config else config.temp_dir
        self.models_dir = models_dir if not config else config.models_dir
        self.ffmpeg_bin = ffmpeg_bin if not config else config.ffmpeg_bin
        self.is_cancelled: bool = False

    def cancel(self) -> None:
        """Sinaliza cancelamento imediato de execuções em andamento."""
        self.is_cancelled = True
        logger.warning("Cancelamento solicitado no DubPipeline.")

    def _cleanup_gpu_vram(self) -> None:
        """Executa limpeza explícita de memória e cache CUDA entre etapas do pipeline."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def process_video(
        self,
        input_video: str,
        target_language: str = "pt",
        source_language: str = "auto",
        output_video: Optional[str] = None,
        enable_lipsync: bool = False,
        use_indextts2: bool = False,
        burn_subtitles_flag: bool = False,
        export_raw_pkg: bool = False,
        keep_temp_files: bool = False,
        job_index: int = 1,
        total_jobs: int = 1,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ) -> Dict[str, Any]:
        """
        Processa um único vídeo de ponta a ponta através de todas as etapas.
        """
        start_time = time.time()
        job_timestamp = int(time.time() * 1000)
        job_work_dir = Path(self.temp_dir) / f"job_{job_timestamp}"
        job_work_dir.mkdir(parents=True, exist_ok=True)

        if not output_video:
            base_name = Path(input_video).stem
            out_dir = Path("output").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            output_video = str(out_dir / f"{base_name}_dubbed_{target_language}.mp4")

        def notify(stage: str, pct: float, msg: str) -> None:
            if self.is_cancelled:
                raise RuntimeError("Operação cancelada pelo usuário.")
            elapsed = time.time() - start_time
            logger.info("[%s] (%.0f%%) %s", stage, pct * 100, msg)
            if progress_callback:
                progress_callback(
                    PipelineProgress(
                        stage=stage,
                        percentage=pct,
                        message=msg,
                        elapsed_time=elapsed,
                        current_job_index=job_index,
                        total_jobs=total_jobs,
                    )
                )

        try:
            notify("init", 0.02, f"Iniciando pipeline para: {os.path.basename(input_video)} -> [{target_language.upper()}]")

            # -------------------------------------------------------------
            # Etapa 1: Extração de Áudio (FFmpeg)
            # -------------------------------------------------------------
            notify("audio_extract", 0.08, "Extraindo áudio original (16kHz mono WAV)...")
            raw_audio_path = str(job_work_dir / "original_audio.wav")
            extracted_path = extract_audio(
                video_path=input_video,
                output_path=raw_audio_path,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            raw_audio_path = extracted_path

            # -------------------------------------------------------------
            # Etapa 2: Transcrição (faster-whisper) + VRAM release
            # -------------------------------------------------------------
            notify("transcribe", 0.15, f"Carregando Whisper ({self.hardware_info.model_profile.whisper_variant})...")
            transcriber = Transcriber(
                model_profile=self.hardware_info.model_profile,
                download_root=os.path.join(self.models_dir, "whisper"),
            )

            def on_transcribe_sub(sub_p: float, sub_m: str) -> None:
                notify("transcribe", 0.15 + (sub_p * 0.15), sub_m)

            segments: List[TranscriptionSegment] = transcriber.transcribe(
                audio_path=raw_audio_path,
                language=source_language,
                auto_unload=True,
                progress_callback=on_transcribe_sub,
            )

            # Salva SRT original
            original_srt_path = str(job_work_dir / "transcription_original.srt")
            transcriber.export_srt(segments, original_srt_path)
            self._cleanup_gpu_vram()

            # -------------------------------------------------------------
            # Etapa 3: Tradução em Lote (Qwen3 GGUF) + VRAM release
            # -------------------------------------------------------------
            detected_src = source_language if source_language != "auto" else "en"
            notify("translate", 0.35, f"Traduzindo falas em lote com Qwen3 ({detected_src} -> {target_language})...")
            translator = Translator(
                model_profile=self.hardware_info.model_profile,
                models_dir=self.models_dir,
            )

            def on_translate_sub(sub_p: float, sub_m: str) -> None:
                notify("translate", 0.35 + (sub_p * 0.15), sub_m)

            translated_segments: List[TranslatedSegment] = translator.translate_segments(
                segments=segments,
                source_language=detected_src,
                target_language=target_language,
                batch_size=8,
                auto_unload=True,
                progress_callback=on_translate_sub,
            )

            # Salva SRT traduzido
            translated_result = TranslationResult(
                source_language=detected_src,
                target_language=target_language,
                segments=translated_segments,
            )
            translated_srt_path = str(job_work_dir / "transcription_translated.srt")
            translated_result.save_srt(translated_srt_path)
            self._cleanup_gpu_vram()

            # -------------------------------------------------------------
            # Etapa 4: Clonagem e Síntese de Voz (F5-TTS / IndexTTS-2) + VRAM release
            # -------------------------------------------------------------
            eng_label = "IndexTTS-2 (FP16)" if use_indextts2 and self.hardware_info.model_profile.enable_indextts_2 else "F5-TTS (Controle de Ritmo)"
            notify("voice_clone", 0.55, f"Clonando voz e sintetizando falas ({eng_label})...")

            tts_engine = get_tts_engine(
                model_profile=self.hardware_info.model_profile,
                use_advanced=use_indextts2,
                models_dir=self.models_dir,
            )
            cloned_audio_dir = str(job_work_dir / "voice_segments")

            def on_voice_sub(sub_p: float, sub_m: str) -> None:
                notify("voice_clone", 0.55 + (sub_p * 0.20), sub_m)

            cloned_segments: List[ClonedAudioSegment] = tts_engine.clone_and_align_all(
                segments=translated_segments,
                reference_audio_path=raw_audio_path,
                output_dir=cloned_audio_dir,
                auto_unload=True,
                progress_callback=on_voice_sub,
            )

            # Áudio dublado mixado na timeline oficial a partir dos segmentos clonados
            dubbed_audio_path = str(job_work_dir / "final_dubbed_audio.wav")
            if cloned_segments:
                notify("voice_clone", 0.76, "Montando timeline do áudio dublado...")
                media_duration = get_audio_duration(raw_audio_path)
                build_timeline_audio(
                    segments=cloned_segments,
                    total_duration=media_duration,
                    output_path=dubbed_audio_path,
                )
            elif os.path.isfile(raw_audio_path):
                shutil.copyfile(raw_audio_path, dubbed_audio_path)
            else:
                Path(dubbed_audio_path).touch()
            self._cleanup_gpu_vram()

            # -------------------------------------------------------------
            # Etapa 5: Sincronização Labial (MuseTalk 1.5 - Experimental) + VRAM release
            # -------------------------------------------------------------
            video_for_assembly = input_video
            if enable_lipsync:
                notify("lipsync", 0.78, "Processando sincronia labial [Experimental] (MuseTalk 1.5)...")
                syncer = LipSyncEngine(
                    model_profile=self.hardware_info.model_profile,
                    models_dir=self.models_dir,
                )
                lipsync_out = str(job_work_dir / "lipsync_output.mp4")

                def on_lipsync_sub(sub_p: float, sub_m: str) -> None:
                    notify("lipsync", 0.78 + (sub_p * 0.12), sub_m)

                syncer.sync(
                    video_path=input_video,
                    dubbed_audio_path=dubbed_audio_path,
                    output_path=lipsync_out,
                    auto_unload=True,
                    progress_callback=on_lipsync_sub,
                )
                video_for_assembly = lipsync_out
                self._cleanup_gpu_vram()

            # -------------------------------------------------------------
            # Etapa 6: Remontagem, Muxing e Pacote Bruto (FFmpeg)
            # -------------------------------------------------------------
            notify("assemble", 0.92, "Renderizando arquivo final com FFmpeg...")

            assembly_cfg = AssemblyConfig(
                burn_subtitles=burn_subtitles_flag,
                subtitle_file=translated_srt_path if burn_subtitles_flag else None,
                ffmpeg_bin=self.ffmpeg_bin,
            )

            final_video_path = assemble_final_video(
                video_source=video_for_assembly,
                audio_source=dubbed_audio_path,
                output_video=output_video,
                config=assembly_cfg,
            )

            raw_package_info = None
            if export_raw_pkg:
                notify("export_raw", 0.96, "Exportando pacote bruto (MP3 + SRT)...")
                raw_out_dir = str(Path(output_video).parent / "raw_packages")
                raw_package_info = export_raw_package(
                    audio_path=dubbed_audio_path,
                    srt_path=translated_srt_path,
                    output_dir=raw_out_dir,
                    base_name=f"{Path(input_video).stem}_{target_language}",
                    ffmpeg_bin=self.ffmpeg_bin,
                )

            total_elapsed = time.time() - start_time
            notify("done", 1.0, f"Concluído com sucesso em {total_elapsed:.1f}s!")

            return {
                "success": True,
                "input_video": input_video,
                "output_video": final_video_path,
                "target_language": target_language,
                "translated_srt": translated_srt_path,
                "raw_package": raw_package_info,
                "elapsed_seconds": total_elapsed,
            }

        finally:
            self._cleanup_gpu_vram()
            if not keep_temp_files and job_work_dir.exists():
                try:
                    shutil.rmtree(job_work_dir, ignore_errors=True)
                except Exception:
                    pass

    def process_batch_queue(
        self,
        video_paths: List[str],
        target_languages: List[str],
        source_language: str = "auto",
        output_dir: str = "output",
        enable_lipsync: bool = False,
        use_indextts2: bool = False,
        burn_subtitles: bool = False,
        export_raw: bool = False,
        keep_temp_files: bool = False,
        queue_progress_callback: Optional[Callable[[int, int, PipelineProgress], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processa uma matriz/fila de múltiplos vídeos x múltiplos idiomas de destino sequencialmente,
        liberando a VRAM após cada item e mantendo o sistema responsivo.
        """
        out_base = Path(output_dir).resolve()
        out_base.mkdir(parents=True, exist_ok=True)

        # Monta a grade de execução: Vídeos x Idiomas
        job_queue: List[Dict[str, str]] = []
        for vid in video_paths:
            for lang in target_languages:
                job_queue.append({"video": vid, "target_lang": lang})

        total_jobs = len(job_queue)
        logger.info("Iniciando Fila de Processamento em Lote: %d trabalho(s) no total (%d vídeos x %d idiomas)",
                    total_jobs, len(video_paths), len(target_languages))

        results: List[Dict[str, Any]] = []

        for idx, item in enumerate(job_queue, 1):
            if self.is_cancelled:
                logger.warning("Fila de processamento interrompida pelo usuário.")
                break

            vid_path = item["video"]
            t_lang = item["target_lang"]
            base_name = Path(vid_path).stem
            out_file = str(out_base / f"{base_name}_dubbed_{t_lang}.mp4")

            def on_item_progress(prog: PipelineProgress) -> None:
                if queue_progress_callback:
                    queue_progress_callback(idx, total_jobs, prog)

            logger.info("=== Processando Trabalho %d/%d: %s -> [%s] ===", idx, total_jobs, base_name, t_lang)

            try:
                job_result = self.process_video(
                    input_video=vid_path,
                    target_language=t_lang,
                    source_language=source_language,
                    output_video=out_file,
                    enable_lipsync=enable_lipsync,
                    use_indextts2=use_indextts2,
                    burn_subtitles_flag=burn_subtitles,
                    export_raw_pkg=export_raw,
                    keep_temp_files=keep_temp_files,
                    job_index=idx,
                    total_jobs=total_jobs,
                    progress_callback=on_item_progress,
                )
                results.append(job_result)
            except Exception as e:
                logger.error("Erro no processamento do trabalho %d/%d (%s): %s", idx, total_jobs, vid_path, e)
                results.append({
                    "success": False,
                    "input_video": vid_path,
                    "target_language": t_lang,
                    "error": str(e),
                })
            finally:
                self._cleanup_gpu_vram()

        logger.info("Fila de processamento concluída: %d/%d trabalhos executados.", len(results), total_jobs)
        return results

    def run(self, progress_callback: Optional[Callable[[PipelineProgress], None]] = None) -> str:
        """
        Método wrapper para compatibilidade com PipelineConfig.
        """
        if not self.config:
            raise ValueError("DubPipeline.run() requer uma instância de PipelineConfig.")

        result = self.process_video(
            input_video=self.config.input_video,
            target_language=self.config.target_language,
            source_language=self.config.source_language,
            output_video=self.config.output_video,
            enable_lipsync=self.config.enable_lipsync,
            use_indextts2=self.config.use_indextts2,
            burn_subtitles_flag=self.config.burn_subtitles,
            export_raw_pkg=getattr(self.config, "export_raw_package", False),
            keep_temp_files=self.config.keep_temp_files,
            progress_callback=progress_callback,
        )
        return result["output_video"]


# Alias para compatibilidade retroativa
DubbingPipeline = DubPipeline
