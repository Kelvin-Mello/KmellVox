"""Módulo de transcrição e alinhamento de fala utilizando faster-whisper integrado a perfis de hardware."""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from core.hardware import ModelProfile

logger = logging.getLogger("KmellVox.Transcribe")


@dataclass
class TranscriptionSegment:
    """Segmento de fala transcrito com timestamps precisos e texto."""
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        """Permite acesso indexado como dicionário: seg['start'], seg['end'], seg['text']."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Chave '{key}' não encontrada no TranscriptionSegment.")


@dataclass
class TranscriptionResult:
    """Resultado completo da transcrição com metadados."""
    language: str
    language_probability: float
    duration_seconds: float
    segments: List[TranscriptionSegment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)

    def to_srt(self) -> str:
        """Exporta os segmentos no formato de legendas SRT."""
        return format_segments_to_srt(self.segments)

    def save_srt(self, filepath: str) -> str:
        """Salva a transcrição como arquivo .srt."""
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_srt(), encoding="utf-8")
        return str(path)


def _format_timestamp(seconds: float) -> str:
    """Converte segundos para o formato de tempo padrão do SRT: HH:MM:SS,mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_segments_to_srt(segments: List[Union[TranscriptionSegment, Dict[str, Any]]]) -> str:
    """
    Converte uma lista de segmentos (objetos ou dicionários) em conteúdo SRT válido.
    
    Cada segmento deve conter:
        - start (float)
        - end (float)
        - text (str)
    """
    srt_blocks = []
    for i, seg in enumerate(segments, 1):
        if isinstance(seg, dict):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = str(seg.get("text", "")).strip()
        else:
            start = getattr(seg, "start", 0.0)
            end = getattr(seg, "end", 0.0)
            text = getattr(seg, "text", "").strip()

        start_str = _format_timestamp(start)
        end_str = _format_timestamp(end)
        srt_blocks.append(f"{i}\n{start_str} --> {end_str}\n{text}\n")
    return "\n".join(srt_blocks)


class Transcriber:
    """
    Transcritor baseado no faster-whisper com configuração dinâmica de ModelProfile
    e liberação explícita de VRAM.
    """

    def __init__(
        self,
        model_profile: Optional[ModelProfile] = None,
        download_root: Optional[str] = "models/whisper",
        device: Optional[str] = None,
    ) -> None:
        """
        Inicializa o Transcriber configurando a variante e compute_type conforme o perfil.
        
        Args:
            model_profile: Instância de ModelProfile (se None, detecta automaticamente).
            download_root: Diretório raiz para cache/download dos pesos.
            device: Dispositivo de execução ('cuda' ou 'cpu').
        """
        self.profile = model_profile or ModelProfile.from_profile()
        self.model_variant = self.profile.whisper_variant
        self.compute_type = self.profile.whisper_compute_type
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if self.profile.profile_name != "cpu" else "cpu"

        self.download_root = download_root
        self.model = None

    def load_model(self) -> None:
        """Carrega o modelo faster-whisper na memória se ainda não estiver carregado."""
        if self.model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper não está instalado no ambiente virtual."
            ) from e

        logger.info(
            "Carregando faster-whisper [%s] no dispositivo '%s' com compute_type '%s' (Perfil: %s)...",
            self.model_variant,
            self.device,
            self.compute_type,
            self.profile.profile_name,
        )

        if self.download_root:
            Path(self.download_root).mkdir(parents=True, exist_ok=True)

        self.model = WhisperModel(
            self.model_variant,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
        )
        logger.info("faster-whisper [%s] carregado com sucesso.", self.model_variant)

    def unload_model(self) -> None:
        """
        Libera explicitamente o modelo da memória RAM e da VRAM da GPU (del + torch.cuda.empty_cache()),
        abrindo espaço livre para a próxima etapa do pipeline.
        """
        if self.model is not None:
            logger.info("Descarregando modelo faster-whisper [%s] e liberando VRAM...", self.model_variant)
            del self.model
            self.model = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() executado com sucesso.")
        except Exception as e:
            logger.debug("Limpeza de cache CUDA não necessária ou não suportada: %s", e)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        auto_unload: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[TranscriptionSegment]:
        """
        Executa a transcrição do áudio gerando segmentos com timestamps e texto.
        
        Args:
            audio_path: Caminho do arquivo de áudio (formato WAV mono 16kHz recomendado).
            language: Idioma de áudio ou None/'auto' para detecção automática.
            beam_size: Tamanho do beam search.
            vad_filter: Ativação do filtro de atividade de voz (VAD).
            word_timestamps: Extração de timestamps por palavra.
            auto_unload: Se True, descarrega o modelo da VRAM ao concluir a transcrição.
            progress_callback: Callback para acompanhamento do progresso (progresso_0_1, mensagem).
            
        Returns:
            List[TranscriptionSegment]: Lista de segmentos de fala com texto, início e fim.
        """
        self.load_model()

        if progress_callback:
            progress_callback(0.05, f"Iniciando transcrição ({self.model_variant})...")

        try:
            segments_gen, info = self.model.transcribe(
                str(audio_path),
                language=language if language and language != "auto" else None,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
            )

            detected_lang = info.language
            lang_prob = info.language_probability
            duration = info.duration
            logger.info(
                "Transcrição: idioma='%s' (prob=%.2f), duração=%.2fs",
                detected_lang,
                lang_prob,
                duration,
            )

            segments: List[TranscriptionSegment] = []
            for seg in segments_gen:
                words_data = []
                if seg.words:
                    words_data = [
                        {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                        for w in seg.words
                    ]

                trans_seg = TranscriptionSegment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    words=words_data,
                )
                segments.append(trans_seg)

                if progress_callback and duration > 0:
                    current_prog = min(0.95, 0.05 + (seg.end / duration) * 0.90)
                    progress_callback(current_prog, f"Transcrevendo: {seg.end:.1f}s / {duration:.1f}s")

            if progress_callback:
                progress_callback(1.0, "Transcrição concluída com sucesso.")

            return segments

        finally:
            if auto_unload:
                self.unload_model()

    def export_srt(
        self,
        segments: List[Union[TranscriptionSegment, Dict[str, Any]]],
        output_path: str,
    ) -> str:
        """
        Grava os segmentos transcritos em um arquivo .srt válido.
        
        Args:
            segments: Lista de TranscriptionSegment ou dicionários com start, end e text.
            output_path: Caminho de destino do arquivo .srt.
            
        Returns:
            str: Caminho absoluto do arquivo .srt gerado.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        srt_content = format_segments_to_srt(segments)
        out.write_text(srt_content, encoding="utf-8")
        logger.info("Arquivo SRT exportado com sucesso em: %s", out)
        return str(out)


# Alias para retrocompatibilidade
WhisperTranscriber = Transcriber
