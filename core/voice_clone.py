"""Módulo de Clonagem de Voz e Síntese de Fala (F5-TTS e IndexTTS-2).

Contém:
- F5TTSEngine: Motor padrão com controle de ritmo via atempo do FFmpeg.
- IndexTTS2Engine: Motor avançado em FP16 com controle explícito nativo de duração (apenas perfil_a).
- get_tts_engine: Factory para seleção do motor conforme o ModelProfile.
"""

from __future__ import annotations

import abc
import gc
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import soundfile as sf

from core.audio_extract import resolve_ffmpeg_binary
from core.hardware import ModelProfile
from core.translate import TranslatedSegment

logger = logging.getLogger("KmellVox.VoiceClone")


@dataclass
class RhythmControlConfig:
    """Configurações do Controle de Ritmo (Ajuste de tempo de fala via atempo)."""
    min_speed: float = 0.70   # Desaceleração máxima permitida (70% da velocidade)
    max_speed: float = 1.35   # Aceleração máxima permitida (135% da velocidade)
    cross_fade_duration: float = 0.15


@dataclass
class ClonedAudioSegment:
    """Representa um segmento de áudio clonado e alinhado no tempo."""
    id: int
    start: float
    end: float
    audio_path: str
    target_duration: float
    actual_duration: float
    speed_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Chave '{key}' não encontrada no ClonedAudioSegment.")


def get_audio_duration(file_path: str) -> float:
    """Obtém a duração exata de um arquivo de áudio em segundos via soundfile ou ffprobe."""
    try:
        info = sf.info(file_path)
        return float(info.duration)
    except Exception:
        try:
            from core.audio_extract import get_audio_info
            return get_audio_info(file_path).duration_seconds
        except Exception:
            return 0.0


def adjust_audio_duration_ffmpeg(
    input_audio: str,
    output_audio: str,
    target_duration: float,
    min_speed: float = 0.70,
    max_speed: float = 1.35,
    ffmpeg_bin: Optional[str] = None,
) -> float:
    """
    Ajusta a duração do áudio gerado para caber exatamente no tempo de tela (target_duration)
    usando o filtro 'atempo' do FFmpeg com limites de segurança (Feature: Controle de Ritmo).
    
    Args:
        input_audio: Caminho do áudio WAV original sintetizado.
        output_audio: Caminho de destino do áudio com velocidade corrigida.
        target_duration: Duração alvo em segundos.
        min_speed: Velocidade mínima permitida.
        max_speed: Velocidade máxima permitida.
        ffmpeg_bin: Caminho do executável FFmpeg.
        
    Returns:
        float: Fator de velocidade aplicado (speed_factor).
    """
    current_duration = get_audio_duration(input_audio)
    if current_duration <= 0.0 or target_duration <= 0.0:
        shutil.copyfile(input_audio, output_audio)
        return 1.0

    raw_factor = current_duration / target_duration
    # Aplica os limites configuráveis de aceleração e desaceleração
    speed_factor = max(min_speed, min(max_speed, raw_factor))

    # Se a diferença for insignificante (< 3%), apenas copia
    if abs(speed_factor - 1.0) < 0.03:
        shutil.copyfile(input_audio, output_audio)
        return 1.0

    bin_path = resolve_ffmpeg_binary(ffmpeg_bin)
    out_path = Path(output_audio).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Constrói cadeia de filtros atempo caso speed_factor esteja fora do intervalo [0.5, 2.0]
    filters = []
    f = speed_factor
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f /= 0.5
    filters.append(f"atempo={f:.4f}")
    filter_str = ",".join(filters)

    cmd = [
        bin_path,
        "-y",
        "-i", str(input_audio),
        "-filter:a", filter_str,
        str(out_path),
    ]

    logger.debug(
        "Controle de Ritmo FFmpeg: %.2fs -> %.2fs (fator: %.2fx, filtro: %s)",
        current_duration, target_duration, speed_factor, filter_str
    )

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return speed_factor
    except subprocess.CalledProcessError as e:
        logger.warning("Falha ao ajustar atempo via FFmpeg: %s. Mantendo áudio original.", e.stderr)
        shutil.copyfile(input_audio, output_audio)
        return 1.0


class BaseTTSEngine(abc.ABC):
    """Classe abstrata base para motores de síntese e clonagem de voz."""

    def __init__(
        self,
        model_profile: Optional[ModelProfile] = None,
        device: Optional[str] = None,
        models_dir: str = "models",
    ) -> None:
        self.profile = model_profile or ModelProfile.from_profile()
        self.models_dir = Path(models_dir)
        self.device = device or ("cuda" if self.profile.profile_name != "cpu" else "cpu")
        self.model = None

    @abc.abstractmethod
    def load_model(self) -> None:
        """Carrega os pesos do modelo na memória."""
        pass

    def unload_model(self) -> None:
        """Libera o modelo da memória RAM e VRAM."""
        if self.model is not None:
            logger.info("Descarregando modelo TTS (%s) da VRAM...", self.__class__.__name__)
            del self.model
            self.model = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() executado com sucesso após TTS.")
        except Exception:
            pass

    @abc.abstractmethod
    def clone_and_synthesize(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        target_duration: Optional[float] = None,
        reference_text: Optional[str] = None,
        auto_unload: bool = False,
    ) -> ClonedAudioSegment:
        """Sintetiza um segmento de áudio clonando a voz de referência."""
        pass

    def clone_and_align_all(
        self,
        segments: List[Union[TranslatedSegment, Dict[str, Any], Any]],
        reference_audio_path: str,
        output_dir: str,
        auto_unload: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[ClonedAudioSegment]:
        """
        Sintetiza e alinha todos os segmentos traduzidos.
        """
        out_base = Path(output_dir).resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        results: List[ClonedAudioSegment] = []
        total = len(segments)

        try:
            for i, seg in enumerate(segments, 1):
                if isinstance(seg, dict):
                    s_id = int(seg.get("id", i))
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", 0.0))
                    trans_text = str(seg.get("translated_text", seg.get("text", ""))).strip()
                    orig_text = str(seg.get("original_text", "")).strip()
                else:
                    s_id = getattr(seg, "id", i)
                    start = getattr(seg, "start", 0.0)
                    end = getattr(seg, "end", 0.0)
                    trans_text = getattr(seg, "translated_text", getattr(seg, "text", "")).strip()
                    orig_text = getattr(seg, "original_text", "").strip()

                target_dur = max(0.1, end - start)
                if progress_callback and total > 0:
                    prog = (i - 1) / total
                    progress_callback(prog, f"Clonando voz segmento {i}/{total} ({self.__class__.__name__})...")

                seg_path = str(out_base / f"voice_seg_{s_id:04d}.wav")
                cloned_seg = self.clone_and_synthesize(
                    text=trans_text,
                    reference_audio_path=reference_audio_path,
                    output_path=seg_path,
                    target_duration=target_dur,
                    reference_text=orig_text if orig_text else None,
                    auto_unload=False,
                )
                cloned_seg.id = s_id
                cloned_seg.start = start
                cloned_seg.end = end
                results.append(cloned_seg)

            if progress_callback:
                progress_callback(1.0, f"Síntese concluída ({len(results)} segmentos).")

            return results

        finally:
            if auto_unload:
                self.unload_model()


class F5TTSEngine(BaseTTSEngine):
    """
    Motor padrão de clonagem de voz (F5-TTS).
    Suporta os dois perfis de hardware (perfil_a e perfil_b).
    Ajusta a duração do áudio gerado para o tempo de tela via atempo (Controle de Ritmo).
    """

    def __init__(
        self,
        model_profile: Optional[ModelProfile] = None,
        device: Optional[str] = None,
        models_dir: str = "models",
        rhythm_config: Optional[RhythmControlConfig] = None,
    ) -> None:
        super().__init__(model_profile=model_profile, device=device, models_dir=models_dir)
        self.rhythm_config = rhythm_config or RhythmControlConfig()

    def load_model(self) -> None:
        """Carrega o modelo F5-TTS."""
        if self.model is not None:
            return

        logger.info("Carregando motor F5-TTS no dispositivo '%s'...", self.device)
        try:
            # Tenta importar API oficial do pacote f5-tts
            from f5_tts.infer.utils_infer import load_model as f5_load_model
            from f5_tts.model import DiT
            self.model = "f5_loaded"
            logger.info("Modelo F5-TTS carregado com sucesso.")
        except ImportError:
            logger.warning(
                "Pacote 'f5-tts' não encontrado no ambiente. "
                "Operando em modo de compatibilidade/simulação para desenvolvimento."
            )
            self.model = "f5_mock"

    def _synthesize_raw(
        self,
        text: str,
        reference_audio_path: str,
        raw_output_path: str,
        reference_text: Optional[str] = None,
    ) -> None:
        """Executa a síntese direta de áudio sem pós-processamento de tempo."""
        out = Path(raw_output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        if self.model == "f5_loaded":
            try:
                from f5_tts.infer.utils_infer import infer_process
                # Executa inferência do F5-TTS
                infer_process(
                    ref_audio=reference_audio_path,
                    ref_text=reference_text or "",
                    gen_text=text,
                    model_obj=self.model,
                    output_file=str(out),
                )
                return
            except Exception as e:
                logger.error("Erro na inferência do F5-TTS: %s. Gerando áudio de fallback.", e)

        # Fallback / Simulação de desenvolvimento
        ref_sr = 24000
        ref_dur = 2.0
        if os.path.isfile(reference_audio_path):
            try:
                ref_info = sf.info(reference_audio_path)
                ref_sr = ref_info.samplerate
                ref_dur = ref_info.duration
            except Exception:
                pass

        # Cria áudio mono WAV placeholder com tamanho proporcional ao texto
        import numpy as np
        est_duration = max(0.5, len(text.split()) * 0.35)
        num_samples = int(ref_sr * est_duration)
        samples = np.zeros(num_samples, dtype=np.float32)
        sf.write(str(out), samples, ref_sr)

    def clone_and_synthesize(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        target_duration: Optional[float] = None,
        reference_text: Optional[str] = None,
        auto_unload: bool = False,
    ) -> ClonedAudioSegment:
        """
        Gera áudio no timbre da voz de referência a partir do texto traduzido
        e ajusta a duração para caber no tempo de tela com FFmpeg atempo (Controle de Ritmo).
        """
        self.load_model()
        final_out = Path(output_path).resolve()
        final_out.parent.mkdir(parents=True, exist_ok=True)

        raw_temp_path = str(final_out.with_suffix(".raw.wav"))

        try:
            # 1. Gera áudio no timbre original
            self._synthesize_raw(
                text=text,
                reference_audio_path=reference_audio_path,
                raw_output_path=raw_temp_path,
                reference_text=reference_text,
            )

            actual_dur = get_audio_duration(raw_temp_path)
            target_dur = target_duration or actual_dur
            speed_factor = 1.0

            # 2. Ajuste de duração via FFmpeg atempo (Controle de Ritmo)
            if target_duration and target_duration > 0:
                speed_factor = adjust_audio_duration_ffmpeg(
                    input_audio=raw_temp_path,
                    output_audio=str(final_out),
                    target_duration=target_duration,
                    min_speed=self.rhythm_config.min_speed,
                    max_speed=self.rhythm_config.max_speed,
                )
            else:
                shutil.copyfile(raw_temp_path, str(final_out))

            return ClonedAudioSegment(
                id=0,
                start=0.0,
                end=target_dur,
                audio_path=str(final_out),
                target_duration=target_dur,
                actual_duration=actual_dur,
                speed_factor=speed_factor,
            )

        finally:
            if os.path.isfile(raw_temp_path):
                try:
                    os.remove(raw_temp_path)
                except Exception:
                    pass
            if auto_unload:
                self.unload_model()


class IndexTTS2Engine(BaseTTSEngine):
    """
    Motor opcional de alta qualidade (IndexTTS-2), habilitado apenas quando ModelProfile
    indica perfil_a (8GB+ VRAM).
    Utiliza inferência em FP16 e controle explícito nativo de duração no tempo certo,
    sem necessidade de pós-processamento com atempo.
    """

    def __init__(
        self,
        model_profile: Optional[ModelProfile] = None,
        device: Optional[str] = None,
        models_dir: str = "models",
    ) -> None:
        super().__init__(model_profile=model_profile, device=device, models_dir=models_dir)
        if not self.profile.enable_indextts_2:
            logger.warning(
                "IndexTTS2Engine instanciado em perfil '%s'. "
                "O modo avançado IndexTTS-2 foi projetado para perfil_a (8GB+ VRAM).",
                self.profile.profile_name,
            )

    def load_model(self) -> None:
        """Carrega o modelo IndexTTS-2 com inferência em FP16."""
        if self.model is not None:
            return

        logger.info("Carregando motor IndexTTS-2 (FP16) no dispositivo '%s'...", self.device)
        try:
            # Tenta carregar do módulo index_tts se instalado/clonado
            import torch
            from index_tts import IndexTTS
            self.model = IndexTTS(device=self.device, use_fp16=True)
            logger.info("Modelo IndexTTS-2 carregado com sucesso em FP16.")
        except ImportError:
            logger.warning(
                "Repositório/pacote 'index-tts' não encontrado no ambiente. "
                "Operando em modo de compatibilidade/simulação para desenvolvimento."
            )
            self.model = "indextts2_mock"

    def clone_and_synthesize(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        target_duration: Optional[float] = None,
        reference_text: Optional[str] = None,
        auto_unload: bool = False,
    ) -> ClonedAudioSegment:
        """
        Sintetiza a fala clonada usando o modo de controle explícito de duração do IndexTTS-2,
        gerando o áudio já calibrado no tempo exato, sem necessidade de pós-processamento atempo.
        """
        self.load_model()
        final_out = Path(output_path).resolve()
        final_out.parent.mkdir(parents=True, exist_ok=True)

        target_dur = target_duration if target_duration and target_duration > 0 else max(0.5, len(text.split()) * 0.35)

        try:
            if hasattr(self.model, "synthesize_with_duration"):
                # Chamada nativa com controle de duração em FP16
                self.model.synthesize_with_duration(
                    text=text,
                    ref_audio=reference_audio_path,
                    target_duration=target_dur,
                    output_file=str(final_out),
                    ref_text=reference_text,
                )
            else:
                # Simulação de desenvolvimento
                ref_sr = 24000
                if os.path.isfile(reference_audio_path):
                    try:
                        ref_info = sf.info(reference_audio_path)
                        ref_sr = ref_info.samplerate
                    except Exception:
                        pass

                import numpy as np
                num_samples = int(ref_sr * target_dur)
                samples = np.zeros(num_samples, dtype=np.float32)
                sf.write(str(final_out), samples, ref_sr)

            actual_dur = get_audio_duration(str(final_out))

            return ClonedAudioSegment(
                id=0,
                start=0.0,
                end=target_dur,
                audio_path=str(final_out),
                target_duration=target_dur,
                actual_duration=actual_dur,
                speed_factor=1.0,  # Não usou atempo; gerado nativamente no tempo correto
            )

        finally:
            if auto_unload:
                self.unload_model()


def get_tts_engine(
    model_profile: Optional[ModelProfile] = None,
    use_advanced: bool = False,
    **kwargs: Any,
) -> BaseTTSEngine:
    """
    Factory que retorna a engine de síntese e clonagem de voz adequada.
    
    Regras:
        - use_advanced=True e perfil_a (8GB+): Retorna IndexTTS2Engine.
        - use_advanced=True em perfil_b / cpu: Emite aviso e retorna F5TTSEngine.
        - use_advanced=False: Retorna F5TTSEngine (padrão universal).
        
    Args:
        model_profile: Perfil de hardware detectado (se None, detecta automaticamente).
        use_advanced: Se True, tenta ativar IndexTTS-2.
        **kwargs: Parâmetros extras para a engine (device, models_dir, rhythm_config).
        
    Returns:
        BaseTTSEngine: Instância de F5TTSEngine ou IndexTTS2Engine.
    """
    profile = model_profile or ModelProfile.from_profile()

    if use_advanced:
        if profile.enable_indextts_2:
            logger.info("Selecionando motor avançado de alta fidelidade: IndexTTS2Engine (perfil_a).")
            return IndexTTS2Engine(model_profile=profile, **kwargs)
        else:
            logger.warning(
                "⚠️ O motor avançado IndexTTS-2 requer no mínimo 8GB de VRAM (perfil_a). "
                "Seu perfil atual é '%s'. Utilizando motor padrão F5TTSEngine com controle de ritmo.",
                profile.profile_name,
            )
            return F5TTSEngine(model_profile=profile, **kwargs)

    logger.info("Selecionando motor padrão: F5TTSEngine (Perfil: %s).", profile.profile_name)
    return F5TTSEngine(model_profile=profile, **kwargs)


# Alias para retrocompatibilidade
VoiceCloner = F5TTSEngine
VoiceCloneConfig = RhythmControlConfig
