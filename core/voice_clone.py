"""Módulo de Clonagem de Voz e Síntese de Fala (F5-TTS e IndexTTS-2).

Contém:
- F5TTSEngine: Motor padrão com controle de ritmo via atempo do FFmpeg.
- IndexTTS2Engine: Motor avançado em FP16 com controle explícito nativo de duração (apenas perfil_a).
- get_tts_engine: Factory para seleção do motor conforme o ModelProfile.
"""

from __future__ import annotations

import abc
import re as _re
import gc
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from core.dependency_manager import ensure_addon_in_sys_path

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
        speed: float = 1.4,
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


# ---------------------------------------------------------------------------
# Chunking inteligente para textos longos
# ---------------------------------------------------------------------------

# Limite máximo recomendado de caracteres por chunk para o F5-TTS.
# O modelo foi treinado com janela de ~30 segundos (referência + gerado).
# Textos com mais de ~300 chars por inferência geram áudio truncado/distorcido.
_MAX_CHUNK_CHARS = 200
_LONG_TEXT_THRESHOLD = 300  # Aciona chunking quando o texto excede este limite
_MAX_REF_AUDIO_SECONDS = 15.0  # Limite máximo recomendado para áudio de referência no F5-TTS


def split_text_into_chunks(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> List[str]:
    """
    Divide um texto longo em chunks respeitando limites de sentenças.

    Regras de divisão (em ordem de prioridade):
    1. Quebra em pontos finais de sentença ('. ', '! ', '? ')
    2. Se uma sentença individual exceder max_chars, quebra em vírgulas ou ponto-e-vírgulas
    3. Como último recurso, quebra no espaço mais próximo do limite

    Args:
        text: Texto completo a ser dividido.
        max_chars: Número máximo de caracteres por chunk (padrão: 200).

    Returns:
        Lista de chunks de texto, cada um com no máximo ~max_chars caracteres.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Texto curto: retorna inteiro
    if len(text) <= max_chars:
        return [text]

    # 1. Divide em sentenças (preserva o delimitador no final de cada sentença)
    sentence_pattern = _re.compile(r'(?<=[.!?])\s+')
    raw_sentences = sentence_pattern.split(text)

    # Limpa e filtra sentenças vazias
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        # Se a sentença sozinha cabe no chunk atual
        candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence

        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            # Salva o chunk atual (se tiver algo)
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Se a sentença sozinha excede max_chars, precisa subdividir
            if len(sentence) > max_chars:
                # Tenta dividir por vírgulas ou ponto-e-vírgulas
                sub_parts = _re.split(r'(?<=[,;])\s+', sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue
                    sub_candidate = (current_chunk + " " + part).strip() if current_chunk else part
                    if len(sub_candidate) <= max_chars:
                        current_chunk = sub_candidate
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # Se mesmo uma sub-parte excede, quebra por espaço
                        if len(part) > max_chars:
                            words = part.split()
                            current_chunk = ""
                            for word in words:
                                word_candidate = (current_chunk + " " + word).strip() if current_chunk else word
                                if len(word_candidate) <= max_chars:
                                    current_chunk = word_candidate
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = word
                        else:
                            current_chunk = part
            else:
                current_chunk = sentence

    # Último chunk remanescente
    if current_chunk:
        chunks.append(current_chunk)

    # Pós-processamento: mescla chunks muito pequenos (<50 chars) com o seguinte
    # Chunks minúsculos geram áudio de qualidade inferior no F5-TTS
    _MIN_CHUNK_CHARS = 50
    merged: List[str] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if len(c) < _MIN_CHUNK_CHARS and i + 1 < len(chunks):
            # Mescla com o próximo se o resultado não ultrapassar max_chars * 1.3
            combined = c + " " + chunks[i + 1]
            if len(combined) <= int(max_chars * 1.3):
                merged.append(combined)
                i += 2
                continue
        merged.append(c)
        i += 1

    return merged


def ensure_ffmpeg_in_path() -> None:
    """Garante que tools/ffmpeg/bin esteja incluído no PATH e registrado para DLLs.

    O torchcodec (dependência do F5-TTS >= 2.11) requer FFmpeg shared DLLs
    acessíveis via os.add_dll_directory() no Windows.
    """
    import sys

    # Candidatos para a pasta ffmpeg/bin
    candidates = [
        Path("tools/ffmpeg/bin"),
        Path(__file__).parent.parent / "tools" / "ffmpeg" / "bin",
    ]

    # No executável congelado, adiciona o diretório ao lado do .exe
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        candidates.insert(0, exe_dir / "tools" / "ffmpeg" / "bin")
        candidates.insert(1, exe_dir / "_internal" / "tools" / "ffmpeg" / "bin")

    for cand in candidates:
        if cand.is_dir():
            cand_str = str(cand.resolve())
            if cand_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = cand_str + os.pathsep + os.environ.get("PATH", "")

            # Registra para carga de DLLs (Windows 10+ / Python 3.8+)
            # Necessário para torchcodec encontrar avcodec-*.dll etc.
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(cand_str)
                except OSError:
                    pass
            break


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
        self._f5_instance = None

    def load_model(self) -> None:
        """Carrega o modelo F5-TTS com os pesos salvos localmente."""
        if self._f5_instance is not None:
            return

        ensure_ffmpeg_in_path()
        ensure_addon_in_sys_path()

        models_base = Path(self.models_dir)
        ckpt_cand = models_base / "tts" / "f5-tts" / "F5TTS_v1_Base" / "model_1250000.safetensors"
        vocab_cand = models_base / "tts" / "f5-tts" / "F5TTS_v1_Base" / "vocab.txt"

        ckpt_file = str(ckpt_cand.resolve()) if ckpt_cand.is_file() else ""
        vocab_file = str(vocab_cand.resolve()) if vocab_cand.is_file() else ""

        target_device = self.device or "cuda"
        logger.info("Carregando motor F5-TTS (dispositivo preferencial: '%s')...", target_device)

        try:
            from f5_tts.api import F5TTS
            try:
                self._f5_instance = F5TTS(
                    ckpt_file=ckpt_file,
                    vocab_file=vocab_file,
                    device=target_device,
                )
                logger.info("Modelo F5-TTS carregado com sucesso em '%s'.", target_device)
            except Exception as e:
                # Fallback para CPU se CUDA falhar (ex: incompatibilidade de kernels sm_120 no PyTorch cu124)
                if target_device != "cpu" and ("no kernel image" in str(e) or "CUDA" in str(e)):
                    logger.warning("Dispositivo CUDA indisponível para F5-TTS (%s). Alternando para CPU...", e)
                    self._f5_instance = F5TTS(
                        ckpt_file=ckpt_file,
                        vocab_file=vocab_file,
                        device="cpu",
                    )
                    logger.info("Modelo F5-TTS carregado com sucesso em modo CPU.")
                else:
                    raise

            self.model = self._f5_instance
        except ImportError:
            logger.warning("Pacote 'f5-tts' não encontrado no ambiente.")
            self.model = "f5_not_installed"
            self._f5_instance = None

    def _check_engine_available(self) -> None:
        """Lança RuntimeError explicativo se o engine não está instalado."""
        if self._f5_instance is None and self.model == "f5_not_installed":
            raise RuntimeError(
                "O pacote 'f5-tts' não está instalado neste ambiente.\n"
                "Para gerar áudio real, instale as dependências pelo menu de configurações."
            )

    def _prepare_reference_audio(self, reference_audio_path: str) -> str:
        """
        Prepara o áudio de referência para o F5-TTS, recortando para no máximo
        _MAX_REF_AUDIO_SECONDS (15s) se necessário.

        O F5-TTS foi treinado com janela de ~30 segundos (referência + gerado).
        Áudios de referência > 20s degradam severamente a qualidade e velocidade.

        Returns:
            str: Caminho do áudio preparado (pode ser o original ou um recorte temporário).
        """
        duration = get_audio_duration(reference_audio_path)
        if duration <= _MAX_REF_AUDIO_SECONDS:
            return reference_audio_path

        logger.warning(
            "Áudio de referência muito longo (%.1fs > %.1fs). Recortando automaticamente.",
            duration, _MAX_REF_AUDIO_SECONDS,
        )

        trimmed_path = Path(reference_audio_path).parent / f"_ref_trimmed_{Path(reference_audio_path).stem}.wav"
        bin_path = resolve_ffmpeg_binary()

        cmd = [
            bin_path,
            "-y",
            "-i", str(Path(reference_audio_path).resolve()),
            "-t", str(_MAX_REF_AUDIO_SECONDS),
            "-ar", "24000",
            "-ac", "1",
            str(trimmed_path.resolve()),
        ]

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
            logger.info(
                "Referência recortada: %.1fs -> %.1fs (%s)",
                duration, _MAX_REF_AUDIO_SECONDS, trimmed_path.name,
            )
            return str(trimmed_path.resolve())
        except subprocess.CalledProcessError as e:
            logger.warning("Falha ao recortar referência: %s. Usando original.", e.stderr)
            return reference_audio_path

    def _synthesize_raw(
        self,
        text: str,
        reference_audio_path: str,
        raw_output_path: str,
        reference_text: Optional[str] = None,
        target_duration: Optional[float] = None,
        speed: float = 1.4,
    ) -> None:
        """Executa a síntese direta de áudio com controle nativo de velocidade."""
        self._check_engine_available()
        ensure_ffmpeg_in_path()

        out = Path(raw_output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        # Recorta áudio de referência se exceder o limite do F5-TTS
        prepared_ref = self._prepare_reference_audio(reference_audio_path)

        if self._f5_instance is not None:
            try:
                # Procura transcrição sidecar (.txt) se ref_text não foi fornecido
                ref_txt = reference_text or ""
                if not ref_txt:
                    txt_sidecar = Path(reference_audio_path).with_suffix(".txt")
                    if txt_sidecar.is_file():
                        try:
                            full_txt = txt_sidecar.read_text(encoding="utf-8").strip()
                            # Se a referência foi recortada, recorta o texto de forma limpa
                            ref_dur = get_audio_duration(reference_audio_path)
                            if ref_dur > _MAX_REF_AUDIO_SECONDS and full_txt:
                                ratio = _MAX_REF_AUDIO_SECONDS / max(ref_dur, 0.1)
                                target_chars = int(len(full_txt) * ratio)
                                cut_text = full_txt[:target_chars]
                                last_period = cut_text.rfind(".")
                                if last_period > len(cut_text) * 0.4:
                                    ref_txt = cut_text[:last_period + 1].strip()
                                else:
                                    last_comma = cut_text.rfind(",")
                                    if last_comma > len(cut_text) * 0.4:
                                        ref_txt = cut_text[:last_comma].strip() + "."
                                    else:
                                        ref_txt = cut_text.strip()
                                        if not ref_txt.endswith("."):
                                            ref_txt += "."
                                logger.info(
                                    "Transcrição recortada alinhada: %d -> %d chars",
                                    len(full_txt), len(ref_txt),
                                )
                            else:
                                ref_txt = full_txt
                            logger.info("Transcrição de referência carregada de %s", txt_sidecar.name)
                        except Exception:
                            pass

                self._f5_instance.infer(
                    ref_file=prepared_ref,
                    ref_text=ref_txt,
                    gen_text=text,
                    file_wave=str(out),
                    nfe_step=32,
                    speed=speed,
                    target_rms=0.15,
                    cross_fade_duration=0.15,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1,
                )
                return
            except Exception as e:
                logger.error("Erro na inferência do F5-TTS: %s", e)
                raise RuntimeError(f"Falha na inferência do F5-TTS: {e}") from e


    def _synthesize_long_text(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        reference_text: Optional[str] = None,
    ) -> None:
        """
        Sintetiza textos longos via chunking inteligente por sentenças.

        Divide o texto em chunks de ~200 chars, sintetiza cada chunk
        individualmente com _synthesize_raw, e concatena todos os WAVs
        resultantes em um único arquivo de saída.
        """
        chunks = split_text_into_chunks(text, max_chars=_MAX_CHUNK_CHARS)
        total_chunks = len(chunks)

        if total_chunks <= 1:
            # Texto curto — sintetiza diretamente
            self._synthesize_raw(
                text=text,
                reference_audio_path=reference_audio_path,
                raw_output_path=output_path,
                reference_text=reference_text,
            )
            return

        logger.info(
            "Chunking ativado: texto de %d chars dividido em %d chunks (max %d chars/chunk)",
            len(text), total_chunks, _MAX_CHUNK_CHARS,
        )

        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_wav_files: List[str] = []

        try:
            for i, chunk_text in enumerate(chunks, 1):
                chunk_wav = str(out_path.parent / f"{out_path.stem}_chunk_{i:03d}.wav")
                logger.debug(
                    "  Chunk %d/%d (%d chars): '%s...'",
                    i, total_chunks, len(chunk_text), chunk_text[:60],
                )
                self._synthesize_raw(
                    text=chunk_text,
                    reference_audio_path=reference_audio_path,
                    raw_output_path=chunk_wav,
                    reference_text=reference_text,
                )
                chunk_wav_files.append(chunk_wav)

            # Concatena todos os chunks usando soundfile + numpy
            self._concat_wav_files(chunk_wav_files, str(out_path))
            logger.info(
                "Chunking concluído: %d chunks concatenados em %s (%.1fs)",
                total_chunks, out_path.name, get_audio_duration(str(out_path)),
            )

        finally:
            # Limpa arquivos temporários de chunks
            for cf in chunk_wav_files:
                if os.path.isfile(cf):
                    try:
                        os.remove(cf)
                    except Exception:
                        pass

    @staticmethod
    def _concat_wav_files(wav_files: List[str], output_path: str) -> None:
        """
        Concatena múltiplos arquivos WAV em um único arquivo de saída.
        Usa numpy + soundfile para concatenação precisa sem recodificação.
        """
        import numpy as np

        if not wav_files:
            return
        if len(wav_files) == 1:
            shutil.copyfile(wav_files[0], output_path)
            return

        all_audio: List[Any] = []
        target_sr: Optional[int] = None

        for wf in wav_files:
            try:
                data, sr = sf.read(wf, dtype="float32")
                if target_sr is None:
                    target_sr = sr
                elif sr != target_sr:
                    # Resample simples se sample rates diferirem
                    logger.warning(
                        "Sample rate diferente em chunk (%d vs %d). Usando o primeiro.", sr, target_sr
                    )
                all_audio.append(data)
            except Exception as e:
                logger.warning("Erro ao ler chunk WAV '%s': %s. Ignorando.", wf, e)

        if not all_audio or target_sr is None:
            raise RuntimeError("Nenhum chunk de áudio válido para concatenar.")

        # Garante que todos os arrays são mono (1D)
        mono_arrays = []
        for arr in all_audio:
            if arr.ndim > 1:
                arr = arr.mean(axis=1)  # Converte stereo para mono
            mono_arrays.append(arr)

        combined = np.concatenate(mono_arrays)
        sf.write(output_path, combined, target_sr)

    def clone_and_synthesize(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        target_duration: Optional[float] = None,
        reference_text: Optional[str] = None,
        auto_unload: bool = False,
        speed: float = 1.4,
    ) -> ClonedAudioSegment:
        """
        Gera áudio no timbre da voz de referência a partir do texto traduzido.
        Para textos longos (>300 chars), ativa chunking automático por sentenças.
        Ajusta a duração para caber no tempo de tela com FFmpeg atempo (Controle de Ritmo).
        """
        self.load_model()
        final_out = Path(output_path).resolve()
        final_out.parent.mkdir(parents=True, exist_ok=True)

        raw_temp_path = str(final_out.with_suffix(".raw.wav"))
        use_chunking = len(text.strip()) > _LONG_TEXT_THRESHOLD

        try:
            # Delega o chunking ao F5-TTS que possui chunking interno com cross-fade.
            # O parâmetro speed controla a taxa de fala nativa durante o Flow Matching.
            self._synthesize_raw(
                text=text.strip(),
                reference_audio_path=reference_audio_path,
                raw_output_path=raw_temp_path,
                reference_text=reference_text,
                speed=speed,
            )

            actual_dur = get_audio_duration(raw_temp_path)
            target_dur = target_duration or actual_dur
            speed_factor = 1.0

            # Ajuste de duração via FFmpeg atempo (Controle de Ritmo)
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
            # Tenta carregar do módulo index_tts ou indextts se instalado/clonado
            import torch
            try:
                from index_tts import IndexTTS
                self.model = IndexTTS(device=self.device, use_fp16=True)
            except ImportError:
                import indextts
                self.model = "indextts_loaded"
            logger.info("Modelo IndexTTS-2 carregado com sucesso em FP16.")
        except ImportError:
            logger.warning(
                "Repositório/pacote 'index-tts' não encontrado no ambiente. "
                "Instale via o README.md (requer clone do repositório IndexTTS-2 + dependências)."
            )
            self.model = "indextts2_not_installed"

    def _check_engine_available(self) -> None:
        """Lança RuntimeError se o IndexTTS-2 não está instalado."""
        if self.model == "indextts2_not_installed":
            raise RuntimeError(
                "O motor IndexTTS-2 não está instalado neste ambiente.\n"
                "Para usar clônagem de voz, instale o IndexTTS-2 seguindo as instruções do README.md,\n"
                "ou instale o F5-TTS (pip install f5-tts torch torchaudio) para o motor padrão."
            )

    def clone_and_synthesize(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        target_duration: Optional[float] = None,
        reference_text: Optional[str] = None,
        auto_unload: bool = False,
        speed: float = 1.4,
    ) -> ClonedAudioSegment:
        """
        Sintetiza a fala clonada usando o modo de controle explícito de duração do IndexTTS-2,
        gerando o áudio já calibrado no tempo exato, sem necessidade de pós-processamento atempo.
        """
        self.load_model()
        self._check_engine_available()

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
                # IndexTTS-2 instalado mas sem synthesize_with_duration — tenta inferência padrão
                self.model.infer(
                    audio_prompt=reference_audio_path,
                    text=text,
                    output_path=str(final_out),
                )

            actual_dur = get_audio_duration(str(final_out))

            return ClonedAudioSegment(
                id=0,
                start=0.0,
                end=target_dur,
                audio_path=str(final_out),
                target_duration=target_dur,
                actual_duration=actual_dur,
                speed_factor=1.0,
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
    Factory que retorna a engine de síntese e clonagem de voz adequada,
    com fallback automático caso o motor preferido não esteja instalado.
    
    Cadeia de resolução:
        1. use_advanced=True + perfil_a → IndexTTS2Engine (se instalado)
        2. Fallback → F5TTSEngine (se instalado)
        3. Nenhum instalado → RuntimeError com instruções de instalação
        
    Args:
        model_profile: Perfil de hardware detectado (se None, detecta automaticamente).
        use_advanced: Se True, tenta ativar IndexTTS-2.
        **kwargs: Parâmetros extras para a engine (device, models_dir, rhythm_config).
        
    Returns:
        BaseTTSEngine: Instância de F5TTSEngine ou IndexTTS2Engine.
        
    Raises:
        RuntimeError: Se nenhum motor TTS estiver instalado no ambiente.
    """
    # Garante que o python_env (instalado após boot) esteja no sys.path e DLLs registradas
    ensure_addon_in_sys_path()

    profile = model_profile or ModelProfile.from_profile()

    # --- Tenta o motor avançado IndexTTS-2 (perfil_a com 8GB+) ---
    if use_advanced and profile.enable_indextts_2:
        logger.info("Tentando inicializar motor avançado IndexTTS2Engine (perfil_a)...")
        try:
            engine = IndexTTS2Engine(model_profile=profile, **kwargs)
            engine.load_model()
            if engine.model and engine.model != "indextts2_not_installed" and not isinstance(engine.model, str):
                return engine
        except Exception as e:
            logger.warning("IndexTTS-2 indisponível (%s). Utilizando motor padrão F5TTSEngine.", e)

    # --- Tenta o motor padrão F5-TTS ---
    logger.info("Selecionando motor de clonagem: F5TTSEngine (Perfil: %s).", profile.profile_name)
    engine = F5TTSEngine(model_profile=profile, **kwargs)
    engine.load_model()
    if engine.model != "f5_not_installed":
        return engine


    # --- Nenhum motor disponível ---
    raise RuntimeError(
        "Nenhum motor de síntese de voz (TTS) está instalado neste ambiente.\n\n"
        "Para gerar áudio com clonagem de voz, instale pelo menos um dos seguintes:\n\n"
        "  Opção 1 — F5-TTS (recomendado para começar):\n"
        "    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
        "    pip install f5-tts\n\n"
        "  Opção 2 — IndexTTS-2 (alta qualidade, requer 8GB+ VRAM):\n"
        "    Consulte o README.md para instruções de instalação do IndexTTS-2.\n\n"
        "  Ou instale todas as dependências de GPU de uma vez:\n"
        "    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
        "    pip install -r requirements-gpu.txt"
    )


# Alias para retrocompatibilidade
VoiceCloner = F5TTSEngine
VoiceCloneConfig = RhythmControlConfig
