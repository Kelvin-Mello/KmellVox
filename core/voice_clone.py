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
# CRITICAL: Previne fragmentação de VRAM em GPUs de 8GB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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


DEFAULT_SENTENCE_PAUSE_SECONDS = 0.80  # Pausa natural entre frases calibrada pela voz original (800ms)


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


def sanitize_tts_text(text: str) -> str:
    """
    Higieniza o texto para modelos de síntese neural (TTS),
    removendo caracteres invisíveis (BOM, zero-width spaces)
    e normalizando espaçamentos que degradam os tokenizers.
    """
    if not text:
        return ""
    import re
    # Remove BOM e caracteres invisíveis de largura zero
    cleaned = re.sub(r"[\ufeff\u200b\u200c\u200d\u2060\ufffe]", "", text)
    # Normaliza espaços múltiplos
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


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

    @staticmethod
    def _concat_wav_files(
        wav_files: List[str],
        output_path: str,
        pause_seconds: float = DEFAULT_SENTENCE_PAUSE_SECONDS,
    ) -> None:
        """
        Concatena múltiplos arquivos WAV em um único arquivo de saída.
        Insere um silêncio acústico suave (pause_seconds) entre os blocos
        para reproduzir fielmente os respiros contemplativos da voz original.
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
                if data.ndim > 1:
                    data = data.mean(axis=1)  # Converte stereo para mono
                all_audio.append(data)
            except Exception as e:
                logger.warning("Erro ao ler chunk WAV '%s': %s. Ignorando.", wf, e)

        if not all_audio or target_sr is None:
            raise RuntimeError("Nenhum chunk de áudio válido para concatenar.")

        # Inserção da pausa de silêncio calibrada entre as sentenças
        silence_samples = int(max(0.0, pause_seconds) * target_sr)
        silence_gap = np.zeros(silence_samples, dtype=np.float32) if silence_samples > 0 else None

        combined_parts = []
        for i, arr in enumerate(all_audio):
            combined_parts.append(arr)
            if silence_gap is not None and i < len(all_audio) - 1:
                combined_parts.append(silence_gap)

        combined = np.concatenate(combined_parts)
        sf.write(output_path, combined, target_sr)

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


def split_text_into_sentences(text: str) -> List[str]:
    """
    Divide um texto em frases/sentenças completas preservando a pontuação terminativa.
    Permite a injeção controlada de pausas naturais (respiros acústicos) entre as orações.
    """
    if not text or not text.strip():
        return []

    # Divide por quebras de linha ou pontuação final (. ! ?) seguida de espaço,
    # preservando reticências dramáticas com continuação minúscula (... and, … but) na mesma oração
    sentence_pattern = _re.compile(r'(?<!\.\.)(?<!…)(?<=[.!?])\s+|\n+|(?<=[…\.\.\.])\s+(?=[A-ZÀ-ÿ0-9"“\'‘])')
    raw_sentences = sentence_pattern.split(text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def split_text_into_narrative_chunks(
    text: str,
    max_chars: int = 250,
    min_chars: int = 80,
) -> List[str]:
    """
    Divide o texto em blocos narrativos coesos (narrative chunks) agrupando
    sentenças completas sem nunca particionar o interior de uma oração.

    Por que blocos narrativos em vez de sentenças isoladas?
    Modelos neurais autorregressivos (como o IndexTTS-2) sofrem anomalia de "cold-start"
    quando recebem frases minúsculas isoladas (ex: "Listen closely." de apenas 4 tokens),
    gerando hesitações acústicas artificiais e pausas internas indesejadas.

    Por que não fatiar cegamente em tokens por vírgula?
    Cortar dentro de orações causa pausas duplas em conjunções ("and,"), alteração de timbre
    e quebras de cadência antinaturais.

    Regras do Agrupamento Narrativo:
    1. A fronteira de um chunk é SEMPRE o término de uma sentença (. ! ? ... :), NUNCA uma vírgula.
    2. Frases curtas (< min_chars, como "Listen closely.") são agrupadas à próxima sentença.
    3. Frases longas individuais (>= max_chars) formam um chunk autônomo e não são partidas.
    4. Cada chunk acumula até ~max_chars (mantendo ~35-80 tokens de contexto acústico ideal).
    """
    sentences = split_text_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_chars = 0

    for s in sentences:
        s_len = len(s)
        if current_sentences and (current_chars + s_len + 1 > max_chars) and current_chars >= min_chars:
            chunks.append(" ".join(current_sentences))
            current_sentences = [s]
            current_chars = s_len
        else:
            current_sentences.append(s)
            current_chars += (s_len + 1) if current_chars > 0 else s_len

    if current_sentences:
        if chunks and current_chars < min_chars and (len(chunks[-1]) + current_chars + 1 <= max_chars + 100):
            chunks[-1] = chunks[-1] + " " + " ".join(current_sentences)
        else:
            chunks.append(" ".join(current_sentences))

    return chunks


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
                ref_txt = sanitize_tts_text(reference_text or "")
                if not ref_txt:
                    txt_sidecar = Path(reference_audio_path).with_suffix(".txt")
                    if txt_sidecar.is_file():
                        try:
                            full_txt = sanitize_tts_text(txt_sidecar.read_text(encoding="utf-8-sig"))
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

                target_text = sanitize_tts_text(text)
                if target_text and not target_text.endswith((".", "!", "?", "...", ":")):
                    target_text += "."
                # Adiciona um espaço suave ao final para dar folga de frames ao Vocos e impedir corte da última palavra
                target_text = target_text + " "

                self._f5_instance.infer(
                    ref_file=prepared_ref,
                    ref_text=ref_txt,
                    gen_text=target_text,
                    file_wave=str(out),
                    nfe_step=32,
                    speed=speed,
                    target_rms=0.1,
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
        speed: float = 1.0,
        sentence_pause_seconds: float = DEFAULT_SENTENCE_PAUSE_SECONDS,
    ) -> None:
        """
        Sintetiza textos longos ou multi-sentenças via chunking inteligente.
        Sintetiza cada sentença individualmente e concatena intercalando a pausa
        natural configurada (respiro de estúdio) entre as frases.
        """
        sentences = split_text_into_sentences(text)
        chunks: List[str] = []
        for s in sentences:
            sub = split_text_into_chunks(s, max_chars=_MAX_CHUNK_CHARS)
            chunks.extend(sub)

        total_chunks = len(chunks)

        if total_chunks <= 1:
            # Texto curto de uma única frase — sintetiza diretamente
            self._synthesize_raw(
                text=text,
                reference_audio_path=reference_audio_path,
                raw_output_path=output_path,
                reference_text=reference_text,
                speed=speed,
            )
            return

        logger.info(
            "Chunking por sentenças ativado: %d sentenças/chunks (pausa entre frases: %.2fs)",
            total_chunks, sentence_pause_seconds,
        )

        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_wav_files: List[str] = []

        try:
            for i, chunk_text in enumerate(chunks, 1):
                chunk_wav = str(out_path.parent / f"{out_path.stem}_chunk_{i:03d}.wav")
                logger.debug(
                    "  Sentença %d/%d (%d chars): '%s...'",
                    i, total_chunks, len(chunk_text), chunk_text[:60],
                )
                self._synthesize_raw(
                    text=chunk_text,
                    reference_audio_path=reference_audio_path,
                    raw_output_path=chunk_wav,
                    reference_text=reference_text,
                    speed=speed,
                )
                chunk_wav_files.append(chunk_wav)

            # Concatena todos os chunks inserindo silêncio de respiro entre as sentenças
            self._concat_wav_files(chunk_wav_files, str(out_path), pause_seconds=sentence_pause_seconds)
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

    def clone_and_synthesize(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        target_duration: Optional[float] = None,
        reference_text: Optional[str] = None,
        auto_unload: bool = False,
        speed: float = 1.0,
        sentence_pause_seconds: float = DEFAULT_SENTENCE_PAUSE_SECONDS,
    ) -> ClonedAudioSegment:
        """
        Gera áudio no timbre da voz de referência a partir do texto.
        Divide sentenças para injetar pausas naturais entre orações (respiro de estúdio).
        Ajusta a duração para caber no tempo de tela com FFmpeg atempo (Controle de Ritmo).
        """
        self.load_model()
        final_out = Path(output_path).resolve()
        final_out.parent.mkdir(parents=True, exist_ok=True)

        raw_temp_path = str(final_out.with_suffix(".raw.wav"))
        clean_text = text.strip()

        # Garante que o texto termine com pontuação e espaço para dar folga acústica
        # à última palavra, impedindo que o modelo corte o som abruptamente.
        if clean_text and not clean_text.endswith((".", "!", "?", "...", ":")):
            clean_text += "."

        use_chunking = len(clean_text) > _LONG_TEXT_THRESHOLD

        try:
            if use_chunking:
                self._synthesize_long_text(
                    text=clean_text,
                    reference_audio_path=reference_audio_path,
                    output_path=raw_temp_path,
                    reference_text=reference_text,
                    speed=speed,
                    sentence_pause_seconds=sentence_pause_seconds,
                )
            else:
                self._synthesize_raw(
                    text=clean_text,
                    reference_audio_path=reference_audio_path,
                    raw_output_path=raw_temp_path,
                    reference_text=reference_text,
                    speed=speed,
                )

            # Pós-processamento acústico do áudio bruto:
            # 1. Remove silêncio morto no início (evita corte da primeira palavra pelo atempo)
            # 2. Adiciona cauda de decaimento (250ms) no final (evita corte abrupto da última palavra)
            try:
                import numpy as np
                raw_data, raw_sr = sf.read(raw_temp_path, dtype="float32")
                if raw_data.ndim > 1:
                    raw_data = raw_data.mean(axis=1)

                # Trimming de silêncio inicial com proteção acústica:
                # Usa limiar sensível (0.002) e margem generosa de 80ms antes da primeira fala
                # para garantir que o algoritmo WSOLA (atempo) e compressores nunca cortem
                # as consoantes ou sílabas iniciais (como 'Look', 'Listen' ou 'Closely').
                _SILENCE_THRESHOLD = 0.002
                _HEAD_MARGIN_SEC = 0.08  # 80ms de margem de respiro antes da fala
                _MIN_REMAINING_SEC = 0.1  # Garante pelo menos 100ms de áudio restante
                trim_idx = 0
                for idx in range(len(raw_data)):
                    if abs(raw_data[idx]) > _SILENCE_THRESHOLD:
                        trim_idx = idx
                        break
                margin_samples = int(_HEAD_MARGIN_SEC * raw_sr)
                trim_start = max(0, trim_idx - margin_samples)
                min_remaining = int(_MIN_REMAINING_SEC * raw_sr)
                # Safety: só aplica trim se sobrar áudio suficiente
                if trim_start > 0 and (len(raw_data) - trim_start) >= min_remaining:
                    logger.debug(
                        "Silêncio inicial ajustado: %d amostras (%.0f ms)",
                        trim_start, 1000 * trim_start / raw_sr,
                    )
                    raw_data = raw_data[trim_start:]

                # Micro-fade in de 5ms no início para evitar cliques
                fade_samples = min(int(0.005 * raw_sr), len(raw_data))
                if fade_samples > 0:
                    fade_curve = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                    raw_data[:fade_samples] *= fade_curve

                # Tail padding: adiciona 250ms de silêncio suave no final
                tail_pad = np.zeros(int(0.25 * raw_sr), dtype=np.float32)
                raw_data = np.concatenate([raw_data, tail_pad])

                sf.write(raw_temp_path, raw_data, raw_sr)
            except Exception as e:
                logger.debug("Falha no pós-processamento acústico: %s", e)

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
                # Mantém o áudio bruto
                shutil.copyfile(raw_temp_path, str(final_out))

            return ClonedAudioSegment(
                id=0,
                start=0.0,
                end=actual_dur,
                audio_path=str(final_out),
                target_duration=target_dur,
                actual_duration=get_audio_duration(str(final_out)),
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
    Motor opcional de alta qualidade (IndexTTS-2 / 2.5), habilitado para GPUs
    com 8GB+ de VRAM (Perfil A). Utiliza inferência com separação de timbre
    e emoção da API oficial do IndexTTS2.
    """

    def __init__(
        self,
        model_profile: Optional[ModelProfile] = None,
        device: Optional[str] = None,
        models_dir: str = "models",
    ) -> None:
        super().__init__(model_profile=model_profile, device=device, models_dir=models_dir)
        self.model_dir = str(Path(models_dir).resolve() / "tts" / "indextts-2")
        self.cfg_path = str(Path(self.model_dir) / "config.yaml")

    def load_model(self) -> None:
        """Carrega o modelo IndexTTS-2 com inferência em FP16."""
        if self.model is not None:
            return

        logger.info("Carregando motor IndexTTS-2 (FP16) no dispositivo '%s'...", self.device)
        try:
            # Compatibilidade de importação para transformers 4.45+
            # IMPORTANTE: Sobrescreve SEMPRE os módulos, pois o PyInstaller empacota
            # uma versão do transformers que registra stubs incompatíveis em sys.modules.
            import sys, types
            try:
                from indextts.gpt.transformers_beam_constraints import DisjunctiveConstraint, PhrasalConstraint
                bc = types.ModuleType("transformers.generation.beam_constraints")
                bc.DisjunctiveConstraint = DisjunctiveConstraint
                bc.PhrasalConstraint = PhrasalConstraint
                sys.modules["transformers.generation.beam_constraints"] = bc
            except ImportError:
                pass

            try:
                from indextts.gpt.transformers_beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
                bs = types.ModuleType("transformers.generation.beam_search")
                bs.BeamScorer = BeamScorer
                bs.BeamSearchScorer = BeamSearchScorer
                bs.ConstrainedBeamSearchScorer = ConstrainedBeamSearchScorer
                sys.modules["transformers.generation.beam_search"] = bs
            except ImportError:
                pass

            try:
                import transformers
                if "modelscope" not in sys.modules:
                    ms = types.ModuleType("modelscope")
                    ms.AutoModelForCausalLM = transformers.AutoModelForCausalLM
                    ms.AutoTokenizer = transformers.AutoTokenizer
                    sys.modules["modelscope"] = ms
            except Exception:
                pass

            # CRITICAL: Desabilita TorchScript JIT em executáveis congelados (PyInstaller).
            # O TorchScript exige acesso ao código-fonte .py para compilação JIT,
            # mas o PyInstaller embute apenas bytecode .pyc. Com PYTORCH_JIT=0,
            # o decorador @torch.jit.script vira um no-op e o código roda em modo
            # eager (execução direta) sem perda de qualidade ou funcionalidade.
            if getattr(sys, "frozen", False):
                os.environ["PYTORCH_JIT"] = "0"
                logger.info("PYTORCH_JIT=0 definido (modo frozen — TorchScript desabilitado).")

            # CRITICAL: Compatibilidade entre huggingface_hub >= 1.0 e BigVGAN.
            # O hub 1.29.0 (empacotado pelo PyInstaller) não passa mais 'proxies'
            # e 'resume_download' ao chamar _from_pretrained, mas o BigVGAN antigo
            # os exige como keyword-only. Monkey-patch injeta defaults ausentes.
            try:
                from indextts.s2mel.modules.bigvgan import bigvgan as _bvg_mod
                _orig_fp = _bvg_mod.BigVGAN._from_pretrained.__func__

                @classmethod
                def _patched_from_pretrained(cls, **kwargs):
                    kwargs.setdefault("proxies", None)
                    kwargs.setdefault("resume_download", None)
                    return _orig_fp(cls, **kwargs)

                _bvg_mod.BigVGAN._from_pretrained = _patched_from_pretrained
                logger.info("Monkey-patch de compatibilidade BigVGAN aplicado.")
            except Exception as patch_err:
                logger.warning("Falha ao aplicar patch BigVGAN (pode ser OK): %s", patch_err)

            from indextts.infer_v2 import IndexTTS2

            # CRITICAL: Após o import, as referências locais de BeamSearchScorer em
            # transformers_generation_utils já resolveram para 'object' (do transformers
            # empacotado pelo PyInstaller). Precisamos sobrescrever diretamente no módulo.
            try:
                from indextts.gpt import transformers_generation_utils as _tgu
                from indextts.gpt.transformers_beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
                _tgu.BeamScorer = BeamScorer
                _tgu.BeamSearchScorer = BeamSearchScorer
                _tgu.ConstrainedBeamSearchScorer = ConstrainedBeamSearchScorer
                logger.info("Monkey-patch BeamSearchScorer aplicado em transformers_generation_utils.")
            except Exception as bsp_err:
                logger.warning("Falha ao aplicar patch BeamSearchScorer: %s", bsp_err)

            try:
                from indextts.gpt import transformers_generation_utils as _tgu2
                from indextts.gpt.transformers_beam_constraints import DisjunctiveConstraint, PhrasalConstraint
                _tgu2.DisjunctiveConstraint = DisjunctiveConstraint
                _tgu2.PhrasalConstraint = PhrasalConstraint
            except Exception:
                pass

            # CRITICAL: Monkey-patch no CFM.solve_euler para prevenir vazamento de VRAM.
            # O IndexTTS original acumulava tensores 3D intermediários em 'sol = []' durante
            # os 25 passos de difusão, causando pico de 8GB+ e CUDA OOM em textos longos.
            try:
                from indextts.s2mel.modules.flow_matching import CFM as _cfm_cls
                def _mem_safe_solve_euler(self_cfm, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
                    import torch
                    t = t_span[0]
                    prompt_len = prompt.size(-1)
                    prompt_x = torch.zeros_like(x)
                    prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
                    x[..., :prompt_len] = 0
                    if self_cfm.zero_prompt_speech_token:
                        mu[..., :prompt_len] = 0
                    if inference_cfg_rate > 0:
                        stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                        stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                        stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                    for step in range(1, len(t_span)):
                        dt = t_span[step] - t_span[step - 1]
                        if inference_cfg_rate > 0:
                            stacked_x = torch.cat([x, x], dim=0)
                            stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                            stacked_dphi_dt = self_cfm.estimator(
                                stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
                            )
                            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                            dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
                        else:
                            dphi_dt = self_cfm.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
                        x = x + dt * dphi_dt
                        t = t + dt
                        x[:, :, :prompt_len] = 0
                    return x
                _cfm_cls.solve_euler = _mem_safe_solve_euler
                logger.info("Monkey-patch CFM.solve_euler aplicado com sucesso para contenção de VRAM.")
            except Exception as cfm_err:
                logger.warning("Falha ao aplicar monkey-patch CFM.solve_euler: %s", cfm_err)

            dev = self.device if self.device and self.device != "auto" else "cuda"
            self.model = IndexTTS2(
                cfg_path=self.cfg_path,
                model_dir=self.model_dir,
                use_fp16=True,
                device=dev,
                use_qwen_emo=False,
            )
            logger.info("Modelo IndexTTS-2 carregado com sucesso em FP16.")
        except Exception as e:
            logger.error("Falha ao carregar IndexTTS-2: %s", e, exc_info=True)
            self.model = "indextts2_not_installed"

    def _check_engine_available(self) -> None:
        """Lança RuntimeError se o IndexTTS-2 não está instalado."""
        if self.model == "indextts2_not_installed" or self.model is None:
            raise RuntimeError(
                "O motor IndexTTS-2 não está operacional neste ambiente.\n"
                "Verifique se os pesos em models/tts/indextts-2 estão completos,\n"
                "ou utilize o motor padrão F5-TTS."
            )

    def _synthesize_single_sentence(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
    ) -> None:
        """
        Sintetiza uma única sentença como uma oração acústica atômica,
        sem quebras artificiais internas de vírgula ou hífen.
        """
        target_text = text.strip()
        if target_text and not target_text.endswith((".", "!", "?", "...", ":")):
            target_text += "."

        # max_text_tokens_per_segment=220: Garante que orações longas complexas
        # (ex: 60+ palavras com múltiplas vírgulas) não sejam fatiadas no meio,
        # eliminando pausas duplas em conjunções ("and,") e preservando o timbre
        # natural e fluido do modelo autorregressivo sem crepitação ou rouquidão.
        self.model.infer(
            spk_audio_prompt=reference_audio_path,
            text=target_text,
            output_path=output_path,
            interval_silence=0,
            num_beams=1,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            max_text_tokens_per_segment=220,
            verbose=False,
        )

    def _synthesize_multi_chunk(
        self,
        chunks: List[str],
        reference_audio_path: str,
        output_path: str,
        pause_seconds: float = DEFAULT_SENTENCE_PAUSE_SECONDS,
    ) -> None:
        """
        Sintetiza múltiplos blocos narrativos individualmente e concatena-os com
        silêncio de respiro calibrado (pause_seconds) entre cada bloco.
        """
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_wav_files: List[str] = []
        total_chunks = len(chunks)

        logger.info(
            "IndexTTS-2 síntese narrativa ativada: %d blocos (pausa entre blocos: %.2fs)",
            total_chunks, pause_seconds,
        )

        try:
            for i, chunk_text in enumerate(chunks, 1):
                chunk_wav = str(out_path.parent / f"{out_path.stem}_chunk_{i:03d}.wav")
                logger.info(
                    "  [IndexTTS-2] Bloco %d/%d (%d chars): '%s...'",
                    i, total_chunks, len(chunk_text), chunk_text[:60],
                )
                self._synthesize_single_sentence(
                    text=chunk_text,
                    reference_audio_path=reference_audio_path,
                    output_path=chunk_wav,
                )
                chunk_wav_files.append(chunk_wav)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # Concatena todos os arquivos WAV inserindo a pausa exata entre os blocos
            self._concat_wav_files(chunk_wav_files, str(out_path), pause_seconds=pause_seconds)
            logger.info(
                "IndexTTS-2 síntese narrativa concluída: %d blocos em %s (%.1fs)",
                total_chunks, out_path.name, get_audio_duration(str(out_path)),
            )
        finally:
            # Limpa arquivos intermediários dos blocos
            for cf in chunk_wav_files:
                if os.path.isfile(cf):
                    try:
                        os.remove(cf)
                    except Exception:
                        pass

    # Alias para retrocompatibilidade
    _synthesize_multi_sentence = _synthesize_multi_chunk

    def clone_and_synthesize(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        target_duration: Optional[float] = None,
        reference_text: Optional[str] = None,
        auto_unload: bool = False,
        speed: float = 1.0,
        sentence_pause_seconds: float = DEFAULT_SENTENCE_PAUSE_SECONDS,
        **kwargs: Any,
    ) -> ClonedAudioSegment:
        """
        Sintetiza a fala clonada usando a arquitetura de blocos narrativos do IndexTTS-2.
        Blocos narrativos mantêm sentenças curtas e orações contextualmente ricas,
        eliminando cold-start de frases minúsculas e preservando a cadência natural.
        """
        self.load_model()
        self._check_engine_available()

        final_out = Path(output_path).resolve()
        final_out.parent.mkdir(parents=True, exist_ok=True)

        pause_sec = kwargs.get("sentence_pause_seconds", sentence_pause_seconds)

        clean_text = sanitize_tts_text(text)
        if clean_text and not clean_text.endswith((".", "!", "?", "...", ":")):
            clean_text += "."

        chunks = split_text_into_narrative_chunks(clean_text)

        # Determina caminho temporário para a síntese bruta (sem estiramento de tempo)
        if final_out.name.endswith(".raw.wav"):
            raw_temp_path = str(final_out)
        else:
            raw_temp_path = str(final_out.with_suffix(".raw.wav"))

        try:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            if len(chunks) <= 1:
                self._synthesize_single_sentence(
                    text=clean_text,
                    reference_audio_path=reference_audio_path,
                    output_path=raw_temp_path,
                )
            else:
                self._synthesize_multi_chunk(
                    chunks=chunks,
                    reference_audio_path=reference_audio_path,
                    output_path=raw_temp_path,
                    pause_seconds=pause_sec,
                )

            # Cópia para o arquivo final se forem caminhos diferentes
            if Path(raw_temp_path).resolve() != final_out:
                shutil.copyfile(raw_temp_path, str(final_out))

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            actual_dur = get_audio_duration(str(final_out))
            target_dur = target_duration or actual_dur
            speed_factor = 1.0

            # Ajuste de duração via FFmpeg atempo para corresponder ao timing do SRT, se fornecido
            if target_duration and target_duration > 0 and actual_dur > 0:
                ratio = actual_dur / target_duration
                # Só aplica se a diferença for significativa (> 5%)
                if abs(ratio - 1.0) > 0.05:
                    stretched_out = final_out.with_suffix(".stretched.wav")
                    speed_factor = adjust_audio_duration_ffmpeg(
                        input_audio=raw_temp_path,
                        output_audio=str(stretched_out),
                        target_duration=target_duration,
                        min_speed=0.65,  # Permite desacelerar até 0.65x
                        max_speed=1.40,  # Permite acelerar até 1.40x
                    )
                    if stretched_out.is_file():
                        shutil.move(str(stretched_out), str(final_out))
                    actual_dur = get_audio_duration(str(final_out))
                    logger.info(
                        "IndexTTS-2 time-stretch: %.2fs → %.2fs (fator: %.2fx)",
                        actual_dur, target_duration, speed_factor,
                    )

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
            if auto_unload:
                self.unload_model()


def get_tts_engine(
    model_profile: Optional[ModelProfile] = None,
    engine_name: Optional[str] = None,
    use_advanced: bool = False,
    **kwargs: Any,
) -> BaseTTSEngine:
    """
    Factory que retorna a engine de síntese e clonagem de voz adequada,
    com suporte ao catálogo unificado de motores do KmellVox.

    Cadeia de resolução:
        1. Se engine_name for 'indextts-2' ou use_advanced=True → IndexTTS2Engine
        2. Se engine_name for 'f5-tts' ou padrão → F5TTSEngine
        3. Se o motor solicitado falhar → RuntimeError (SEM fallback silencioso)

    IMPORTANTE: Esta factory NUNCA faz fallback silencioso para outro motor.
    Se o motor solicitado falhar, lança RuntimeError com a razão detalhada.
    O tratamento de fallback/cancelamento fica a cargo da camada de UI.
    """
    ensure_addon_in_sys_path()
    profile = model_profile or ModelProfile.from_profile()

    selected = (engine_name or "").lower().strip()
    if not selected:
        if use_advanced and profile.enable_indextts_2:
            selected = "indextts-2"
        else:
            selected = "f5-tts"

    # --- Tenta IndexTTS-2 ---
    if selected in ("indextts-2", "indextts", "indextts2"):
        logger.info("Tentando inicializar motor IndexTTS2Engine...")
        try:
            engine = IndexTTS2Engine(model_profile=profile, **kwargs)
            engine.load_model()
            if engine.model and engine.model != "indextts2_not_installed" and not isinstance(engine.model, str):
                return engine
            raise RuntimeError(
                "O motor IndexTTS-2 não pôde ser inicializado.\n\n"
                "Os pesos ou dependências do IndexTTS-2 não puderam ser carregados.\n"
                "Verifique se todos os arquivos de modelo estão presentes em models/tts/indextts-2/ "
                "(incluindo hf_cache/ com modelos auxiliares).\n\n"
                "A operação foi cancelada. Selecione outro motor ou corrija a instalação."
            )
        except RuntimeError:
            raise
        except Exception as e:
            logger.error("IndexTTS-2 indisponível: %s", e, exc_info=True)
            raise RuntimeError(
                f"O motor IndexTTS-2 falhou ao carregar:\n\n"
                f"• {e}\n\n"
                f"A operação foi cancelada. Verifique os logs para detalhes."
            ) from e

    elif selected in ("f5-tts", "f5tts", "f5"):
        # --- Tenta F5-TTS ---
        logger.info("Selecionando motor de clonagem: F5TTSEngine (Perfil: %s).", profile.profile_name)
        engine = F5TTSEngine(model_profile=profile, **kwargs)
        engine.load_model()
        if engine.model != "f5_not_installed":
            return engine

        raise RuntimeError(
            "O motor F5-TTS não pôde ser inicializado.\n\n"
            "O pacote 'f5-tts' não está instalado ou os pesos neurais não foram encontrados.\n"
            "Consulte o Gerenciador de Modelos ou instale via requirements-gpu.txt.\n\n"
            "A operação foi cancelada."
        )

    else:
        # Motor não reconhecido ou não operacional
        raise RuntimeError(
            f"O motor '{selected}' não está disponível neste ambiente.\n\n"
            f"Este motor está em desenvolvimento para futuras versões e ainda não possui "
            f"inferência integrada no KmellVox.\n\n"
            f"A operação foi cancelada. Selecione um motor operacional (F5-TTS ou IndexTTS-2)."
        )


# Alias para retrocompatibilidade
VoiceCloner = F5TTSEngine
VoiceCloneConfig = RhythmControlConfig
