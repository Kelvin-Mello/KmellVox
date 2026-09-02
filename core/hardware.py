"""Módulo de detecção de hardware, VRAM e seleção automática de perfil de execução para o KmellVox.

Contém a ÚNICA fonte de verdade para resolução de perfis de hardware a partir de VRAM:
- VRAM >= 7.5 GB         -> perfil_a (float16, Whisper large-v3, Qwen3-8B, IndexTTS-2 habilitado)
- 5.0 GB <= VRAM < 7.5 GB -> perfil_b (int8_float16, Whisper distil-large-v3, Qwen3-4B, MuseTalk FP16)
- Sem GPU CUDA / < 5.0 GB -> cpu (int8, Whisper small, Qwen3-1.5B)
"""

from __future__ import annotations

import enum
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import yaml

logger = logging.getLogger("KmellVox.Hardware")


class HardwareProfile(str, enum.Enum):
    """Perfis de execução baseados na capacidade do hardware."""
    PERFIL_A = "perfil_a"      # VRAM >= 7.5 GB
    PERFIL_B = "perfil_b"      # 5.0 GB <= VRAM < 7.5 GB
    CPU = "cpu"                # Sem GPU CUDA ou VRAM < 5.0 GB


class WhisperModelVariant(str, enum.Enum):
    """Variantes válidas e homologadas do Faster-Whisper no KmellVox."""
    LARGE_V3 = "large-v3"               # Exclusivo de perfil_a
    DISTIL_LARGE_V3 = "distil-large-v3" # Exclusivo de perfil_b
    SMALL = "small"                     # Exclusivo de cpu (ou fallback)


VALID_WHISPER_VARIANTS: Set[str] = {v.value for v in WhisperModelVariant}


@dataclass
class ModelProfile:
    """
    Perfil de configuração de modelos e flags resolvido automaticamente conforme o hardware.
    
    Campos:
        profile_name: Nome do perfil ("perfil_a", "perfil_b", "cpu").
        whisper_variant: Variante do faster-whisper (large-v3, distil-large-v3, small).
        whisper_compute_type: Tipo de computação (float16, int8_float16, int8).
        translation_model: Nome/identificador do modelo LLM para tradução.
        translation_model_family: Família do modelo LLM ("Qwen3").
        translation_repo_id: Identificador do repositório Hugging Face ("Qwen/Qwen3-8B-Instruct-GGUF").
        translation_filename: Nome do arquivo GGUF ("Qwen3-8B-Instruct-Q4_K_M.gguf").
        default_tts_engine: Engine padrão de síntese de voz (F5-TTS).
        enable_indextts_2: Se a opção avançada de IndexTTS-2 deve ser habilitada na UI.
        musetalk_use_float16: Se a flag --use_float16 do MuseTalk é obrigatória/ativada.
    """
    profile_name: str = "cpu"
    whisper_variant: str = "small"
    whisper_compute_type: str = "int8"
    translation_model: str = "Qwen3-1.5B-Instruct-Q4_K_M"
    translation_model_family: str = "Qwen3"
    translation_repo_id: str = "Qwen/Qwen3-1.5B-Instruct-GGUF"
    translation_filename: str = "Qwen3-1.5B-Instruct-Q4_K_M.gguf"
    default_tts_engine: str = "F5-TTS"
    enable_indextts_2: bool = False
    musetalk_use_float16: bool = False

    def __post_init__(self) -> None:
        """Valida que whisper_variant pertence estritamente ao conjunto de variantes homologadas."""
        if self.whisper_variant not in VALID_WHISPER_VARIANTS:
            raise ValueError(
                f"Variante do Whisper inválida: '{self.whisper_variant}'. "
                f"A especificação do KmellVox aceita apenas: {sorted(list(VALID_WHISPER_VARIANTS))}"
            )

    @classmethod
    def from_profile(cls, profile_name: Optional[str] = None, config_path: str = "config.yaml") -> "ModelProfile":
        """
        Resolve as opções de modelos a partir de um perfil especificado ou do detectado.
        
        Args:
            profile_name: "perfil_a", "perfil_b", "cpu", ou None para detectar/ler de config.
            config_path: Caminho do arquivo config.yaml.
            
        Returns:
            ModelProfile: Instância com as configurações resolvidas.
        """
        if not profile_name:
            profile_name = detect_gpu_profile(config_path=config_path)

        norm = str(profile_name).lower().strip()

        if norm in ("perfil_a", "high_vram", "mid_vram"):
            return cls(
                profile_name="perfil_a",
                whisper_variant=WhisperModelVariant.LARGE_V3.value,
                whisper_compute_type="float16",
                translation_model="Qwen3-8B-Instruct-Q4_K_M",
                translation_model_family="Qwen3",
                translation_repo_id="Qwen/Qwen3-8B-Instruct-GGUF",
                translation_filename="Qwen3-8B-Instruct-Q4_K_M.gguf",
                default_tts_engine="F5-TTS",
                enable_indextts_2=True,
                musetalk_use_float16=False,  # Opcional em perfil_a
            )
        elif norm in ("perfil_b", "low_vram"):
            return cls(
                profile_name="perfil_b",
                whisper_variant=WhisperModelVariant.DISTIL_LARGE_V3.value,
                whisper_compute_type="int8_float16",
                translation_model="Qwen3-4B-Instruct-Q4_K_M",
                translation_model_family="Qwen3",
                translation_repo_id="Qwen/Qwen3-4B-Instruct-GGUF",
                translation_filename="Qwen3-4B-Instruct-Q4_K_M.gguf",
                default_tts_engine="F5-TTS",
                enable_indextts_2=False,     # Apenas perfil_a (8GB+)
                musetalk_use_float16=True,   # Obrigatório em perfil_b
            )
        else:  # Modo CPU
            return cls(
                profile_name="cpu",
                whisper_variant=WhisperModelVariant.SMALL.value,
                whisper_compute_type="int8",
                translation_model="Qwen3-1.5B-Instruct-Q4_K_M",
                translation_model_family="Qwen3",
                translation_repo_id="Qwen/Qwen3-1.5B-Instruct-GGUF",
                translation_filename="Qwen3-1.5B-Instruct-Q4_K_M.gguf",
                default_tts_engine="F5-TTS",
                enable_indextts_2=False,
                musetalk_use_float16=False,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Converte o perfil para um dicionário serializável."""
        return asdict(self)


def resolve_profile_from_vram(cuda_available: bool, vram_gb: float) -> str:
    """
    ÚNICA fonte de verdade para a lógica de decisão do perfil de hardware a partir da VRAM.
    
    Regras estritas do KmellVox:
        - VRAM >= 7.5 GB         -> "perfil_a"
        - 5.0 GB <= VRAM < 7.5 GB -> "perfil_b"
        - Sem GPU CUDA / VRAM < 5.0 GB -> "cpu"
    """
    if not cuda_available:
        return "cpu"

    if vram_gb >= 7.5:
        return "perfil_a"
    elif vram_gb >= 5.0:
        return "perfil_b"
    else:
        return "cpu"


def query_physical_gpu() -> Tuple[bool, str, float]:
    """
    Consulta diretamente os recursos da GPU física instalada na máquina.
    
    Returns:
        Tuple[bool, str, float]: (cuda_disponível, nome_do_dispositivo, vram_total_gb)
    """
    # 1. Tenta usar torch.cuda.get_device_properties(0).total_memory
    try:
        import torch
        if hasattr(torch, "cuda"):
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = total_bytes / (1024 ** 3)
                logger.debug("PyTorch CUDA detectado: %s (VRAM: %.2f GB)", device_name, vram_gb)
                return True, device_name, vram_gb
            else:
                # PyTorch está presente mas não possui suporte CUDA ativo
                return False, "CPU", 0.0
    except ImportError:
        logger.debug("PyTorch não encontrado no ambiente.")
    except Exception as e:
        logger.debug("Falha ao consultar torch.cuda: %s", e)

    # 2. Fallback via nvidia-smi se torch não estiver presente
    gpu_data = _query_nvidia_smi()
    if gpu_data:
        cuda_available = True
        device_name = gpu_data.get("name", "NVIDIA GPU")
        vram_gb = gpu_data.get("total_gb", 0.0)
        return cuda_available, device_name, vram_gb

    return False, "CPU", 0.0


def sync_hardware_config(
    profile: str,
    device_name: str = "CPU",
    cuda_available: bool = False,
    vram_gb: float = 0.0,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """
    ÚNICA função responsável por persistir e sincronizar todas as chaves de hardware e modelos no config.yaml.
    
    Garante que:
        - A chave raiz 'gpu_profile' seja exatamente igual a 'hardware.profile'.
        - 'hardware.compute_type' seja consistente com o ModelProfile correspondente.
        - 'models.translation' tenha model_family='Qwen3', repo_id e filename estritamente correspondentes ao perfil.
        - 'models.transcription.model_size' seja estritamente 'large-v3' (perfil_a), 'distil-large-v3' (perfil_b) ou 'small' (cpu).
        - 'hardware.vram_detected_gb' e 'hardware.device' reflitam o hardware real.
    """
    path = Path(config_path).resolve()
    data: Dict[str, Any] = {}

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.debug("Erro ao ler %s para sincronizar hardware: %s", path, e)

    norm_profile = profile.lower().strip()
    if norm_profile not in ("perfil_a", "perfil_b", "cpu"):
        norm_profile = "perfil_a" if norm_profile in ("high_vram", "mid_vram") else "perfil_b" if norm_profile == "low_vram" else "cpu"

    model_prof = ModelProfile.from_profile(norm_profile)

    # 1. Atualiza chave raiz
    data["gpu_profile"] = norm_profile

    # 2. Atualiza seção aninhada 'hardware'
    if "hardware" not in data or not isinstance(data["hardware"], dict):
        data["hardware"] = {}

    data["hardware"]["profile"] = norm_profile
    data["hardware"]["compute_type"] = model_prof.whisper_compute_type
    data["hardware"]["device"] = "cuda" if cuda_available and norm_profile != "cpu" else "cpu"
    data["hardware"]["device_name"] = device_name
    data["hardware"]["vram_detected_gb"] = round(vram_gb, 2)

    # 3. Atualiza seção 'models' (translation e transcription)
    if "models" not in data or not isinstance(data["models"], dict):
        data["models"] = {}

    if "translation" not in data["models"] or not isinstance(data["models"]["translation"], dict):
        data["models"]["translation"] = {}
    data["models"]["translation"]["model_family"] = model_prof.translation_model_family
    data["models"]["translation"]["repo_id"] = model_prof.translation_repo_id
    data["models"]["translation"]["filename"] = model_prof.translation_filename

    if "transcription" not in data["models"] or not isinstance(data["models"]["transcription"], dict):
        data["models"]["transcription"] = {}
    data["models"]["transcription"]["model_size"] = model_prof.whisper_variant

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        logger.info(
            "Configuração sincronizada em '%s': gpu_profile='%s', hardware.profile='%s', compute_type='%s', whisper='%s', translation='%s', vram=%.2f GB",
            path.name, norm_profile, norm_profile, model_prof.whisper_compute_type, model_prof.whisper_variant, model_prof.translation_filename, vram_gb
        )
    except Exception as e:
        logger.warning("Não foi possível salvar configurações em '%s': %s", path, e)

    return data


def _load_gpu_profile_from_config(config_path: str = "config.yaml") -> Optional[str]:
    """Tenta carregar e validar o perfil de GPU previamente salvo no config.yaml."""
    path = Path(config_path).resolve()
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                root_profile = data.get("gpu_profile")
                nested_profile = data.get("hardware", {}).get("profile")

                # Se ambas as chaves existirem e forem consistentes
                if root_profile in ("perfil_a", "perfil_b", "cpu"):
                    if nested_profile is None or nested_profile == root_profile:
                        return root_profile
        except Exception as e:
            logger.debug("Erro ao ler gpu_profile de %s: %s", path, e)
    return None


def detect_gpu_profile(config_path: str = "config.yaml", force_redetect: bool = False) -> str:
    """
    Detecta o perfil de hardware baseado na VRAM da GPU via PyTorch CUDA.
    
    Critérios estritos:
        - VRAM >= 7.5 GB: "perfil_a"
        - 5.0 GB <= VRAM < 7.5 GB: "perfil_b"
        - Sem GPU CUDA / VRAM < 5.0 GB: "cpu" (com aviso claro de lentidão)
        
    Salva e sincroniza o resultado em config.yaml (raiz e chaves aninhadas).
    """
    # 1. Consulta o hardware físico real
    cuda_available, device_name, vram_gb = query_physical_gpu()
    physical_profile = resolve_profile_from_vram(cuda_available, vram_gb)

    if not force_redetect:
        saved_profile = _load_gpu_profile_from_config(config_path)
        if saved_profile is not None:
            if saved_profile == physical_profile:
                logger.debug("Perfil de GPU verificado e compatível com o config: %s", saved_profile)
                return saved_profile
            else:
                logger.warning(
                    "Divergência detectada entre o hardware atual (%s, %.2f GB -> %s) e o perfil salvo (%s). Ressincronizando config.yaml...",
                    device_name, vram_gb, physical_profile, saved_profile
                )

    # 2. Avaliação e logs claros ao usuário
    if not cuda_available:
        logger.warning(
            "⚠️ AVISO: Nenhuma GPU CUDA compatível foi detectada! "
            "O KmellVox será executado no modo CPU. O processamento de áudio, tradução "
            "e lip sync será MUITO mais lento que o habitual."
        )
    elif physical_profile == "perfil_a":
        logger.info("VRAM Total (%.2f GB) >= 7.5 GB -> Selecionado: perfil_a", vram_gb)
    elif physical_profile == "perfil_b":
        logger.info("VRAM Total (%.2f GB) entre 5.0 GB e 7.5 GB -> Selecionado: perfil_b", vram_gb)
    else:
        logger.warning(
            "⚠️ AVISO: A GPU detectada possui apenas %.2f GB de VRAM (mínimo recomendado: 5.0 GB). "
            "Executando no modo CPU para evitar erros de falta de memória (Out-Of-Memory). "
            "O processamento será consideravelmente mais lento.", vram_gb
        )

    # 3. Salva e sincroniza todas as chaves no config.yaml
    sync_hardware_config(
        profile=physical_profile,
        device_name=device_name,
        cuda_available=cuda_available,
        vram_gb=vram_gb,
        config_path=config_path,
    )

    return physical_profile


@dataclass
class HardwareInfo:
    """Informações detalhadas do ambiente de hardware detectado."""
    device_name: str = "CPU"
    cuda_available: bool = False
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cpu_cores: int = os.cpu_count() or 4
    system_ram_gb: float = 0.0
    profile: HardwareProfile = HardwareProfile.CPU
    gpu_profile: str = "cpu"
    model_profile: ModelProfile = field(default_factory=ModelProfile)
    recommended_compute_type: str = "int8"
    recommended_whisper_model: str = "small"
    recommended_llm_quant: str = "q4_k_m"
    extra_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "cuda_available": self.cuda_available,
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_free_gb": round(self.vram_free_gb, 2),
            "cpu_cores": self.cpu_cores,
            "system_ram_gb": round(self.system_ram_gb, 2),
            "profile": self.profile.value,
            "gpu_profile": self.gpu_profile,
            "model_profile": self.model_profile.to_dict(),
            "recommended_compute_type": self.recommended_compute_type,
            "recommended_whisper_model": self.recommended_whisper_model,
            "recommended_llm_quant": self.recommended_llm_quant,
            "extra_details": self.extra_details,
        }


def _get_ram_info() -> float:
    """Obtém a quantidade total de RAM do sistema em GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 8.0


def _query_nvidia_smi() -> Optional[Dict[str, Any]]:
    """Tenta consultar nvidia-smi para obter VRAM e modelo da GPU como fallback."""
    if not shutil.which("nvidia-smi"):
        return None

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,nounits,noheader"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if not lines:
            return None

        parts = [p.strip() for p in lines[0].split(",")]
        name = parts[0]
        total_mb = float(parts[1]) if len(parts) > 1 else 0.0
        free_mb = float(parts[2]) if len(parts) > 2 else 0.0
        driver = parts[3] if len(parts) > 3 else "Unknown"

        return {
            "name": name,
            "total_gb": total_mb / 1024.0,
            "free_gb": free_mb / 1024.0,
            "driver": driver,
        }
    except Exception as e:
        logger.debug("nvidia-smi query falhou: %s", e)
        return None


def detect_hardware(force_profile: Optional[str] = None, config_path: str = "config.yaml") -> HardwareInfo:
    """
    Detecta os recursos do sistema, valida e seleciona o perfil de hardware e de modelos.
    """
    info = HardwareInfo()
    info.system_ram_gb = _get_ram_info()

    cuda_avail, dev_name, vram_total = query_physical_gpu()
    info.cuda_available = cuda_avail
    info.device_name = dev_name
    info.vram_total_gb = vram_total
    info.vram_free_gb = vram_total

    if force_profile and force_profile.lower() in ("perfil_a", "perfil_b", "cpu"):
        gpu_prof = force_profile.lower()
        sync_hardware_config(
            profile=gpu_prof,
            device_name=dev_name,
            cuda_available=cuda_avail,
            vram_gb=vram_total,
            config_path=config_path,
        )
    else:
        gpu_prof = detect_gpu_profile(config_path=config_path, force_redetect=False)

    info.gpu_profile = gpu_prof
    info.model_profile = ModelProfile.from_profile(gpu_prof, config_path=config_path)

    # Mapeia HardwareProfile enum
    if gpu_prof == "perfil_a":
        info.profile = HardwareProfile.PERFIL_A
    elif gpu_prof == "perfil_b":
        info.profile = HardwareProfile.PERFIL_B
    else:
        info.profile = HardwareProfile.CPU

    info.recommended_compute_type = info.model_profile.whisper_compute_type
    info.recommended_whisper_model = info.model_profile.whisper_variant
    info.recommended_llm_quant = "q4_k_m"

    return info
