"""Módulo de detecção de hardware, VRAM e seleção automática de perfil de execução para o KmellVox."""

from __future__ import annotations

import enum
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger("KmellVox.Hardware")


class HardwareProfile(str, enum.Enum):
    """Perfis de execução baseados na capacidade do hardware."""
    PERFIL_A = "perfil_a"      # VRAM >= 7.5 GB
    PERFIL_B = "perfil_b"      # 5.0 GB <= VRAM < 7.5 GB
    CPU = "cpu"                # Sem GPU CUDA ou VRAM < 5.0 GB
    
    # Aliases legados para compatibilidade
    LOW_VRAM = "perfil_b"
    MID_VRAM = "perfil_a"
    HIGH_VRAM = "perfil_a"


@dataclass
class ModelProfile:
    """
    Perfil de configuração de modelos e flags resolvido automaticamente conforme o hardware.
    
    Campos:
        profile_name: Nome do perfil ("perfil_a", "perfil_b", "cpu").
        whisper_variant: Variante do faster-whisper (large-v3, distil-large-v3, etc.).
        whisper_compute_type: Tipo de computação (float16, int8_float16, int8).
        translation_model: Nome/identificador do modelo LLM para tradução.
        default_tts_engine: Engine padrão de síntese de voz (F5-TTS).
        enable_indextts_2: Se a opção avançada de IndexTTS-2 deve ser habilitada na UI.
        musetalk_use_float16: Se a flag --use_float16 do MuseTalk é obrigatória/ativada.
    """
    profile_name: str = "cpu"
    whisper_variant: str = "small"
    whisper_compute_type: str = "int8"
    translation_model: str = "Qwen3-1.5B-Instruct Q4_K_M"
    default_tts_engine: str = "F5-TTS"
    enable_indextts_2: bool = False
    musetalk_use_float16: bool = False

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
                whisper_variant="large-v3",
                whisper_compute_type="float16",
                translation_model="Qwen3-8B-Instruct Q4_K_M",
                default_tts_engine="F5-TTS",
                enable_indextts_2=True,
                musetalk_use_float16=False,  # Opcional em perfil_a
            )
        elif norm in ("perfil_b", "low_vram"):
            return cls(
                profile_name="perfil_b",
                whisper_variant="distil-large-v3",
                whisper_compute_type="int8_float16",
                translation_model="Qwen3-4B-Instruct Q4_K_M",
                default_tts_engine="F5-TTS",
                enable_indextts_2=False,     # Apenas perfil_a
                musetalk_use_float16=True,   # Obrigatório em perfil_b
            )
        else:  # CPU mode
            return cls(
                profile_name="cpu",
                whisper_variant="small",
                whisper_compute_type="int8",
                translation_model="Qwen3-1.5B-Instruct Q4_K_M",
                default_tts_engine="F5-TTS",
                enable_indextts_2=False,
                musetalk_use_float16=False,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Converte o perfil para um dicionário serializável."""
        return asdict(self)


def _save_gpu_profile_to_config(profile: str, config_path: str = "config.yaml") -> None:
    """Salva a chave gpu_profile no arquivo config.yaml."""
    path = Path(config_path)
    data: Dict[str, Any] = {}
    
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.debug("Erro ao ler %s para salvar gpu_profile: %s", path, e)

    data["gpu_profile"] = profile

    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        logger.info("Perfil de GPU '%s' persistido com sucesso em '%s'.", profile, path)
    except Exception as e:
        logger.warning("Não foi possível salvar gpu_profile em '%s': %s", path, e)


def _load_gpu_profile_from_config(config_path: str = "config.yaml") -> Optional[str]:
    """Tenta carregar o perfil de GPU previamente salvo no config.yaml."""
    path = Path(config_path)
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                profile = data.get("gpu_profile")
                if profile in ("perfil_a", "perfil_b", "cpu"):
                    return profile
        except Exception as e:
            logger.debug("Erro ao ler gpu_profile de %s: %s", path, e)
    return None


def detect_gpu_profile(config_path: str = "config.yaml", force_redetect: bool = False) -> str:
    """
    Detecta o perfil de hardware baseado na VRAM da GPU via PyTorch CUDA.
    
    Critérios:
        - VRAM >= 7.5 GB: "perfil_a"
        - 5.0 GB <= VRAM < 7.5 GB: "perfil_b"
        - Sem GPU CUDA / VRAM < 5.0 GB: "cpu" (com aviso claro de lentidão)
        
    Salva o resultado na chave 'gpu_profile' de config.yaml.
    
    Args:
        config_path: Caminho do arquivo de configuração config.yaml.
        force_redetect: Se True, ignora o valor salvo em config.yaml e redetecta.
        
    Returns:
        str: "perfil_a", "perfil_b" ou "cpu".
    """
    if not force_redetect:
        saved_profile = _load_gpu_profile_from_config(config_path)
        if saved_profile is not None:
            logger.debug("Utilizando perfil de GPU salvo no config: %s", saved_profile)
            return saved_profile

    # 1. Tenta usar torch.cuda.get_device_properties(0).total_memory
    cuda_available = False
    vram_gb = 0.0
    device_name = "CPU"

    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            device_name = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = total_bytes / (1024 ** 3)
            logger.info("PyTorch CUDA detectado: %s (VRAM Total: %.2f GB)", device_name, vram_gb)
    except ImportError:
        logger.debug("PyTorch não está instalado no ambiente.")
    except Exception as e:
        logger.warning("Falha ao consultar torch.cuda: %s", e)

    # 2. Avaliação do perfil
    if not cuda_available:
        logger.warning(
            "⚠️ AVISO: Nenhuma GPU CUDA compatível foi detectada! "
            "O KmellVox será executado no modo CPU. O processamento de áudio, tradução "
            "e lip sync será MUITO mais lento que o habitual."
        )
        profile = "cpu"
    elif vram_gb >= 7.5:
        logger.info("VRAM Total (%.2f GB) >= 7.5 GB -> Selecionado: perfil_a", vram_gb)
        profile = "perfil_a"
    elif vram_gb >= 5.0:
        logger.info("VRAM Total (%.2f GB) entre 5.0 GB e 7.5 GB -> Selecionado: perfil_b", vram_gb)
        profile = "perfil_b"
    else:
        logger.warning(
            "⚠️ AVISO: A GPU detectada possui apenas %.2f GB de VRAM (mínimo recomendado: 5.0 GB). "
            "Executando no modo CPU para evitar erros de falta de memória (Out-Of-Memory). "
            "O processamento será consideravelmente mais lento.", vram_gb
        )
        profile = "cpu"

    # 3. Salva em config.yaml
    _save_gpu_profile_to_config(profile, config_path)

    return profile


@dataclass
class HardwareInfo:
    """Informações detalhadas do ambiente de hardware detectado (compatibilidade retroativa)."""
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
    Detecta os recursos do sistema e seleciona o perfil de hardware e de modelos.
    """
    info = HardwareInfo()
    info.system_ram_gb = _get_ram_info()

    gpu_prof = detect_gpu_profile(config_path=config_path, force_redetect=(force_profile is not None))
    if force_profile and force_profile.lower() in ("perfil_a", "perfil_b", "cpu"):
        gpu_prof = force_profile.lower()

    info.gpu_profile = gpu_prof
    info.model_profile = ModelProfile.from_profile(gpu_prof, config_path=config_path)

    # Tenta obter dados detalhados para exibição
    gpu_data = _query_nvidia_smi()
    try:
        import torch
        if torch.cuda.is_available():
            info.cuda_available = True
            info.device_name = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            info.vram_total_gb = total_bytes / (1024 ** 3)
            info.vram_free_gb = info.vram_total_gb
    except Exception:
        pass

    if not info.cuda_available and gpu_data:
        info.cuda_available = True
        info.device_name = gpu_data.get("name", "NVIDIA GPU")
        info.vram_total_gb = gpu_data.get("total_gb", 0.0)
        info.vram_free_gb = gpu_data.get("free_gb", info.vram_total_gb)
        info.extra_details = gpu_data

    # Mapeia HardwareProfile
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
