"""Catálogo centralizado de motores de síntese de voz (TTS) para o KmellVox.

Gerencia os metadados de cada motor, verificação de integridade dos pesos no disco,
classificação de compatibilidade por VRAM (Recomendado, Pouco Recomendado, Não Recomendado)
e operações de inventário, download e desinstalação de modelos.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("KmellVox.TTSCatalog")


@dataclass(frozen=True)
class TTSEngineMeta:
    """Metadados de um motor TTS suportado no catálogo."""
    id: str
    name: str
    description: str
    min_vram_gb: float
    recommended_vram_gb: float
    disk_size_gb: float
    relative_model_dir: str
    hf_repo_id: Optional[str]
    is_experimental: bool = False
    supports_zero_shot: bool = True
    supports_speaker_caching: bool = False


# Catálogo oficial curado do KmellVox (Motores de 2026 para Narração e Dublagem)
TTS_CATALOG: Dict[str, TTSEngineMeta] = {
    "f5-tts": TTSEngineMeta(
        id="f5-tts",
        name="F5-TTS v1 Base",
        description="Padrão oficial de alto desempenho. Rápido, leve e excelente para narrações em inglês e chinês.",
        min_vram_gb=4.0,
        recommended_vram_gb=6.0,
        disk_size_gb=1.2,
        relative_model_dir="tts/f5-tts/F5TTS_v1_Base",
        hf_repo_id="SWivid/F5-TTS",
        is_experimental=False,
        supports_zero_shot=True,
        supports_speaker_caching=False,
    ),
    "indextts-2": TTSEngineMeta(
        id="indextts-2",
        name="IndexTTS-2.5 (Qualidade Máxima)",
        description="Expressividade superior, prosódia rica e controle de emoção/timbre desacoplados. Mais pesado.",
        min_vram_gb=8.0,
        recommended_vram_gb=10.0,
        disk_size_gb=5.5,
        relative_model_dir="tts/indextts-2",
        hf_repo_id="IndexTeam/IndexTTS-2",
        is_experimental=True,
        supports_zero_shot=True,
        supports_speaker_caching=False,
    ),
    "qwen3-tts-0.6b": TTSEngineMeta(
        id="qwen3-tts-0.6b",
        name="Qwen3-TTS 0.6B Base",
        description="Nova geração da Alibaba. Clonagem rápida a partir de 3s e cache de prompt de locutor.",
        min_vram_gb=4.0,
        recommended_vram_gb=6.0,
        disk_size_gb=1.5,
        relative_model_dir="tts/qwen3-tts-0.6b",
        hf_repo_id="Qwen/Qwen3-TTS-0.6B",
        is_experimental=False,
        supports_zero_shot=True,
        supports_speaker_caching=True,
    ),
    "qwen3-tts-1.7b": TTSEngineMeta(
        id="qwen3-tts-1.7b",
        name="Qwen3-TTS 1.7B Base",
        description="Versão topo de linha do Qwen3. Máxima fidelidade harmônica para GPUs potentes.",
        min_vram_gb=8.0,
        recommended_vram_gb=12.0,
        disk_size_gb=4.5,
        relative_model_dir="tts/qwen3-tts-1.7b",
        hf_repo_id="Qwen/Qwen3-TTS-1.7B",
        is_experimental=True,
        supports_zero_shot=True,
        supports_speaker_caching=True,
    ),
    "chatterbox-turbo": TTSEngineMeta(
        id="chatterbox-turbo",
        name="Chatterbox Turbo (350M)",
        description="Especializado em narrações em inglês. Ultraleve, com pausas respiratórias dinâmicas.",
        min_vram_gb=4.0,
        recommended_vram_gb=6.0,
        disk_size_gb=0.8,
        relative_model_dir="tts/chatterbox-turbo",
        hf_repo_id="chatterbox-ai/chatterbox-turbo",
        is_experimental=False,
        supports_zero_shot=True,
        supports_speaker_caching=False,
    ),
}


def list_tts_catalog() -> List[TTSEngineMeta]:
    """Retorna todos os motores presentes no catálogo oficial."""
    return list(TTS_CATALOG.values())


def get_engine_meta(engine_id: str) -> Optional[TTSEngineMeta]:
    """Obtém os metadados de um motor específico pelo ID."""
    return TTS_CATALOG.get(engine_id.lower().strip())


def get_hardware_compatibility(engine_id: str, vram_gb: float) -> Tuple[str, str]:
    """
    Avalia a compatibilidade de um motor com a VRAM detectada.
    
    Returns:
        Tuple[str, str]: (badge_curto, explicacao)
        Ex: ("🟢 Recomendado", "Sua GPU de 7.96GB atende com folga")
    """
    meta = get_engine_meta(engine_id)
    if not meta:
        return "⚪ Desconhecido", "Motor não catalogado."

    # Margem de 0.2GB para compensar VRAM reservada pelo Windows (~40 a 200MB)
    eff_vram = vram_gb + 0.2

    if eff_vram >= meta.recommended_vram_gb:
        return "🟢 Recomendado", f"Excelente desempenho na sua GPU ({vram_gb:.1f}GB VRAM >= {meta.recommended_vram_gb:.0f}GB)."
    elif eff_vram >= meta.min_vram_gb:
        return "🟡 Pouco Recomendado", f"Executável na sua GPU ({vram_gb:.1f}GB VRAM), mas com margem de memória reduzida."
    else:
        return "🔴 Não Recomendado", f"Exige pelo menos {meta.min_vram_gb:.0f}GB de VRAM (detectado: {vram_gb:.1f}GB)."


def is_engine_installed(engine_id: str, models_dir: str = "models") -> bool:
    """Verifica se os arquivos de pesos essenciais do motor existem no disco."""
    meta = get_engine_meta(engine_id)
    if not meta:
        return False

    base_dir = Path(models_dir).resolve() / meta.relative_model_dir
    if not base_dir.is_dir():
        return False

    if engine_id == "f5-tts":
        safetensors = list(base_dir.glob("*.safetensors"))
        return len(safetensors) > 0
    elif engine_id == "indextts-2":
        gpt_pth = base_dir / "gpt.pth"
        s2mel_pth = base_dir / "s2mel.pth"
        # Verifica também os modelos auxiliares em hf_cache/
        hf_cache = base_dir / "hf_cache"
        has_aux = (
            hf_cache.is_dir()
            and (hf_cache / "semantic_codec_model.safetensors").is_file()
            and (hf_cache / "campplus_cn_common.bin").is_file()
            and (hf_cache / "bigvgan").is_dir()
        )
        return gpt_pth.is_file() and s2mel_pth.is_file() and has_aux
    elif "qwen3-tts" in engine_id:
        has_weights = len(list(base_dir.glob("*.safetensors"))) > 0 or len(list(base_dir.glob("*.bin"))) > 0
        return has_weights
    elif engine_id == "chatterbox-turbo":
        has_weights = len(list(base_dir.glob("*.pt"))) > 0 or len(list(base_dir.glob("*.safetensors"))) > 0
        return has_weights

    return any(base_dir.iterdir())


def is_engine_operational(engine_id: str, models_dir: str = "models") -> Tuple[bool, str]:
    """
    Verifica se o motor TTS está totalmente operacional e pronto para síntese.
    Retorna: (is_operational: bool, explanation: str)
    """
    clean_id = (engine_id or "").lower().strip()
    if clean_id in ("f5-tts", "f5tts", "f5"):
        if is_engine_installed("f5-tts", models_dir):
            return True, "Operacional — Padrão oficial de alto desempenho e fidelidade."
        return False, "Pesos neurais do F5-TTS não encontrados no disco."

    if clean_id in ("indextts-2", "indextts", "indextts2"):
        if not is_engine_installed("indextts-2", models_dir):
            return False, "Pesos neurais do IndexTTS-2 não encontrados no disco (necessário download)."
        return True, "Operacional — Expressividade avançada e controle nativo de duração em FP16."

    if "qwen3-tts" in clean_id:
        return False, "Motor Qwen3-TTS em fase de desenvolvimento para atualizações futuras."

    if "chatterbox" in clean_id:
        return False, "Motor Chatterbox Turbo planejado para integração em versões futuras."

    return False, f"Motor '{clean_id}' não suportado neste ambiente."


def get_engine_status_summary(vram_gb: float, models_dir: str = "models") -> List[Dict[str, Any]]:
    """Gera o inventário consolidado de motores para exibição na UI."""
    results = []
    for meta in list_tts_catalog():
        installed = is_engine_installed(meta.id, models_dir)
        badge, explanation = get_hardware_compatibility(meta.id, vram_gb)
        results.append({
            "id": meta.id,
            "name": meta.name,
            "description": meta.description,
            "installed": installed,
            "badge": badge,
            "explanation": explanation,
            "size_gb": meta.disk_size_gb,
            "min_vram_gb": meta.min_vram_gb,
            "recommended_vram_gb": meta.recommended_vram_gb,
            "is_experimental": meta.is_experimental,
        })
    return results


def uninstall_engine_model(engine_id: str, models_dir: str = "models") -> bool:
    """Remove os arquivos de modelo de um motor específico para liberar espaço em disco."""
    meta = get_engine_meta(engine_id)
    if not meta:
        return False

    target_dir = Path(models_dir).resolve() / meta.relative_model_dir
    if target_dir.is_dir():
        try:
            logger.info("Removendo arquivos do modelo '%s' em %s...", engine_id, target_dir)
            shutil.rmtree(target_dir)
            return True
        except Exception as e:
            logger.error("Falha ao desinstalar modelo '%s': %s", engine_id, e)
            return False
    return False
