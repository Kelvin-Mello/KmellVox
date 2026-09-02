"""Script e utilitário para download, verificação e atualização de pesos dos modelos conforme o perfil de hardware."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.safe_streams import SafeStream, ensure_safe_streams

# Garante streams válidos antes de qualquer import de terceiros
ensure_safe_streams()

import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

from core.hardware import ModelProfile, detect_gpu_profile, sync_hardware_config

logger = logging.getLogger("KmellVox.Downloader")


class SafeTqdm(tqdm):
    """Subclasse do tqdm que garante que 'file' nunca seja None em ambientes GUI ou PyInstaller."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "file" not in kwargs or kwargs["file"] is None:
            kwargs["file"] = sys.stderr if sys.stderr and hasattr(sys.stderr, "write") else SafeStream()
        super().__init__(*args, **kwargs)


def create_hf_progress_tqdm(
    spec_idx: int,
    total_specs: int,
    spec_name: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """
    Cria uma classe tqdm personalizada para o huggingface_hub que encaminha o progresso
    granular de bytes baixados para a interface gráfica em tempo real.
    """
    class HFDownloadProgressTqdm(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if "file" not in kwargs or kwargs["file"] is None:
                kwargs["file"] = sys.stderr if sys.stderr and hasattr(sys.stderr, "write") else SafeStream()
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> None:
            super().update(n)
            if progress_callback:
                if self.total and self.total > 0:
                    file_pct = max(0.0, min(1.0, self.n / self.total))
                    overall_pct = ((spec_idx - 1) + file_pct) / total_specs
                    cur_mb = self.n / (1024 * 1024)
                    tot_mb = self.total / (1024 * 1024)
                    msg = f"[{spec_idx}/{total_specs}] {spec_name}: {cur_mb:.1f} MB / {tot_mb:.1f} MB ({file_pct * 100:.1f}%)"
                    progress_callback(overall_pct, msg)
                else:
                    cur_mb = self.n / (1024 * 1024)
                    overall_pct = (spec_idx - 0.5) / total_specs
                    msg = f"[{spec_idx}/{total_specs}] {spec_name}: {cur_mb:.1f} MB transferidos..."
                    progress_callback(overall_pct, msg)

    return HFDownloadProgressTqdm


@dataclass
class ModelDownloadSpec:
    """Especificação de download de um modelo da HuggingFace."""
    key: str                    # whisper, llm, f5_tts, index_tts, musetalk
    name: str
    category: str               # transcription, translation, voice_clone, lipsync
    repo_id: str
    filename: Optional[str] = None
    destination_rel: str = ""
    profiles: List[str] = field(default_factory=list) # ["perfil_a", "perfil_b", "cpu"]
    description: str = ""
    expected_min_bytes: int = 1024


# ==============================================================================
# Catálogo Canônico de Modelos do KmellVox por Perfil de Hardware
# ==============================================================================
MODEL_CATALOG: List[ModelDownloadSpec] = [
    # 1. Faster-Whisper (Transcrição)
    ModelDownloadSpec(
        key="whisper_large_v3",
        name="Faster-Whisper large-v3",
        category="transcription",
        repo_id="Systran/faster-whisper-large-v3",
        destination_rel="whisper/large-v3",
        profiles=["perfil_a"],
        description="Whisper large-v3 FP16 (Precisão máxima para perfil_a 8GB+)",
        expected_min_bytes=1500 * 1024 * 1024,
    ),
    ModelDownloadSpec(
        key="whisper_distil_large_v3",
        name="Faster-Whisper distil-large-v3",
        category="transcription",
        repo_id="Systran/faster-distil-whisper-large-v3",
        destination_rel="whisper/distil-large-v3",
        profiles=["perfil_b"],
        description="Whisper distil-large-v3 (Otimizado para perfil_b 6GB VRAM)",
        expected_min_bytes=750 * 1024 * 1024,
    ),
    ModelDownloadSpec(
        key="whisper_small",
        name="Faster-Whisper small",
        category="transcription",
        repo_id="Systran/faster-whisper-small",
        destination_rel="whisper/small",
        profiles=["cpu"],
        description="Whisper small INT8 (Leve para execução em CPU)",
        expected_min_bytes=240 * 1024 * 1024,
    ),

    # 2. LLM Qwen3 GGUF (Tradução)
    ModelDownloadSpec(
        key="llm_qwen3_8b",
        name="Qwen3-8B-Instruct GGUF Q4_K_M",
        category="translation",
        repo_id="Qwen/Qwen3-8B-Instruct-GGUF",
        filename="Qwen3-8B-Instruct-Q4_K_M.gguf",
        destination_rel="llm/Qwen3-8B-Instruct-Q4_K_M.gguf",
        profiles=["perfil_a"],
        description="Qwen3 8B Q4_K_M (Tradução contextual de estúdio para perfil_a)",
        expected_min_bytes=4000 * 1024 * 1024,
    ),
    ModelDownloadSpec(
        key="llm_qwen3_4b",
        name="Qwen3-4B-Instruct GGUF Q4_K_M",
        category="translation",
        repo_id="Qwen/Qwen3-4B-Instruct-GGUF",
        filename="Qwen3-4B-Instruct-Q4_K_M.gguf",
        destination_rel="llm/Qwen3-4B-Instruct-Q4_K_M.gguf",
        profiles=["perfil_b"],
        description="Qwen3 4B Q4_K_M (Tradução balanceada para perfil_b 6GB)",
        expected_min_bytes=2000 * 1024 * 1024,
    ),
    ModelDownloadSpec(
        key="llm_qwen3_1_5b",
        name="Qwen3-1.5B-Instruct GGUF Q4_K_M",
        category="translation",
        repo_id="Qwen/Qwen3-1.5B-Instruct-GGUF",
        filename="Qwen3-1.5B-Instruct-Q4_K_M.gguf",
        destination_rel="llm/Qwen3-1.5B-Instruct-Q4_K_M.gguf",
        profiles=["cpu"],
        description="Qwen3 1.5B Q4_K_M (Tradução leve para CPU)",
        expected_min_bytes=800 * 1024 * 1024,
    ),

    # 3. F5-TTS (Clonagem de Voz Padrão) - Ambos os perfis e CPU
    ModelDownloadSpec(
        key="f5_tts_base",
        name="F5-TTS (Modelo Base Multilíngue)",
        category="voice_clone",
        repo_id="SWivid/F5-TTS",
        destination_rel="tts/f5-tts",
        profiles=["perfil_a", "perfil_b", "cpu"],
        description="Motor F5-TTS padrão de clonagem com controle de ritmo via atempo",
        expected_min_bytes=500 * 1024 * 1024,
    ),

    # 4. IndexTTS-2 (Clonagem Avançada FP16) - Somente perfil_a (8GB+ VRAM)
    ModelDownloadSpec(
        key="indextts_2_fp16",
        name="IndexTTS-2 (Qualidade Máxima de Voz)",
        category="voice_clone_indextts2",
        repo_id="IndexTeam/IndexTTS-2",
        destination_rel="tts/indextts-2",
        profiles=["perfil_a"],
        description="Motor avançado IndexTTS-2 com controle explícito nativo de duração (8GB+)",
        expected_min_bytes=1200 * 1024 * 1024,
    ),

    # 5. MuseTalk 1.5 (Sincronia Labial Facial) - Ambos os perfis
    ModelDownloadSpec(
        key="musetalk_1_5_core",
        name="MuseTalk 1.5 (Checkpoints Principais)",
        category="lipsync",
        repo_id="TMElyralab/MuseTalk",
        destination_rel="musetalk",
        profiles=["perfil_a", "perfil_b"],
        description="Checkpoints de sincronia labial MuseTalk 1.5 (FP16/FP32)",
        expected_min_bytes=600 * 1024 * 1024,
    ),
]


def verify_file_or_dir_exists(target_path: Path, min_bytes: int = 1024) -> bool:
    """Verifica se o arquivo ou pasta existe e possui tamanho válido para retomar/pular download."""
    if not target_path.exists():
        return False

    if target_path.is_file():
        return target_path.stat().st_size >= min_bytes

    if target_path.is_dir():
        total_size = sum(f.stat().st_size for f in target_path.glob("**/*") if f.is_file())
        return total_size >= min_bytes

    return False


def get_model_size_mb(target_path: Path) -> float:
    """Retorna o tamanho do modelo em megabytes."""
    if not target_path.exists():
        return 0.0
    if target_path.is_file():
        return target_path.stat().st_size / (1024 * 1024)
    if target_path.is_dir():
        return sum(f.stat().st_size for f in target_path.glob("**/*") if f.is_file()) / (1024 * 1024)
    return 0.0


def check_models_status(
    profile: Optional[str] = None,
    base_models_dir: str = "models",
    config_path: str = "config.yaml",
) -> List[Dict[str, Any]]:
    """
    Retorna o status de presença de cada modelo no disco para o perfil especificado ou detectado.
    """
    if profile is None:
        profile = detect_gpu_profile(config_path=config_path)

    base = Path(base_models_dir).resolve()
    relevant_specs = [m for m in MODEL_CATALOG if profile in m.profiles]
    status_list = []

    for spec in relevant_specs:
        target = base / spec.destination_rel
        installed = verify_file_or_dir_exists(target, spec.expected_min_bytes)
        size_mb = get_model_size_mb(target)

        status_list.append({
            "key": spec.key,
            "name": spec.name,
            "category": spec.category,
            "repo_id": spec.repo_id,
            "path": str(target),
            "installed": installed,
            "size_mb": round(size_mb, 1),
            "description": spec.description,
            "profile_match": profile,
        })

    return status_list


def update_config_model_paths(
    config_path: str,
    saved_paths: Dict[str, str],
    detected_profile: str,
) -> None:
    """
    Salva os caminhos finais dos modelos baixados no config.yaml.
    """
    cfg_file = Path(config_path).resolve()
    cfg_data = {}
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Erro ao ler config.yaml: %s", e)

    # Sincroniza hardware e perfis
    sync_hardware_config(
        profile=detected_profile,
        config_path=config_path,
    )

    # Re-lê para incluir caminhos de modelos sem perder chaves de hardware
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

        if "models" not in cfg_data:
            cfg_data["models"] = {}

        for cat, local_path in saved_paths.items():
            if cat not in cfg_data["models"]:
                cfg_data["models"][cat] = {}
            cfg_data["models"][cat]["model_path"] = local_path

        cfg_data["gpu_profile"] = detected_profile

        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg_data, f, allow_unicode=True, default_flow_style=False)
        logger.info("Caminhos finais dos modelos gravados com sucesso em: %s", cfg_file)
    except Exception as e:
        logger.error("Erro ao gravar modelos no config.yaml: %s", e)


def fetch_models_for_profile(
    profile: Optional[str] = None,
    base_models_dir: str = "models",
    config_path: str = "config.yaml",
    force_download: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, str]:
    """
    Baixa apenas os arquivos e checkpoints necessários para o perfil de hardware detectado/selecionado,
    verificando integridade e salvando os caminhos finais em config.yaml.
    """
    ensure_safe_streams()

    if profile is None:
        profile = detect_gpu_profile(config_path=config_path)

    base = Path(base_models_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    target_specs = [m for m in MODEL_CATALOG if profile in m.profiles]
    total_specs = len(target_specs)
    saved_paths: Dict[str, str] = {}

    logger.info("=================================================================")
    logger.info("KmellVox Downloader - Perfil: '%s' (%d modelos)", profile, total_specs)
    logger.info("Diretório de destino: %s", base)
    logger.info("=================================================================")

    if progress_callback:
        progress_callback(0.02, f"Iniciando download para perfil '{profile}' ({total_specs} modelos)...")

    # Utiliza SafeTqdm para garantir que 'file' nunca seja None em ambientes GUI
    with SafeTqdm(total=total_specs, desc=f"Modelos ({profile})", unit="model") as pbar:
        for idx, spec in enumerate(target_specs, 1):
            dest_target = base / spec.destination_rel
            dest_target.parent.mkdir(parents=True, exist_ok=True)

            stage_pct = (idx - 1) / total_specs
            logger.info("Processando: %s (Repo: %s)...", spec.name, spec.repo_id)

            if progress_callback:
                progress_callback(stage_pct, f"Verificando/Baixando ({idx}/{total_specs}): {spec.name}...")

            # 1. Verifica se já está instalado
            if not force_download and verify_file_or_dir_exists(dest_target, spec.expected_min_bytes):
                size_mb = get_model_size_mb(dest_target)
                logger.info("-> Modelo '%s' já presente localmente (%.1f MB). Pulando.", spec.name, size_mb)
                saved_paths[spec.category] = str(dest_target)
                pbar.update(1)
                continue

            # 2. Download via huggingface_hub com suporte nativo a resume e progresso em tempo real
            progress_tqdm_cls = create_hf_progress_tqdm(
                spec_idx=idx,
                total_specs=total_specs,
                spec_name=spec.name,
                progress_callback=progress_callback,
            )

            try:
                if spec.filename:
                    logger.info("Baixando arquivo '%s' do repositório '%s'...", spec.filename, spec.repo_id)
                    downloaded_path = hf_hub_download(
                        repo_id=spec.repo_id,
                        filename=spec.filename,
                        local_dir=str(dest_target.parent),
                        tqdm_class=progress_tqdm_cls,
                    )
                    if Path(downloaded_path) != dest_target and not dest_target.exists():
                        try:
                            os.rename(downloaded_path, str(dest_target))
                        except Exception:
                            pass
                    saved_paths[spec.category] = str(dest_target)
                else:
                    logger.info("Baixando snapshot completo do repositório '%s'...", spec.repo_id)
                    snapshot_download(
                        repo_id=spec.repo_id,
                        local_dir=str(dest_target),
                        tqdm_class=progress_tqdm_cls,
                    )
                    saved_paths[spec.category] = str(dest_target)

                logger.info("-> '%s' baixado e verificado com sucesso.", spec.name)

            except Exception as e:
                logger.error("Falha ao baixar '%s' da HuggingFace: %s", spec.name, e)
                # Propaga o erro caso não consiga baixar e o arquivo não exista
                if not dest_target.exists():
                    raise RuntimeError(f"Erro ao baixar {spec.name} ({spec.repo_id}): {e}") from e
                saved_paths[spec.category] = str(dest_target)

            pbar.update(1)

    # 3. Salva caminhos finais no config.yaml
    update_config_model_paths(config_path, saved_paths, profile)

    if progress_callback:
        progress_callback(1.0, f"Todos os modelos do perfil '{profile}' foram verificados/atualizados.")

    logger.info("Download e configuração de modelos finalizados com sucesso!")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="KmellVox Model Downloader & Verifier")
    parser.add_argument("--profile", default=None, choices=["perfil_a", "perfil_b", "cpu"],
                        help="Perfil de hardware para baixar pesos (se omitido, lê do config.yaml).")
    parser.add_argument("--dir", default="models", help="Diretório raiz de modelos.")
    parser.add_argument("--config", default="config.yaml", help="Caminho do config.yaml.")
    parser.add_argument("--status", action="store_true", help="Exibe relatório de status dos modelos locais.")
    parser.add_argument("--force", action="store_true", help="Força novo download mesmo se o arquivo já existir.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.status:
        statuses = check_models_status(profile=args.profile, base_models_dir=args.dir, config_path=args.config)
        prof_name = args.profile or detect_gpu_profile(config_path=args.config)
        print(f"\n=========================================================================")
        print(f"Status dos Modelos do KmellVox - Perfil: [{prof_name}]")
        print(f"Pasta: '{os.path.abspath(args.dir)}'")
        print("=========================================================================")
        for s in statuses:
            badge = "[INSTALADO]" if s["installed"] else "[AUSENTE]  "
            size_txt = f"({s['size_mb']} MB)" if s["installed"] else ""
            print(f"{badge:12} | {s['category']:14} | {s['name']:38} {size_txt}")
        print("=========================================================================\n")
        return

    fetch_models_for_profile(
        profile=args.profile,
        base_models_dir=args.dir,
        config_path=args.config,
        force_download=args.force,
    )


if __name__ == "__main__":
    main()
