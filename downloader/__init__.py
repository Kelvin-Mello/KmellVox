"""Módulo de download e gerenciamento de pesos de modelos do KmellVox."""

from .fetch_models import (
    fetch_models_for_profile,
    check_models_status,
    MODEL_CATALOG,
    update_config_model_paths,
    verify_file_or_dir_exists,
)

# Aliases para compatibilidade retroativa
download_model_profile = fetch_models_for_profile
MODEL_REGISTRY = MODEL_CATALOG

__all__ = [
    "fetch_models_for_profile",
    "download_model_profile",
    "check_models_status",
    "MODEL_CATALOG",
    "MODEL_REGISTRY",
    "update_config_model_paths",
    "verify_file_or_dir_exists",
]
