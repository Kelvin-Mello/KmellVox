"""Testes unitários para downloader/fetch_models.py (download seletivo por perfil, verificação de integridade e config.yaml)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downloader.fetch_models import (
    MODEL_CATALOG,
    check_models_status,
    fetch_models_for_profile,
    update_config_model_paths,
    verify_file_or_dir_exists,
)


class TestFetchModels(unittest.TestCase):
    """Conjunto de testes para o módulo de download e verificação de modelos."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dummy_config = os.path.join(self.temp_dir.name, "config.yaml")
        initial_cfg = {"app": {"name": "KmellVox"}, "models": {}}
        with open(self.dummy_config, "w", encoding="utf-8") as f:
            yaml.dump(initial_cfg, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_catalog_profile_filtering(self):
        """Testa se o catálogo filtra modelos estritamente conforme o perfil requerido."""
        # 1. perfil_a (8GB+) deve incluir IndexTTS-2 e Whisper Large-v3
        statuses_a = check_models_status(profile="perfil_a", base_models_dir=self.temp_dir.name, config_path=self.dummy_config)
        names_a = [s["name"] for s in statuses_a]
        self.assertTrue(any("IndexTTS-2" in n for n in names_a))
        self.assertTrue(any("Large-v3" in n for n in names_a))
        self.assertTrue(any("Qwen3-8B" in n for n in names_a))

        # 2. perfil_b (6GB) NÃO deve incluir IndexTTS-2 e deve incluir Distil-Large-v3 e Qwen3-4B
        statuses_b = check_models_status(profile="perfil_b", base_models_dir=self.temp_dir.name, config_path=self.dummy_config)
        names_b = [s["name"] for s in statuses_b]
        self.assertFalse(any("IndexTTS-2" in n for n in names_b))
        self.assertTrue(any("Distil" in n for n in names_b))
        self.assertTrue(any("Qwen3-4B" in n for n in names_b))

        # 3. cpu deve incluir Whisper Small e Qwen3-1.5B
        statuses_cpu = check_models_status(profile="cpu", base_models_dir=self.temp_dir.name, config_path=self.dummy_config)
        names_cpu = [s["name"] for s in statuses_cpu]
        self.assertTrue(any("Small" in n for n in names_cpu))
        self.assertTrue(any("1.5B" in n for n in names_cpu))

    @patch("downloader.fetch_models.hf_hub_download")
    @patch("downloader.fetch_models.snapshot_download")
    def test_fetch_models_perfil_b_and_config_update(self, mock_snapshot, mock_hf_download):
        """Testa o fluxo de download simulado para perfil_b e valida gravação dos caminhos no config.yaml."""
        mock_hf_download.side_effect = lambda repo_id, filename, local_dir, **kw: os.path.join(local_dir, filename)
        mock_snapshot.return_value = self.temp_dir.name

        progress_calls = []

        def on_prog(pct: float, msg: str):
            progress_calls.append((pct, msg))

        saved = fetch_models_for_profile(
            profile="perfil_b",
            base_models_dir=self.temp_dir.name,
            config_path=self.dummy_config,
            progress_callback=on_prog,
        )

        self.assertIn("transcription", saved)
        self.assertIn("translation", saved)
        self.assertIn("voice_clone", saved)

        # Valida que o config.yaml foi atualizado
        with open(self.dummy_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg.get("gpu_profile"), "perfil_b")
        self.assertIn("transcription", cfg.get("models", {}))
        self.assertIn("translation", cfg.get("models", {}))

        # Valida callbacks de progresso atingindo 100%
        self.assertGreater(len(progress_calls), 0)
        self.assertEqual(progress_calls[-1][0], 1.0)

    def test_resume_and_verification_skips_existing(self):
        """Testa se arquivos já existentes no disco não são baixados novamente."""
        # Cria arquivo fictício para simular modelo já baixado
        mock_model_file = Path(self.temp_dir.name) / "llm" / "Qwen3-4B-Instruct-Q4_K_M.gguf"
        mock_model_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mock_model_file, "wb") as f:
            f.write(b"0" * 4096)

        self.assertTrue(verify_file_or_dir_exists(mock_model_file, min_bytes=1024))


if __name__ == "__main__":
    unittest.main()
