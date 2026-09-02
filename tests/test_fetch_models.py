"""Testes unitários para downloader/fetch_models.py (download seletivo por perfil, verificação de integridade e config.yaml)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# Garante que a raiz do projeto esteja no sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.safe_streams import SafeStream, ensure_safe_streams
from downloader.fetch_models import (
    MODEL_CATALOG,
    ModelDownloadSpec,
    ModelSpecVerificationError,
    SafeTqdm,
    check_models_status,
    fetch_models_for_profile,
    update_config_model_paths,
    verify_file_or_dir_exists,
    verify_model_catalog_online,
    verify_spec_on_huggingface,
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
        self.assertTrue(any("indextts-2" in n.lower() for n in names_a))
        self.assertTrue(any("large-v3" in n.lower() for n in names_a))
        self.assertTrue(any("qwen3" in n.lower() for n in names_a))

        # 2. perfil_b (6GB) NÃO deve incluir IndexTTS-2 e deve incluir Distil-Large-v3 e Qwen3
        statuses_b = check_models_status(profile="perfil_b", base_models_dir=self.temp_dir.name, config_path=self.dummy_config)
        names_b = [s["name"] for s in statuses_b]
        self.assertFalse(any("indextts-2" in n.lower() for n in names_b))
        self.assertTrue(any("distil" in n.lower() for n in names_b))
        self.assertTrue(any("qwen3" in n.lower() for n in names_b))

        # 3. cpu deve incluir Whisper Small e Qwen3
        statuses_cpu = check_models_status(profile="cpu", base_models_dir=self.temp_dir.name, config_path=self.dummy_config)
        names_cpu = [s["name"] for s in statuses_cpu]
        self.assertTrue(any("small" in n.lower() for n in names_cpu))
        self.assertTrue(any("qwen3" in n.lower() or "0.6b" in n.lower() for n in names_cpu))

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

    @patch("downloader.fetch_models.hf_hub_download")
    @patch("downloader.fetch_models.snapshot_download")
    def test_gui_null_streams_safe_download(self, mock_snapshot, mock_hf_download):
        """
        Simula ambiente PyInstaller/GUI onde sys.stdout e sys.stderr são None,
        garantindo que o download execute perfeitamente sem 'NoneType object has no attribute write'.
        """
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        try:
            # Simula streams nulos
            sys.stdout = None
            sys.stderr = None

            mock_hf_download.side_effect = lambda repo_id, filename, local_dir, **kw: os.path.join(local_dir, filename)
            mock_snapshot.return_value = self.temp_dir.name

            # SafeTqdm e ensure_safe_streams não devem levantar exceção
            ensure_safe_streams()
            with SafeTqdm(total=3) as pbar:
                pbar.update(1)

            saved = fetch_models_for_profile(
                profile="cpu",
                base_models_dir=self.temp_dir.name,
                config_path=self.dummy_config,
            )
            self.assertTrue(len(saved) > 0)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    def test_resume_and_verification_skips_existing(self):
        """Testa se arquivos já existentes no disco não são baixados novamente."""
        # Cria arquivo fictício para simular modelo já baixado
        mock_model_file = Path(self.temp_dir.name) / "llm" / "Qwen3-8B-Q4_K_M.gguf"
        mock_model_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mock_model_file, "wb") as f:
            f.write(b"0" * 4096)

        self.assertTrue(verify_file_or_dir_exists(mock_model_file, min_bytes=1024))

    def test_pre_flight_verification_valid_spec(self):
        """Testa que uma especificação válida com repositório e arquivo existentes passa na pré-verificação."""
        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_sibling1 = MagicMock(rfilename="Qwen3-8B-Q4_K_M.gguf")
        mock_sibling2 = MagicMock(rfilename="README.md")
        mock_info.siblings = [mock_sibling1, mock_sibling2]
        mock_api.model_info.return_value = mock_info

        spec = ModelDownloadSpec(
            key="llm_qwen3_8b",
            name="Qwen3-8B GGUF Q4_K_M",
            category="translation",
            repo_id="Qwen/Qwen3-8B-GGUF",
            filename="Qwen3-8B-Q4_K_M.gguf",
            profiles=["perfil_a"],
        )

        ok, err = verify_spec_on_huggingface(spec, hf_api=mock_api)
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_pre_flight_verification_invalid_repo(self):
        """Testa que um repositório inexistente é rejeitado com mensagem clara e não altera o modelo."""
        mock_api = MagicMock()
        mock_api.model_info.side_effect = Exception("401 Client Error: Repository Not Found")

        spec = ModelDownloadSpec(
            key="llm_invalid",
            name="Modelo Inexistente",
            category="translation",
            repo_id="OrganizaçãoInexistente/Modelo-Fantasma",
            filename="modelo.gguf",
            profiles=["perfil_a"],
        )

        ok, err = verify_spec_on_huggingface(spec, hf_api=mock_api)
        self.assertFalse(ok)
        self.assertIsNotNone(err)
        self.assertIn("OrganizaçãoInexistente/Modelo-Fantasma", err)
        self.assertIn("Repositório Hugging Face inexistente", err)

    def test_pre_flight_verification_missing_filename_suggests_alternatives(self):
        """Testa que um filename inexistente é rejeitado, lista alternativas e diagnostica sufixos sem trocar arquitetura."""
        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_sibling1 = MagicMock(rfilename="Qwen3-8B-Q4_K_M.gguf")
        mock_sibling2 = MagicMock(rfilename="Qwen3-8B-Q5_K_M.gguf")
        mock_info.siblings = [mock_sibling1, mock_sibling2]
        mock_api.model_info.return_value = mock_info

        # Tenta pedir 'Qwen3-8B-Instruct-Q4_K_M.gguf' que não existe no repo
        spec = ModelDownloadSpec(
            key="llm_qwen3_8b",
            name="Qwen3-8B GGUF",
            category="translation",
            repo_id="Qwen/Qwen3-8B-GGUF",
            filename="Qwen3-8B-Instruct-Q4_K_M.gguf",
            profiles=["perfil_a"],
        )

        ok, err = verify_spec_on_huggingface(spec, hf_api=mock_api)
        self.assertFalse(ok)
        self.assertIsNotNone(err)
        self.assertIn("Qwen3-8B-Instruct-Q4_K_M.gguf", err)
        self.assertIn("Qwen3-8B-Q4_K_M.gguf", err)  # Sugestão encontrada
        self.assertIn("Diagnóstico", err)

    @patch("downloader.fetch_models.verify_model_catalog_online")
    def test_fetch_models_aborts_on_pre_flight_failure(self, mock_verify_online):
        """Testa que o processo de download é abortado imediatamente com ModelSpecVerificationError em caso de falha."""
        mock_verify_online.return_value = [
            "Repositório inexistente: Qwen/Qwen3-8B-Instruct-GGUF. Sugestão: use Qwen/Qwen3-8B-GGUF"
        ]

        with self.assertRaises(ModelSpecVerificationError) as ctx:
            fetch_models_for_profile(
                profile="perfil_a",
                base_models_dir=self.temp_dir.name,
                config_path=self.dummy_config,
                force_download=True,
                verify_online_first=True,
            )

        self.assertIn("Falha na pré-verificação de modelos no Hugging Face", str(ctx.exception))
        self.assertIn("AVISO DE INTEGRIDADE DE ARQUITETURA", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
