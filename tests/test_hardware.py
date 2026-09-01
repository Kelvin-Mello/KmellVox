"""Testes unitários para simulação e validação dos perfis de hardware e modelos (perfil_a, perfil_b, cpu)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hardware import ModelProfile, detect_gpu_profile


class TestHardwareProfiles(unittest.TestCase):
    """Conjunto de testes para validação de detect_gpu_profile e ModelProfile."""

    def setUp(self):
        # Cria um arquivo temporário de configuração config.yaml para cada teste
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_path = os.path.join(self.temp_dir.name, "config.yaml")
        initial_data = {"app": {"name": "KmellVoxTest"}}
        with open(self.temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(initial_data, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_perfil_a_simulation(self):
        """Simula GPU com 8.0 GB VRAM (>= 7.5 GB) -> deve retornar 'perfil_a'."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 5060 (Simulated)"
        
        # 8 GB em bytes
        mock_props = MagicMock()
        mock_props.total_memory = int(8.0 * (1024 ** 3))
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict(sys.modules, {"torch": mock_torch}):
            profile = detect_gpu_profile(config_path=self.temp_config_path, force_redetect=True)
            self.assertEqual(profile, "perfil_a")

            # Verifica persistência no config.yaml
            with open(self.temp_config_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
                self.assertEqual(saved.get("gpu_profile"), "perfil_a")

            # Valida resolução do ModelProfile
            mp = ModelProfile.from_profile(profile)
            self.assertEqual(mp.profile_name, "perfil_a")
            self.assertEqual(mp.whisper_variant, "large-v3")
            self.assertEqual(mp.whisper_compute_type, "float16")
            self.assertEqual(mp.translation_model, "Qwen3-8B-Instruct Q4_K_M")
            self.assertEqual(mp.default_tts_engine, "F5-TTS")
            self.assertTrue(mp.enable_indextts_2)
            self.assertFalse(mp.musetalk_use_float16)

    def test_perfil_b_simulation(self):
        """Simula GPU com 6.0 GB VRAM (entre 5.0 GB e 7.5 GB) -> deve retornar 'perfil_b'."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3060 (Simulated)"

        # 6 GB em bytes
        mock_props = MagicMock()
        mock_props.total_memory = int(6.0 * (1024 ** 3))
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict(sys.modules, {"torch": mock_torch}):
            profile = detect_gpu_profile(config_path=self.temp_config_path, force_redetect=True)
            self.assertEqual(profile, "perfil_b")

            # Verifica persistência no config.yaml
            with open(self.temp_config_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
                self.assertEqual(saved.get("gpu_profile"), "perfil_b")

            # Valida resolução do ModelProfile
            mp = ModelProfile.from_profile(profile)
            self.assertEqual(mp.profile_name, "perfil_b")
            self.assertEqual(mp.whisper_variant, "distil-large-v3")
            self.assertEqual(mp.whisper_compute_type, "int8_float16")
            self.assertEqual(mp.translation_model, "Qwen3-4B-Instruct Q4_K_M")
            self.assertEqual(mp.default_tts_engine, "F5-TTS")
            self.assertFalse(mp.enable_indextts_2)
            self.assertTrue(mp.musetalk_use_float16)

    def test_cpu_simulation_no_cuda(self):
        """Simula ambiente sem GPU CUDA disponível -> deve retornar 'cpu' com log de aviso."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with self.assertLogs("KmellVox.Hardware", level="WARNING") as log_cm:
                profile = detect_gpu_profile(config_path=self.temp_config_path, force_redetect=True)
                self.assertEqual(profile, "cpu")
                
                # Verifica se o aviso sobre lentidão na CPU foi emitido
                warning_found = any("Nenhuma GPU CUDA" in record.getMessage() or "CPU" in record.getMessage() 
                                    for record in log_cm.records)
                self.assertTrue(warning_found)

            # Verifica persistência no config.yaml
            with open(self.temp_config_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
                self.assertEqual(saved.get("gpu_profile"), "cpu")

            # Valida resolução do ModelProfile
            mp = ModelProfile.from_profile(profile)
            self.assertEqual(mp.profile_name, "cpu")
            self.assertEqual(mp.whisper_variant, "small")
            self.assertEqual(mp.whisper_compute_type, "int8")
            self.assertEqual(mp.translation_model, "Qwen3-1.5B-Instruct Q4_K_M")
            self.assertEqual(mp.default_tts_engine, "F5-TTS")
            self.assertFalse(mp.enable_indextts_2)
            self.assertFalse(mp.musetalk_use_float16)

    def test_cpu_simulation_low_vram(self):
        """Simula GPU com VRAM insuficiente (< 5.0 GB) -> deve retornar 'cpu' com aviso."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GTX 1050 4GB (Simulated)"

        # 4 GB em bytes
        mock_props = MagicMock()
        mock_props.total_memory = int(4.0 * (1024 ** 3))
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with self.assertLogs("KmellVox.Hardware", level="WARNING") as log_cm:
                profile = detect_gpu_profile(config_path=self.temp_config_path, force_redetect=True)
                self.assertEqual(profile, "cpu")

                warning_found = any("insuficiente" in record.getMessage() or "CPU" in record.getMessage() 
                                    for record in log_cm.records)
                self.assertTrue(warning_found)


if __name__ == "__main__":
    unittest.main()
