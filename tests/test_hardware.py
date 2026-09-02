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

from core.hardware import (
    ModelProfile,
    detect_gpu_profile,
    detect_hardware,
    resolve_profile_from_vram,
    sync_hardware_config,
)


class TestHardwareProfiles(unittest.TestCase):
    """Conjunto de testes para validação de detect_gpu_profile, detect_hardware e consistência de config.yaml."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_path = os.path.join(self.temp_dir.name, "config.yaml")
        initial_data = {"app": {"name": "KmellVoxTest"}}
        with open(self.temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(initial_data, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resolve_profile_from_vram_boundaries(self):
        """Valida os limites matemáticos exatos da única função de decisão de perfil."""
        # 1. Sem CUDA
        self.assertEqual(resolve_profile_from_vram(cuda_available=False, vram_gb=16.0), "cpu")

        # 2. Com CUDA, VRAM baixa (< 5.0 GB) -> cpu
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=0.0), "cpu")
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=4.99), "cpu")

        # 3. Com CUDA, 5.0 GB <= VRAM < 7.5 GB -> perfil_b
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=5.0), "perfil_b")
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=6.0), "perfil_b")
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=7.49), "perfil_b")

        # 4. Com CUDA, VRAM >= 7.5 GB -> perfil_a
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=7.5), "perfil_a")
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=7.96), "perfil_a")
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=8.0), "perfil_a")
        self.assertEqual(resolve_profile_from_vram(cuda_available=True, vram_gb=24.0), "perfil_a")

    def test_perfil_a_simulation(self):
        """Simula GPU com 8.0 GB VRAM (>= 7.5 GB) -> deve retornar 'perfil_a' e gravar chaves consistentes."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4070 (Simulated)"
        
        mock_props = MagicMock()
        mock_props.total_memory = int(8.0 * (1024 ** 3))
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict(sys.modules, {"torch": mock_torch}):
            profile = detect_gpu_profile(config_path=self.temp_config_path, force_redetect=True)
            self.assertEqual(profile, "perfil_a")

            # Verifica sincronia e consistência de TODAS as chaves no config.yaml
            with open(self.temp_config_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
                self.assertEqual(saved.get("gpu_profile"), "perfil_a")
                self.assertEqual(saved.get("hardware", {}).get("profile"), "perfil_a")
                self.assertEqual(saved.get("hardware", {}).get("compute_type"), "float16")
                self.assertEqual(saved.get("hardware", {}).get("device"), "cuda")
                self.assertEqual(saved.get("hardware", {}).get("vram_detected_gb"), 8.0)

            # Valida ModelProfile correspondente
            mp = ModelProfile.from_profile(profile)
            self.assertEqual(mp.profile_name, "perfil_a")
            self.assertEqual(mp.whisper_variant, "large-v3")
            self.assertEqual(mp.whisper_compute_type, "float16")
            self.assertEqual(mp.translation_model, "Qwen2.5-8B-Instruct-Q4_K_M" if hasattr(mp, "translation_model") and "8B" in mp.translation_model else "Qwen2.5-7B-Instruct-Q4_K_M")
            self.assertEqual(mp.default_tts_engine, "F5-TTS")
            self.assertTrue(mp.enable_indextts_2)
            self.assertFalse(mp.musetalk_use_float16)

    def test_edge_case_7_96_gb_rtx5060(self):
        """
        Teste específico para o caso relatado: RTX 5060 com 7.96 GB de VRAM.
        Garante que 7.96 GB >= 7.5 GB -> perfil_a em TODAS as chaves raiz e aninhadas.
        """
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 5060 Laptop GPU"

        mock_props = MagicMock()
        mock_props.total_memory = int(7.96 * (1024 ** 3))
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict(sys.modules, {"torch": mock_torch}):
            hw_info = detect_hardware(config_path=self.temp_config_path)
            self.assertEqual(hw_info.gpu_profile, "perfil_a")
            self.assertEqual(hw_info.recommended_compute_type, "float16")
            self.assertEqual(hw_info.model_profile.whisper_variant, "large-v3")
            self.assertTrue(hw_info.model_profile.enable_indextts_2)

            with open(self.temp_config_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
                # 1. Chave raiz
                self.assertEqual(saved.get("gpu_profile"), "perfil_a")
                # 2. Chave aninhada profile
                self.assertEqual(saved.get("hardware", {}).get("profile"), "perfil_a")
                # 3. Chave aninhada compute_type
                self.assertEqual(saved.get("hardware", {}).get("compute_type"), "float16")
                # 4. Chave aninhada vram_detected_gb
                self.assertEqual(saved.get("hardware", {}).get("vram_detected_gb"), 7.96)
                # 5. Chave aninhada device
                self.assertEqual(saved.get("hardware", {}).get("device"), "cuda")

    def test_perfil_b_simulation(self):
        """Simula GPU com 6.0 GB VRAM (entre 5.0 GB e 7.5 GB) -> deve retornar 'perfil_b' sincronizado."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3060 (Simulated)"

        mock_props = MagicMock()
        mock_props.total_memory = int(6.0 * (1024 ** 3))
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict(sys.modules, {"torch": mock_torch}):
            profile = detect_gpu_profile(config_path=self.temp_config_path, force_redetect=True)
            self.assertEqual(profile, "perfil_b")

            # Verifica sincronia em config.yaml
            with open(self.temp_config_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
                self.assertEqual(saved.get("gpu_profile"), "perfil_b")
                self.assertEqual(saved.get("hardware", {}).get("profile"), "perfil_b")
                self.assertEqual(saved.get("hardware", {}).get("compute_type"), "int8_float16")
                self.assertEqual(saved.get("hardware", {}).get("device"), "cuda")
                self.assertEqual(saved.get("hardware", {}).get("vram_detected_gb"), 6.0)

            # Valida ModelProfile
            mp = ModelProfile.from_profile(profile)
            self.assertEqual(mp.profile_name, "perfil_b")
            self.assertEqual(mp.whisper_variant, "distil-large-v3")
            self.assertEqual(mp.whisper_compute_type, "int8_float16")
            self.assertEqual(mp.translation_model, "Qwen2.5-3B-Instruct-Q4_K_M")
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
                
                warning_found = any("Nenhuma GPU CUDA" in record.getMessage() or "CPU" in record.getMessage() 
                                    for record in log_cm.records)
                self.assertTrue(warning_found)

            with open(self.temp_config_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)
                self.assertEqual(saved.get("gpu_profile"), "cpu")
                self.assertEqual(saved.get("hardware", {}).get("profile"), "cpu")
                self.assertEqual(saved.get("hardware", {}).get("compute_type"), "int8")
                self.assertEqual(saved.get("hardware", {}).get("device"), "cpu")

            mp = ModelProfile.from_profile(profile)
            self.assertEqual(mp.profile_name, "cpu")
            self.assertEqual(mp.whisper_variant, "small")
            self.assertEqual(mp.whisper_compute_type, "int8")
            self.assertEqual(mp.translation_model, "Qwen2.5-1.5B-Instruct-Q4_K_M")
            self.assertEqual(mp.default_tts_engine, "F5-TTS")
            self.assertFalse(mp.enable_indextts_2)
            self.assertFalse(mp.musetalk_use_float16)

    def test_cpu_simulation_low_vram(self):
        """Simula GPU com VRAM insuficiente (< 5.0 GB) -> deve retornar 'cpu' com aviso."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GTX 1050 4GB (Simulated)"

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

    def test_revalidation_fixes_contradictory_saved_config(self):
        """
        Garante que se o config.yaml em disco tiver valores contraditórios ou obsoletos
        (ex: gpu_profile=cpu enquanto o hardware atual é uma GPU de 7.96 GB),
        a verificação de integridade re-detecta e sobrescreve o config.yaml com dados consistentes.
        """
        # Grava o cenário contraditório reportado no config.yaml
        contradictory_data = {
            "gpu_profile": "cpu",
            "hardware": {
                "profile": "mid_vram",
                "compute_type": "float16",
                "device": "cuda",
                "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
                "vram_detected_gb": 7.96,
            }
        }
        with open(self.temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(contradictory_data, f)

        # Simula GPU física real presente no sistema (7.96 GB)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 5060 Laptop GPU"
        mock_props = MagicMock()
        mock_props.total_memory = int(7.96 * (1024 ** 3))
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict(sys.modules, {"torch": mock_torch}):
            # Chama sem force_redetect: a integridade deve detectar que "cpu" salvo != hardware físico e corrigir!
            detected = detect_gpu_profile(config_path=self.temp_config_path, force_redetect=False)
            self.assertEqual(detected, "perfil_a")

            # Verifica se o arquivo em disco foi completamente corrigido e sincronizado
            with open(self.temp_config_path, "r", encoding="utf-8") as f:
                corrected = yaml.safe_load(f)
                self.assertEqual(corrected.get("gpu_profile"), "perfil_a")
                self.assertEqual(corrected.get("hardware", {}).get("profile"), "perfil_a")
                self.assertEqual(corrected.get("hardware", {}).get("compute_type"), "float16")
                self.assertEqual(corrected.get("hardware", {}).get("vram_detected_gb"), 7.96)


if __name__ == "__main__":
    unittest.main()
