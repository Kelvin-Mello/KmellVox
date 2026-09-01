"""Testes unitários para core/lipsync.py (LipSyncEngine, FP16 condicional, MuseTalk 1.5, VRAM cleanup)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hardware import ModelProfile
from core.lipsync import LipSyncEngine, LipSyncResult


class TestLipSync(unittest.TestCase):
    """Conjunto de testes para o motor de sincronização labial MuseTalk 1.5."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # Cria arquivos fictícios de vídeo e áudio para os testes
        self.dummy_video = os.path.join(self.temp_dir.name, "input_video.mp4")
        with open(self.dummy_video, "w") as f:
            f.write("dummy video data")

        self.dummy_audio = os.path.join(self.temp_dir.name, "dubbed_audio.wav")
        with open(self.dummy_audio, "w") as f:
            f.write("dummy audio data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_float16_profile_resolution(self):
        """Testa se a flag use_float16 é obrigatória no perfil_b e configurável no perfil_a."""
        # 1. perfil_b -> use_float16 DEVE ser True (obrigatório para 6GB VRAM)
        prof_b = ModelProfile.from_profile("perfil_b")
        engine_b = LipSyncEngine(model_profile=prof_b, use_float16=False)
        self.assertTrue(engine_b.use_float16)

        # 2. perfil_a -> use_float16 configurável (padrão False ou conforme especificado)
        prof_a = ModelProfile.from_profile("perfil_a")
        engine_a_default = LipSyncEngine(model_profile=prof_a)
        self.assertFalse(engine_a_default.use_float16)

        engine_a_fp16 = LipSyncEngine(model_profile=prof_a, use_float16=True)
        self.assertTrue(engine_a_fp16.use_float16)

        # 3. cpu -> use_float16 = False
        prof_cpu = ModelProfile.from_profile("cpu")
        engine_cpu = LipSyncEngine(model_profile=prof_cpu)
        self.assertFalse(engine_cpu.use_float16)

    @patch("subprocess.run")
    def test_lipsync_sync_and_vram_cleanup(self, mock_subproc):
        """Testa o método sync() com simulação FFmpeg e liberação explícita de VRAM."""
        mock_subproc.return_value = MagicMock(returncode=0)

        prof_b = ModelProfile.from_profile("perfil_b")
        engine = LipSyncEngine(model_profile=prof_b, models_dir=self.temp_dir.name)

        out_video = os.path.join(self.temp_dir.name, "lipsync_output.mp4")

        result: LipSyncResult = engine.sync(
            video_path=self.dummy_video,
            dubbed_audio_path=self.dummy_audio,
            output_path=out_video,
            auto_unload=True,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.speed_float16_used)  # Validou uso de FP16 no perfil_b
        mock_subproc.assert_called_once()

        # Valida que o modelo foi liberado da VRAM
        self.assertIsNone(engine.model)

    def test_checkpoints_status(self):
        """Testa o relatório de status dos checkpoints do MuseTalk."""
        prof_a = ModelProfile.from_profile("perfil_a")
        engine = LipSyncEngine(model_profile=prof_a, models_dir=self.temp_dir.name)
        status = engine.get_checkpoints_status()

        self.assertIn("musetalk_unet", status)
        self.assertIn("dwpose", status)
        self.assertIn("face_parsing", status)
        self.assertIn("sd_vae", status)


if __name__ == "__main__":
    unittest.main()
