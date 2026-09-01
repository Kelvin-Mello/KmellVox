"""Testes unitários para core/voice_clone.py (F5TTSEngine, IndexTTS2Engine, get_tts_engine, controle de ritmo)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hardware import ModelProfile
from core.translate import TranslatedSegment
from core.voice_clone import (
    F5TTSEngine,
    IndexTTS2Engine,
    RhythmControlConfig,
    adjust_audio_duration_ffmpeg,
    get_audio_duration,
    get_tts_engine,
)


class TestVoiceClone(unittest.TestCase):
    """Conjunto de testes para os motores de clonagem de voz e controle de ritmo."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # Cria um arquivo de áudio WAV sintético de 2.0 segundos para teste
        self.sample_wav = os.path.join(self.temp_dir.name, "ref_sample.wav")
        sr = 24000
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(sr * 2.0), dtype=np.float32))
        sf.write(self.sample_wav, samples, sr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_audio_duration(self):
        """Testa leitura precisa da duração do áudio em segundos."""
        dur = get_audio_duration(self.sample_wav)
        self.assertAlmostEqual(dur, 2.0, delta=0.05)

    @patch("subprocess.run")
    def test_adjust_audio_duration_ffmpeg_speedup(self, mock_subproc):
        """Testa cálculo e invocação do filtro atempo para acelerar áudio (Controle de Ritmo)."""
        mock_subproc.return_value = MagicMock(returncode=0)

        out_wav = os.path.join(self.temp_dir.name, "adjusted_fast.wav")
        # Áudio tem 2.0s, alvo é 1.6s -> fator = 2.0 / 1.6 = 1.25x
        factor = adjust_audio_duration_ffmpeg(
            input_audio=self.sample_wav,
            output_audio=out_wav,
            target_duration=1.6,
            min_speed=0.70,
            max_speed=1.35,
        )

        self.assertAlmostEqual(factor, 1.25, delta=0.01)
        mock_subproc.assert_called_once()
        call_args = mock_subproc.call_args[0][0]
        self.assertIn("-filter:a", call_args)
        filter_val = call_args[call_args.index("-filter:a") + 1]
        self.assertIn("atempo=1.2500", filter_val)

    @patch("subprocess.run")
    def test_adjust_audio_duration_limits(self, mock_subproc):
        """Testa se o limitador de velocidade respeita min_speed e max_speed."""
        mock_subproc.return_value = MagicMock(returncode=0)

        out_wav = os.path.join(self.temp_dir.name, "adjusted_limit.wav")
        # Áudio de 2.0s, alvo de 0.5s -> fator teórico seria 4.0x, mas max_speed=1.35
        factor = adjust_audio_duration_ffmpeg(
            input_audio=self.sample_wav,
            output_audio=out_wav,
            target_duration=0.5,
            min_speed=0.70,
            max_speed=1.35,
        )

        self.assertEqual(factor, 1.35)

    def test_f5_tts_engine_synthesis_and_vram_cleanup(self):
        """Testa síntese e liberação de VRAM no F5TTSEngine."""
        prof_b = ModelProfile.from_profile("perfil_b")
        engine = F5TTSEngine(model_profile=prof_b, rhythm_config=RhythmControlConfig(max_speed=1.35))

        out_seg_wav = os.path.join(self.temp_dir.name, "f5_out.wav")
        cloned = engine.clone_and_synthesize(
            text="Olá, este é um teste de voz com o F5-TTS.",
            reference_audio_path=self.sample_wav,
            output_path=out_seg_wav,
            target_duration=2.5,
            auto_unload=True,
        )

        self.assertTrue(os.path.isfile(out_seg_wav))
        self.assertEqual(cloned.target_duration, 2.5)
        # Verifica limpeza de VRAM
        self.assertIsNone(engine.model)

    def test_indextts2_engine_native_duration_and_vram_cleanup(self):
        """Testa síntese e controle explícito nativo de duração no IndexTTS2Engine."""
        prof_a = ModelProfile.from_profile("perfil_a")
        engine = IndexTTS2Engine(model_profile=prof_a)

        out_seg_wav = os.path.join(self.temp_dir.name, "indextts2_out.wav")
        cloned = engine.clone_and_synthesize(
            text="Texto para síntese nativa com IndexTTS-2 em alta qualidade.",
            reference_audio_path=self.sample_wav,
            output_path=out_seg_wav,
            target_duration=3.0,
            auto_unload=True,
        )

        self.assertTrue(os.path.isfile(out_seg_wav))
        self.assertEqual(cloned.target_duration, 3.0)
        self.assertEqual(cloned.speed_factor, 1.0)  # Sem pós-processamento atempo
        # Verifica limpeza de VRAM
        self.assertIsNone(engine.model)

    def test_get_tts_engine_factory(self):
        """Testa as regras de seleção da factory get_tts_engine."""
        prof_a = ModelProfile.from_profile("perfil_a")
        prof_b = ModelProfile.from_profile("perfil_b")
        prof_cpu = ModelProfile.from_profile("cpu")

        # 1. perfil_a com use_advanced=True -> IndexTTS2Engine
        eng_a_adv = get_tts_engine(model_profile=prof_a, use_advanced=True)
        self.assertIsInstance(eng_a_adv, IndexTTS2Engine)

        # 2. perfil_a com use_advanced=False -> F5TTSEngine
        eng_a_std = get_tts_engine(model_profile=prof_a, use_advanced=False)
        self.assertIsInstance(eng_a_std, F5TTSEngine)

        # 3. perfil_b com use_advanced=True -> Fallback para F5TTSEngine com aviso
        with self.assertLogs("KmellVox.VoiceClone", level="WARNING") as log_cm:
            eng_b_adv = get_tts_engine(model_profile=prof_b, use_advanced=True)
            self.assertIsInstance(eng_b_adv, F5TTSEngine)
            warning_found = any("8GB de VRAM" in r.getMessage() or "perfil_a" in r.getMessage() for r in log_cm.records)
            self.assertTrue(warning_found)

        # 4. cpu com use_advanced=True -> Fallback para F5TTSEngine
        eng_cpu = get_tts_engine(model_profile=prof_cpu, use_advanced=True)
        self.assertIsInstance(eng_cpu, F5TTSEngine)

    def test_clone_and_align_all(self):
        """Testa o processamento e alinhamento de múltiplos segmentos traduzidos."""
        prof_a = ModelProfile.from_profile("perfil_a")
        engine = F5TTSEngine(model_profile=prof_a)

        segments = [
            TranslatedSegment(id=1, start=0.0, end=2.0, original_text="Hello", translated_text="Olá"),
            TranslatedSegment(id=2, start=2.5, end=5.0, original_text="Welcome", translated_text="Bem-vindo"),
        ]

        out_dir = os.path.join(self.temp_dir.name, "batch_voice")
        results = engine.clone_and_align_all(
            segments=segments,
            reference_audio_path=self.sample_wav,
            output_dir=out_dir,
            auto_unload=True,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].start, 0.0)
        self.assertEqual(results[0].end, 2.0)
        self.assertEqual(results[1].start, 2.5)
        self.assertEqual(results[1].end, 5.0)
        self.assertTrue(os.path.isfile(results[0].audio_path))
        self.assertTrue(os.path.isfile(results[1].audio_path))


if __name__ == "__main__":
    unittest.main()
