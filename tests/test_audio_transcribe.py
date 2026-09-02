"""Testes unitários para core/audio_extract.py e core/transcribe.py."""

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

from core.audio_extract import extract_audio
from core.hardware import VALID_WHISPER_VARIANTS, ModelProfile, WhisperModelVariant
from core.transcribe import Transcriber, TranscriptionSegment, format_segments_to_srt


class TestAudioExtractAndTranscribe(unittest.TestCase):
    """Conjunto de testes para extração de áudio e transcrição."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("ffmpeg.input")
    def test_extract_audio_ffmpeg_python(self, mock_ffmpeg_input):
        """Testa se extract_audio invoca ffmpeg-python com parâmetros para WAV mono 16kHz."""
        mock_stream_in = MagicMock()
        mock_stream_out = MagicMock()
        mock_ffmpeg_input.return_value = mock_stream_in
        mock_stream_in.output.return_value = mock_stream_out

        video_path = os.path.join(self.temp_dir.name, "input.mp4")
        output_path = os.path.join(self.temp_dir.name, "output.wav")

        result = extract_audio(video_path, output_path)

        mock_ffmpeg_input.assert_called_once_with(str(video_path))
        mock_stream_in.output.assert_called_once_with(
            str(Path(output_path).resolve()),
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
            vn=None,
        )
        mock_stream_out.run.assert_called_once()
        self.assertEqual(result, str(Path(output_path).resolve()))

    def test_transcriber_profile_configuration(self):
        """Testa se o Transcriber configura corretamente variante e compute_type conforme ModelProfile."""
        # Perfil A (VRAM >= 7.5GB -> large-v3 e float16)
        prof_a = ModelProfile.from_profile("perfil_a")
        t_a = Transcriber(model_profile=prof_a)
        self.assertEqual(t_a.model_variant, "large-v3")
        self.assertEqual(t_a.compute_type, "float16")
        self.assertEqual(t_a.device, "cuda")

        # Perfil B (5.0GB <= VRAM < 7.5GB -> distil-large-v3 e int8_float16)
        prof_b = ModelProfile.from_profile("perfil_b")
        t_b = Transcriber(model_profile=prof_b)
        self.assertEqual(t_b.model_variant, "distil-large-v3")
        self.assertEqual(t_b.compute_type, "int8_float16")
        self.assertEqual(t_b.device, "cuda")

        # CPU (< 5.0GB VRAM -> small e int8)
        prof_cpu = ModelProfile.from_profile("cpu")
        t_cpu = Transcriber(model_profile=prof_cpu)
        self.assertEqual(t_cpu.model_variant, "small")
        self.assertEqual(t_cpu.compute_type, "int8")
        self.assertEqual(t_cpu.device, "cpu")

    def test_invalid_whisper_variant_rejected(self):
        """
        Valida que qualquer valor fora da lista permitida (como 'medium', 'tiny', 'base', 'large-v2')
        é IMEDIATAMENTE rejeitado com ValueError, prevenindo corrupção no config.yaml.
        """
        invalid_variants = ["medium", "tiny", "base", "large-v2", "turbo", "invalid_model"]

        for bad in invalid_variants:
            with self.assertRaises(ValueError, msg=f"Deveria ter rejeitado variante inválida '{bad}'"):
                ModelProfile(
                    profile_name="perfil_a",
                    whisper_variant=bad,
                )

        # Valida que o conjunto homologado contém apenas large-v3, distil-large-v3 e small
        self.assertEqual(VALID_WHISPER_VARIANTS, {"large-v3", "distil-large-v3", "small"})

    def test_config_yaml_whisper_variant_integrity(self):
        """
        Valida que o config.yaml gravado na raiz possui um model_size estritamente homologado.
        """
        cfg_path = PROJECT_ROOT / "config.yaml"
        self.assertTrue(cfg_path.is_file(), "config.yaml deve existir na raiz do projeto.")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        trans_size = cfg.get("models", {}).get("transcription", {}).get("model_size", "")
        self.assertIn(
            trans_size,
            VALID_WHISPER_VARIANTS,
            f"models.transcription.model_size no config.yaml ('{trans_size}') não é uma variante válida!",
        )
        # Como o hardware detectado é perfil_a (7.96 GB VRAM), o valor deve ser 'large-v3'
        self.assertEqual(trans_size, "large-v3")

    @patch("faster_whisper.WhisperModel")
    def test_transcribe_and_unload_vram(self, mock_whisper_cls):
        """Testa o método transcribe retornando segmentos e liberando a VRAM no final."""
        mock_instance = MagicMock()
        mock_whisper_cls.return_value = mock_instance

        # Cria segmentos simulados
        seg1 = MagicMock()
        seg1.id = 1
        seg1.start = 0.0
        seg1.end = 2.5
        seg1.text = "Olá, bem-vindo ao KmellVox."
        seg1.avg_logprob = -0.1
        seg1.no_speech_prob = 0.0
        seg1.words = []

        seg2 = MagicMock()
        seg2.id = 2
        seg2.start = 2.8
        seg2.end = 5.2
        seg2.text = "Dublagem e sincronia labial com IA."
        seg2.avg_logprob = -0.1
        seg2.no_speech_prob = 0.0
        seg2.words = []

        info_mock = MagicMock()
        info_mock.language = "pt"
        info_mock.language_probability = 0.98
        info_mock.duration = 6.0

        mock_instance.transcribe.return_value = ([seg1, seg2], info_mock)

        prof_a = ModelProfile.from_profile("perfil_a")
        transcriber = Transcriber(model_profile=prof_a)

        dummy_audio = os.path.join(self.temp_dir.name, "sample.wav")
        with open(dummy_audio, "wb") as f:
            f.write(b"dummy wav data")

        result = transcriber.transcribe(dummy_audio, auto_unload=True)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Olá, bem-vindo ao KmellVox.")
        self.assertEqual(result[1].text, "Dublagem e sincronia labial com IA.")
        
        # Garante que o modelo foi liberado da memória após auto_unload
        self.assertIsNone(transcriber.model)

    def test_format_segments_to_srt(self):
        """Testa a geração correta de formato SRT a partir de segmentos de transcrição."""
        segments = [
            TranscriptionSegment(id=1, start=0.0, end=1.500, text="Primeira linha"),
            TranscriptionSegment(id=2, start=2.000, end=4.250, text="Segunda linha com acentuação"),
        ]

        srt_content = format_segments_to_srt(segments)

        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nPrimeira linha", srt_content)
        self.assertIn("2\n00:00:02,000 --> 00:00:04,250\nSegunda linha com acentuação", srt_content)


if __name__ == "__main__":
    unittest.main()
