"""Testes unitários para core/audio_extract.py e core/transcribe.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.audio_extract import extract_audio
from core.hardware import ModelProfile
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
        # Perfil A
        prof_a = ModelProfile.from_profile("perfil_a")
        t_a = Transcriber(model_profile=prof_a)
        self.assertEqual(t_a.model_variant, "large-v3")
        self.assertEqual(t_a.compute_type, "float16")
        self.assertEqual(t_a.device, "cuda")

        # Perfil B
        prof_b = ModelProfile.from_profile("perfil_b")
        t_b = Transcriber(model_profile=prof_b)
        self.assertEqual(t_b.model_variant, "distil-large-v3")
        self.assertEqual(t_b.compute_type, "int8_float16")
        self.assertEqual(t_b.device, "cuda")

        # CPU
        prof_cpu = ModelProfile.from_profile("cpu")
        t_cpu = Transcriber(model_profile=prof_cpu)
        self.assertEqual(t_cpu.model_variant, "small")
        self.assertEqual(t_cpu.compute_type, "int8")
        self.assertEqual(t_cpu.device, "cpu")

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
        info_mock.language_probability = 0.99
        info_mock.duration = 5.2

        mock_instance.transcribe.return_value = ([seg1, seg2], info_mock)

        prof_a = ModelProfile.from_profile("perfil_a")
        transcriber = Transcriber(model_profile=prof_a)

        audio_file = os.path.join(self.temp_dir.name, "sample.wav")
        segments = transcriber.transcribe(audio_file, auto_unload=True)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[0].end, 2.5)
        self.assertEqual(segments[0].text, "Olá, bem-vindo ao KmellVox.")
        self.assertEqual(segments[0]["start"], 0.0)  # Acesso indexado compatível

        self.assertEqual(segments[1].start, 2.8)
        self.assertEqual(segments[1].end, 5.2)
        self.assertEqual(segments[1].text, "Dublagem e sincronia labial com IA.")

        # Verifica se o modelo foi explicitamente descarregado da VRAM
        self.assertIsNone(transcriber.model)

    def test_export_srt(self):
        """Testa exportação de arquivo .srt a partir de segmentos."""
        segments = [
            TranscriptionSegment(id=1, start=1.2, end=4.5, text="Primeira frase de teste."),
            TranscriptionSegment(id=2, start=5.0, end=8.75, text="Segunda frase de teste."),
        ]

        transcriber = Transcriber()
        srt_out_path = os.path.join(self.temp_dir.name, "subtitles.srt")
        saved_path = transcriber.export_srt(segments, srt_out_path)

        self.assertTrue(os.path.isfile(saved_path))
        with open(saved_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_part_1 = "1\n00:00:01,200 --> 00:00:04,500\nPrimeira frase de teste."
        expected_part_2 = "2\n00:00:05,000 --> 00:00:08,750\nSegunda frase de teste."

        self.assertIn(expected_part_1, content)
        self.assertIn(expected_part_2, content)


if __name__ == "__main__":
    unittest.main()
