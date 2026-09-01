"""Testes unitários para core/assemble.py e core/pipeline.py (DubPipeline, fila, muxing, legendas e pacote bruto)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.assemble import (
    AssemblyConfig,
    assemble_final_video,
    burn_subtitles,
    export_raw_package,
    mux_audio_video,
)
from core.hardware import ModelProfile
from core.pipeline import DubPipeline, PipelineConfig, PipelineProgress
from core.transcribe import TranscriptionSegment
from core.translate import TranslatedSegment
from core.voice_clone import ClonedAudioSegment


class TestAssembleAndPipeline(unittest.TestCase):
    """Conjunto de testes para as etapas de montagem e orquestração do pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # Cria arquivos fictícios de teste
        self.dummy_video = os.path.join(self.temp_dir.name, "sample_video.mp4")
        with open(self.dummy_video, "w") as f:
            f.write("mock video")

        self.dummy_audio = os.path.join(self.temp_dir.name, "sample_audio.wav")
        with open(self.dummy_audio, "w") as f:
            f.write("mock audio")

        self.dummy_srt = os.path.join(self.temp_dir.name, "subtitles.srt")
        with open(self.dummy_srt, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:02,000\nTeste de legenda.\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("subprocess.run")
    def test_burn_subtitles(self, mock_subproc):
        """Testa se burn_subtitles executa o comando FFmpeg com o filtro subtitles correspondente."""
        mock_subproc.return_value = MagicMock(returncode=0)
        out_video = os.path.join(self.temp_dir.name, "burned.mp4")

        result = burn_subtitles(
            video_path=self.dummy_video,
            srt_path=self.dummy_srt,
            output_path=out_video,
        )

        self.assertEqual(result, str(Path(out_video).resolve()))
        mock_subproc.assert_called_once()
        cmd = mock_subproc.call_args[0][0]
        self.assertIn("-vf", cmd)
        vf_arg = cmd[cmd.index("-vf") + 1]
        self.assertIn("subtitles=", vf_arg)
        self.assertIn("-c:v", cmd)
        self.assertIn("libx264", cmd)

    @patch("subprocess.run")
    def test_mux_audio_video(self, mock_subproc):
        """Testa se mux_audio_video usa cópia direta de vídeo (-c:v copy) para velocidade sem perda."""
        mock_subproc.return_value = MagicMock(returncode=0)
        out_video = os.path.join(self.temp_dir.name, "muxed.mp4")

        result = mux_audio_video(
            video_path=self.dummy_video,
            audio_path=self.dummy_audio,
            output_path=out_video,
        )

        self.assertEqual(result, str(Path(out_video).resolve()))
        mock_subproc.assert_called_once()
        cmd = mock_subproc.call_args[0][0]
        self.assertIn("-c:v", cmd)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
        self.assertIn("-c:a", cmd)
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "aac")

    @patch("subprocess.run")
    def test_export_raw_package(self, mock_subproc):
        """Testa a geração do pacote bruto com áudio MP3 e arquivo SRT sincronizado."""
        mock_subproc.return_value = MagicMock(returncode=0)
        pkg_dir = os.path.join(self.temp_dir.name, "raw_package")

        pkg = export_raw_package(
            audio_path=self.dummy_audio,
            srt_path=self.dummy_srt,
            output_dir=pkg_dir,
            base_name="meu_video_pt",
        )

        self.assertIn("audio_mp3", pkg)
        self.assertIn("subtitles_srt", pkg)
        self.assertTrue(pkg["audio_mp3"].endswith(".mp3"))
        self.assertTrue(pkg["subtitles_srt"].endswith(".srt"))
        self.assertTrue(os.path.isfile(pkg["subtitles_srt"]))

    @patch("core.pipeline.extract_audio")
    @patch("core.pipeline.Transcriber")
    @patch("core.pipeline.Translator")
    @patch("core.pipeline.get_tts_engine")
    @patch("core.pipeline.assemble_final_video")
    def test_dub_pipeline_single_video(
        self,
        mock_assemble,
        mock_get_tts,
        mock_translator_cls,
        mock_transcriber_cls,
        mock_extract,
    ):
        """Testa o fluxo completo do DubPipeline para um único vídeo com tracking de callbacks."""
        # 1. Mock do extrator criando arquivo real de áudio
        def fake_extract(video_path, output_path, **kwargs):
            Path(output_path).write_text("dummy audio")
            return output_path
        mock_extract.side_effect = fake_extract

        # 2. Mock do Transcriber
        transcriber_inst = MagicMock()
        mock_transcriber_cls.return_value = transcriber_inst
        transcriber_inst.transcribe.return_value = [
            TranscriptionSegment(id=1, start=0.0, end=2.0, text="Hello world")
        ]

        # 3. Mock do Translator
        translator_inst = MagicMock()
        mock_translator_cls.return_value = translator_inst
        translator_inst.translate_segments.return_value = [
            TranslatedSegment(id=1, start=0.0, end=2.0, original_text="Hello world", translated_text="Olá mundo")
        ]

        # 4. Mock do TTS Engine
        tts_inst = MagicMock()
        mock_get_tts.return_value = tts_inst
        tts_inst.clone_and_align_all.return_value = [
            ClonedAudioSegment(id=1, start=0.0, end=2.0, audio_path=self.dummy_audio, target_duration=2.0, actual_duration=2.0)
        ]

        # 5. Mock da montagem final
        out_target = os.path.join(self.temp_dir.name, "final_dubbed.mp4")
        mock_assemble.return_value = out_target

        pipeline = DubPipeline(temp_dir=self.temp_dir.name)
        progress_events = []

        def on_prog(p: PipelineProgress):
            progress_events.append((p.stage, p.percentage))

        result = pipeline.process_video(
            input_video=self.dummy_video,
            target_language="pt",
            output_video=out_target,
            enable_lipsync=False,
            progress_callback=on_prog,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["output_video"], out_target)

        # Valida que as etapas emitiram progresso ordenado
        stages = [e[0] for e in progress_events]
        self.assertIn("audio_extract", stages)
        self.assertIn("transcribe", stages)
        self.assertIn("translate", stages)
        self.assertIn("voice_clone", stages)
        self.assertIn("assemble", stages)
        self.assertIn("done", stages)

        # Valida que o progresso final atingiu 100%
        self.assertEqual(progress_events[-1][1], 1.0)

    @patch("core.pipeline.DubPipeline.process_video")
    def test_dub_pipeline_batch_queue(self, mock_process_video):
        """Testa o processamento sequencial de fila com múltiplos vídeos x múltiplos idiomas."""
        mock_process_video.return_value = {
            "success": True,
            "output_video": "mock_output.mp4",
        }

        pipeline = DubPipeline(temp_dir=self.temp_dir.name)
        video_list = [self.dummy_video, os.path.join(self.temp_dir.name, "video2.mp4")]
        lang_list = ["pt", "es", "fr"]

        queue_events = []

        def on_queue_prog(cur, total, p: PipelineProgress):
            queue_events.append((cur, total, p.stage))

        # 2 vídeos x 3 idiomas = 6 trabalhos no total
        results = pipeline.process_batch_queue(
            video_paths=video_list,
            target_languages=lang_list,
            output_dir=self.temp_dir.name,
            queue_progress_callback=on_queue_prog,
        )

        self.assertEqual(len(results), 6)
        self.assertEqual(mock_process_video.call_count, 6)


if __name__ == "__main__":
    unittest.main()
