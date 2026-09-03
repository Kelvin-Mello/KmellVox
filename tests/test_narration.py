"""Testes unitários para core/narration.py e ui/narration_tab.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hardware import ModelProfile
from core.narration import (
    NarrationEngine,
    NarrationJob,
    detect_text_format,
    list_preset_voices,
    parse_srt,
    slugify_text,
)
from ui.narration_tab import NarrationTab

app = QApplication.instance()
if app is None:
    app = QApplication(["-platform", "offscreen"])


SAMPLE_SRT_TEXT = """1
00:00:01,000 --> 00:00:03,500
Olá, bem-vindo ao teste do KmellVox.

2
00:00:05,000 --> 00:00:08,200
Este é o segundo trecho com pausa.
"""

SAMPLE_PLAIN_TEXT = """
Este é um texto de narração contínua sem nenhum timestamp ou numeração de legendas.
"""


class TestNarrationCoreAndUI(unittest.TestCase):
    """Testes para o motor de narração e para a interface da aba de narração."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # Cria arquivo de áudio de teste
        self.dummy_audio = os.path.join(self.temp_dir.name, "ref_voice.wav")
        with open(self.dummy_audio, "wb") as f:
            f.write(b"RIFF dummy audio data")

        # Cria arquivo de texto de teste
        self.dummy_txt = os.path.join(self.temp_dir.name, "sample.txt")
        with open(self.dummy_txt, "w", encoding="utf-8") as f:
            f.write(SAMPLE_PLAIN_TEXT)

        # Cria arquivo SRT de teste
        self.dummy_srt = os.path.join(self.temp_dir.name, "sample.srt")
        with open(self.dummy_srt, "w", encoding="utf-8") as f:
            f.write(SAMPLE_SRT_TEXT)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detect_text_format(self):
        """Testa a identificação automática entre formato SRT e texto puro."""
        self.assertEqual(detect_text_format(SAMPLE_SRT_TEXT), "srt")
        self.assertEqual(detect_text_format(SAMPLE_PLAIN_TEXT), "txt")
        self.assertEqual(detect_text_format(""), "txt")
        self.assertEqual(detect_text_format("00:01:23,456 --> 00:01:25,789\nFala."), "srt")

    def test_parse_srt(self):
        """Testa o parsing de blocos SRT em segmentos estruturados com timestamps."""
        segments = parse_srt(SAMPLE_SRT_TEXT)
        self.assertEqual(len(segments), 2)

        self.assertEqual(segments[0].id, 1)
        self.assertAlmostEqual(segments[0].start, 1.0)
        self.assertAlmostEqual(segments[0].end, 3.5)
        self.assertIn("Olá, bem-vindo", segments[0].text)

        self.assertEqual(segments[1].id, 2)
        self.assertAlmostEqual(segments[1].start, 5.0)
        self.assertAlmostEqual(segments[1].end, 8.2)
        self.assertIn("segundo trecho", segments[1].text)

    def test_slugify_text(self):
        """Testa geração de slugs para nomes de arquivos."""
        slug = slugify_text("Olá Mundo! Este é um Teste de Áudio.", max_words=3)
        self.assertEqual(slug, "ola_mundo_este")

    def test_list_preset_voices(self):
        """Testa a descoberta de vozes pré-definidas."""
        # Cria pasta fictícia com preset
        presets_dir = Path(self.temp_dir.name) / "tts" / "presets"
        presets_dir.mkdir(parents=True, exist_ok=True)
        (presets_dir / "voz_narrador.wav").write_bytes(b"dummy")

        voices = list_preset_voices(models_dir=self.temp_dir.name)
        matched = [v for v in voices if v["id"] == "voz_narrador"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["id"], "voz_narrador")
        self.assertIn("Voz Preset", matched[0]["label"])


    def test_resolve_destination_folder(self):
        """Testa as regras de resolução de diretório de destino."""
        engine = NarrationEngine()

        # 1. Salvar na mesma pasta de origem do arquivo
        job1 = NarrationJob(
            job_id="j1",
            source_text="teste",
            source_file_path=self.dummy_txt,
            save_to_source_folder=True,
            create_audio_subfolder=False,
        )
        dest1 = engine.resolve_destination_folder(job1)
        self.assertEqual(str(dest1), str(Path(self.dummy_txt).parent))

        # 2. Salvar na mesma pasta de origem com subpasta 'Áudio'
        job2 = NarrationJob(
            job_id="j2",
            source_text="teste",
            source_file_path=self.dummy_txt,
            save_to_source_folder=True,
            create_audio_subfolder=True,
        )
        dest2 = engine.resolve_destination_folder(job2)
        self.assertEqual(str(dest2), str(Path(self.dummy_txt).parent / "Áudio"))

        # 3. Salvar em pasta personalizada
        custom_folder = os.path.join(self.temp_dir.name, "minha_saida")
        job3 = NarrationJob(
            job_id="j3",
            source_text="teste",
            source_file_path=None,
            destination_folder=custom_folder,
            save_to_source_folder=False,
            create_audio_subfolder=True,
        )
        dest3 = engine.resolve_destination_folder(job3)
        self.assertEqual(str(dest3), str(Path(custom_folder) / "Áudio"))

    @patch("core.narration.get_tts_engine")
    @patch.object(NarrationEngine, "_convert_to_mp3")
    def test_narration_engine_plain_text(self, mock_mp3, mock_get_tts):
        """Testa a geração de narração a partir de texto puro."""
        mock_tts = MagicMock()
        mock_get_tts.return_value = mock_tts

        def fake_mp3(in_wav, out_mp3, **kw):
            Path(out_mp3).write_text("mp3 content")
            return out_mp3
        mock_mp3.side_effect = fake_mp3

        engine = NarrationEngine()
        job = NarrationJob(
            job_id="job_txt",
            source_text=SAMPLE_PLAIN_TEXT,
            source_format="txt",
            source_file_path=self.dummy_txt,
            voice_mode="clone",
            reference_audio_path=self.dummy_audio,
            save_to_source_folder=True,
        )

        outputs = engine.run(job)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].endswith(".mp3"))
        mock_tts.clone_and_synthesize.assert_called_once()
        mock_tts.unload_model.assert_called_once()

    @patch("core.narration.get_tts_engine")
    @patch.object(NarrationEngine, "_convert_to_mp3")
    def test_narration_engine_srt_separate_files(self, mock_mp3, mock_get_tts):
        """Testa geração de arquivos separados por trecho de SRT."""
        mock_tts = MagicMock()
        mock_get_tts.return_value = mock_tts

        def fake_mp3(in_wav, out_mp3, **kw):
            Path(out_mp3).write_text("mp3 content")
            return out_mp3
        mock_mp3.side_effect = fake_mp3

        engine = NarrationEngine()
        job = NarrationJob(
            job_id="job_srt_sep",
            source_text=SAMPLE_SRT_TEXT,
            source_format="srt",
            source_file_path=self.dummy_srt,
            voice_mode="clone",
            reference_audio_path=self.dummy_audio,
            split_mode="separado",
            save_to_source_folder=True,
        )

        outputs = engine.run(job)
        self.assertEqual(len(outputs), 2)
        self.assertTrue(any("001_" in o for o in outputs))
        self.assertTrue(any("002_" in o for o in outputs))
        self.assertEqual(mock_tts.clone_and_synthesize.call_count, 2)

    @patch("core.narration.get_tts_engine")
    @patch.object(NarrationEngine, "_concat_audio_segments")
    @patch.object(NarrationEngine, "_create_silence_wav")
    def test_narration_engine_srt_single_file_with_silence(self, mock_silence, mock_concat, mock_get_tts):
        """Testa geração de áudio único contínuo para SRT com inserção proporcional de silêncio."""
        mock_tts = MagicMock()
        mock_get_tts.return_value = mock_tts
        mock_concat.side_effect = lambda pieces, out_mp3, *a, **kw: out_mp3

        engine = NarrationEngine()
        job = NarrationJob(
            job_id="job_srt_single",
            source_text=SAMPLE_SRT_TEXT,
            source_format="srt",
            source_file_path=self.dummy_srt,
            voice_mode="clone",
            reference_audio_path=self.dummy_audio,
            split_mode="unico",
            save_to_source_folder=True,
        )

        outputs = engine.run(job)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].endswith(".mp3"))
        mock_concat.assert_called_once()

    def test_narration_tab_ui_interactions(self):
        """Testa interações e comportamentos na interface NarrationTab."""
        tab = NarrationTab()

        # 1. Digita texto puro -> Formato TXT, opções SRT ocultas
        tab.txt_content.setPlainText("Texto simples para narrar.")
        self.assertIn("Texto Puro", tab.lbl_format_detected.text())
        self.assertTrue(tab.grp_srt_options.isHidden())

        # 2. Cola conteúdo SRT -> Formato SRT, opções SRT visíveis
        tab.txt_content.setPlainText(SAMPLE_SRT_TEXT)
        self.assertIn("Legenda (.srt)", tab.lbl_format_detected.text())
        self.assertFalse(tab.grp_srt_options.isHidden())

        # 3. Adiciona à fila com referência de áudio
        tab.txt_ref_audio.setText(self.dummy_audio)
        tab._add_current_to_queue()
        self.assertEqual(len(tab.queue_jobs), 1)
        self.assertEqual(tab.table_queue.rowCount(), 1)

        # 4. Limpa fila
        tab._clear_all_jobs()
        self.assertEqual(len(tab.queue_jobs), 0)
        self.assertEqual(tab.table_queue.rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
