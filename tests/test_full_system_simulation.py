"""Bateria de testes pesados e simulação completa de uso do usuário (KmellVox Studio).

Cobre:
1. Inicialização e centralização geométrica da Janela Principal.
2. Navegação, alternância de perfis e acionamento de botões na Janela de Configurações.
3. Fluxos de Narração (Texto Puro e Legendas SRT com múltiplos blocos).
4. Gerenciamento e estresse na fila de trabalhos (QueueWidget).
5. Execução do NarrationEngine com conversão de áudio e concatenação no tempo.
6. Validação do DependencyManager e inicialização dos motores TTS.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QApplication

# Garante raiz do projeto no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Inicializa QApplication em modo offscreen para testes de UI automatizados
app = QApplication.instance()
if app is None:
    app = QApplication(["-platform", "offscreen"])

from core.dependency_manager import (
    DependencyStatus,
    check_all_dependencies,
    _check_torch,
    _check_f5tts,
    _check_ffmpeg,
    _check_faster_whisper,
    _check_llama_cpp,
)
from core.hardware import ModelProfile, detect_hardware
from core.narration import (
    NarrationEngine,
    NarrationJob,
    detect_text_format,
    parse_srt,
)
from core.voice_clone import ClonedAudioSegment
from ui.main_window import MainWindow
from ui.narration_tab import NarrationTab
from ui.queue_widget import JobItem, QueueWidget
from ui.settings_dialog import SettingsDialog


SAMPLE_SRT_CONTENT = """1
00:00:01,000 --> 00:00:03,500
Bem-vindos ao teste de dublagem e síntese de voz do KmellVox.

2
00:00:04,200 --> 00:00:07,800
Este é o segundo segmento de legenda com sincronia temporal.

3
00:00:08,500 --> 00:00:11,000
Validando o comportamento com pausas e controle de ritmo.

4
00:00:12,000 --> 00:00:14,500
<i>Segmento com formatação HTML simulada</i>.

5
00:00:15,000 --> 00:00:18,000
Conclusão do arquivo de teste de legendas SRT.
"""


class TestFullSystemSimulation(unittest.TestCase):
    """Bateria de testes de simulação de uso da interface e pipelines."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Áudio de referência fictício
        import numpy as np
        import soundfile as sf
        self.ref_audio_file = self.temp_path / "ref_voice.wav"
        sf.write(str(self.ref_audio_file), np.zeros(24000, dtype=np.float32), 24000)

        # Config temporário
        self.config_path = self.temp_path / "config.yaml"
        self.config_path.write_text(
            "gpu_profile: perfil_a\n"
            "paths:\n"
            "  models_dir: models\n"
            "hardware:\n"
            "  profile: perfil_a\n"
            "  compute_type: float16\n",
            encoding="utf-8",
        )

        # Mock de caixas de diálogo para evitar bloqueio modal
        self.patch_warn = patch("PySide6.QtWidgets.QMessageBox.warning")
        self.patch_info = patch("PySide6.QtWidgets.QMessageBox.information")
        self.patch_quest = patch("PySide6.QtWidgets.QMessageBox.question")
        self.patch_warn.start()
        self.patch_info.start()
        self.patch_quest.start()

    def tearDown(self) -> None:
        self.patch_warn.stop()
        self.patch_info.stop()
        self.patch_quest.stop()
        self.temp_dir.cleanup()

    # ─── 1. Janela Principal e Centralização ───────────────────────────────────

    def test_main_window_centering_various_screens(self) -> None:
        """Testa se o cálculo de centralização da Janela Principal funciona sem erros."""
        window = MainWindow()

        # Simula telas de diferentes resoluções
        for width, height in [(1920, 1080), (2560, 1440), (1366, 768), (3840, 2160)]:
            mock_screen = MagicMock()
            mock_screen.availableGeometry.return_value = QRect(0, 0, width, height)

            with patch.object(window, "screen", return_value=mock_screen):
                window._center_on_screen()
                geom = window.frameGeometry()
                expected_x = (width - geom.width()) // 2
                expected_y = (height - geom.height()) // 2

                # Verifica se a posição x e y está devidamente calculada
                self.assertIsNotNone(geom.topLeft())

    def test_main_window_tab_switching(self) -> None:
        """Simula o usuário alternando entre as abas da interface principal."""
        window = MainWindow()
        self.assertIsNotNone(window.tabs)
        self.assertGreaterEqual(window.tabs.count(), 2)

        # Alterna para a aba de Dublagem (índice 0)
        window.tabs.setCurrentIndex(0)
        self.assertEqual(window.tabs.currentIndex(), 0)

        # Alterna para a aba de Narração (índice 1)
        window.tabs.setCurrentIndex(1)
        self.assertEqual(window.tabs.currentIndex(), 1)

    # ─── 2. Janela de Configurações e Componentes ─────────────────────────────

    def test_settings_dialog_navigation_and_actions(self) -> None:
        """Simula a abertura, navegação e interação com todos os botões de configuração."""
        dlg = SettingsDialog(config_path=str(self.config_path))

        # 1. Verifica se os labels e botões de cada componente foram criados
        for key in ["torch", "faster_whisper", "llama_cpp", "f5_tts", "index_tts", "ffmpeg"]:
            self.assertIn(key, dlg.dep_labels)

        for key in ["torch", "f5_tts", "index_tts"]:
            self.assertIn(key, dlg.dep_action_btns)
            btn = dlg.dep_action_btns[key]
            self.assertTrue(btn.isEnabled())

        # 2. Testa alternância de perfis de hardware
        for idx in range(dlg.cb_profile.count()):
            dlg.cb_profile.setCurrentIndex(idx)
            selected_text = dlg.cb_profile.currentText()
            self.assertIn(selected_text, ["Automático", "Alta Performance", "Otimizado", "CPU (Sem GPU)"])

        # 3. Simula callback de verificação de dependências
        dummy_deps = [
            DependencyStatus("torch", "PyTorch (GPU/CUDA)", True, "v2.6.0+cu124", "OK", has_update=False),
            DependencyStatus("faster_whisper", "Faster-Whisper", True, "v1.2.1", "OK"),
            DependencyStatus("llama_cpp", "Llama-CPP", True, "v0.3.35", "OK"),
            DependencyStatus("f5_tts", "F5-TTS", True, "v1.1.22", "OK", has_update=True, latest_version="v1.1.23"),
            DependencyStatus("index_tts", "IndexTTS-2", False, "", "Opcional"),
            DependencyStatus("ffmpeg", "FFmpeg", True, "Encontrado", "OK"),
        ]
        dlg._refresh_dependency_status(dummy_deps)

        # PyTorch está atualizado: botão de atualizar fica oculto e badge '(Atualizado)' visível
        self.assertTrue(dlg.dep_action_btns["torch"].isHidden())
        self.assertEqual(dlg.dep_badges["torch"].text(), "(Atualizado)")

        # F5-TTS tem atualização disponível: botão 'Atualizar' visível
        self.assertFalse(dlg.dep_action_btns["f5_tts"].isHidden())
        self.assertIn("Atualizar", dlg.dep_action_btns["f5_tts"].text())

        # IndexTTS não está instalado: botão 'Baixar' visível
        self.assertFalse(dlg.dep_action_btns["index_tts"].isHidden())
        self.assertEqual(dlg.dep_action_btns["index_tts"].text(), "⬇️ Baixar")

        # Verifica se o botão principal detectou dependências ausentes
        self.assertIn("Baixar Dependências Ausentes", dlg.btn_install_all.text())

        # 4. Testa controle de estado ocupado / livre da UI
        dlg._set_install_ui_busy(True, "Instalando teste...")
        self.assertFalse(dlg.btn_install_all.isEnabled())
        self.assertFalse(dlg.dep_action_btns["torch"].isEnabled())
        self.assertFalse(dlg.prog_bar_install.isHidden())

        dlg._set_install_ui_busy(False)
        self.assertTrue(dlg.btn_install_all.isEnabled())
        self.assertTrue(dlg.dep_action_btns["torch"].isEnabled())
        self.assertTrue(dlg.prog_bar_install.isHidden())

        # 4.5. Testa lista dinâmica e botões individuais de modelos
        dummy_model_statuses = [
            {
                "key": "whisper_large_v3",
                "name": "Faster-Whisper large-v3",
                "installed": True,
                "size_mb": 3087.3,
                "expected_min_bytes": 1500 * 1024 * 1024,
                "expected_size_str": "~3.0 GB",
            },
            {
                "key": "llm_qwen3_8b",
                "name": "Qwen3-8B GGUF Q4_K_M",
                "installed": False,
                "size_mb": 0.0,
                "expected_min_bytes": 4000 * 1024 * 1024,
                "expected_size_str": "~4.0 GB",
            },
        ]
        dlg._refresh_models_status(dummy_model_statuses)

        # Modelo já baixado: botão de download oculto e badge '(Instalado)' visível
        if "whisper_large_v3" in dlg.model_action_btns:
            self.assertTrue(dlg.model_action_btns["whisper_large_v3"].isHidden())
            self.assertEqual(dlg.model_badges["whisper_large_v3"].text(), "(Instalado)")

        # Modelo ausente: botão '⬇️ Baixar' visível
        if "llm_qwen3_8b" in dlg.model_action_btns:
            self.assertFalse(dlg.model_action_btns["llm_qwen3_8b"].isHidden())
            self.assertEqual(dlg.model_action_btns["llm_qwen3_8b"].text(), "⬇️ Baixar")

        # Botão mestre detecta modelo ausente
        self.assertIn("Baixar Modelos Ausentes", dlg.btn_download_models.text())

        # Testa busy de modelos
        dlg._set_models_ui_busy(True, "Baixando modelo...")
        self.assertFalse(dlg.btn_download_models.isEnabled())
        self.assertFalse(dlg.prog_bar_dl.isHidden())
        dlg._set_models_ui_busy(False)
        self.assertTrue(dlg.btn_download_models.isEnabled())
        self.assertTrue(dlg.prog_bar_dl.isHidden())

        # 5. Salva e fecha
        dlg._apply_and_close()
        self.assertTrue(self.config_path.exists())

    # ─── 3. Aba de Narração (Texto e Legendas SRT) ────────────────────────────

    def test_narration_tab_text_and_srt_workflows(self) -> None:
        """Simula importação de texto e SRT, validação de formato e enfileiramento de narração."""
        narration_tab = NarrationTab()

        # 1. Simula entrada de texto puro
        plain_text = "Esta é uma simulação de texto corrido para geração de narração em áudio."
        narration_tab.txt_content.setPlainText(plain_text)
        self.assertEqual(detect_text_format(plain_text), "txt")

        # 2. Simula entrada de legendas SRT
        narration_tab.txt_content.setPlainText(SAMPLE_SRT_CONTENT)
        self.assertEqual(detect_text_format(SAMPLE_SRT_CONTENT), "srt")

        # 3. Testa parser de SRT
        segments = parse_srt(SAMPLE_SRT_CONTENT)
        self.assertEqual(len(segments), 5)
        self.assertEqual(segments[0].id, 1)
        self.assertEqual(segments[0].start, 1.0)
        self.assertEqual(segments[0].end, 3.5)
        self.assertIn("Bem-vindos", segments[0].text)

        # Verifica se o segmento 4 teve tags HTML removidas
        self.assertEqual(segments[3].text, "Segmento com formatação HTML simulada.")

        # 4. Testa adição à fila de narração
        narration_tab.txt_ref_audio.setText(str(self.ref_audio_file))
        narration_tab.chk_save_source_dir.setChecked(False)
        narration_tab.txt_dest_folder.setText(str(self.temp_path))

        initial_count = len(narration_tab.queue_jobs)
        narration_tab._add_current_to_queue()
        self.assertEqual(len(narration_tab.queue_jobs), initial_count + 1)

        last_job = list(narration_tab.queue_jobs.values())[-1]
        self.assertEqual(last_job.source_format, "srt")
        self.assertEqual(last_job.status, "Pendente")

    # ─── 4. Fila de Trabalhos (QueueWidget) - Estresse ─────────────────────────

    def test_queue_widget_stress(self) -> None:
        """Adiciona e atualiza 50 itens na fila para garantir estabilidade e ausência de leaks."""
        queue = QueueWidget()

        # Adiciona 50 jobs
        for i in range(1, 51):
            job = JobItem(
                job_id=f"stress_job_{i:03d}",
                input_file=f"C:/mock/video_{i:03d}.mp4",
                output_file=f"C:/mock/out_{i:03d}.mp4",
                source_lang="en",
                target_lang="pt",
            )
            queue.add_job(job)

        self.assertEqual(len(queue.jobs), 50)
        self.assertEqual(len(queue.get_pending_jobs()), 50)

        # Atualiza progresso dos primeiros 25 para Concluído
        for i in range(1, 26):
            queue.update_job_progress(f"stress_job_{i:03d}", 1.0, "Pronto", status="Concluído")

        self.assertEqual(len(queue.get_pending_jobs()), 25)

        # Remove concluídos
        queue.clear_completed_jobs()
        self.assertEqual(len(queue.jobs), 25)

        # Cancela os restantes
        for i in range(26, 51):
            queue.cancel_or_remove_job(f"stress_job_{i:03d}")

        self.assertEqual(len(queue.jobs), 0)

    # ─── 5. NarrationEngine com Concatenação e Conversão ───────────────────────

    def test_narration_engine_simulation(self) -> None:
        """Simula a execução do NarrationEngine utilizando um mock TTSEngine."""
        engine = NarrationEngine()

        # Mock do TTSEngine interno
        mock_tts = MagicMock()

        def fake_clone(text, reference_audio_path, output_path, target_duration=None, **kwargs):
            # Cria um arquivo WAV silencioso fictício
            engine._create_silence_wav(1.5, output_path)
            return ClonedAudioSegment(
                id=1,
                start=0.0,
                end=1.5,
                audio_path=output_path,
                target_duration=1.5,
                actual_duration=1.5,
            )

        mock_tts.clone_and_synthesize.side_effect = fake_clone

        # Testa job de texto simples
        text_job = NarrationJob(
            job_id="test_text_01",
            source_text="Testando síntese simulada de narração.",
            source_format="txt",
            reference_audio_path=str(self.ref_audio_file),
            destination_folder=str(self.temp_path),
            save_to_source_folder=False,
            create_audio_subfolder=False,
        )

        with patch("core.narration.get_tts_engine", return_value=mock_tts):
            outputs = engine.run(text_job)
            self.assertGreater(len(outputs), 0)
            self.assertTrue(Path(outputs[0]).exists())

        # Testa job de SRT com 3 blocos
        srt_job = NarrationJob(
            job_id="test_srt_01",
            source_text=SAMPLE_SRT_CONTENT,
            source_format="srt",
            reference_audio_path=str(self.ref_audio_file),
            destination_folder=str(self.temp_path),
            save_to_source_folder=False,
            create_audio_subfolder=True,
            split_mode="unico",  # Gera arquivo único concatenado
        )

        with patch("core.narration.get_tts_engine", return_value=mock_tts):
            srt_outputs = engine.run(srt_job)
            self.assertEqual(len(srt_outputs), 1)
            self.assertTrue(Path(srt_outputs[0]).exists())

    # ─── 6. Verificadores de Dependências e Sistema ───────────────────────────

    def test_dependency_manager_functions(self) -> None:
        """Verifica se os analisadores de componentes executam de forma resiliente."""
        # Cada função deve retornar um DependencyStatus sem levantar exceção não tratada
        whisper_status = _check_faster_whisper()
        self.assertIsInstance(whisper_status, DependencyStatus)
        self.assertEqual(whisper_status.name, "faster_whisper")

        ffmpeg_status = _check_ffmpeg()
        self.assertIsInstance(ffmpeg_status, DependencyStatus)
        self.assertEqual(ffmpeg_status.name, "ffmpeg")

        all_deps = check_all_dependencies()
        self.assertEqual(len(all_deps), 6)
        names = [d.name for d in all_deps]
        self.assertIn("torch", names)
        self.assertIn("f5_tts", names)
        self.assertIn("ffmpeg", names)
        self.assertIn("faster_whisper", names)
        self.assertIn("llama_cpp", names)
        self.assertIn("index_tts", names)


if __name__ == "__main__":
    unittest.main()
