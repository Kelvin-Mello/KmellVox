"""Testes unitários para a interface gráfica PySide6 (MainWindow, QueueWidget, SettingsDialog, PipelineWorkerThread)."""

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
from ui.main_window import DropArea, MainWindow, PipelineWorkerThread
from ui.queue_widget import JobItem, QueueWidget
from ui.settings_dialog import SettingsDialog

# Inicializa QApplication em modo offscreen para testes de UI automatizados
app = QApplication.instance()
if app is None:
    app = QApplication(["-platform", "offscreen"])


class TestUIComponents(unittest.TestCase):
    """Conjunto de testes para os componentes e janelas da interface PySide6."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_main_window_initialization_and_widgets(self):
        """Testa se a Janela Principal inicializa todos os painéis e widgets requeridos."""
        window = MainWindow()

        # 1. Área de Drop
        self.assertIsInstance(window.drop_area, DropArea)

        # 2. Lista de Idiomas com multi-seleção
        self.assertGreater(window.list_languages.count(), 0)
        selected_langs = window._get_selected_languages()
        self.assertIn("pt", selected_langs)

        # 3. Painel de Opções (Sincronia Labial, Legendas, Formato de Saída, Sliders de Ritmo)
        self.assertIsNotNone(window.chk_lipsync)
        self.assertFalse(window.chk_lipsync.isChecked())  # Experimental, desmarcado por padrão
        self.assertIsNotNone(window.chk_burn_subs)
        self.assertIsNotNone(window.cb_output_format)

        # 4. Sliders de Ritmo
        self.assertEqual(window.slider_min_speed.value(), 70)
        self.assertEqual(window.slider_max_speed.value(), 135)

        # 5. Barras de Progresso
        self.assertIsNotNone(window.prog_bar_current)
        self.assertIsNotNone(window.prog_bar_queue)
        self.assertIsNotNone(window.lbl_current_stage)

        # 6. Fila de Trabalhos
        self.assertIsInstance(window.queue_widget, QueueWidget)

    def test_queue_widget_job_management(self):
        """Testa adição, atualização de progresso e cancelamento de itens no QueueWidget."""
        queue = QueueWidget()

        job1 = JobItem(
            job_id="job_001",
            input_file="C:/test/video1.mp4",
            output_file="C:/test/video1_pt.mp4",
            source_lang="en",
            target_lang="pt",
        )

        job2 = JobItem(
            job_id="job_002",
            input_file="C:/test/video2.mp4",
            output_file="C:/test/video2_es.mp4",
            source_lang="en",
            target_lang="es",
        )

        queue.add_job(job1)
        queue.add_job(job2)
        self.assertEqual(len(queue.jobs), 2)
        self.assertEqual(len(queue.get_pending_jobs()), 2)

        # Atualiza progresso do job1
        queue.update_job_progress("job_001", 0.50, "Traduzindo", status="Processando")
        self.assertEqual(queue.jobs["job_001"].status, "Processando")

        # Conclui job1
        queue.update_job_progress("job_001", 1.0, "Concluído", status="Concluído")
        self.assertEqual(len(queue.get_pending_jobs()), 1)

        # Remove concluídos
        queue.clear_completed_jobs()
        self.assertEqual(len(queue.jobs), 1)
        self.assertIn("job_002", queue.jobs)

        # Cancela / Remove job2
        queue.cancel_or_remove_job("job_002")
        self.assertEqual(len(queue.jobs), 0)

    def test_settings_dialog_save_and_load(self):
        """Testa diálogo de configurações salvando e forçando perfis no YAML."""
        dummy_config = os.path.join(self.temp_dir.name, "config.yaml")
        with open(dummy_config, "w", encoding="utf-8") as f:
            f.write("gpu_profile: perfil_b\npaths:\n  models_dir: models\n")

        dlg = SettingsDialog(config_path=dummy_config)
        self.assertEqual(dlg.cb_profile.currentText(), "perfil_b")

        # Altera para perfil_a e salva
        dlg.cb_profile.setCurrentText("perfil_a")
        dlg.txt_models.setText("custom_models_dir")
        dlg._apply_and_close()

        # Recarrega para validar persistência
        dlg2 = SettingsDialog(config_path=dummy_config)
        self.assertEqual(dlg2.cb_profile.currentText(), "perfil_a")
        self.assertEqual(dlg2.txt_models.text(), "custom_models_dir")

    def test_indextts2_toggle_tooltip_hardware_rules(self):
        """Testa regras do toggle IndexTTS-2 com tooltip em perfil_a vs perfil_b."""
        window = MainWindow()

        # Simula perfil_b -> deve desabilitar IndexTTS-2 com tooltip explicativo
        prof_b = ModelProfile.from_profile("perfil_b")
        with patch("ui.main_window.detect_hardware") as mock_hw:
            mock_inst = MagicMock()
            mock_inst.cuda_available = True
            mock_inst.device_name = "NVIDIA RTX 3060"
            mock_inst.vram_total_gb = 6.0
            mock_inst.profile.value = "perfil_b"
            mock_inst.model_profile = prof_b
            mock_hw.return_value = mock_inst

            window._refresh_hardware_status()
            self.assertFalse(window.chk_indextts2.isEnabled())
            self.assertIn("8GB de VRAM", window.chk_indextts2.toolTip())

        # Simula perfil_a -> deve habilitar IndexTTS-2
        prof_a = ModelProfile.from_profile("perfil_a")
        with patch("ui.main_window.detect_hardware") as mock_hw:
            mock_inst = MagicMock()
            mock_inst.cuda_available = True
            mock_inst.device_name = "NVIDIA RTX 4090"
            mock_inst.vram_total_gb = 16.0
            mock_inst.profile.value = "perfil_a"
            mock_inst.model_profile = prof_a
            mock_hw.return_value = mock_inst

            window._refresh_hardware_status()
            self.assertTrue(window.chk_indextts2.isEnabled())


if __name__ == "__main__":
    unittest.main()
