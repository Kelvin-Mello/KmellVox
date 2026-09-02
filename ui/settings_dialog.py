"""Diálogo de Configurações e Preferências do KmellVox com downloader de modelos integrado (PySide6)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.hardware import ModelProfile, detect_gpu_profile, detect_hardware
from downloader.fetch_models import fetch_models_for_profile

logger = logging.getLogger("KmellVox.Settings")


class ModelDownloaderWorker(QThread):
    """Thread em segundo plano para download de modelos sem congelar o diálogo."""

    progress_signal = Signal(float, str)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, profile: str, models_dir: str, config_path: str, parent=None) -> None:
        super().__init__(parent)
        self.profile = profile
        self.models_dir = models_dir
        self.config_path = config_path

    def run(self) -> None:
        try:
            def on_progress(pct: float, msg: str) -> None:
                self.progress_signal.emit(pct, msg)

            saved_paths = fetch_models_for_profile(
                profile=self.profile,
                base_models_dir=self.models_dir,
                config_path=self.config_path,
                progress_callback=on_progress,
            )
            self.finished_signal.emit(saved_paths)
        except Exception as e:
            self.error_signal.emit(str(e))


class SettingsDialog(QDialog):
    """Diálogo para visualização de hardware, download de pesos e edição de parâmetros do config.yaml."""

    def __init__(self, config_path: str = "config.yaml", parent=None) -> None:
        super().__init__(parent)
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = self._load_config()
        self.setWindowTitle("Configurações & Modelos - KmellVox")
        self.resize(600, 580)
        self.downloader_worker: Optional[ModelDownloaderWorker] = None
        self._init_ui()

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.is_file():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error("Erro ao carregar %s: %s", self.config_path, e)
        return {}

    def _save_config(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config_data, f, allow_unicode=True, default_flow_style=False)
            logger.info("Configurações salvas com sucesso em %s", self.config_path)
        except Exception as e:
            logger.error("Erro ao salvar %s: %s", self.config_path, e)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Hardware & Perfil de GPU
        hw_group = QGroupBox("Hardware e Perfil de Execução")
        hw_layout = QFormLayout(hw_group)

        hw_info = detect_hardware()
        gpu_label = f"{hw_info.device_name} ({hw_info.vram_total_gb:.1f} GB VRAM)" if hw_info.cuda_available else "Nenhuma GPU CUDA compatível (Modo CPU)"
        lbl_hw_detect = QLabel(gpu_label)
        lbl_hw_detect.setStyleSheet("font-weight: bold; color: #04D361;" if hw_info.cuda_available else "font-weight: bold; color: #FFA200;")
        hw_layout.addRow("Dispositivo Detectado:", lbl_hw_detect)

        # ComboBox para forçar manualmente o perfil (perfil_a, perfil_b, cpu)
        self.cb_profile = QComboBox()
        self.cb_profile.addItems(["auto", "perfil_a", "perfil_b", "cpu"])
        
        saved_profile = self.config_data.get("gpu_profile") or self.config_data.get("hardware", {}).get("profile", "auto")
        self.cb_profile.setCurrentText(saved_profile)
        self.cb_profile.currentIndexChanged.connect(self._on_profile_changed)
        hw_layout.addRow("Perfil de Hardware (VRAM):", self.cb_profile)

        self.lbl_profile_details = QLabel()
        self.lbl_profile_details.setStyleSheet("color: #A8A8B3; font-size: 11px;")
        hw_layout.addRow("Modelos Atribuídos:", self.lbl_profile_details)
        self._update_profile_details()

        layout.addWidget(hw_group)

        # 2. Gerenciamento e Download de Modelos
        models_group = QGroupBox("Armazenamento e Download de Modelos")
        models_layout = QVBoxLayout(models_group)

        row_md = QHBoxLayout()
        row_md.addWidget(QLabel("Diretório de Modelos:"))
        self.txt_models = QLineEdit()
        self.txt_models.setText(self.config_data.get("paths", {}).get("models_dir", "models"))
        btn_models = QPushButton("Procurar...")
        btn_models.clicked.connect(self._browse_models_dir)
        row_md.addWidget(self.txt_models)
        row_md.addWidget(btn_models)
        models_layout.addLayout(row_md)

        # Botão para baixar / atualizar modelos conforme perfil
        row_dl = QHBoxLayout()
        self.btn_download_models = QPushButton("📥 Baixar / Atualizar Modelos")
        self.btn_download_models.setStyleSheet("background-color: #04D361; color: #121214; font-weight: bold; padding: 6px 14px;")
        self.btn_download_models.clicked.connect(self._start_model_download)
        row_dl.addWidget(self.btn_download_models)
        row_dl.addStretch()
        models_layout.addLayout(row_dl)

        # Progresso de download
        self.lbl_dl_status = QLabel("Status: Modelos prontos para verificação.")
        self.lbl_dl_status.setStyleSheet("color: #A8A8B3; font-size: 11px;")
        models_layout.addWidget(self.lbl_dl_status)

        self.prog_bar_dl = QProgressBar()
        self.prog_bar_dl.setRange(0, 100)
        self.prog_bar_dl.setValue(0)
        self.prog_bar_dl.setFixedHeight(16)
        models_layout.addWidget(self.prog_bar_dl)

        layout.addWidget(models_group)

        # 3. Ferramentas e Executáveis
        tools_group = QGroupBox("Caminhos de Ferramentas de Sistema")
        tools_layout = QFormLayout(tools_group)

        self.txt_ffmpeg = QLineEdit()
        self.txt_ffmpeg.setText(self.config_data.get("paths", {}).get("ffmpeg_bin", "tools/ffmpeg/bin/ffmpeg.exe"))
        btn_ffmpeg = QPushButton("Procurar...")
        btn_ffmpeg.clicked.connect(self._browse_ffmpeg)
        row_ff = QHBoxLayout()
        row_ff.addWidget(self.txt_ffmpeg)
        row_ff.addWidget(btn_ffmpeg)
        tools_layout.addRow("Executável FFmpeg:", row_ff)

        layout.addWidget(tools_group)

        # 4. Preferências de Pipeline
        pref_group = QGroupBox("Preferências Gerais do Pipeline")
        pref_layout = QFormLayout(pref_group)

        self.chk_burn_subs = QCheckBox("Estampar legendas no vídeo final (Burn-in) por padrão")
        self.chk_burn_subs.setChecked(self.config_data.get("pipeline", {}).get("subtitle_burn_in", False))
        pref_layout.addRow("", self.chk_burn_subs)

        self.chk_keep_temp = QCheckBox("Manter arquivos temporários de trabalho após conclusão")
        self.chk_keep_temp.setChecked(self.config_data.get("pipeline", {}).get("keep_temp_files", False))
        pref_layout.addRow("", self.chk_keep_temp)

        layout.addWidget(pref_group)

        # Botões Inferiores
        btn_box = QHBoxLayout()
        btn_redetect = QPushButton("🔄 Redetectar GPU")
        btn_redetect.clicked.connect(self._redetect_gpu)
        btn_box.addWidget(btn_redetect)

        btn_box.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_save = QPushButton("Salvar Alterações")
        btn_save.setStyleSheet("background-color: #8257E5; color: white; padding: 6px 16px; font-weight: bold;")
        btn_save.clicked.connect(self._apply_and_close)
        btn_box.addWidget(btn_save)

        layout.addLayout(btn_box)

    def _on_profile_changed(self) -> None:
        self._update_profile_details()

    def _update_profile_details(self) -> None:
        prof_name = self.cb_profile.currentText()
        mp = ModelProfile.from_profile(prof_name if prof_name != "auto" else None)
        indextts_txt = "Sim (Habilitado)" if mp.enable_indextts_2 else "Não (F5-TTS Padrão)"
        fp16_txt = "Sim (Obrigatório)" if mp.musetalk_use_float16 else "Opcional"
        info = (
            f"• Whisper: {mp.whisper_variant} ({mp.whisper_compute_type})\n"
            f"• Tradutor LLM: {mp.translation_model}\n"
            f"• IndexTTS-2 Habilitado: {indextts_txt}\n"
            f"• MuseTalk FP16: {fp16_txt}"
        )
        self.lbl_profile_details.setText(info)

    def _browse_ffmpeg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Executável FFmpeg", "", "Executáveis (*.exe);;Todos (*.*)")
        if path:
            self.txt_ffmpeg.setText(path)

    def _browse_models_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Selecionar Diretório Onde os Modelos Estão Instalados")
        if dir_path:
            self.txt_models.setText(dir_path)

    def _redetect_gpu(self) -> None:
        detected = detect_gpu_profile(config_path=str(self.config_path), force_redetect=True)
        self.cb_profile.setCurrentText(detected)
        self._update_profile_details()
        QMessageBox.information(self, "Detecção Concluída", f"Perfil detectado com base na VRAM: {detected}")

    def _start_model_download(self) -> None:
        """Inicia o download e atualização de modelos em background."""
        selected_prof = self.cb_profile.currentText()
        actual_profile = selected_prof if selected_prof != "auto" else detect_gpu_profile(config_path=str(self.config_path))
        models_dir = self.txt_models.text().strip() or "models"

        self.btn_download_models.setEnabled(False)
        self.prog_bar_dl.setValue(0)
        self.lbl_dl_status.setText(f"Iniciando download para perfil '{actual_profile}'...")

        self.downloader_worker = ModelDownloaderWorker(
            profile=actual_profile,
            models_dir=models_dir,
            config_path=str(self.config_path),
            parent=self,
        )
        self.downloader_worker.progress_signal.connect(self._on_dl_progress)
        self.downloader_worker.finished_signal.connect(self._on_dl_finished)
        self.downloader_worker.error_signal.connect(self._on_dl_error)
        self.downloader_worker.start()

    def _on_dl_progress(self, pct: float, msg: str) -> None:
        self.prog_bar_dl.setValue(int(pct * 100))
        self.lbl_dl_status.setText(f"{int(pct * 100)}% - {msg}")

    def _on_dl_finished(self, saved_paths: dict) -> None:
        self.prog_bar_dl.setValue(100)
        self.lbl_dl_status.setText("✅ Todos os modelos foram verificados e salvos com sucesso.")
        self.btn_download_models.setEnabled(True)
        self.downloader_worker = None
        QMessageBox.information(
            self,
            "Download Concluído",
            f"Todos os modelos necessários para o perfil '{self.cb_profile.currentText()}' estão prontos!",
        )

    def _on_dl_error(self, err: str) -> None:
        self.lbl_dl_status.setText(f"❌ Erro no download: {err}")
        self.btn_download_models.setEnabled(True)
        self.downloader_worker = None
        QMessageBox.warning(self, "Falha no Download", f"Ocorreu um erro durante o download dos modelos:\n{err}")

    def _apply_and_close(self) -> None:
        selected_prof = self.cb_profile.currentText()
        if selected_prof == "auto":
            selected_prof = detect_gpu_profile(config_path=str(self.config_path), force_redetect=True)

        hw_info = detect_hardware(force_profile=selected_prof, config_path=str(self.config_path))
        self.config_data["gpu_profile"] = hw_info.gpu_profile

        if "hardware" not in self.config_data:
            self.config_data["hardware"] = {}
        self.config_data["hardware"]["profile"] = hw_info.gpu_profile
        self.config_data["hardware"]["compute_type"] = hw_info.recommended_compute_type
        self.config_data["hardware"]["device"] = "cuda" if hw_info.cuda_available and hw_info.gpu_profile != "cpu" else "cpu"
        self.config_data["hardware"]["device_name"] = hw_info.device_name
        self.config_data["hardware"]["vram_detected_gb"] = round(hw_info.vram_total_gb, 2)

        if "paths" not in self.config_data:
            self.config_data["paths"] = {}
        self.config_data["paths"]["ffmpeg_bin"] = self.txt_ffmpeg.text()
        self.config_data["paths"]["models_dir"] = self.txt_models.text()

        if "pipeline" not in self.config_data:
            self.config_data["pipeline"] = {}
        self.config_data["pipeline"]["subtitle_burn_in"] = self.chk_burn_subs.isChecked()
        self.config_data["pipeline"]["keep_temp_files"] = self.chk_keep_temp.isChecked()

        self._save_config()
        self.accept()
