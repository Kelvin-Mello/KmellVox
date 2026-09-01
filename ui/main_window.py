"""Janela Principal da Aplicação KmellVox em PySide6.

Inclui:
- Área de arrastar e soltar (drag-and-drop) para vídeos.
- Lista multi-seleção de idiomas de destino para lote (N vídeos x M idiomas).
- Painel de opções (Sincronia Labial experimental, Legendas, Formato de Saída, Controle de Ritmo).
- Toggle de IndexTTS-2 condicional com tooltip por VRAM.
- Barra de progresso por vídeo e progresso geral da fila com QThread sem travamento da UI.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.hardware import detect_hardware
from core.pipeline import DubPipeline, PipelineConfig, PipelineProgress
from .queue_widget import JobItem, QueueWidget
from .settings_dialog import SettingsDialog

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"}

ALL_LANGUAGES = [
    ("pt", "Português (Brasil)"),
    ("en", "Inglês (English)"),
    ("es", "Espanhol (Español)"),
    ("fr", "Francês (Français)"),
    ("de", "Alemão (Deutsch)"),
    ("it", "Italiano (Italiano)"),
    ("ja", "Japonês (日本語)"),
    ("zh", "Chinês (中文)"),
]

DARK_THEME_QSS = """
QMainWindow, QWidget {
    background-color: #121214;
    color: #E1E1E6;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #29292E;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #A8A8B3;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}

QGroupBox#experimental_group {
    border: 1px dashed #FFA200;
    border-radius: 8px;
    margin-top: 8px;
    padding-top: 10px;
    color: #FFA200;
}

QPushButton {
    background-color: #202024;
    color: #E1E1E6;
    border: 1px solid #323238;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #29292E;
    border-color: #48484F;
}

QPushButton:pressed {
    background-color: #121214;
}

QPushButton#btn_primary {
    background-color: #8257E5;
    color: #FFFFFF;
    border: none;
    font-weight: bold;
}

QPushButton#btn_primary:hover {
    background-color: #9466FF;
}

QTableWidget, QListWidget {
    background-color: #19191B;
    border: 1px solid #29292E;
    border-radius: 6px;
    gridline-color: #29292E;
    color: #E1E1E6;
}

QHeaderView::section {
    background-color: #202024;
    color: #A8A8B3;
    padding: 6px;
    border: none;
    font-weight: bold;
}

QProgressBar {
    border: 1px solid #29292E;
    border-radius: 4px;
    text-align: center;
    background-color: #202024;
    color: #FFFFFF;
}

QProgressBar::chunk {
    background-color: #04D361;
    border-radius: 3px;
}

QComboBox, QLineEdit {
    background-color: #202024;
    border: 1px solid #323238;
    border-radius: 6px;
    padding: 5px;
    color: #E1E1E6;
}

QTextEdit {
    background-color: #09090A;
    border: 1px solid #29292E;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    color: #04D361;
}

QSlider::groove:horizontal {
    border: 1px solid #323238;
    height: 6px;
    background: #202024;
    margin: 2px 0;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #8257E5;
    border: 1px solid #8257E5;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #9466FF;
}
"""


class DropArea(QFrame):
    """Área customizada de Drag-and-Drop para aceitar arquivos de vídeo."""

    files_dropped = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #323238;
                border-radius: 8px;
                background-color: #19191B;
                padding: 12px;
            }
            QFrame:hover {
                border-color: #8257E5;
                background-color: #202024;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(6)

        self.lbl_icon = QLabel("📥 Arraste e solte vídeos aqui")
        self.lbl_icon.setAlignment(Qt.AlignCenter)
        self.lbl_icon.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_icon.setStyleSheet("color: #E1E1E6; border: none; background: transparent;")
        layout.addWidget(self.lbl_icon)

        self.lbl_sub = QLabel("ou clique no botão abaixo para selecionar arquivos do computador")
        self.lbl_sub.setAlignment(Qt.AlignCenter)
        self.lbl_sub.setStyleSheet("color: #737380; font-size: 11px; border: none; background: transparent;")
        layout.addWidget(self.lbl_sub)

        btn_browse = QPushButton("📁 Selecionar Arquivos...")
        btn_browse.setFixedWidth(180)
        btn_browse.clicked.connect(self._browse_files)
        layout.addWidget(btn_browse, alignment=Qt.AlignCenter)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QFrame {
                    border: 2px dashed #04D361;
                    border-radius: 8px;
                    background-color: #202028;
                }
            """)

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #323238;
                border-radius: 8px;
                background-color: #19191B;
            }
        """)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #323238;
                border-radius: 8px;
                background-color: #19191B;
            }
        """)
        urls = event.mimeData().urls()
        valid_paths = []
        for u in urls:
            path = u.toLocalFile()
            if os.path.isfile(path) and Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTS:
                valid_paths.append(path)
            elif os.path.isdir(path):
                # Varre pasta
                for root, _, files in os.walk(path):
                    for f in files:
                        if Path(f).suffix.lower() in SUPPORTED_VIDEO_EXTS:
                            valid_paths.append(os.path.join(root, f))

        if valid_paths:
            self.files_dropped.emit(valid_paths)

    def _browse_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Selecionar Vídeos", "", "Arquivos de Vídeo (*.mp4 *.mkv *.mov *.avi *.webm);;Todos (*.*)"
        )
        if paths:
            self.files_dropped.emit(paths)


class PipelineWorkerThread(QThread):
    """Thread em segundo plano que executa o DubPipeline de forma assíncrona."""

    progress_signal = Signal(str, float, str, int, int)
    job_finished_signal = Signal(str, dict)
    job_error_signal = Signal(str, str)
    queue_completed_signal = Signal()

    def __init__(self, queue_jobs: List[JobItem], parent=None) -> None:
        super().__init__(parent)
        self.queue_jobs = queue_jobs
        self.pipeline = DubPipeline()

    def run(self) -> None:
        total_jobs = len(self.queue_jobs)
        for idx, job in enumerate(self.queue_jobs, 1):
            if self.pipeline.is_cancelled:
                break

            def on_progress(p: PipelineProgress) -> None:
                self.progress_signal.emit(job.job_id, p.percentage, p.message, idx, total_jobs)

            try:
                result = self.pipeline.process_video(
                    input_video=job.input_file,
                    target_language=job.target_lang,
                    source_language=job.source_lang,
                    output_video=job.output_file,
                    enable_lipsync=job.enable_lipsync,
                    use_indextts2=job.use_indextts2,
                    burn_subtitles_flag=job.burn_subtitles,
                    export_raw_pkg=job.export_raw,
                    job_index=idx,
                    total_jobs=total_jobs,
                    progress_callback=on_progress,
                )
                self.job_finished_signal.emit(job.job_id, result)
            except Exception as e:
                self.job_error_signal.emit(job.job_id, str(e))

        self.queue_completed_signal.emit()

    def cancel(self) -> None:
        if self.pipeline:
            self.pipeline.cancel()


class MainWindow(QMainWindow):
    """Janela Principal Completa do KmellVox."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KmellVox - Dublagem, Clonagem de Voz e Lip Sync com IA")
        self.resize(1140, 840)
        self.setStyleSheet(DARK_THEME_QSS)

        self.worker_thread: Optional[PipelineWorkerThread] = None
        self._init_ui()
        self._refresh_hardware_status()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        # 1. Header (Título, Badge de Hardware e Botão de Configurações)
        header_layout = QHBoxLayout()
        title_label = QLabel("🎙️ KmellVox Studio")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.badge_hardware = QLabel("Detectando hardware...")
        self.badge_hardware.setStyleSheet(
            "background-color: #29292E; color: #04D361; padding: 5px 12px; border-radius: 12px; font-weight: bold;"
        )
        header_layout.addWidget(self.badge_hardware)

        btn_settings = QPushButton("⚙️ Configurações")
        btn_settings.clicked.connect(self._open_settings)
        header_layout.addWidget(btn_settings)

        main_layout.addLayout(header_layout)

        # 2. Área Superior Dividida: (Drop Area + Multi-Seleção de Idiomas + Painel de Opções)
        top_splitter = QSplitter(Qt.Horizontal)

        # 2.1 Coluna Esquerda: Área de Drop e Idiomas de Destino
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Área de Drop
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        left_layout.addWidget(self.drop_area)

        # Lista de Idiomas de Destino (Multi-seleção)
        lang_group = QGroupBox("Idiomas de Destino (Processamento em Lote)")
        lang_layout = QVBoxLayout(lang_group)

        self.list_languages = QListWidget()
        self.list_languages.setFixedHeight(120)
        for code, label in ALL_LANGUAGES:
            item = QListWidgetItem(f"[{code.upper()}] {label}")
            item.setData(Qt.UserRole, code)
            item.setCheckState(Qt.Checked if code == "pt" else Qt.Unchecked)
            self.list_languages.addItem(item)
        lang_layout.addWidget(self.list_languages)

        # Botões de Seleção Rápida de Idiomas
        lang_btns_layout = QHBoxLayout()
        btn_sel_all = QPushButton("Selecionar Todos")
        btn_sel_all.clicked.connect(self._select_all_languages)
        btn_desel_all = QPushButton("Desmarcar Todos")
        btn_desel_all.clicked.connect(self._deselect_all_languages)
        lang_btns_layout.addWidget(btn_sel_all)
        lang_btns_layout.addWidget(btn_desel_all)
        lang_layout.addLayout(lang_btns_layout)

        left_layout.addWidget(lang_group)
        top_splitter.addWidget(left_box)

        # 2.2 Coluna Direita: Painel de Opções, Ritmo e IndexTTS-2
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        opts_group = QGroupBox("Opções de Síntese e Formato")
        opts_layout = QVBoxLayout(opts_group)
        opts_layout.setSpacing(8)

        # Formato de Saída
        row_format = QHBoxLayout()
        row_format.addWidget(QLabel("Formato de Saída:"))
        self.cb_output_format = QComboBox()
        self.cb_output_format.addItems([
            "Vídeo Completo Dublado (.mp4)",
            "Pacote Bruto (MP3 + SRT)",
            "Ambos (Vídeo + Pacote Bruto)",
        ])
        row_format.addWidget(self.cb_output_format)
        opts_layout.addLayout(row_format)

        # Legendas
        self.chk_burn_subs = QCheckBox("Estampar Legendas Embutidas no Vídeo (Burn-in)")
        self.chk_burn_subs.setChecked(False)
        opts_layout.addWidget(self.chk_burn_subs)

        # Toggle IndexTTS-2
        self.chk_indextts2 = QCheckBox("Qualidade máxima de voz (IndexTTS-2)")
        self.chk_indextts2.setChecked(False)
        opts_layout.addWidget(self.chk_indextts2)

        # Controle de Ritmo (Sliders)
        rhythm_box = QGroupBox("Controle de Ritmo (Ajuste de Fala)")
        rhythm_layout = QVBoxLayout(rhythm_box)

        # Desaceleração Máxima
        row_slow = QHBoxLayout()
        row_slow.addWidget(QLabel("Desaceleração Máx:"))
        self.slider_min_speed = QSlider(Qt.Horizontal)
        self.slider_min_speed.setRange(50, 100)  # 0.50x a 1.00x
        self.slider_min_speed.setValue(70)      # 0.70x padrão
        self.lbl_min_speed_val = QLabel("0.70x (70%)")
        self.slider_min_speed.valueChanged.connect(
            lambda v: self.lbl_min_speed_val.setText(f"{v/100:.2f}x ({v}%)")
        )
        row_slow.addWidget(self.slider_min_speed)
        row_slow.addWidget(self.lbl_min_speed_val)
        rhythm_layout.addLayout(row_slow)

        # Aceleração Máxima
        row_fast = QHBoxLayout()
        row_fast.addWidget(QLabel("Aceleração Máx:"))
        self.slider_max_speed = QSlider(Qt.Horizontal)
        self.slider_max_speed.setRange(100, 200) # 1.00x a 2.00x
        self.slider_max_speed.setValue(135)     # 1.35x padrão
        self.lbl_max_speed_val = QLabel("1.35x (135%)")
        self.slider_max_speed.valueChanged.connect(
            lambda v: self.lbl_max_speed_val.setText(f"{v/100:.2f}x ({v}%)")
        )
        row_fast.addWidget(self.slider_max_speed)
        row_fast.addWidget(self.lbl_max_speed_val)
        rhythm_layout.addLayout(row_fast)

        opts_layout.addWidget(rhythm_box)

        # Sincronia Labial Experimental
        exp_group = QGroupBox("Recursos Experimentais")
        exp_group.setObjectName("experimental_group")
        exp_layout = QVBoxLayout(exp_group)

        self.chk_lipsync = QCheckBox("⚠️ Sincronia Labial Facial - MuseTalk 1.5 (Experimental / Instável)")
        self.chk_lipsync.setChecked(False)
        self.chk_lipsync.setToolTip(
            "Recurso experimental: O MuseTalk 1.5 gera movimentos labiais realistas adaptados à nova fala, "
            "mas pode apresentar instabilidade ou artefatos dependendo do ângulo da face e iluminação da cena."
        )
        self.chk_lipsync.setStyleSheet("color: #FFA200; font-weight: bold;")
        exp_layout.addWidget(self.chk_lipsync)
        opts_layout.addWidget(exp_group)

        right_layout.addWidget(opts_group)
        top_splitter.addWidget(right_box)

        top_splitter.setSizes([500, 500])
        main_layout.addWidget(top_splitter)

        # 3. Fila de Renderização & Console de Logs
        splitter_bottom = QSplitter(Qt.Vertical)

        # Fila
        self.queue_widget = QueueWidget()
        self.queue_widget.job_cancelled.connect(self._on_job_cancelled)
        splitter_bottom.addWidget(self.queue_widget)

        # Console de Logs
        log_box = QWidget()
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Terminal & Logs de Execução:"))
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        log_layout.addWidget(self.txt_logs)
        splitter_bottom.addWidget(log_box)

        splitter_bottom.setSizes([260, 120])
        main_layout.addWidget(splitter_bottom, stretch=1)

        # 4. Barras de Progresso e Ações Inferiores
        progress_panel = QGroupBox("Progresso de Execução")
        prog_layout = QVBoxLayout(progress_panel)

        # Progresso por Vídeo Atual (com indicador de etapa do pipeline)
        row_prog_cur = QHBoxLayout()
        self.lbl_current_stage = QLabel("Etapa Atual: Ocioso")
        self.lbl_current_stage.setStyleSheet("font-weight: 500; color: #04D361;")
        row_prog_cur.addWidget(self.lbl_current_stage)
        row_prog_cur.addStretch()
        prog_layout.addLayout(row_prog_cur)

        self.prog_bar_current = QProgressBar()
        self.prog_bar_current.setRange(0, 100)
        self.prog_bar_current.setValue(0)
        self.prog_bar_current.setFixedHeight(22)
        prog_layout.addWidget(self.prog_bar_current)

        # Progresso Geral da Fila
        row_prog_queue = QHBoxLayout()
        self.lbl_queue_progress = QLabel("Fila Geral: 0/0 vídeos concluídos")
        self.lbl_queue_progress.setStyleSheet("color: #A8A8B3; font-size: 11px;")
        row_prog_queue.addWidget(self.lbl_queue_progress)
        row_prog_queue.addStretch()
        prog_layout.addLayout(row_prog_queue)

        self.prog_bar_queue = QProgressBar()
        self.prog_bar_queue.setRange(0, 100)
        self.prog_bar_queue.setValue(0)
        self.prog_bar_queue.setFixedHeight(14)
        prog_layout.addWidget(self.prog_bar_queue)

        main_layout.addWidget(progress_panel)

        # Botões de Ação
        actions_layout = QHBoxLayout()
        self.lbl_status = QLabel("Pronto para processar.")
        actions_layout.addWidget(self.lbl_status, stretch=1)

        self.btn_start = QPushButton("▶️ Iniciar Fila de Dublagem")
        self.btn_start.setObjectName("btn_primary")
        self.btn_start.setFixedHeight(40)
        self.btn_start.clicked.connect(self._start_processing)
        actions_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("⏹️ Cancelar Fila")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        actions_layout.addWidget(self.btn_cancel)

        main_layout.addLayout(actions_layout)

    def _refresh_hardware_status(self) -> None:
        hw = detect_hardware()
        if hw.cuda_available:
            self.badge_hardware.setText(f"🚀 CUDA: {hw.device_name} ({hw.vram_total_gb:.1f} GB) - Perfil: {hw.profile.value}")
        else:
            self.badge_hardware.setText(f"💻 CPU Mode ({hw.cpu_cores} cores)")
            self.badge_hardware.setStyleSheet("background-color: #383840; color: #FFA200; padding: 5px 12px; border-radius: 12px;")

        # Toggle IndexTTS-2 desabilitado automaticamente em perfil_b com tooltip explicativo
        if hw.model_profile.enable_indextts_2:
            self.chk_indextts2.setEnabled(True)
            self.chk_indextts2.setToolTip("IndexTTS-2 em FP16 com controle nativo de duração (Habilitado para perfil_a 8GB+ VRAM).")
        else:
            self.chk_indextts2.setEnabled(False)
            self.chk_indextts2.setChecked(False)
            self.chk_indextts2.setToolTip(
                f"Qualidade máxima (IndexTTS-2) requer no mínimo 8GB de VRAM (perfil_a). "
                f"Seu perfil detectado é '{hw.model_profile.profile_name}', utilizando F5-TTS com Controle de Ritmo."
            )

    def _select_all_languages(self) -> None:
        for i in range(self.list_languages.count()):
            self.list_languages.item(i).setCheckState(Qt.Checked)

    def _deselect_all_languages(self) -> None:
        for i in range(self.list_languages.count()):
            self.list_languages.item(i).setCheckState(Qt.Unchecked)

    def _get_selected_languages(self) -> List[str]:
        langs = []
        for i in range(self.list_languages.count()):
            item = self.list_languages.item(i)
            if item.checkState() == Qt.Checked:
                langs.append(item.data(Qt.UserRole))
        return langs

    def _on_files_dropped(self, file_paths: List[str]) -> None:
        selected_langs = self._get_selected_languages()
        if not selected_langs:
            selected_langs = ["pt"]

        fmt = self.cb_output_format.currentText()
        is_raw_only = "Pacote Bruto" in fmt and "Vídeo" not in fmt
        export_both = "Ambos" in fmt

        out_dir = Path("output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        added_count = 0
        for video_path in file_paths:
            base_name = Path(video_path).stem
            for lang in selected_langs:
                job_id = f"job_{int(time.time() * 1000)}_{added_count}"
                out_file = str(out_dir / f"{base_name}_dubbed_{lang}.mp4")

                job = JobItem(
                    job_id=job_id,
                    input_file=video_path,
                    output_file=out_file,
                    source_lang="auto",
                    target_lang=lang,
                    enable_lipsync=self.chk_lipsync.isChecked() and not is_raw_only,
                    use_indextts2=self.chk_indextts2.isChecked(),
                    burn_subtitles=self.chk_burn_subs.isChecked() and not is_raw_only,
                    export_raw=is_raw_only or export_both,
                )
                self.queue_widget.add_job(job)
                added_count += 1

        self._log(f"Adicionado(s) {added_count} trabalho(s) à fila de renderização.")

    def _start_processing(self) -> None:
        pending_jobs = self.queue_widget.get_pending_jobs()
        if not pending_jobs:
            QMessageBox.information(self, "Fila Vazia", "Adicione vídeos à fila antes de iniciar.")
            return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.prog_bar_queue.setValue(0)
        self.prog_bar_current.setValue(0)
        self.lbl_status.setText("Processando fila...")

        self.worker_thread = PipelineWorkerThread(pending_jobs, self)
        self.worker_thread.progress_signal.connect(self._on_worker_progress)
        self.worker_thread.job_finished_signal.connect(self._on_job_finished)
        self.worker_thread.job_error_signal.connect(self._on_job_error)
        self.worker_thread.queue_completed_signal.connect(self._on_queue_completed)
        self.worker_thread.start()

    def _on_worker_progress(self, job_id: str, pct: float, msg: str, cur_idx: int, total_idx: int) -> None:
        self.queue_widget.update_job_progress(job_id, pct, msg, status="Processando")
        self.prog_bar_current.setValue(int(pct * 100))
        self.lbl_current_stage.setText(f"Etapa: {msg}")

        queue_pct = int(((cur_idx - 1 + pct) / total_idx) * 100)
        self.prog_bar_queue.setValue(queue_pct)
        self.lbl_queue_progress.setText(f"Fila Geral: Trabalho {cur_idx} de {total_idx} ({queue_pct}%)")

        self.lbl_status.setText(f"[{cur_idx}/{total_idx}] {msg}")
        self._log(f"[{job_id}] {msg}")

    def _on_job_finished(self, job_id: str, result: dict) -> None:
        self.queue_widget.update_job_progress(job_id, 1.0, "Concluído", status="Concluído")
        self._log(f"✅ Concluído: {result.get('output_video', job_id)}")

    def _on_job_error(self, job_id: str, err: str) -> None:
        self.queue_widget.update_job_progress(job_id, 0.0, "Erro", status="Erro")
        self._log(f"❌ Erro no trabalho {job_id}: {err}")

    def _on_queue_completed(self) -> None:
        self.lbl_status.setText("Fila de processamento finalizada com sucesso!")
        self.lbl_current_stage.setText("Etapa Atual: Concluído")
        self.prog_bar_current.setValue(100)
        self.prog_bar_queue.setValue(100)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker_thread = None
        self._log("🏁 Fila de processamento finalizada.")

    def _on_job_cancelled(self, job_id: str) -> None:
        self._log(f"Item cancelado pelo usuário: {job_id}")

    def _cancel_processing(self) -> None:
        if self.worker_thread:
            self._log("Cancelando execução da fila...")
            self.worker_thread.cancel()
            self.btn_cancel.setEnabled(False)

    def _open_settings(self) -> None:
        dlg = SettingsDialog("config.yaml", self)
        if dlg.exec():
            self._refresh_hardware_status()
            self._log("Configurações atualizadas com sucesso.")

    def open_first_run_downloader(self) -> None:
        """Abre o diálogo de configurações focado no download de modelos na primeira execução."""
        self._log("Primeira execução detectada. Abrindo gerenciador de modelos...")
        dlg = SettingsDialog("config.yaml", self)
        dlg.setWindowTitle("Bem-vindo ao KmellVox - Instalação de Modelos")
        dlg.exec()
        self._refresh_hardware_status()

    def _log(self, text: str) -> None:
        self.txt_logs.append(f"[{time.strftime('%H:%M:%S')}] {text}")
