"""Janela Principal da Aplicação KmellVox em PySide6.

Inclui:
- Aba 1: 🎬 Dublagem de Vídeo (Pipeline de vídeo com IA, extração, tradução, clonagem, lip sync e fila).
- Aba 2: 🎙️ Gerador de Narração (Síntese de áudio a partir de texto puro ou legendas SRT).
- Header unificado com perfil de hardware em tempo real e acesso a configurações.
- Execução assíncrona com QThread sem bloqueio de interface.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("KmellVox.MainWindow")

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QIcon, QImage, QPainter, QPen
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
    QTabBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.hardware import PROFILE_DISPLAY_NAMES, detect_hardware
from core.pipeline import DubPipeline, PipelineConfig, PipelineProgress
from .narration_tab import NarrationTab
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

def _generate_default_icons(icons_dir: Path) -> None:
    """Gera automaticamente ícones vetoriais de estado caso não existam no disco."""
    icons_dir.mkdir(parents=True, exist_ok=True)
    check_file = icons_dir / "check_icon.png"
    arrow_file = icons_dir / "arrow_down.png"

    if not check_file.is_file():
        img = QImage(18, 18, QImage.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(18, 18, 20), 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(4, 9, 7, 13)
        p.drawLine(7, 13, 14, 5)
        p.end()
        img.save(str(check_file))

    if not arrow_file.is_file():
        img = QImage(14, 14, QImage.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(225, 225, 230), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(3, 5, 7, 9)
        p.drawLine(7, 9, 11, 5)
        p.end()
        img.save(str(arrow_file))


def build_dark_theme_qss() -> str:
    """Gera a folha de estilos completa e dinâmica com ícones em alta resolução para alto contraste."""
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.parent))
    icons_dir = base_dir / "ui" / "assets"
    check_icon = (icons_dir / "check_icon.png").resolve()
    arrow_down = (icons_dir / "arrow_down.png").resolve()

    if not check_icon.is_file() or not arrow_down.is_file():
        try:
            _generate_default_icons(icons_dir)
        except Exception:
            pass

    check_rule = f"image: url('{str(check_icon).replace(chr(92), '/')}');" if check_icon.is_file() else ""
    arrow_rule = f"image: url('{str(arrow_down).replace(chr(92), '/')}');" if arrow_down.is_file() else ""

    return f"""
QMainWindow, QWidget {{
    background-color: #121214;
    color: #E1E1E6;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}

/* --- Abas --- */
QTabWidget::pane {{
    border: 1px solid #29292E;
    border-radius: 8px;
    background-color: #121214;
    top: -1px;
}}

QTabBar::tab {{
    background-color: #19191B;
    color: #A8A8B3;
    border: 1px solid #29292E;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 20px;
    margin-right: 4px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background-color: #202024;
    color: #04D361;
    border-color: #323238;
    font-weight: bold;
}}

QTabBar::tab:hover:!selected {{
    background-color: #29292E;
    color: #E1E1E6;
}}

/* --- Agrupamentos --- */
QGroupBox {{
    border: 1px solid #29292E;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #A8A8B3;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}

QGroupBox#experimental_group {{
    border: 1px dashed #FFA200;
    border-radius: 8px;
    margin-top: 8px;
    padding-top: 10px;
    color: #FFA200;
}}

/* --- Botões --- */
QPushButton {{
    background-color: #202024;
    color: #E1E1E6;
    border: 1px solid #323238;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: #29292E;
    border-color: #52525B;
    color: #FFFFFF;
}}

QPushButton:pressed {{
    background-color: #121214;
}}

QPushButton:disabled {{
    background-color: #18181B;
    border-color: #27272A;
    color: #71717A;
}}

QPushButton#btn_primary {{
    background-color: #8257E5;
    color: #FFFFFF;
    border: none;
    font-weight: bold;
}}

QPushButton#btn_primary:hover {{
    background-color: #9466FF;
}}

/* --- Botões de Rádio (Seleção Única de Alto Contraste) --- */
QRadioButton {{
    color: #E1E1E6;
    spacing: 10px;
    font-size: 13px;
    font-weight: 500;
    padding: 4px 2px;
}}

QRadioButton:hover {{
    color: #FFFFFF;
}}

QRadioButton:disabled {{
    color: #71717A;
}}

QRadioButton::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 11px;
    border: 2px solid #52525B;
    background-color: #18181B;
}}

QRadioButton::indicator:hover {{
    border: 2px solid #04D361;
    background-color: #27272A;
}}

QRadioButton::indicator:checked {{
    border: 2px solid #04D361;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.4, fx:0.5, fy:0.5, stop:0 #04D361, stop:0.55 #04D361, stop:0.6 transparent, stop:1 transparent);
}}

QRadioButton::indicator:checked:hover {{
    border: 2px solid #00E676;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.45, fx:0.5, fy:0.5, stop:0 #00E676, stop:0.6 #00E676, stop:0.65 transparent, stop:1 transparent);
}}

QRadioButton::indicator:disabled {{
    border-color: #3F3F46;
    background-color: #121214;
}}

/* --- Caixas de Seleção (Checkboxes de Alto Contraste) --- */
QCheckBox {{
    color: #E1E1E6;
    spacing: 10px;
    font-size: 13px;
    font-weight: 500;
    padding: 4px 2px;
}}

QCheckBox:hover {{
    color: #FFFFFF;
}}

QCheckBox:disabled {{
    color: #71717A;
}}

QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 5px;
    border: 2px solid #52525B;
    background-color: #18181B;
}}

QCheckBox::indicator:hover {{
    border: 2px solid #04D361;
    background-color: #27272A;
}}

QCheckBox::indicator:checked {{
    border: 2px solid #04D361;
    background-color: #04D361;
    {check_rule}
}}

QCheckBox::indicator:checked:hover {{
    border-color: #00E676;
    background-color: #00E676;
}}

QCheckBox::indicator:disabled {{
    border-color: #3F3F46;
    background-color: #121214;
}}

/* --- Barra de Seleção (QComboBox) --- */
QComboBox {{
    background-color: #202024;
    border: 1.5px solid #3F3F46;
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 26px;
    color: #FFFFFF;
    font-weight: 500;
    font-size: 13px;
}}

QComboBox:hover {{
    border-color: #8257E5;
    background-color: #27272A;
}}

QComboBox:focus {{
    border-color: #9466FF;
}}

QComboBox:disabled {{
    background-color: #18181B;
    border-color: #2E2E35;
    color: #71717A;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 32px;
    border-left: 1px solid #323238;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    background-color: #27272A;
}}

QComboBox::drop-down:hover {{
    background-color: #383842;
}}

QComboBox::down-arrow {{
    {arrow_rule}
    width: 14px;
    height: 14px;
}}

QComboBox QAbstractItemView {{
    background-color: #202024;
    border: 1.5px solid #3F3F46;
    border-radius: 6px;
    color: #E1E1E6;
    selection-background-color: #8257E5;
    selection-color: #FFFFFF;
    outline: none;
    padding: 6px;
}}

/* --- Campos de Texto (QLineEdit) --- */
QLineEdit {{
    background-color: #202024;
    border: 1.5px solid #3F3F46;
    border-radius: 6px;
    padding: 6px 10px;
    color: #FFFFFF;
    font-size: 13px;
}}

QLineEdit:hover {{
    border-color: #52525B;
}}

QLineEdit:focus {{
    border-color: #8257E5;
}}

QLineEdit:disabled {{
    background-color: #18181B;
    border-color: #2E2E35;
    color: #71717A;
}}

/* --- Tabelas, Listas e Editores --- */
QTableWidget, QListWidget, QPlainTextEdit {{
    background-color: #19191B;
    border: 1px solid #29292E;
    border-radius: 6px;
    gridline-color: #29292E;
    color: #E1E1E6;
}}

QHeaderView::section {{
    background-color: #202024;
    color: #A8A8B3;
    padding: 6px;
    border: none;
    font-weight: bold;
}}

QProgressBar {{
    border: 1px solid #29292E;
    border-radius: 4px;
    text-align: center;
    background-color: #202024;
    color: #FFFFFF;
}}

QProgressBar::chunk {{
    background-color: #04D361;
    border-radius: 3px;
}}

QTextEdit {{
    background-color: #09090A;
    border: 1px solid #29292E;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    color: #04D361;
}}

QSlider::groove:horizontal {{
    border: 1px solid #323238;
    height: 6px;
    background: #202024;
    margin: 2px 0;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #8257E5;
    border: 1px solid #8257E5;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background: #9466FF;
}}

/* --- Barras de Rolagem (QScrollBar de Alta Visibilidade) --- */
QScrollBar:vertical {{
    border: none;
    background-color: #18181B;
    width: 12px;
    margin: 0px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: #3F3F46;
    min-height: 26px;
    border-radius: 5px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: #04D361;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    border: none;
    background: none;
}}

QScrollBar:horizontal {{
    border: none;
    background-color: #18181B;
    height: 12px;
    margin: 0px;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: #3F3F46;
    min-width: 26px;
    border-radius: 5px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: #04D361;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    border: none;
    background: none;
}}
"""

DARK_THEME_QSS = build_dark_theme_qss()



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
    """Janela Principal Completa do KmellVox Studio."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KmellVox Studio - Dublagem, Narração e Clonagem de Voz com IA")
        self.resize(1180, 880)
        self.setStyleSheet(DARK_THEME_QSS)

        self.worker_thread: Optional[PipelineWorkerThread] = None
        self._init_ui()

        # Centraliza a janela no monitor
        self._center_on_screen()

        # Adia a detecção de hardware para depois do primeiro render completo da janela.
        # Isso elimina o efeito de piscar causado pelo bloqueio síncrono de CUDA.
        QTimer.singleShot(150, self._refresh_hardware_status)

    def _center_on_screen(self) -> None:
        """Centraliza a janela no monitor primário."""
        screen = self.screen()
        if screen is not None:
            screen_geometry = screen.availableGeometry()
            window_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(8)

        # 1. Header Global
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

        # 2. QTabWidget com Abas: (1) Dublagem de Vídeo, (2) Gerador de Narração
        self.tabs = QTabWidget()

        # =====================================================================
        # ABA 1: DUBLAGEM DE VÍDEO
        # =====================================================================
        self.tab_dubbing = QWidget()
        dubbing_layout = QVBoxLayout(self.tab_dubbing)
        dubbing_layout.setContentsMargins(8, 8, 8, 8)
        dubbing_layout.setSpacing(8)

        top_splitter = QSplitter(Qt.Horizontal)

        # Coluna Esquerda da Dublagem: Drop + Idiomas
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        left_layout.addWidget(self.drop_area)

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

        # Coluna Direita da Dublagem: Opções, Ritmo, IndexTTS-2 e LipSync
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        opts_group = QGroupBox("Opções de Síntese e Formato")
        opts_layout = QVBoxLayout(opts_group)
        opts_layout.setSpacing(8)

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

        self.chk_burn_subs = QCheckBox("Estampar Legendas Embutidas no Vídeo (Burn-in)")
        self.chk_burn_subs.setChecked(False)
        opts_layout.addWidget(self.chk_burn_subs)

        self.chk_indextts2 = QCheckBox("Qualidade máxima de voz (IndexTTS-2)")
        self.chk_indextts2.setChecked(False)
        opts_layout.addWidget(self.chk_indextts2)

        rhythm_box = QGroupBox("Controle de Ritmo (Ajuste de Fala)")
        rhythm_layout = QVBoxLayout(rhythm_box)

        row_slow = QHBoxLayout()
        row_slow.addWidget(QLabel("Desaceleração Máx:"))
        self.slider_min_speed = QSlider(Qt.Horizontal)
        self.slider_min_speed.setRange(50, 100)
        self.slider_min_speed.setValue(70)
        self.lbl_min_speed_val = QLabel("0.70x (70%)")
        self.slider_min_speed.valueChanged.connect(
            lambda v: self.lbl_min_speed_val.setText(f"{v/100:.2f}x ({v}%)")
        )
        row_slow.addWidget(self.slider_min_speed)
        row_slow.addWidget(self.lbl_min_speed_val)
        rhythm_layout.addLayout(row_slow)

        row_fast = QHBoxLayout()
        row_fast.addWidget(QLabel("Aceleração Máx:"))
        self.slider_max_speed = QSlider(Qt.Horizontal)
        self.slider_max_speed.setRange(100, 200)
        self.slider_max_speed.setValue(135)
        self.lbl_max_speed_val = QLabel("1.35x (135%)")
        self.slider_max_speed.valueChanged.connect(
            lambda v: self.lbl_max_speed_val.setText(f"{v/100:.2f}x ({v}%)")
        )
        row_fast.addWidget(self.slider_max_speed)
        row_fast.addWidget(self.lbl_max_speed_val)
        rhythm_layout.addLayout(row_fast)

        opts_layout.addWidget(rhythm_box)

        exp_group = QGroupBox("Recursos Experimentais")
        exp_group.setObjectName("experimental_group")
        exp_layout = QVBoxLayout(exp_group)

        self.chk_lipsync = QCheckBox("⚠️ Sincronia Labial Facial - MuseTalk 1.5 (Experimental / Instável)")
        self.chk_lipsync.setChecked(False)
        self.chk_lipsync.setToolTip(
            "Recurso experimental: O MuseTalk 1.5 gera movimentos labiais realistas adaptados à nova fala."
        )
        self.chk_lipsync.setStyleSheet("color: #FFA200; font-weight: bold;")
        exp_layout.addWidget(self.chk_lipsync)
        opts_layout.addWidget(exp_group)

        right_layout.addWidget(opts_group)
        top_splitter.addWidget(right_box)

        top_splitter.setSizes([500, 500])
        dubbing_layout.addWidget(top_splitter)

        # Fila e Controles da Dublagem
        splitter_bottom = QSplitter(Qt.Vertical)
        self.queue_widget = QueueWidget()
        self.queue_widget.job_cancelled.connect(self._on_job_cancelled)
        splitter_bottom.addWidget(self.queue_widget)

        log_box = QWidget()
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Terminal & Logs de Execução:"))
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        log_layout.addWidget(self.txt_logs)
        splitter_bottom.addWidget(log_box)

        splitter_bottom.setSizes([200, 100])
        dubbing_layout.addWidget(splitter_bottom, stretch=1)

        # Barra de Progresso da Dublagem
        progress_panel = QGroupBox("Progresso de Execução")
        prog_layout = QVBoxLayout(progress_panel)

        row_prog_cur = QHBoxLayout()
        self.lbl_current_stage = QLabel("Etapa Atual: Ocioso")
        self.lbl_current_stage.setStyleSheet("font-weight: 500; color: #04D361;")
        row_prog_cur.addWidget(self.lbl_current_stage)
        row_prog_cur.addStretch()
        prog_layout.addLayout(row_prog_cur)

        self.prog_bar_current = QProgressBar()
        self.prog_bar_current.setRange(0, 100)
        self.prog_bar_current.setValue(0)
        self.prog_bar_current.setFixedHeight(20)
        prog_layout.addWidget(self.prog_bar_current)

        row_prog_queue = QHBoxLayout()
        self.lbl_queue_progress = QLabel("Fila Geral: 0/0 vídeos concluídos")
        self.lbl_queue_progress.setStyleSheet("color: #A8A8B3; font-size: 11px;")
        row_prog_queue.addWidget(self.lbl_queue_progress)
        row_prog_queue.addStretch()
        prog_layout.addLayout(row_prog_queue)

        self.prog_bar_queue = QProgressBar()
        self.prog_bar_queue.setRange(0, 100)
        self.prog_bar_queue.setValue(0)
        self.prog_bar_queue.setFixedHeight(12)
        prog_layout.addWidget(self.prog_bar_queue)

        dubbing_layout.addWidget(progress_panel)

        actions_layout = QHBoxLayout()
        self.lbl_status = QLabel("Pronto para processar.")
        actions_layout.addWidget(self.lbl_status, stretch=1)

        self.btn_start = QPushButton("▶️ Iniciar Fila de Dublagem")
        self.btn_start.setObjectName("btn_primary")
        self.btn_start.setFixedHeight(38)
        self.btn_start.clicked.connect(self._start_processing)
        actions_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("⏹️ Cancelar Fila")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setFixedHeight(38)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        actions_layout.addWidget(self.btn_cancel)

        dubbing_layout.addLayout(actions_layout)

        # Adiciona Aba 1
        self.tabs.addTab(self.tab_dubbing, "🎬 Dublagem de Vídeo")

        # =====================================================================
        # ABA 2: GERADOR DE NARRAÇÃO (TEXTO / SRT)
        # =====================================================================
        self.tab_narration = NarrationTab(self)
        self.tab_narration.log_signal.connect(self._log)
        self.tabs.addTab(self.tab_narration, "🎙️ Gerador de Narração")

        main_layout.addWidget(self.tabs, stretch=1)

    def _refresh_hardware_status(self) -> None:
        hw = detect_hardware()
        display = hw.model_profile.display_name
        if hw.cuda_available:
            self.badge_hardware.setText(f"🚀 CUDA: {hw.device_name} ({hw.vram_total_gb:.1f} GB) — {display}")
        else:
            self.badge_hardware.setText(f"💻 {display} ({hw.cpu_cores} cores)")
            self.badge_hardware.setStyleSheet("background-color: #383840; color: #FFA200; padding: 5px 12px; border-radius: 12px;")

        if hw.model_profile.enable_indextts_2:
            self.chk_indextts2.setEnabled(True)
            self.chk_indextts2.setToolTip("IndexTTS-2 em FP16 com controle nativo de duração (Habilitado para Alta Performance, 8GB+ VRAM).")
        else:
            self.chk_indextts2.setEnabled(False)
            self.chk_indextts2.setChecked(False)
            self.chk_indextts2.setToolTip(
                f"Qualidade máxima (IndexTTS-2) requer no mínimo 8GB de VRAM (Alta Performance). "
                f"Seu perfil detectado é '{display}', utilizando F5-TTS com Controle de Ritmo."
            )

        if hasattr(self, "tab_narration"):
            self.tab_narration._check_preset_voices()

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

        self._log(f"Adicionado(s) {added_count} trabalho(s) à fila de renderização de vídeo.")

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
        try:
            dlg = SettingsDialog("config.yaml", self)
            if dlg.exec():
                self._refresh_hardware_status()
                self._log("Configurações atualizadas com sucesso.")
        except Exception as e:
            logger.error("Erro ao abrir configurações: %s", e, exc_info=True)
            self._log(f"❌ Erro ao abrir configurações: {e}")
            QMessageBox.critical(
                self,
                "Erro ao Abrir Configurações",
                f"Não foi possível abrir o diálogo de configurações:\n\n{e}",
            )

    def open_first_run_downloader(self) -> None:
        """Abre o diálogo de configurações focado no download de modelos na primeira execução."""
        self._log("Primeira execução detectada. Abrindo gerenciador de modelos...")
        try:
            dlg = SettingsDialog("config.yaml", self)
            dlg.setWindowTitle("Bem-vindo ao KmellVox - Instalação de Modelos")
            dlg.exec()
            self._refresh_hardware_status()
        except Exception as e:
            logger.error("Erro no gerenciador de modelos: %s", e, exc_info=True)
            self._log(f"❌ Erro no gerenciador de modelos: {e}")

    def _log(self, text: str) -> None:
        self.txt_logs.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def closeEvent(self, event) -> None:
        """Encerramento limpo: mata threads, libera VRAM e encerra o processo."""
        import gc
        logger.info("Encerrando KmellVox Studio...")

        # 1. Cancela thread de narração se ativa
        if hasattr(self, "tab_narration") and self.tab_narration:
            try:
                self.tab_narration.cleanup()
            except Exception as e:
                logger.warning("Erro ao limpar aba de narração: %s", e)

        # 2. Cancela thread de dublagem se ativa
        if self.worker_thread and self.worker_thread.isRunning():
            try:
                self.worker_thread.terminate()
                self.worker_thread.wait(2000)
            except Exception as e:
                logger.warning("Erro ao encerrar thread de dublagem: %s", e)

        # 3. Limpa VRAM e memória
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("KmellVox Studio encerrado com sucesso.")
        event.accept()

        # 4. Força encerramento do processo para garantir que nenhuma thread órfã sobreviva
        import os
        os._exit(0)
