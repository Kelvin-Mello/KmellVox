"""Aba de Geração de Narração em Áudio (PySide6).

Permite sintetizar áudio a partir de texto puro ou legendas SRT,
com controle de velocidade de fala, masterização vocal com perfis personalizados,
gerenciamento completo de vozes clonadas e fila de processamento em lote.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PySide6.QtCore import QThread, QUrl, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.hardware import ModelProfile, detect_hardware
from core.narration import (
    AudioMasteringConfig,
    NarrationEngine,
    NarrationJob,
    delete_saved_voice,
    detect_text_format,
    get_voices_directory,
    list_all_saved_voices,
    list_preset_voices,
    rename_saved_voice,
    save_cloned_voice,
)

logger = logging.getLogger("KmellVox.NarrationTab")


def _get_config_path() -> Path:
    """Retorna o caminho do config.yaml (funciona tanto dev quanto PyInstaller frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "config.yaml"
    return Path(__file__).parent.parent / "config.yaml"


def _load_last_dir(key: str) -> str:
    """Lê a última pasta salva para uma chave específica do config.yaml."""
    try:
        cfg_path = _get_config_path()
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            val = cfg.get("ui", {}).get("last_dirs", {}).get(key, "")
            if val and Path(val).is_dir():
                return val
    except Exception:
        pass
    return ""


def _save_last_dir(key: str, folder: str) -> None:
    """Persiste a última pasta usada para uma chave específica no config.yaml."""
    try:
        cfg_path = _get_config_path()
        cfg: dict = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        cfg.setdefault("ui", {}).setdefault("last_dirs", {})[key] = folder
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except Exception:
        pass


def _load_mastering_profiles_from_config() -> Dict[str, dict]:
    """Carrega perfis de masterização do config.yaml com fallback padrão."""
    default_profiles = {
        "Padrao_KmellVox": {
            "name": "Padrão KmellVox (Recomendado)",
            "bass": 4.5,
            "treble": 2.0,
            "compression": 2.5,
            "loudness": -16.0,
            "speed": 1.0,
        },
        "Podcast_Encorpado": {
            "name": "Podcast / Narração Encorpada",
            "bass": 6.0,
            "treble": 2.5,
            "compression": 3.0,
            "loudness": -15.0,
            "speed": 1.35,
        },
        "Clareza_Brilho": {
            "name": "Clareza e Brilho",
            "bass": 2.0,
            "treble": 4.0,
            "compression": 2.0,
            "loudness": -16.0,
            "speed": 1.45,
        },
        "Neutro": {
            "name": "Neutro / Sem Efeitos",
            "bass": 0.0,
            "treble": 0.0,
            "compression": 1.0,
            "loudness": -16.0,
            "speed": 1.0,
        },
    }
    try:
        cfg_path = _get_config_path()
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            saved = cfg.get("audio_mastering", {}).get("profiles", {})
            if isinstance(saved, dict) and saved:
                default_profiles.update(saved)
    except Exception as e:
        logger.warning("Falha ao carregar perfis de masterização: %s", e)
    return default_profiles


def _save_mastering_profile_to_config(key: str, profile_data: dict) -> None:
    """Salva um perfil de masterização no config.yaml."""
    try:
        cfg_path = _get_config_path()
        cfg: dict = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            sec = cfg.setdefault("audio_mastering", {})
            sec.setdefault("profiles", {})[key] = profile_data
            sec["last_profile"] = key
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.warning("Falha ao salvar perfil de masterização: %s", e)


def _delete_mastering_profile_from_config(key: str) -> None:
    """Remove um perfil personalizado de masterização do config.yaml."""
    try:
        cfg_path = _get_config_path()
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            profiles = cfg.get("audio_mastering", {}).get("profiles", {})
            if key in profiles:
                del profiles[key]
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.warning("Falha ao excluir perfil de masterização: %s", e)


class NarrationWorkerThread(QThread):
    """Thread em segundo plano para executar a fila de narrações sem congelar a interface."""

    progress_signal = Signal(str, float, str)
    job_finished_signal = Signal(str, list)
    job_error_signal = Signal(str, str)
    queue_completed_signal = Signal()


    def __init__(
        self,
        jobs: List[NarrationJob],
        model_profile: Optional[ModelProfile] = None,
        models_dir: str = "models",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.jobs = jobs
        self.engine = NarrationEngine(model_profile=model_profile, models_dir=models_dir)



    def run(self) -> None:
        for job in self.jobs:
            if self.engine.is_cancelled:
                break

            def on_progress(pct: float, msg: str) -> None:
                self.progress_signal.emit(job.job_id, pct, msg)

            try:
                outputs = self.engine.run(
                    job,
                    progress_callback=on_progress,
                )
                self.job_finished_signal.emit(job.job_id, outputs)
            except Exception as e:
                import traceback, logging
                logging.getLogger("KmellVox.NarrationWorker").error(
                    "Erro completo no job %s:\n%s", job.job_id, traceback.format_exc()
                )
                self.job_error_signal.emit(job.job_id, str(e))

        self.queue_completed_signal.emit()

    def cancel(self) -> None:
        if self.engine:
            self.engine.cancel()


class NarrationTab(QWidget):
    """Widget completo da aba de Narração de Texto e SRT reorganizado em sub-abas."""

    log_signal = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_source_file_path: Optional[str] = None
        self.queue_jobs: Dict[str, NarrationJob] = {}
        self.worker_thread: Optional[NarrationWorkerThread] = None

        # Player de áudio integrado para preview de vozes
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.9)
        self.playing_voice_path: Optional[str] = None

        # Dicionário em memória dos perfis de masterização
        self.mastering_profiles = _load_mastering_profiles_from_config()

        self._init_ui()
        self._refresh_engine_catalog()
        self._refresh_all_voices()
        self._load_last_used_profile()

    def _init_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # TabWidget Principal que divide Síntese vs Clonagem/Gerenciador
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                font-weight: bold;
                font-size: 13px;
                padding: 8px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #27272A;
                color: #04D361;
                border-bottom: 2px solid #04D361;
            }
            QTabBar::tab:!selected {
                background-color: #18181B;
                color: #A1A1AA;
            }
        """)

        # Sub-aba 1: Síntese de Narração
        self.tab_synthesis = QWidget()
        self._build_synthesis_tab(self.tab_synthesis)
        self.tabs.addTab(self.tab_synthesis, "🎙️ Síntese de Narração")

        # Sub-aba 2: Clonagem e Gerenciamento de Vozes
        self.tab_cloning = QWidget()
        self._build_cloning_tab(self.tab_cloning)
        self.tabs.addTab(self.tab_cloning, "🧬 Clonagem e Gerenciamento de Vozes")

        root_layout.addWidget(self.tabs)

    # -----------------------------------------------------------------------
    # SUB-ABA 1: SÍNTESE DE NARRAÇÃO
    # -----------------------------------------------------------------------

    def _build_synthesis_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(8)

        # Barra Compacta Superior: Motor TTS Ativo e Classificação por Hardware
        engine_banner = QFrame()
        engine_banner.setFixedHeight(44)
        engine_banner.setStyleSheet("""
            QFrame {
                background-color: #1A1824;
                border: 1px solid #8257E5;
                border-radius: 6px;
            }
        """)
        banner_layout = QHBoxLayout(engine_banner)
        banner_layout.setContentsMargins(12, 0, 12, 0)
        banner_layout.setSpacing(10)

        lbl_engine_icon = QLabel("🎙️ Motor de IA:")
        lbl_engine_icon.setStyleSheet("font-size: 13px; font-weight: 900; color: #FFFFFF; border: none; background: transparent;")
        banner_layout.addWidget(lbl_engine_icon)

        self.cb_tts_engine = QComboBox()
        self.cb_tts_engine.setFixedHeight(30)
        self.cb_tts_engine.setCursor(Qt.PointingHandCursor)
        self.cb_tts_engine.setStyleSheet("""
            QComboBox {
                background-color: #27272A;
                color: #04D361;
                font-weight: 900;
                font-size: 12px;
                padding: 2px 10px;
                border: 1px solid #8257E5;
                border-radius: 6px;
                min-width: 250px;
            }
        """)
        self.cb_tts_engine.currentIndexChanged.connect(self._on_engine_changed)
        banner_layout.addWidget(self.cb_tts_engine)

        self.lbl_engine_badge = QLabel("🟢 Recomendado")
        self.lbl_engine_badge.setFixedHeight(24)
        self.lbl_engine_badge.setStyleSheet("""
            background-color: #04D36122;
            color: #04D361;
            font-weight: bold;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 5px;
            border: 1px solid #04D36155;
        """)
        banner_layout.addWidget(self.lbl_engine_badge)

        self.lbl_engine_desc = QLabel("F5-TTS v1 Base — Padrão oficial de alto desempenho e fidelidade.")
        self.lbl_engine_desc.setStyleSheet("color: #A1A1AA; font-size: 11px; border: none; background: transparent;")
        banner_layout.addWidget(self.lbl_engine_desc, stretch=1)

        layout.addWidget(engine_banner, 0)

        splitter = QSplitter(Qt.Horizontal)

        # -----------------------------
        # Coluna Esquerda: Configuração
        # -----------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 1. Entrada de Texto / SRT
        input_group = QGroupBox("1. Conteúdo de Origem (Texto ou Legenda SRT)")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(4)

        input_toolbar = QHBoxLayout()
        btn_import = QPushButton("📂 Importar Arquivo (.txt / .srt)")
        btn_import.setFixedHeight(28)
        btn_import.clicked.connect(self._import_file)
        input_toolbar.addWidget(btn_import)

        btn_clear = QPushButton("🧹 Limpar")
        btn_clear.setFixedHeight(28)
        btn_clear.clicked.connect(self._clear_input)
        input_toolbar.addWidget(btn_clear)

        input_toolbar.addStretch()
        self.lbl_format_detected = QLabel("📄 Formato detectado: Texto Puro (.txt)")
        self.lbl_format_detected.setStyleSheet("color: #04D361; font-weight: bold;")
        input_toolbar.addWidget(self.lbl_format_detected)
        input_layout.addLayout(input_toolbar)

        self.txt_content = QPlainTextEdit()
        self.txt_content.setPlaceholderText(
            "Cole aqui o seu texto puro para narração contínua ou o conteúdo de um arquivo .SRT..."
        )
        self.txt_content.setMaximumHeight(115)
        self.txt_content.setMinimumHeight(70)
        self.txt_content.textChanged.connect(self._on_text_changed)
        input_layout.addWidget(self.txt_content)
        left_layout.addWidget(input_group)

        # 2. Seleção de Voz
        voice_group = QGroupBox("2. Seleção da Voz Clonada")
        voice_layout = QVBoxLayout(voice_group)
        voice_layout.setSpacing(6)

        row_voice = QHBoxLayout()
        row_voice.addWidget(QLabel("Voz:"))
        self.cb_synthesis_voices = QComboBox()
        self.cb_synthesis_voices.setFixedHeight(30)
        self.cb_synthesis_voices.setCursor(Qt.PointingHandCursor)
        row_voice.addWidget(self.cb_synthesis_voices, stretch=3)

        btn_refresh_v = QPushButton("🔄")
        btn_refresh_v.setToolTip("Atualizar lista de vozes e motores")
        btn_refresh_v.setFixedSize(30, 30)
        btn_refresh_v.clicked.connect(self._refresh_all_voices)
        row_voice.addWidget(btn_refresh_v)

        btn_go_clone = QPushButton("➕ Nova Voz...")
        btn_go_clone.setToolTip("Abrir aba de clonagem para criar uma nova voz")
        btn_go_clone.setFixedHeight(30)
        btn_go_clone.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        row_voice.addWidget(btn_go_clone)
        voice_layout.addLayout(row_voice)
        left_layout.addWidget(voice_group)

        # 3. Ritmo e Pausas da Fala
        speed_group = QGroupBox("3. Ritmo e Pausas da Fala")
        speed_layout = QVBoxLayout(speed_group)
        speed_layout.setSpacing(6)

        # Linha 1: Velocidade
        row_spd = QHBoxLayout()
        row_spd.addWidget(QLabel("Velocidade:"))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(70, 200)  # 0.70x a 2.00x
        self.slider_speed.setValue(100)     # 1.00x padrão (Original / Calibrada)
        self.slider_speed.setSingleStep(5)
        self.slider_speed.valueChanged.connect(self._on_speed_slider_changed)
        row_spd.addWidget(self.slider_speed, stretch=3)

        self.lbl_speed_display = QLabel("1.00x (Original)")
        self.lbl_speed_display.setStyleSheet("font-weight: bold; color: #04D361; min-width: 100px;")
        row_spd.addWidget(self.lbl_speed_display)
        speed_layout.addLayout(row_spd)

        # Atalhos rápidos de velocidade
        row_presets_spd = QHBoxLayout()
        presets_list = [(0.85, "0.85x"), (1.0, "1.00x (Original)"), (1.15, "1.15x"), (1.30, "1.30x"), (1.50, "1.50x")]
        for spd_val, spd_lbl in presets_list:
            btn_spd = QPushButton(spd_lbl)
            btn_spd.setFixedHeight(24)
            btn_spd.setStyleSheet("font-size: 11px; padding: 2px 8px;")
            btn_spd.clicked.connect(lambda _, v=int(spd_val * 100): self.slider_speed.setValue(v))
            row_presets_spd.addWidget(btn_spd)
        row_presets_spd.addStretch()
        speed_layout.addLayout(row_presets_spd)

        # Linha 2: Pausa entre Frases (Respiro Natural pós-ponto)
        row_pause = QHBoxLayout()
        row_pause.addWidget(QLabel("Pausa pós-ponto:"))
        self.slider_pause = QSlider(Qt.Horizontal)
        self.slider_pause.setRange(20, 150)  # 0.20s a 1.50s
        self.slider_pause.setValue(80)       # 0.80s padrão calibrado da voz original
        self.slider_pause.setSingleStep(5)
        self.slider_pause.valueChanged.connect(self._on_pause_slider_changed)
        row_pause.addWidget(self.slider_pause, stretch=3)

        self.lbl_pause_display = QLabel("0.80s (Original)")
        self.lbl_pause_display.setStyleSheet("font-weight: bold; color: #04D361; min-width: 100px;")
        row_pause.addWidget(self.lbl_pause_display)
        speed_layout.addLayout(row_pause)

        # Atalhos rápidos de pausa
        row_presets_pause = QHBoxLayout()
        pause_presets = [(0.40, "0.40s"), (0.60, "0.60s"), (0.80, "0.80s (Original)"), (1.00, "1.00s"), (1.20, "1.20s")]
        for p_val, p_lbl in pause_presets:
            btn_p = QPushButton(p_lbl)
            btn_p.setFixedHeight(24)
            btn_p.setStyleSheet("font-size: 11px; padding: 2px 8px;")
            btn_p.clicked.connect(lambda _, v=int(p_val * 100): self.slider_pause.setValue(v))
            row_presets_pause.addWidget(btn_p)
        row_presets_pause.addStretch()
        speed_layout.addLayout(row_presets_pause)

        left_layout.addWidget(speed_group)

        # 4. Masterização Vocal e Dinâmica de Estúdio
        mast_group = QGroupBox("4. Masterização Vocal e Dinâmica de Estúdio")
        mast_layout = QVBoxLayout(mast_group)
        mast_layout.setSpacing(6)

        # Linha de Perfis
        row_prof = QHBoxLayout()
        row_prof.addWidget(QLabel("Perfil:"))
        self.cb_mastering_profiles = QComboBox()
        self.cb_mastering_profiles.setFixedHeight(28)
        self.cb_mastering_profiles.currentIndexChanged.connect(self._on_profile_selected)
        row_prof.addWidget(self.cb_mastering_profiles, stretch=3)

        btn_save_prof = QPushButton("💾 Salvar Perfil...")
        btn_save_prof.setFixedHeight(28)
        btn_save_prof.clicked.connect(self._save_custom_profile_dialog)
        row_prof.addWidget(btn_save_prof)

        btn_del_prof = QPushButton("🗑️")
        btn_del_prof.setToolTip("Excluir perfil selecionado")
        btn_del_prof.setFixedSize(28, 28)
        btn_del_prof.clicked.connect(self._delete_custom_profile)
        row_prof.addWidget(btn_del_prof)
        mast_layout.addLayout(row_prof)

        # Sliders de Ajuste Fino
        grid_params = QHBoxLayout()

        # Graves (Bass)
        col_bass = QVBoxLayout()
        col_bass.addWidget(QLabel("Graves (Warmth):"))
        self.slider_bass = QSlider(Qt.Horizontal)
        self.slider_bass.setRange(0, 120)  # 0.0 a 12.0 dB
        self.slider_bass.setValue(45)     # 4.5 dB
        self.slider_bass.valueChanged.connect(self._on_param_slider_changed)
        col_bass.addWidget(self.slider_bass)
        self.lbl_bass_val = QLabel("+4.5 dB")
        col_bass.addWidget(self.lbl_bass_val)
        grid_params.addLayout(col_bass)

        # Agudos (Treble)
        col_treb = QVBoxLayout()
        col_treb.addWidget(QLabel("Agudos (Brilho):"))
        self.slider_treble = QSlider(Qt.Horizontal)
        self.slider_treble.setRange(0, 80)  # 0.0 a 8.0 dB
        self.slider_treble.setValue(20)    # 2.0 dB
        self.slider_treble.valueChanged.connect(self._on_param_slider_changed)
        col_treb.addWidget(self.slider_treble)
        self.lbl_treble_val = QLabel("+2.0 dB")
        col_treb.addWidget(self.lbl_treble_val)
        grid_params.addLayout(col_treb)

        # Compressão
        col_comp = QVBoxLayout()
        col_comp.addWidget(QLabel("Compressão:"))
        self.slider_comp = QSlider(Qt.Horizontal)
        self.slider_comp.setRange(10, 50)  # 1.0 a 5.0 ratio
        self.slider_comp.setValue(25)     # 2.5 ratio
        self.slider_comp.valueChanged.connect(self._on_param_slider_changed)
        col_comp.addWidget(self.slider_comp)
        self.lbl_comp_val = QLabel("2.5:1")
        col_comp.addWidget(self.lbl_comp_val)
        grid_params.addLayout(col_comp)

        # Loudness
        col_lufs = QVBoxLayout()
        col_lufs.addWidget(QLabel("Volume (LUFS):"))
        self.slider_lufs = QSlider(Qt.Horizontal)
        self.slider_lufs.setRange(-24, -12)  # -24 a -12 LUFS
        self.slider_lufs.setValue(-16)       # -16 LUFS
        self.slider_lufs.valueChanged.connect(self._on_param_slider_changed)
        col_lufs.addWidget(self.slider_lufs)
        self.lbl_lufs_val = QLabel("-16 LUFS")
        col_lufs.addWidget(self.lbl_lufs_val)
        grid_params.addLayout(col_lufs)

        mast_layout.addLayout(grid_params)
        left_layout.addWidget(mast_group)

        # 5. Opções de Legenda SRT (Modos e Intervalo de Blocos)
        self.grp_srt_options = QGroupBox("5. Opções de Legenda SRT")
        srt_opt_layout = QVBoxLayout(self.grp_srt_options)
        srt_opt_layout.setSpacing(6)

        self.lbl_srt_info = QLabel("⏱️ Legenda detectada: 0 blocos")
        self.lbl_srt_info.setStyleSheet("color: #FFA200; font-weight: bold; font-size: 11px;")
        srt_opt_layout.addWidget(self.lbl_srt_info)

        # Modo de Exportação
        srt_opt_layout.addWidget(QLabel("<b>Modo de Exportação:</b>"))
        self.bg_export_mode = QButtonGroup(self)
        self.rb_srt_single = QRadioButton("Juntar tudo em um único áudio contínuo (com pausas do SRT)")
        self.rb_srt_single.setChecked(True)
        self.rb_srt_split = QRadioButton("Gerar áudios separados por trecho (ex: 001_trecho.mp3)")
        self.bg_export_mode.addButton(self.rb_srt_single)
        self.bg_export_mode.addButton(self.rb_srt_split)
        srt_opt_layout.addWidget(self.rb_srt_single)
        srt_opt_layout.addWidget(self.rb_srt_split)

        # Seleção de Blocos a Sintetizar
        srt_opt_layout.addWidget(QLabel("<b>Blocos de Legenda a Sintetizar:</b>"))

        self.bg_blocks_mode = QButtonGroup(self)
        self.rb_blocks_all = QRadioButton("Todos os blocos do arquivo SRT")
        self.rb_blocks_all.setChecked(True)
        self.rb_blocks_all.toggled.connect(self._on_srt_blocks_mode_changed)
        self.bg_blocks_mode.addButton(self.rb_blocks_all)
        srt_opt_layout.addWidget(self.rb_blocks_all)

        row_range = QHBoxLayout()
        self.rb_blocks_range = QRadioButton("Intervalo:")
        self.rb_blocks_range.toggled.connect(self._on_srt_blocks_mode_changed)
        self.bg_blocks_mode.addButton(self.rb_blocks_range)
        row_range.addWidget(self.rb_blocks_range)

        row_range.addWidget(QLabel("De:"))
        self.spin_block_from = QSpinBox()
        self.spin_block_from.setRange(1, 9999)
        self.spin_block_from.setValue(1)
        self.spin_block_from.setEnabled(False)
        self.spin_block_from.setFixedWidth(65)
        row_range.addWidget(self.spin_block_from)

        row_range.addWidget(QLabel("Até:"))
        self.spin_block_to = QSpinBox()
        self.spin_block_to.setRange(1, 9999)
        self.spin_block_to.setValue(1)
        self.spin_block_to.setEnabled(False)
        self.spin_block_to.setFixedWidth(65)
        row_range.addWidget(self.spin_block_to)
        row_range.addStretch()
        srt_opt_layout.addLayout(row_range)

        row_custom = QHBoxLayout()
        self.rb_blocks_custom = QRadioButton("Digitar blocos:")
        self.rb_blocks_custom.setToolTip("Digite os números dos blocos desejados (ex: 1-5 ou 1, 3, 5-8 ou 12)")
        self.rb_blocks_custom.toggled.connect(self._on_srt_blocks_mode_changed)
        self.bg_blocks_mode.addButton(self.rb_blocks_custom)
        row_custom.addWidget(self.rb_blocks_custom)

        self.txt_blocks_custom = QLineEdit()
        self.txt_blocks_custom.setPlaceholderText("Ex: 1-5 ou 1, 3, 7-10 ou 12")
        self.txt_blocks_custom.setEnabled(False)
        row_custom.addWidget(self.txt_blocks_custom)
        srt_opt_layout.addLayout(row_custom)

        self.grp_srt_options.setVisible(False)
        left_layout.addWidget(self.grp_srt_options)

        dest_group = QGroupBox("Destino dos Arquivos de Áudio (MP3)")
        dest_layout = QVBoxLayout(dest_group)
        self.chk_save_source_dir = QCheckBox("Salvar na mesma pasta de origem do arquivo")
        self.chk_save_source_dir.setChecked(True)
        self.chk_save_source_dir.setEnabled(False)
        self.chk_save_source_dir.toggled.connect(self._on_dest_mode_changed)
        dest_layout.addWidget(self.chk_save_source_dir)

        row_dest = QHBoxLayout()
        self.txt_dest_folder = QLineEdit(str(Path.home() / "Downloads"))
        self.txt_dest_folder.setEnabled(False)
        row_dest.addWidget(self.txt_dest_folder)
        self.btn_browse_dest = QPushButton("Procurar...")
        self.btn_browse_dest.setEnabled(False)
        self.btn_browse_dest.clicked.connect(self._browse_dest_folder)
        row_dest.addWidget(self.btn_browse_dest)
        dest_layout.addLayout(row_dest)

        self.chk_audio_subfolder = QCheckBox("Criar subpasta 'Áudio' dentro do destino")
        self.chk_audio_subfolder.setChecked(True)
        dest_layout.addWidget(self.chk_audio_subfolder)

        self.chk_export_raw_wav = QCheckBox("Salvar também Áudio Bruto (.wav neural puro sem efeitos)")
        self.chk_export_raw_wav.setChecked(True)
        self.chk_export_raw_wav.setToolTip("Exporta cópia fiel do áudio direto da rede neural (.wav) ao lado do MP3 masterizado para comparação A/B.")
        dest_layout.addWidget(self.chk_export_raw_wav)
        left_layout.addWidget(dest_group)

        # Contêiner da Coluna Esquerda: QScrollArea fluida + Botão Fixo no Rodapé
        left_container = QWidget()
        left_container_layout = QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(6)

        # Envolve as opções em uma QScrollArea fluida para garantir que
        # a janela nunca cresça verticalmente fora da tela ao abrir blocos SRT ou em telas menores.
        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QFrame.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_left.setWidget(left_widget)
        left_container_layout.addWidget(scroll_left, stretch=1)

        # Botão Adicionar à Fila: FIXO no rodapé da coluna de configuração (nunca some nem rola para fora!)
        self.btn_add_to_queue = QPushButton("➕ Adicionar à Fila de Narração")
        self.btn_add_to_queue.setFixedHeight(42)
        self.btn_add_to_queue.setStyleSheet("""
            QPushButton {
                background-color: #8257E5;
                color: #FFFFFF;
                font-size: 13px;
                font-weight: 900;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #9466FF;
            }
        """)
        self.btn_add_to_queue.clicked.connect(self._add_current_to_queue)
        left_container_layout.addWidget(self.btn_add_to_queue)

        splitter.addWidget(left_container)

        # -----------------------------
        # Coluna Direita: Fila e Execução
        # -----------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        queue_tools = QHBoxLayout()
        self.lbl_queue_header = QLabel("Fila de Narrações (0 itens)")
        self.lbl_queue_header.setStyleSheet("font-weight: bold; color: #A8A8B3;")
        queue_tools.addWidget(self.lbl_queue_header)
        queue_tools.addStretch()

        btn_clear_done = QPushButton("🧹 Limpar Concluídos")
        btn_clear_done.setFixedHeight(26)
        btn_clear_done.clicked.connect(self._clear_completed_jobs)
        queue_tools.addWidget(btn_clear_done)

        btn_clear_queue = QPushButton("🗑️ Limpar Tudo")
        btn_clear_queue.setFixedHeight(26)
        btn_clear_queue.clicked.connect(self._clear_all_jobs)
        queue_tools.addWidget(btn_clear_queue)
        right_layout.addLayout(queue_tools)

        # Tabela da fila
        self.table_queue = QTableWidget()
        self.table_queue.setColumnCount(6)
        self.table_queue.setHorizontalHeaderLabels([
            "Origem", "Voz", "Velocidade", "Status", "Progresso", "Ações"
        ])
        self.table_queue.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_queue.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table_queue.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_queue.setAlternatingRowColors(True)
        self.table_queue.setMinimumHeight(120)
        right_layout.addWidget(self.table_queue)

        # Painel de Progresso
        prog_box = QGroupBox("Status de Processamento")
        prog_box_layout = QVBoxLayout(prog_box)
        prog_box_layout.setContentsMargins(8, 6, 8, 6)
        prog_box_layout.setSpacing(4)
        self.lbl_job_stage = QLabel("Etapa Atual: Ocioso")
        self.lbl_job_stage.setStyleSheet("color: #04D361; font-weight: 500;")
        prog_box_layout.addWidget(self.lbl_job_stage)

        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedHeight(18)
        prog_box_layout.addWidget(self.prog_bar)
        right_layout.addWidget(prog_box)

        # Botões de Execução
        exec_layout = QHBoxLayout()
        self.btn_process_queue = QPushButton("▶️ Processar Fila de Narração")
        self.btn_process_queue.setStyleSheet("background-color: #04D361; color: #121214; font-weight: bold; height: 38px;")
        self.btn_process_queue.clicked.connect(self._start_queue_processing)
        exec_layout.addWidget(self.btn_process_queue, stretch=2)

        self.btn_cancel_queue = QPushButton("⏹️ Cancelar Fila")
        self.btn_cancel_queue.setEnabled(False)
        self.btn_cancel_queue.setFixedHeight(38)
        self.btn_cancel_queue.clicked.connect(self._cancel_queue_processing)
        exec_layout.addWidget(self.btn_cancel_queue, stretch=1)
        right_layout.addLayout(exec_layout)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([650, 430])
        layout.addWidget(splitter, 1)

    # -----------------------------------------------------------------------
    # SUB-ABA 2: CLONAGEM E GERENCIAMENTO DE VOZES
    # -----------------------------------------------------------------------

    def _build_cloning_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Banner Informativo de Arquitetura de Clonagem
        info_box = QGroupBox("ℹ️ Como Funciona a Clonagem e Seleção de Modelos no KmellVox")
        info_box.setStyleSheet("""
            QGroupBox {
                background-color: #18181B;
                border: 1px solid #27272A;
                border-radius: 8px;
                padding: 8px 12px;
            }
        """)
        info_layout = QVBoxLayout(info_box)
        info_layout.setSpacing(4)
        lbl_info = QLabel(
            "<b>Como o KmellVox clona e gera vozes:</b><br>"
            "• <b>Nesta Aba (Clonagem):</b> Você cadastra as vozes desejadas. O sistema extrai e calibra uma amostra limpa (de 8 a 12s) que serve como o <b>DNA vocal</b>.<br>"
            "• <b>Na Aba 'Síntese de Narração':</b> Você escolhe no topo da tela qual <b>Motor TTS</b> (ex: F5-TTS, IndexTTS-2.5, Qwen3) usará esse DNA para ler seu roteiro em tempo real.<br>"
            "• Qualquer voz cadastrada aqui pode ser utilizada instantaneamente por qualquer um dos motores instalados!"
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #D4D4D8; font-size: 12px; line-height: 140%;")
        info_layout.addWidget(lbl_info)
        layout.addWidget(info_box)

        # Bloco Superior: Clonar Nova Voz
        clone_box = QGroupBox("🧬 Clonar e Adicionar Nova Voz")
        clone_layout = QVBoxLayout(clone_box)
        clone_layout.setSpacing(8)

        row_file = QHBoxLayout()
        row_file.addWidget(QLabel("Áudio de Referência:"))
        self.txt_clone_file = QLineEdit()
        self.txt_clone_file.setPlaceholderText("Selecione um arquivo de áudio da voz que deseja clonar (.wav, .mp3, .flac, .m4a)...")
        row_file.addWidget(self.txt_clone_file)
        btn_browse_clone = QPushButton("📁 Procurar Áudio...")
        btn_browse_clone.clicked.connect(self._browse_clone_audio)
        row_file.addWidget(btn_browse_clone)
        clone_layout.addLayout(row_file)

        row_meta = QHBoxLayout()
        row_meta.addWidget(QLabel("Nome da Voz:"))
        self.txt_clone_name = QLineEdit()
        self.txt_clone_name.setPlaceholderText("Ex: Narrador Documentário, Voz Carlos, etc.")
        row_meta.addWidget(self.txt_clone_name, stretch=2)

        row_meta.addWidget(QLabel("Transcrição (Opcional):"))
        self.txt_clone_transcription = QLineEdit()
        self.txt_clone_transcription.setPlaceholderText("O que foi falado no áudio (se vazio, transcreve automaticamente)...")
        row_meta.addWidget(self.txt_clone_transcription, stretch=3)
        clone_layout.addLayout(row_meta)

        lbl_hint = QLabel(
            "💡 <b>Inteligência de Alinhamento KmellVox</b>: Para clonagem perfeita e com graves preservados, "
            "o áudio é calibrado entre 5 e 12 segundos exatamente no final de uma frase com ponto final ou pausa, "
            "sem cortar palavras ao meio."
        )
        lbl_hint.setStyleSheet("color: #A1A1AA; font-size: 11px;")
        clone_layout.addWidget(lbl_hint)

        row_btn_clone = QHBoxLayout()
        row_btn_clone.addStretch()
        self.btn_execute_clone = QPushButton("💾 Clonar e Salvar Voz no Sistema")
        self.btn_execute_clone.setFixedHeight(34)
        self.btn_execute_clone.setStyleSheet("background-color: #04D361; color: #121214; font-weight: bold; padding: 0 16px;")
        self.btn_execute_clone.clicked.connect(self._save_new_cloned_voice)
        row_btn_clone.addWidget(self.btn_execute_clone)
        clone_layout.addLayout(row_btn_clone)
        layout.addWidget(clone_box)

        # Bloco Inferior: Gerenciador de Vozes Salvas
        mgmt_box = QGroupBox("📚 Minhas Vozes Salvas (Voices)")
        mgmt_layout = QVBoxLayout(mgmt_box)
        mgmt_layout.setSpacing(6)

        toolbar_mgmt = QHBoxLayout()
        self.lbl_voices_count = QLabel("Vozes Salvas (0)")
        self.lbl_voices_count.setStyleSheet("font-weight: bold; color: #A1A1AA;")
        toolbar_mgmt.addWidget(self.lbl_voices_count)
        toolbar_mgmt.addStretch()

        self.lbl_player_status = QLabel("⏹️ Player Ocioso")
        self.lbl_player_status.setStyleSheet("color: #04D361; font-weight: 500; margin-right: 12px;")
        toolbar_mgmt.addWidget(self.lbl_player_status)

        btn_stop_audio = QPushButton("⏹️ Parar Reprodução")
        btn_stop_audio.clicked.connect(self._stop_preview)
        toolbar_mgmt.addWidget(btn_stop_audio)

        btn_open_folder = QPushButton("📁 Abrir Pasta de Vozes")
        btn_open_folder.clicked.connect(self._open_voices_folder)
        toolbar_mgmt.addWidget(btn_open_folder)

        btn_refresh = QPushButton("🔄 Atualizar")
        btn_refresh.clicked.connect(self._refresh_all_voices)
        toolbar_mgmt.addWidget(btn_refresh)
        mgmt_layout.addLayout(toolbar_mgmt)

        # Tabela de Vozes
        self.table_voices = QTableWidget()
        self.table_voices.setColumnCount(5)
        self.table_voices.setHorizontalHeaderLabels([
            "Nome da Voz", "Duração", "Tamanho", "Data de Cadastro", "Ações"
        ])
        self.table_voices.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_voices.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_voices.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_voices.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_voices.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_voices.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_voices.setAlternatingRowColors(True)
        mgmt_layout.addWidget(self.table_voices)

        layout.addWidget(mgmt_box)

    # -----------------------------------------------------------------------
    # CONTROLES DE VELOCIDADE E MASTERIZAÇÃO VOCAL
    # -----------------------------------------------------------------------

    def _on_speed_slider_changed(self, val: int) -> None:
        spd = val / 100.0
        if val == 100:
            self.lbl_speed_display.setText("1.00x (Original)")
        else:
            self.lbl_speed_display.setText(f"{spd:.2f}x")

    def _on_pause_slider_changed(self, val: int) -> None:
        sec = val / 100.0
        if abs(sec - 0.80) < 0.02:
            self.lbl_pause_display.setText("0.80s (Original)")
            self.lbl_pause_display.setStyleSheet("font-weight: bold; color: #04D361; min-width: 100px;")
        else:
            self.lbl_pause_display.setText(f"{sec:.2f}s")
            self.lbl_pause_display.setStyleSheet("font-weight: bold; color: #E1E1E6; min-width: 100px;")

    def _on_param_slider_changed(self) -> None:
        bass = self.slider_bass.value() / 10.0
        treble = self.slider_treble.value() / 10.0
        comp = self.slider_comp.value() / 10.0
        lufs = self.slider_lufs.value()

        self.lbl_bass_val.setText(f"+{bass:.1f} dB")
        self.lbl_treble_val.setText(f"+{treble:.1f} dB")
        self.lbl_comp_val.setText(f"{comp:.1f}:1")
        self.lbl_lufs_val.setText(f"{lufs} LUFS")

    def _get_current_mastering_config(self) -> AudioMasteringConfig:
        tempo_calib = 1.00
        try:
            cfg_p = _get_config_path()
            if cfg_p.exists():
                with open(cfg_p, "r", encoding="utf-8") as f:
                    c = yaml.safe_load(f) or {}
                tempo_calib = float(c.get("audio_mastering", {}).get("tempo_calibration", 1.00))
        except Exception:
            pass

        pause_val = 0.80
        if hasattr(self, "slider_pause"):
            pause_val = self.slider_pause.value() / 100.0

        return AudioMasteringConfig(
            bass_gain_db=self.slider_bass.value() / 10.0,
            treble_gain_db=self.slider_treble.value() / 10.0,
            compressor_threshold=-18.0,
            compressor_ratio=self.slider_comp.value() / 10.0,
            target_lufs=float(self.slider_lufs.value()),
            speech_speed=self.slider_speed.value() / 100.0,
            tempo_calibration=tempo_calib,
            sentence_pause_seconds=pause_val,
            enabled=True,
        )

    def _load_last_used_profile(self) -> None:
        self.cb_mastering_profiles.clear()
        for k, prof in self.mastering_profiles.items():
            self.cb_mastering_profiles.addItem(prof.get("name", k), k)

        # Seleciona o perfil padrão
        idx = self.cb_mastering_profiles.findData("Padrao_KmellVox")
        if idx >= 0:
            self.cb_mastering_profiles.setCurrentIndex(idx)
        elif self.cb_mastering_profiles.count() > 0:
            self.cb_mastering_profiles.setCurrentIndex(0)

    def _on_profile_selected(self, index: int) -> None:
        key = self.cb_mastering_profiles.currentData()
        if not key or key not in self.mastering_profiles:
            return
        p = self.mastering_profiles[key]

        # Bloqueia sinais temporariamente para não disparar eventos circulares
        self.slider_bass.blockSignals(True)
        self.slider_treble.blockSignals(True)
        self.slider_comp.blockSignals(True)
        self.slider_lufs.blockSignals(True)
        self.slider_speed.blockSignals(True)

        self.slider_bass.setValue(int(p.get("bass", 4.5) * 10))
        self.slider_treble.setValue(int(p.get("treble", 2.0) * 10))
        self.slider_comp.setValue(int(p.get("compression", 2.5) * 10))
        self.slider_lufs.setValue(int(p.get("loudness", -16.0)))
        self.slider_speed.setValue(int(p.get("speed", 1.4) * 100))

        self.slider_bass.blockSignals(False)
        self.slider_treble.blockSignals(False)
        self.slider_comp.blockSignals(False)
        self.slider_lufs.blockSignals(False)
        self.slider_speed.blockSignals(False)

        self._on_param_slider_changed()
        self._on_speed_slider_changed(self.slider_speed.value())

    def _save_custom_profile_dialog(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Salvar Perfil de Áudio",
            "Digite um nome para o novo perfil de masterização vocal:",
        )
        if not ok or not name.strip():
            return

        clean_name = name.strip()
        key = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        profile_data = {
            "name": clean_name,
            "bass": self.slider_bass.value() / 10.0,
            "treble": self.slider_treble.value() / 10.0,
            "compression": self.slider_comp.value() / 10.0,
            "loudness": float(self.slider_lufs.value()),
            "speed": self.slider_speed.value() / 100.0,
        }
        self.mastering_profiles[key] = profile_data
        _save_mastering_profile_to_config(key, profile_data)

        self.cb_mastering_profiles.addItem(clean_name, key)
        self.cb_mastering_profiles.setCurrentIndex(self.cb_mastering_profiles.count() - 1)
        QMessageBox.information(self, "Perfil Salvo", f"Perfil '{clean_name}' salvo com sucesso!")

    def _delete_custom_profile(self) -> None:
        key = self.cb_mastering_profiles.currentData()
        if key in ("Padrao_KmellVox", "Neutro"):
            QMessageBox.warning(self, "Perfil Padrão", "Os perfis padrão do sistema não podem ser excluídos.")
            return

        ans = QMessageBox.question(
            self,
            "Excluir Perfil",
            f"Deseja realmente excluir o perfil '{self.cb_mastering_profiles.currentText()}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans == QMessageBox.Yes:
            _delete_mastering_profile_from_config(key)
            if key in self.mastering_profiles:
                del self.mastering_profiles[key]
            idx = self.cb_mastering_profiles.currentIndex()
            self.cb_mastering_profiles.removeItem(idx)
            self._on_profile_selected(0)

    # -----------------------------------------------------------------------
    # REPRODUÇÃO DE ÁUDIO PREVIEW (QMediaPlayer)
    # -----------------------------------------------------------------------

    def _play_preview(self, audio_path: str, voice_name: str) -> None:
        if not os.path.isfile(audio_path):
            QMessageBox.warning(self, "Arquivo Ausente", "Arquivo de áudio da voz não foi encontrado no disco.")
            return

        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(audio_path))
        self.player.play()
        self.playing_voice_path = audio_path
        self.lbl_player_status.setText(f"▶️ Reproduzindo: {voice_name}")

    def _stop_preview(self) -> None:
        self.player.stop()
        self.playing_voice_path = None
        self.lbl_player_status.setText("⏹️ Player Ocioso")

    def _open_voices_folder(self) -> None:
        voices_dir = get_voices_directory()
        try:
            os.startfile(str(voices_dir))
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # GERENCIAMENTO DE VOZES (Sub-Aba 2)
    # -----------------------------------------------------------------------

    def _browse_clone_audio(self) -> None:
        last = _load_last_dir("narration_ref_audio")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Áudio de Referência para Clonagem",
            last,
            "Áudios Suportados (*.wav *.mp3 *.flac *.ogg *.m4a);;Todos (*.*)",
        )
        if path:
            self.txt_clone_file.setText(path)
            _save_last_dir("narration_ref_audio", str(Path(path).parent))
            if not self.txt_clone_name.text().strip():
                self.txt_clone_name.setText(Path(path).stem.replace("_", " ").title())

    def _save_new_cloned_voice(self) -> None:
        audio_p = self.txt_clone_file.text().strip()
        name = self.txt_clone_name.text().strip()
        transcription = self.txt_clone_transcription.text().strip()

        if not audio_p or not os.path.isfile(audio_p):
            QMessageBox.warning(self, "Áudio Inválido", "Selecione um arquivo de áudio válido para clonar a voz.")
            return

        if not name:
            QMessageBox.warning(self, "Nome Ausente", "Digite um nome para identificar esta voz.")
            return

        try:
            self.btn_execute_clone.setEnabled(False)
            self.btn_execute_clone.setText("⏳ Processando e Calibrando...")

            res = save_cloned_voice(
                voice_name=name,
                audio_path=audio_p,
                transcript=transcription,
            )

            self.log_signal.emit(f"🧬 Nova voz clonada com alinhamento cirúrgico: '{res['name']}'")
            self._refresh_all_voices()

            # Limpa campos
            self.txt_clone_file.clear()
            self.txt_clone_name.clear()
            self.txt_clone_transcription.clear()

            QMessageBox.information(
                self,
                "Voz Clonada",
                f"Voz '{res['name']}' clonada e calibrada com sucesso!\n"
                "Ela já está pronta para seleção na aba de Síntese de Narração.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Erro ao Clonar Voz", f"Falha ao processar o áudio de referência:\n{e}")
        finally:
            self.btn_execute_clone.setEnabled(True)
            self.btn_execute_clone.setText("💾 Clonar e Salvar Voz no Sistema")

    def _check_preset_voices(self) -> None:
        """Compatibilidade com chamadas de atualização do MainWindow."""
        self._refresh_engine_catalog()
        self._refresh_all_voices()

    def _refresh_engine_catalog(self) -> None:
        """Atualiza a lista de motores TTS disponíveis com badges de compatibilidade por hardware."""
        if not hasattr(self, "cb_tts_engine"):
            return
        try:
            from core.hardware import detect_hardware
            from core.tts_catalog import list_tts_catalog, get_hardware_compatibility
            hw = detect_hardware()
            vram = hw.vram_total_gb if hw.cuda_available else 0.0

            cur_data = self.cb_tts_engine.currentData()
            self.cb_tts_engine.clear()

            from core.tts_catalog import is_engine_operational
            for meta in list_tts_catalog():
                is_op, op_exp = is_engine_operational(meta.id)
                badge, explanation = get_hardware_compatibility(meta.id, vram)
                if not is_op:
                    label = f"{meta.name} [⚪ Em Breve]"
                else:
                    label = f"{meta.name} [{badge}]"
                self.cb_tts_engine.addItem(label, meta.id)
                idx = self.cb_tts_engine.count() - 1
                self.cb_tts_engine.setItemData(idx, f"{meta.description}\n\nStatus: {op_exp}\nCompatibilidade: {explanation}", Qt.ToolTipRole)

            if cur_data:
                idx = self.cb_tts_engine.findData(cur_data)
                if idx >= 0:
                    self.cb_tts_engine.setCurrentIndex(idx)
            else:
                f5_idx = self.cb_tts_engine.findData("f5-tts")
                if f5_idx >= 0:
                    self.cb_tts_engine.setCurrentIndex(f5_idx)

            self._on_engine_changed()
        except Exception as e:
            logger.warning("Falha ao carregar catálogo de motores TTS na UI: %s", e)

    def _on_engine_changed(self, idx: int = 0) -> None:
        """Atualiza dinamicamente os badges e a descrição do motor selecionado."""
        if not hasattr(self, "cb_tts_engine") or not hasattr(self, "lbl_engine_badge"):
            return
        try:
            from core.hardware import detect_hardware
            from core.tts_catalog import get_engine_meta, get_hardware_compatibility, is_engine_operational
            engine_id = self.cb_tts_engine.currentData()
            if not engine_id:
                return
            meta = get_engine_meta(engine_id)
            if meta:
                hw = detect_hardware()
                vram = hw.vram_total_gb if hw.cuda_available else 0.0
                is_op, op_exp = is_engine_operational(meta.id)
                badge, exp = get_hardware_compatibility(meta.id, vram)
                if not is_op:
                    badge = "⚪ Em Breve"
                    exp = op_exp

                if "Pouco" in badge or "🟡" in badge:
                    self.lbl_engine_badge.setStyleSheet(
                        "background-color: transparent; color: #FFA200; font-weight: bold; font-size: 11px; padding: 2px 4px; border: none;"
                    )
                elif "Não" in badge or "🔴" in badge:
                    self.lbl_engine_badge.setStyleSheet(
                        "background-color: transparent; color: #FF5555; font-weight: bold; font-size: 11px; padding: 2px 4px; border: none;"
                    )
                elif "Breve" in badge or "⚪" in badge:
                    self.lbl_engine_badge.setStyleSheet(
                        "background-color: transparent; color: #A1A1AA; font-weight: bold; font-size: 11px; padding: 2px 4px; border: none;"
                    )
                else:
                    self.lbl_engine_badge.setStyleSheet(
                        "background-color: #04D36122; color: #04D361; font-weight: bold; font-size: 11px; padding: 2px 8px; border-radius: 5px; border: 1px solid #04D36155;"
                    )
                self.lbl_engine_badge.setText(badge)
                self.lbl_engine_desc.setText(f"{meta.description} ({exp})")
                if hasattr(self, "btn_add_to_queue"):
                    self.btn_add_to_queue.setText(f"➕ Adicionar à Fila (Motor: {meta.name})")
        except Exception as e:
            logger.warning("Falha ao atualizar visual do motor selecionado: %s", e)

    def _refresh_all_voices(self) -> None:
        """Recarrega a lista de vozes da pasta voices/ e atualiza ambas as sub-abas."""
        self._refresh_engine_catalog()
        voices = list_all_saved_voices()

        # 1. Atualiza ComboBox da Sub-Aba 1 (Síntese)
        cur_selected = self.cb_synthesis_voices.currentData()
        self.cb_synthesis_voices.clear()
        for v in voices:
            self.cb_synthesis_voices.addItem(f"🎙️ {v['display_name']} ({v['duration']:.1f}s)", v["audio_path"])

        if cur_selected:
            idx = self.cb_synthesis_voices.findData(cur_selected)
            if idx >= 0:
                self.cb_synthesis_voices.setCurrentIndex(idx)

        # 2. Atualiza Tabela da Sub-Aba 2 (Gerenciador)
        self.lbl_voices_count.setText(f"Vozes Salvas ({len(voices)})")
        self.table_voices.setRowCount(0)

        for row, v in enumerate(voices):
            self.table_voices.insertRow(row)

            # Nome
            item_name = QTableWidgetItem(v["display_name"])
            item_name.setToolTip(v["transcript"] if v["transcript"] else v["audio_path"])
            self.table_voices.setItem(row, 0, item_name)

            # Duração
            item_dur = QTableWidgetItem(f"{v['duration']:.1f}s")
            item_dur.setTextAlignment(Qt.AlignCenter)
            self.table_voices.setItem(row, 1, item_dur)

            # Tamanho
            item_sz = QTableWidgetItem(f"{v['size_kb']} KB")
            item_sz.setTextAlignment(Qt.AlignCenter)
            self.table_voices.setItem(row, 2, item_sz)

            # Data
            item_dt = QTableWidgetItem(v["date_str"])
            item_dt.setTextAlignment(Qt.AlignCenter)
            self.table_voices.setItem(row, 3, item_dt)

            # Ações
            act_widget = QWidget()
            act_layout = QHBoxLayout(act_widget)
            act_layout.setContentsMargins(2, 2, 2, 2)
            act_layout.setSpacing(4)

            btn_play = QPushButton("▶️ Ouvir")
            btn_play.setFixedHeight(24)
            btn_play.setStyleSheet("font-size: 11px; padding: 2px 6px;")
            btn_play.clicked.connect(lambda _, ap=v["audio_path"], vn=v["display_name"]: self._play_preview(ap, vn))
            act_layout.addWidget(btn_play)

            btn_ren = QPushButton("✏️")
            btn_ren.setToolTip("Renomear voz")
            btn_ren.setFixedSize(24, 24)
            btn_ren.clicked.connect(lambda _, old_n=v["name"]: self._rename_voice_dialog(old_n))
            act_layout.addWidget(btn_ren)

            btn_upd = QPushButton("🔄")
            btn_upd.setToolTip("Atualizar áudio de referência desta voz")
            btn_upd.setFixedSize(24, 24)
            btn_upd.clicked.connect(lambda _, vn=v["name"]: self._update_voice_sample(vn))
            act_layout.addWidget(btn_upd)

            btn_del = QPushButton("🗑️")
            btn_del.setToolTip("Excluir esta voz permanentemente")
            btn_del.setFixedSize(24, 24)
            btn_del.setStyleSheet("color: #FF5555;")
            btn_del.clicked.connect(lambda _, vn=v["name"]: self._delete_voice_dialog(vn))
            act_layout.addWidget(btn_del)

            self.table_voices.setCellWidget(row, 4, act_widget)

    def _rename_voice_dialog(self, old_name: str) -> None:
        new_name, ok = QInputDialog.getText(
            self,
            "Renomear Voz",
            f"Digite o novo nome para '{old_name}':",
            text=old_name,
        )
        if ok and new_name.strip() and new_name.strip() != old_name:
            if rename_saved_voice(old_name, new_name.strip()):
                self._refresh_all_voices()
                QMessageBox.information(self, "Voz Renomeada", f"Voz renomeada para '{new_name.strip()}'.")
            else:
                QMessageBox.warning(self, "Falha", "Não foi possível renomear a voz selecionada.")

    def _update_voice_sample(self, voice_name: str) -> None:
        last = _load_last_dir("narration_ref_audio")
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Atualizar Amostra de Áudio para '{voice_name}'",
            last,
            "Áudios Suportados (*.wav *.mp3 *.flac *.m4a);;Todos (*.*)",
        )
        if not path:
            return

        try:
            save_cloned_voice(voice_name=voice_name, audio_path=path)
            self._refresh_all_voices()
            QMessageBox.information(self, "Amostra Atualizada", f"Áudio de referência da voz '{voice_name}' atualizado com sucesso!")
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Atualizar", f"Falha ao atualizar o áudio:\n{e}")

    def _delete_voice_dialog(self, voice_name: str) -> None:
        ans = QMessageBox.question(
            self,
            "Excluir Voz",
            f"Tem certeza que deseja excluir permanentemente a voz '{voice_name}' do sistema?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans == QMessageBox.Yes:
            self._stop_preview()
            if delete_saved_voice(voice_name):
                self._refresh_all_voices()
                QMessageBox.information(self, "Voz Excluída", f"Voz '{voice_name}' removida com sucesso.")
            else:
                QMessageBox.warning(self, "Falha", "Não foi possível remover o arquivo de voz.")

    # -----------------------------------------------------------------------
    # OPERAÇÕES DE NARRAÇÃO, FILA E EXECUÇÃO
    # -----------------------------------------------------------------------

    def _on_srt_blocks_mode_changed(self) -> None:
        """Controla a habilitação dos campos de intervalo de blocos SRT."""
        if not hasattr(self, "rb_blocks_range"):
            return
        is_range = self.rb_blocks_range.isChecked()
        is_custom = self.rb_blocks_custom.isChecked()
        self.spin_block_from.setEnabled(is_range)
        self.spin_block_to.setEnabled(is_range)
        self.txt_blocks_custom.setEnabled(is_custom)

    def _on_text_changed(self) -> None:
        text = self.txt_content.toPlainText()
        fmt = detect_text_format(text)
        if fmt == "srt":
            self.lbl_format_detected.setText("⏱️ Formato detectado: Legenda (.srt)")
            self.lbl_format_detected.setStyleSheet("color: #FFA200; font-weight: bold;")
            self.grp_srt_options.setVisible(True)

            # Analisa quantidade de blocos detectados
            from core.narration import parse_srt
            segs = parse_srt(text)
            total = len(segs)
            self.lbl_srt_info.setText(f"⏱️ Legenda detectada: {total} bloco(s) de áudio")
            self.spin_block_from.setRange(1, max(1, total))
            self.spin_block_to.setRange(1, max(1, total))
            if not self.rb_blocks_range.isChecked() and not self.rb_blocks_custom.isChecked():
                self.spin_block_to.setValue(max(1, total))
        else:
            self.lbl_format_detected.setText("📄 Formato detectado: Texto Puro (.txt)")
            self.lbl_format_detected.setStyleSheet("color: #04D361; font-weight: bold;")
            self.grp_srt_options.setVisible(False)

    def _on_dest_mode_changed(self) -> None:
        save_source = self.chk_save_source_dir.isChecked() and self.chk_save_source_dir.isEnabled()
        self.txt_dest_folder.setEnabled(not save_source)
        self.btn_browse_dest.setEnabled(not save_source)

    def _import_file(self) -> None:
        last = _load_last_dir("narration_source_file")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Importar Arquivo para Narração",
            last,
            "Arquivos Suportados (*.txt *.srt);;Legendas SRT (*.srt);;Texto Puro (*.txt);;Todos (*.*)",
        )
        if not path:
            return

        _save_last_dir("narration_source_file", str(Path(path).parent))

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            self.txt_content.setPlainText(content)
            self.current_source_file_path = path

            self.chk_save_source_dir.setEnabled(True)
            self.chk_save_source_dir.setChecked(True)
            self._on_dest_mode_changed()

            self.log_signal.emit(f"Arquivo importado: {os.path.basename(path)} ({len(content)} caracteres)")
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Importar", f"Falha ao ler o arquivo selecionado:\n{e}")

    def _clear_input(self) -> None:
        self.txt_content.clear()
        self.current_source_file_path = None
        self.chk_save_source_dir.setChecked(False)
        self.chk_save_source_dir.setEnabled(False)
        self._on_dest_mode_changed()

    def _browse_dest_folder(self) -> None:
        last = _load_last_dir("narration_dest_folder")
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Destino", last)
        if folder:
            self.txt_dest_folder.setText(folder)
            _save_last_dir("narration_dest_folder", folder)

    @property
    def txt_ref_audio(self) -> QLineEdit:
        """Propriedade de compatibilidade para código legado ou testes unitários."""
        return self.txt_clone_file

    def _add_current_to_queue(self) -> None:
        text = self.txt_content.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Texto Vazio", "Por favor, digite, cole ou importe um texto/SRT para narração.")
            return

        ref_audio = self.cb_synthesis_voices.currentData()
        if not ref_audio or not os.path.isfile(str(ref_audio)):
            # Fallback para áudio de clonagem se preenchido (compatibilidade de automação e testes)
            clone_audio = self.txt_clone_file.text().strip()
            if clone_audio and os.path.isfile(clone_audio):
                ref_audio = clone_audio
            else:
                QMessageBox.warning(
                    self,
                    "Voz Não Selecionada",
                    "Selecione uma voz clonada disponível ou clone uma nova voz na aba ao lado.",
                )
                return

        fmt = detect_text_format(text)
        split_mode = "separado" if fmt == "srt" and self.rb_srt_split.isChecked() else "unico"
        save_source = self.chk_save_source_dir.isChecked() and self.chk_save_source_dir.isEnabled()
        dest_dir = self.txt_dest_folder.text().strip() or str(Path.home() / "Downloads")

        srt_range_str: Optional[str] = None
        if fmt == "srt":
            if hasattr(self, "rb_blocks_range") and self.rb_blocks_range.isChecked():
                b_from = self.spin_block_from.value()
                b_to = self.spin_block_to.value()
                srt_range_str = f"{b_from}-{b_to}"
            elif hasattr(self, "rb_blocks_custom") and self.rb_blocks_custom.isChecked():
                srt_range_str = self.txt_blocks_custom.text().strip() or None

        speed = self.slider_speed.value() / 100.0
        pause_sec = self.slider_pause.value() / 100.0
        mastering_cfg = self._get_current_mastering_config()
        selected_engine = self.cb_tts_engine.currentData() if hasattr(self, "cb_tts_engine") and self.cb_tts_engine.currentData() else "f5-tts"

        from core.tts_catalog import get_engine_meta, is_engine_operational
        is_op, op_exp = is_engine_operational(selected_engine)
        if not is_op:
            meta = get_engine_meta(selected_engine)
            meta_name = meta.name if meta else selected_engine
            ans = QMessageBox.question(
                self,
                "Motor TTS Não Operacional",
                f"O motor selecionado '{meta_name}' não está pronto para execução imediata neste ambiente:\n\n"
                f"• {op_exp}\n\n"
                f"Deseja alternar automaticamente para o motor padrão F5-TTS v1 Base [🟢 Operacional]?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans == QMessageBox.Yes:
                selected_engine = "f5-tts"
                f5_idx = self.cb_tts_engine.findData("f5-tts")
                if f5_idx >= 0:
                    self.cb_tts_engine.setCurrentIndex(f5_idx)
            else:
                return

        export_raw = self.chk_export_raw_wav.isChecked() if hasattr(self, "chk_export_raw_wav") else True

        job_id = f"narr_job_{int(time.time() * 1000)}_{len(self.queue_jobs)}"
        job = NarrationJob(
            job_id=job_id,
            source_text=text,
            source_format=fmt,
            source_file_path=self.current_source_file_path,
            voice_mode="clone",
            reference_audio_path=ref_audio,
            selected_engine=selected_engine,
            split_mode=split_mode,
            srt_range=srt_range_str,
            speech_speed=speed,
            sentence_pause_seconds=pause_sec,
            mastering_config=mastering_cfg,
            export_raw_wav=export_raw,
            destination_folder=dest_dir,
            save_to_source_folder=save_source,
            create_audio_subfolder=self.chk_audio_subfolder.isChecked(),
        )

        self.queue_jobs[job_id] = job
        self._insert_job_row(job)
        range_info = f", Blocos: {srt_range_str}" if srt_range_str else ""
        self.log_signal.emit(f"Item adicionado à fila: {job_id} ({fmt.upper()}{range_info}, {speed:.2f}x, motor: {selected_engine})")

        # Aviso ao usuário quando a referência de voz é curta (< 8s)
        # Referências de 10-15s fornecem melhor fidelidade de clonagem no F5-TTS
        if ref_audio:
            try:
                from core.voice_clone import get_audio_duration
                ref_dur = get_audio_duration(ref_audio)
                if 0 < ref_dur < 8.0:
                    self.log_signal.emit(
                        f"⚠️ Referência de voz curta ({ref_dur:.1f}s). "
                        f"Para melhor fidelidade de clonagem, use uma referência de 10-15 segundos."
                    )
            except Exception:
                pass

    def _insert_job_row(self, job: NarrationJob) -> None:
        row = self.table_queue.rowCount()
        self.table_queue.insertRow(row)

        origin_label = (
            os.path.basename(job.source_file_path)
            if job.source_file_path
            else f"Texto ({len(job.source_text)} chars)"
        )
        item_orig = QTableWidgetItem(origin_label)
        item_orig.setData(Qt.UserRole, job.job_id)
        item_orig.setToolTip(job.source_text[:200] + "...")
        self.table_queue.setItem(row, 0, item_orig)

        # Voz
        voice_label = Path(job.reference_audio_path).stem if job.reference_audio_path else "Padrão"
        item_voice = QTableWidgetItem(voice_label.replace("_", " ").title())
        item_voice.setTextAlignment(Qt.AlignCenter)
        self.table_queue.setItem(row, 1, item_voice)

        # Velocidade
        item_spd = QTableWidgetItem(f"{job.speech_speed:.2f}x")
        item_spd.setTextAlignment(Qt.AlignCenter)
        self.table_queue.setItem(row, 2, item_spd)

        # Status
        item_status = QTableWidgetItem("⏳ Pendente")
        item_status.setTextAlignment(Qt.AlignCenter)
        self.table_queue.setItem(row, 3, item_status)

        # Progresso
        prog = QProgressBar()
        prog.setRange(0, 100)
        prog.setValue(0)
        prog.setFixedWidth(100)
        self.table_queue.setCellWidget(row, 4, prog)

        # Ações
        actions_widget = QWidget()
        act_layout = QHBoxLayout(actions_widget)
        act_layout.setContentsMargins(2, 2, 2, 2)
        btn_del = QPushButton("Remover")
        btn_del.setToolTip("Remover esta narração da fila")
        btn_del.setFixedHeight(24)
        btn_del.setStyleSheet(
            "QPushButton { background-color: #3a3a40; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 3px; padding: 2px 6px; font-size: 11px; } "
            "QPushButton:hover { background-color: #e74c3c; color: white; }"
        )
        btn_del.clicked.connect(lambda _, jid=job.job_id: self._remove_job(jid))
        act_layout.addWidget(btn_del)
        self.table_queue.setCellWidget(row, 5, actions_widget)

        self._update_queue_label()

    def _remove_job(self, job_id: str) -> None:
        if job_id not in self.queue_jobs:
            return
        del self.queue_jobs[job_id]
        for row in range(self.table_queue.rowCount()):
            item = self.table_queue.item(row, 0)
            if item and item.data(Qt.UserRole) == job_id:
                self.table_queue.removeRow(row)
                break
        self._update_queue_label()

    def _clear_completed_jobs(self) -> None:
        done_ids = [jid for jid, j in self.queue_jobs.items() if j.status in ("Concluído", "Cancelado")]
        for jid in done_ids:
            self._remove_job(jid)

    def _clear_all_jobs(self) -> None:
        pending_ids = [jid for jid, j in self.queue_jobs.items() if j.status != "Processando..."]
        for jid in pending_ids:
            self._remove_job(jid)

    def _update_queue_label(self) -> None:
        total = len(self.queue_jobs)
        pending = sum(1 for j in self.queue_jobs.values() if j.status == "Pendente")
        self.lbl_queue_header.setText(f"Fila de Narrações ({total} total, {pending} pendentes)")

    def _start_queue_processing(self) -> None:
        pending = [j for j in self.queue_jobs.values() if j.status == "Pendente"]
        if not pending:
            QMessageBox.information(self, "Fila Vazia", "Nenhuma narração pendente na fila.")
            return

        self.btn_process_queue.setEnabled(False)
        self.btn_cancel_queue.setEnabled(True)
        self.prog_bar.setValue(0)
        self.lbl_job_stage.setText("Iniciando processamento da fila de narração...")

        hw = detect_hardware()
        if getattr(sys, "frozen", False):
            models_dir = str((Path(sys.executable).parent / "models").resolve())
        else:
            models_dir = str((Path(__file__).parent.parent / "models").resolve())

        self.worker_thread = NarrationWorkerThread(
            jobs=pending,
            model_profile=hw.model_profile,
            models_dir=models_dir,
            parent=self,
        )
        self.worker_thread.progress_signal.connect(self._on_job_progress)
        self.worker_thread.job_finished_signal.connect(self._on_job_finished)
        self.worker_thread.job_error_signal.connect(self._on_job_error)
        self.worker_thread.queue_completed_signal.connect(self._on_queue_completed)


        for job in pending:
            job.status = "Processando..."
            self._update_job_row(job.job_id)

        self.worker_thread.start()
        self.log_signal.emit(f"🚀 Fila iniciada com {len(pending)} tarefas.")



    def _update_job_row(self, job_id: str) -> None:
        for row in range(self.table_queue.rowCount()):
            item = self.table_queue.item(row, 0)
            if item and item.data(Qt.UserRole) == job_id:
                job = self.queue_jobs[job_id]
                status_item = self.table_queue.item(row, 3)
                if status_item:
                    status_item.setText(job.status)
                cell_prog = self.table_queue.cellWidget(row, 4)
                if isinstance(cell_prog, QProgressBar):
                    cell_prog.setValue(int(job.progress * 100))
                break

    def _on_job_progress(self, job_id: str, pct: float, msg: str) -> None:
        if job_id in self.queue_jobs:
            self.queue_jobs[job_id].progress = pct
            self.queue_jobs[job_id].status_message = msg
            self._update_job_row(job_id)

        self.prog_bar.setValue(int(pct * 100))
        self.lbl_job_stage.setText(f"Etapa: {msg}")

    def _on_job_finished(self, job_id: str, outputs: list) -> None:
        if job_id in self.queue_jobs:
            job = self.queue_jobs[job_id]
            job.status = "Concluído"
            job.progress = 1.0
            job.output_files = outputs
            self._update_job_row(job_id)

        self.log_signal.emit(f"✅ Tarefa {job_id} concluída. {len(outputs)} arquivo(s) gerado(s).")

    def _on_job_error(self, job_id: str, err: str) -> None:
        if job_id in self.queue_jobs:
            job = self.queue_jobs[job_id]
            job.status = f"Erro"
            self._update_job_row(job_id)

        self.lbl_job_stage.setText(f"❌ Erro na narração: {err[:80]}")
        self.log_signal.emit(f"❌ Erro na narração ({job_id}): {err}")

        # Popup de notificação detalhada para o usuário
        QMessageBox.critical(
            self,
            "Erro na Geração de Narração",
            f"A operação de narração foi cancelada devido a um erro:\n\n{err}\n\n"
            f"Verifique se o motor TTS selecionado está corretamente instalado e operacional.",
        )

    def _on_queue_completed(self) -> None:
        self.btn_process_queue.setEnabled(True)
        self.btn_cancel_queue.setEnabled(False)
        self.prog_bar.setValue(100)
        self.lbl_job_stage.setText("Fila de narração finalizada com sucesso!")
        self.worker_thread = None
        self._update_queue_label()
        self.log_signal.emit("🏁 Fila de narração concluída.")

    def _cancel_queue_processing(self) -> None:
        """Cancela a fila de narração imediatamente, interrompendo e descartando o áudio atual."""
        if self.worker_thread:
            self.log_signal.emit("⏹️ Cancelando geração imediatamente...")
            self.lbl_job_stage.setText("⏹️ Interrompendo geração...")

            self.worker_thread.cancel()

            if self.worker_thread.isRunning():
                logger.info("Encerrando thread de narração forçosamente (cancelamento imediato)...")
                self.worker_thread.terminate()
                self.worker_thread.wait(400)
                self.log_signal.emit("⏹️ Thread interrompida imediatamente.")

            self.worker_thread = None

            for jid, job in self.queue_jobs.items():
                if job.status in ("Pendente", "Processando..."):
                    job.status = "Cancelado"
                    self._update_job_row(jid)

            self.btn_process_queue.setEnabled(True)
            self.btn_cancel_queue.setEnabled(False)
            self.prog_bar.setValue(0)
            self.lbl_job_stage.setText("⏹️ Processamento cancelado. Áudio parcial descartado.")
            self._update_queue_label()
            self.log_signal.emit("⏹️ Fila de narração cancelada e descartada.")

            self._cleanup_torch()

    def cleanup(self) -> None:
        """Encerramento limpo: cancela threads, para reprodução e libera VRAM."""
        self._stop_preview()
        if self.worker_thread and self.worker_thread.isRunning():
            logger.info("Encerrando thread de narração (cleanup)...")
            self.worker_thread.cancel()
            if not self.worker_thread.wait(1500):
                self.worker_thread.terminate()
                self.worker_thread.wait(500)
            self.worker_thread = None

        self._cleanup_torch()

    @staticmethod
    def _cleanup_torch() -> None:
        """Libera VRAM e memória do PyTorch."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
