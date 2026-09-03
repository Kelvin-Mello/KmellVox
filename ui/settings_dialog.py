"""Diálogo de Configurações e Preferências do KmellVox com downloader de modelos integrado (PySide6)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.dependency_manager import (
    DependencyStatus,
    check_all_dependencies,
    install_f5tts,
    install_indextts,
    install_pytorch,
    install_smart_all,
    install_tts_dependencies,
)
from core.hardware import PROFILE_DISPLAY_NAMES, ModelProfile, detect_gpu_profile, detect_hardware
from downloader.fetch_models import MODEL_CATALOG, check_models_status, fetch_models_for_profile

logger = logging.getLogger("KmellVox.Settings")


class ModelDownloaderWorker(QThread):
    """Thread em segundo plano para download de modelos sem congelar o diálogo."""

    progress_signal = Signal(float, str)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(
        self,
        profile: str,
        models_dir: str,
        config_path: str,
        target_model_key: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.models_dir = models_dir
        self.config_path = config_path
        self.target_model_key = target_model_key

    def run(self) -> None:
        try:
            def on_progress(pct: float, msg: str) -> None:
                self.progress_signal.emit(pct, msg)

            saved_paths = fetch_models_for_profile(
                profile=self.profile,
                base_models_dir=self.models_dir,
                config_path=self.config_path,
                progress_callback=on_progress,
                target_model_key=self.target_model_key,
            )
            self.finished_signal.emit(saved_paths)
        except Exception as e:
            self.error_signal.emit(str(e))


class ModelCheckerWorker(QThread):
    """Thread em segundo plano para verificar presença e integridade dos modelos no disco."""

    finished_signal = Signal(list)

    def __init__(self, profile: str, models_dir: str, config_path: str, parent=None) -> None:
        super().__init__(parent)
        self.profile = profile
        self.models_dir = models_dir
        self.config_path = config_path

    def run(self) -> None:
        try:
            statuses = check_models_status(
                profile=self.profile,
                base_models_dir=self.models_dir,
                config_path=self.config_path,
            )
            self.finished_signal.emit(statuses)
        except Exception as e:
            logger.warning("Erro ao verificar status dos modelos: %s", e)
            self.finished_signal.emit([])


class DependencyCheckerWorker(QThread):
    """Thread para verificar status de dependências sem congelar a UI."""

    finished_signal = Signal(list)  # List[DependencyStatus]

    def __init__(self, check_updates: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.check_updates = check_updates

    def run(self) -> None:
        try:
            deps = check_all_dependencies(check_updates=self.check_updates)
            self.finished_signal.emit(deps)
        except Exception as e:
            logger.warning("Erro ao verificar dependências: %s", e)
            self.finished_signal.emit([])


class DependencyInstallerWorker(QThread):
    """Thread para instalar dependências (PyTorch, F5-TTS, IndexTTS ou tudo) sem congelar a UI."""

    progress_signal = Signal(float, str)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, target: str = "all", parent=None) -> None:
        super().__init__(parent)
        self.target = target

    def run(self) -> None:
        try:
            def on_progress(pct: float, msg: str) -> None:
                self.progress_signal.emit(pct, msg)

            if self.target == "torch":
                result = install_pytorch(progress_callback=on_progress)
            elif self.target == "f5_tts":
                result = install_f5tts(progress_callback=on_progress)
            elif self.target == "index_tts":
                result = install_indextts(progress_callback=on_progress)
            else:
                result = install_smart_all(progress_callback=on_progress)

            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))


class SettingsDialog(QDialog):
    """Diálogo para visualização de hardware, download de pesos e edição de parâmetros do config.yaml."""

    def __init__(self, config_path: str = "config.yaml", parent=None) -> None:
        super().__init__(parent)
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = self._load_config()
        self.setWindowTitle("Configurações & Modelos - KmellVox")
        self.resize(980, 690)
        self.setMinimumWidth(880)
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

        try:
            hw_info = detect_hardware()
            gpu_label = f"{hw_info.device_name} ({hw_info.vram_total_gb:.1f} GB VRAM)" if hw_info.cuda_available else "Nenhuma GPU CUDA compatível (Modo CPU)"
            cuda_ok = hw_info.cuda_available
        except Exception as e:
            logger.warning("Falha ao detectar hardware no SettingsDialog: %s", e)
            gpu_label = "Erro ao detectar hardware"
            cuda_ok = False
        lbl_hw_detect = QLabel(gpu_label)
        lbl_hw_detect.setStyleSheet("font-weight: bold; color: #04D361;" if cuda_ok else "font-weight: bold; color: #FFA200;")
        hw_layout.addRow("Dispositivo Detectado:", lbl_hw_detect)

        # ComboBox com nomes amigáveis (mapeados internamente para perfil_a/perfil_b/cpu)
        self.cb_profile = QComboBox()
        self._profile_items = [("auto", "Automático")] + [
            (key, display) for key, display in PROFILE_DISPLAY_NAMES.items()
        ]
        for _key, display in self._profile_items:
            self.cb_profile.addItem(display)

        saved_profile = self.config_data.get("gpu_profile") or self.config_data.get("hardware", {}).get("profile", "auto")
        # Seleciona o item correto baseado no valor salvo
        for idx, (key, _disp) in enumerate(self._profile_items):
            if key == saved_profile:
                self.cb_profile.setCurrentIndex(idx)
                break

        self.cb_profile.currentIndexChanged.connect(self._on_profile_changed)
        hw_layout.addRow("Perfil de Hardware (VRAM):", self.cb_profile)

        self.lbl_profile_details = QLabel()
        self.lbl_profile_details.setStyleSheet("color: #A8A8B3; font-size: 11px;")
        hw_layout.addRow("Modelos Atribuídos:", self.lbl_profile_details)
        self._update_profile_details()

        layout.addWidget(hw_group)

        # 1.5. Status de Componentes (Ambiente de Execução)
        status_group = QGroupBox("Status de Componentes (Bibliotecas e Motores)")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(6)

        self.dep_labels: Dict[str, QLabel] = {}
        self.dep_action_btns: Dict[str, QPushButton] = {}
        self.dep_badges: Dict[str, QLabel] = {}
        self._last_checked_deps: list = []

        _default_deps = [
            ("torch", "PyTorch (GPU/CUDA)", True, "PyTorch"),
            ("faster_whisper", "Faster-Whisper (Transcrição)", False, ""),
            ("llama_cpp", "Llama-CPP (Tradução LLM)", False, ""),
            ("f5_tts", "F5-TTS (Clonagem de Voz)", True, "F5-TTS"),
            ("index_tts", "IndexTTS-2 (Voz Avançada)", True, "IndexTTS-2"),
            ("ffmpeg", "FFmpeg (Processamento A/V)", False, ""),
        ]

        for key, name, can_install, btn_label in _default_deps:
            row = QHBoxLayout()
            lbl = QLabel(f"⏳  {name}  —  Verificando...")
            lbl.setStyleSheet("color: #A8A8B3; font-size: 12px; padding: 2px 0;")
            row.addWidget(lbl)
            self.dep_labels[key] = lbl
            row.addStretch()

            if can_install:
                btn = QPushButton(f"⬇️ Baixar {btn_label}")
                btn.setStyleSheet(
                    "background-color: #29292E; color: #E1E1E6; border: 1px solid #3F3F46; "
                    "padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;"
                )
                btn.clicked.connect(lambda _, k=key, n=name: self._start_individual_install(k, n))
                btn.setVisible(False)  # Oculto durante a verificação inicial
                row.addWidget(btn)
                self.dep_action_btns[key] = btn

                lbl_badge = QLabel("(Verificando...)")
                lbl_badge.setStyleSheet("color: #71717A; font-size: 11px; font-style: italic;")
                lbl_badge.setVisible(True)
                row.addWidget(lbl_badge)
                self.dep_badges[key] = lbl_badge
            else:
                lbl_badge = QLabel("(Integrado)")
                lbl_badge.setStyleSheet("color: #71717A; font-size: 11px; font-style: italic;")
                row.addWidget(lbl_badge)
                self.dep_badges[key] = lbl_badge

            status_layout.addLayout(row)

        self._dep_checker = DependencyCheckerWorker(parent=self)
        self._dep_checker.finished_signal.connect(self._on_deps_checked)
        self._dep_checker.start()

        # Botão Inteligente para Baixar / Atualizar Todas as Dependências
        install_row = QHBoxLayout()
        self.btn_install_all = QPushButton("⏳ Verificando componentes e atualizações...")
        self.btn_install_all.setStyleSheet(
            "background-color: #29292E; color: #A8A8B3; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px; border: 1px solid #3F3F46;"
        )
        self.btn_install_all.setEnabled(False)  # Desabilitado enquanto verifica
        self.btn_install_all.setToolTip(
            "Aguarde a verificação dos componentes instalados..."
        )
        self.btn_install_all.clicked.connect(self._start_smart_all_install)
        install_row.addWidget(self.btn_install_all)
        install_row.addStretch()
        status_layout.addLayout(install_row)

        # Progresso de instalação
        self.lbl_install_status = QLabel("")
        self.lbl_install_status.setStyleSheet("color: #A8A8B3; font-size: 11px;")
        self.lbl_install_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_install_status)

        self.prog_bar_install = QProgressBar()
        self.prog_bar_install.setRange(0, 100)
        self.prog_bar_install.setValue(0)
        self.prog_bar_install.setFixedHeight(14)
        self.prog_bar_install.setVisible(False)
        status_layout.addWidget(self.prog_bar_install)

        # 2. Ferramentas e Executáveis
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

        # 3. Preferências de Pipeline
        pref_group = QGroupBox("Preferências Gerais do Pipeline")
        pref_layout = QFormLayout(pref_group)

        self.chk_burn_subs = QCheckBox("Estampar legendas no vídeo final (Burn-in) por padrão")
        self.chk_burn_subs.setChecked(self.config_data.get("pipeline", {}).get("subtitle_burn_in", False))
        pref_layout.addRow("", self.chk_burn_subs)

        self.chk_keep_temp = QCheckBox("Manter arquivos temporários de trabalho após conclusão")
        self.chk_keep_temp.setChecked(self.config_data.get("pipeline", {}).get("keep_temp_files", False))
        pref_layout.addRow("", self.chk_keep_temp)

        # 4. Gerenciamento e Download de Modelos (Pesos de IA)
        models_group = QGroupBox("Armazenamento e Download de Modelos (Pesos Neurais)")
        models_layout = QVBoxLayout(models_group)

        row_md = QHBoxLayout()
        row_md.addWidget(QLabel("Diretório de Modelos:"))
        self.txt_models = QLineEdit()
        self.txt_models.setText(self.config_data.get("paths", {}).get("models_dir", "models"))
        self.txt_models.textChanged.connect(lambda _: self._check_models_in_background())
        btn_models = QPushButton("Procurar...")
        btn_models.clicked.connect(self._browse_models_dir)
        row_md.addWidget(self.txt_models)
        row_md.addWidget(btn_models)
        models_layout.addLayout(row_md)

        # Container para a lista dinâmica de modelos do perfil
        self.models_list_container = QWidget()
        self.models_list_layout = QVBoxLayout(self.models_list_container)
        self.models_list_layout.setContentsMargins(0, 4, 0, 4)
        models_layout.addWidget(self.models_list_container)

        self.model_labels: Dict[str, QLabel] = {}
        self.model_action_btns: Dict[str, QPushButton] = {}
        self.model_badges: Dict[str, QLabel] = {}

        # Botão Inteligente para Baixar / Atualizar Modelos
        row_dl = QHBoxLayout()
        self.btn_download_models = QPushButton("⏳ Verificando modelos locais...")
        self.btn_download_models.setStyleSheet(
            "background-color: #29292E; color: #A8A8B3; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px; border: 1px solid #3F3F46;"
        )
        self.btn_download_models.setEnabled(False)
        self.btn_download_models.clicked.connect(self._start_smart_models_action)
        row_dl.addWidget(self.btn_download_models)
        row_dl.addStretch()
        models_layout.addLayout(row_dl)

        # Progresso de download
        self.lbl_dl_status = QLabel("")
        self.lbl_dl_status.setStyleSheet("color: #A8A8B3; font-size: 11px;")
        self.lbl_dl_status.setWordWrap(True)
        models_layout.addWidget(self.lbl_dl_status)

        self.prog_bar_dl = QProgressBar()
        self.prog_bar_dl.setRange(0, 100)
        self.prog_bar_dl.setValue(0)
        self.prog_bar_dl.setFixedHeight(16)
        self.prog_bar_dl.setVisible(False)
        models_layout.addWidget(self.prog_bar_dl)

        # ─── Estrutura em Duas Colunas: Componentes (Esq) vs Modelos (Dir) ────
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(14)

        col_left = QVBoxLayout()
        col_left.setSpacing(10)
        col_left.addWidget(status_group)
        col_left.addWidget(tools_group)
        col_left.addWidget(pref_group)
        col_left.addStretch()

        col_right = QVBoxLayout()
        col_right.setSpacing(10)
        col_right.addWidget(models_group)
        col_right.addStretch()

        columns_layout.addLayout(col_left, stretch=1)
        columns_layout.addLayout(col_right, stretch=1)
        layout.addLayout(columns_layout)

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

        # Inicia a verificação dos modelos locais em segundo plano
        self._check_models_in_background()

    def _get_selected_profile_key(self) -> str:
        """Retorna o nome interno do perfil selecionado no combo (perfil_a, perfil_b, cpu, ou auto)."""
        idx = self.cb_profile.currentIndex()
        if 0 <= idx < len(self._profile_items):
            return self._profile_items[idx][0]
        return "auto"

    def _on_profile_changed(self) -> None:
        self._update_profile_details()
        self._check_models_in_background()

    def _update_profile_details(self) -> None:
        prof_key = self._get_selected_profile_key()
        mp = ModelProfile.from_profile(prof_key if prof_key != "auto" else None)
        indextts_txt = "Habilitado" if mp.enable_indextts_2 else "F5-TTS Padrão"
        fp16_txt = "FP16" if mp.musetalk_use_float16 else "Opcional"
        info = (
            f"• Whisper: {mp.whisper_variant} ({mp.whisper_compute_type})   "
            f"• Tradutor: {mp.translation_model}   "
            f"• IndexTTS-2: {indextts_txt}   "
            f"• MuseTalk: {fp16_txt}"
        )
        self.lbl_profile_details.setText(info)

    def _on_deps_checked(self, deps: list) -> None:
        """Chamado pela thread de background ao concluir a verificação de dependências."""
        self._refresh_dependency_status(deps)

    def _recheck_deps_in_background(self) -> None:
        """Relança verificação de dependências em background para atualizar labels existentes."""
        for key, btn in self.dep_action_btns.items():
            btn.setVisible(False)
            badge = self.dep_badges.get(key)
            if badge:
                badge.setText("(Verificando...)")
                badge.setStyleSheet("color: #71717A; font-size: 11px; font-style: italic;")
                badge.setVisible(True)
            if key in self.dep_labels:
                lbl = self.dep_labels[key]
                current_text = lbl.text()
                comp_name = current_text.split("—")[0].replace("✅", "").replace("❌", "").replace("⏳", "").strip()
                lbl.setText(f"⏳  {comp_name}  —  Verificando...")
                lbl.setStyleSheet("color: #A8A8B3; font-size: 12px; padding: 2px 0;")

        if hasattr(self, "btn_install_all"):
            self.btn_install_all.setText("⏳ Verificando componentes e atualizações...")
            self.btn_install_all.setStyleSheet(
                "background-color: #29292E; color: #A8A8B3; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px; border: 1px solid #3F3F46;"
            )
            self.btn_install_all.setEnabled(False)

        checker = DependencyCheckerWorker(parent=self)
        checker.finished_signal.connect(lambda deps: self._refresh_dependency_status(deps))
        checker.start()
        # Guarda referência para evitar GC
        self._dep_rechecker = checker

    def _refresh_dependency_status(self, deps: list, layout: Optional[QVBoxLayout] = None) -> None:
        """Atualiza labels de status e botões de ação com a lista de dependências verificadas."""
        self._last_checked_deps = deps
        for dep in deps:
            icon = "✅" if dep.installed else "❌"
            version_txt = dep.version if dep.installed else "Não instalado"
            if dep.installed and getattr(dep, "has_update", False) and getattr(dep, "latest_version", ""):
                version_txt += f"  (Nova versão: {dep.latest_version})"
            text = f"{icon}  {dep.display_name}  —  {version_txt}"
            color = "#04D361" if dep.installed else "#FF6961"

            if dep.name in self.dep_labels:
                lbl = self.dep_labels[dep.name]
                lbl.setText(text)
                lbl.setStyleSheet(f"color: {color}; font-size: 12px; padding: 2px 0;")

            # Atualiza botão individual e badge da dependência
            has_btn = dep.name in self.dep_action_btns
            badge = self.dep_badges.get(dep.name)

            if has_btn:
                btn = self.dep_action_btns[dep.name]
                if not dep.installed:
                    # Não instalado: botão Baixar fica visível
                    btn.setVisible(True)
                    btn.setText("⬇️ Baixar")
                    btn.setStyleSheet(
                        "background-color: #332616; color: #FFA200; border: 1px solid #FFA200; "
                        "padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;"
                    )
                    if badge:
                        badge.setVisible(False)
                elif getattr(dep, "has_update", False):
                    # Instalado E possui atualização real: exibe o botão Atualizar
                    btn.setVisible(True)
                    tag = f" ({dep.latest_version})" if getattr(dep, "latest_version", "") else ""
                    btn.setText(f"🔄 Atualizar{tag}")
                    btn.setStyleSheet(
                        "background-color: #1F2723; color: #04D361; border: 1px solid #04D361; "
                        "padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;"
                    )
                    if badge:
                        badge.setVisible(False)
                else:
                    # Instalado e SEM atualização: oculta o botão e exibe badge (Atualizado)
                    btn.setVisible(False)
                    if badge:
                        badge.setText("(Atualizado)")
                        badge.setStyleSheet("color: #71717A; font-size: 11px; font-style: italic;")
                        badge.setVisible(True)

        # Atualiza status do botão mestre
        missing_installables = [d for d in deps if not d.installed and d.name in ("torch", "f5_tts", "index_tts")]
        available_updates = [d for d in deps if d.installed and getattr(d, "has_update", False) and d.name in ("torch", "f5_tts", "index_tts")]

        if hasattr(self, "btn_install_all"):
            self.btn_install_all.setEnabled(True)
            if missing_installables:
                count = len(missing_installables)
                self.btn_install_all.setText(f"🚀 Baixar Dependências Ausentes ({count})")
                self.btn_install_all.setStyleSheet(
                    "background-color: #FFA200; color: #121214; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px;"
                )
                self.btn_install_all.setToolTip("Baixa e instala os componentes ausentes no sistema.")
            elif available_updates:
                count = len(available_updates)
                self.btn_install_all.setText(f"🔄 Atualizar Dependências Disponíveis ({count})")
                self.btn_install_all.setStyleSheet(
                    "background-color: #04D361; color: #121214; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px;"
                )
                self.btn_install_all.setToolTip("Atualiza os componentes para as novas versões encontradas.")
            else:
                self.btn_install_all.setText("✅ Todas as Dependências Estão Atualizadas (Verificar Novamente)")
                self.btn_install_all.setStyleSheet(
                    "background-color: #1F2723; color: #04D361; border: 1px solid #04D361; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px;"
                )
                self.btn_install_all.setToolTip("Todos os componentes estão atualizados. Clique para verificar novamente se há novidades.")

        # Se havia status de checagem em andamento, atualiza para sucesso
        if hasattr(self, "lbl_install_status") and "Verificando" in self.lbl_install_status.text():
            if not missing_installables and not available_updates:
                self.lbl_install_status.setText("✅ Todas as dependências já estão na versão mais recente.")

    def _browse_ffmpeg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Executável FFmpeg", "", "Executáveis (*.exe);;Todos (*.*)")
        if path:
            self.txt_ffmpeg.setText(path)

    def _browse_models_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Selecionar Diretório Onde os Modelos Estão Instalados")
        if dir_path:
            self.txt_models.setText(dir_path)
            self._check_models_in_background()

    def _redetect_gpu(self) -> None:
        detected = detect_gpu_profile(config_path=str(self.config_path), force_redetect=True)
        # Seleciona o item correto no combo
        for idx, (key, _disp) in enumerate(self._profile_items):
            if key == detected:
                self.cb_profile.setCurrentIndex(idx)
                break
        self._update_profile_details()
        display = PROFILE_DISPLAY_NAMES.get(detected, detected)
        QMessageBox.information(self, "Detecção Concluída", f"Perfil detectado com base na VRAM: {display}")

    # ─── Gerenciamento Individual e Inteligente de Modelos ────────────────────

    def _check_models_in_background(self) -> None:
        """Verifica a presença e integridade de todos os modelos do perfil selecionado no disco."""
        selected_prof = self._get_selected_profile_key()
        actual_profile = selected_prof if selected_prof != "auto" else detect_gpu_profile(config_path=str(self.config_path))
        models_dir = self.txt_models.text().strip() or "models"

        # Identifica especificações do perfil selecionado
        specs = [m for m in MODEL_CATALOG if actual_profile in m.profiles]

        # Re-popula widgets da lista se os modelos do perfil mudaram
        current_keys = list(self.model_labels.keys())
        spec_keys = [s.key for s in specs]

        if current_keys != spec_keys:
            # Remove widgets anteriores
            while self.models_list_layout.count():
                item = self.models_list_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    while item.layout().count():
                        sub = item.layout().takeAt(0)
                        if sub.widget():
                            sub.widget().deleteLater()

            self.model_labels.clear()
            self.model_action_btns.clear()
            self.model_badges.clear()

            for spec in specs:
                row = QHBoxLayout()
                lbl = QLabel(f"⏳  {spec.name}  —  Verificando...")
                lbl.setStyleSheet("color: #A8A8B3; font-size: 12px; padding: 2px 0;")
                row.addWidget(lbl)
                self.model_labels[spec.key] = lbl
                row.addStretch()

                btn = QPushButton("⬇️ Baixar")
                btn.setStyleSheet(
                    "background-color: #332616; color: #FFA200; border: 1px solid #FFA200; "
                    "padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;"
                )
                btn.clicked.connect(lambda _, k=spec.key, n=spec.name: self._start_individual_model_download(k, n))
                btn.setVisible(False)  # Oculto durante checagem
                row.addWidget(btn)
                self.model_action_btns[spec.key] = btn

                lbl_badge = QLabel("(Verificando...)")
                lbl_badge.setStyleSheet("color: #71717A; font-size: 11px; font-style: italic;")
                lbl_badge.setVisible(True)
                row.addWidget(lbl_badge)
                self.model_badges[spec.key] = lbl_badge

                self.models_list_layout.addLayout(row)
        else:
            # Apenas redefine o estado para Verificando...
            for spec in specs:
                if spec.key in self.model_labels:
                    self.model_labels[spec.key].setText(f"⏳  {spec.name}  —  Verificando...")
                    self.model_labels[spec.key].setStyleSheet("color: #A8A8B3; font-size: 12px; padding: 2px 0;")
                if spec.key in self.model_action_btns:
                    self.model_action_btns[spec.key].setVisible(False)
                if spec.key in self.model_badges:
                    self.model_badges[spec.key].setText("(Verificando...)")
                    self.model_badges[spec.key].setStyleSheet("color: #71717A; font-size: 11px; font-style: italic;")
                    self.model_badges[spec.key].setVisible(True)

        if hasattr(self, "btn_download_models"):
            self.btn_download_models.setText("⏳ Verificando modelos locais...")
            self.btn_download_models.setStyleSheet(
                "background-color: #29292E; color: #A8A8B3; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px; border: 1px solid #3F3F46;"
            )
            self.btn_download_models.setEnabled(False)

        worker = ModelCheckerWorker(
            profile=actual_profile,
            models_dir=models_dir,
            config_path=str(self.config_path),
            parent=self,
        )
        worker.finished_signal.connect(self._refresh_models_status)
        worker.start()
        self._model_checker = worker

    def _refresh_models_status(self, statuses: list) -> None:
        """Atualiza os indicadores individuais de cada modelo e o botão mestre."""
        self._last_checked_models = statuses
        missing = []
        total_missing_bytes = 0

        for s in statuses:
            key = s["key"]
            installed = s["installed"]
            name = s["name"]
            size_mb = s["size_mb"]
            expected_size = s.get("expected_size_str", "")

            if not installed:
                missing.append(s)
                total_missing_bytes += s.get("expected_min_bytes", 0)

            if key in self.model_labels:
                lbl = self.model_labels[key]
                if installed:
                    lbl.setText(f"✅  {name}  —  {size_mb:.1f} MB (Instalado)")
                    lbl.setStyleSheet("color: #04D361; font-size: 12px; padding: 2px 0;")
                else:
                    lbl.setText(f"❌  {name}  —  Não baixado ({expected_size})")
                    lbl.setStyleSheet("color: #FF6961; font-size: 12px; padding: 2px 0;")

            btn = self.model_action_btns.get(key)
            badge = self.model_badges.get(key)

            if btn and badge:
                if installed:
                    btn.setVisible(False)
                    badge.setText("(Instalado)")
                    badge.setStyleSheet("color: #71717A; font-size: 11px; font-style: italic;")
                    badge.setVisible(True)
                else:
                    btn.setText("⬇️ Baixar")
                    btn.setStyleSheet(
                        "background-color: #332616; color: #FFA200; border: 1px solid #FFA200; "
                        "padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;"
                    )
                    btn.setVisible(True)
                    badge.setVisible(False)

        if hasattr(self, "btn_download_models"):
            self.btn_download_models.setEnabled(True)
            if missing:
                count = len(missing)
                if total_missing_bytes >= 1024 * 1024 * 1024:
                    missing_str = f"{total_missing_bytes / (1024 * 1024 * 1024):.1f} GB"
                else:
                    missing_str = f"{total_missing_bytes / (1024 * 1024):.0f} MB"

                self.btn_download_models.setText(f"🚀 Baixar Modelos Ausentes ({count} modelo(s) - ~{missing_str})")
                self.btn_download_models.setStyleSheet(
                    "background-color: #FFA200; color: #121214; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px;"
                )
                self.btn_download_models.setToolTip(
                    "Baixa apenas os modelos ausentes no disco, mantendo intactos os arquivos já baixados."
                )
            else:
                self.btn_download_models.setText("✅ Todos os Modelos Estão Prontos (Verificar Novamente)")
                self.btn_download_models.setStyleSheet(
                    "background-color: #1F2723; color: #04D361; border: 1px solid #04D361; font-weight: bold; padding: 7px 16px; margin-top: 6px; border-radius: 6px;"
                )
                self.btn_download_models.setToolTip(
                    "Todos os modelos necessários para este perfil estão instalados e prontos para uso."
                )

    def _set_models_ui_busy(self, busy: bool, message: str = "") -> None:
        """Bloqueia/desbloqueia os controles durante o download de modelos."""
        if hasattr(self, "btn_download_models"):
            self.btn_download_models.setEnabled(not busy)
        for btn in self.model_action_btns.values():
            btn.setEnabled(not busy)

        self.prog_bar_dl.setVisible(busy)
        if busy:
            self.prog_bar_dl.setValue(0)
        if message:
            self.lbl_dl_status.setText(message)

    def _start_individual_model_download(self, target_key: str, display_name: str) -> None:
        """Inicia o download exclusivo de um modelo individual."""
        reply = QMessageBox.question(
            self,
            f"Baixar {display_name}",
            f"Deseja baixar apenas o modelo '{display_name}' agora?\n\n"
            "Os outros modelos já baixados permanecerão intactos.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._start_model_download(target_model_key=target_key, target_model_name=display_name)

    def _start_smart_models_action(self) -> None:
        """Executa o download inteligente apenas dos modelos ausentes ou reverifica integridade."""
        missing = [s for s in getattr(self, "_last_checked_models", []) if not s.get("installed", False)]
        if not missing:
            self.lbl_dl_status.setText("🔍 Verificando integridade dos modelos no disco...")
            self._check_models_in_background()
            return

        reply = QMessageBox.question(
            self,
            "Baixar Modelos Ausentes",
            f"O KmellVox encontrou {len(missing)} modelo(s) ausente(s) no disco.\n\n"
            "Deseja baixar apenas os modelos faltantes agora?\n"
            "(Os modelos já presentes não serão baixados novamente).",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._start_model_download(target_model_key=None)

    def _start_model_download(
        self,
        target_model_key: Optional[str] = None,
        target_model_name: Optional[str] = None,
    ) -> None:
        """Inicia o worker de download em background."""
        selected_prof = self._get_selected_profile_key()
        actual_profile = selected_prof if selected_prof != "auto" else detect_gpu_profile(config_path=str(self.config_path))
        models_dir = self.txt_models.text().strip() or "models"

        label_scope = f"'{target_model_name}'" if target_model_name else f"perfil '{actual_profile}'"
        self._set_models_ui_busy(True, f"Iniciando download para {label_scope}...")

        self.downloader_worker = ModelDownloaderWorker(
            profile=actual_profile,
            models_dir=models_dir,
            config_path=str(self.config_path),
            target_model_key=target_model_key,
            parent=self,
        )
        self.downloader_worker.progress_signal.connect(self._on_dl_progress)
        self.downloader_worker.finished_signal.connect(self._on_dl_finished)
        self.downloader_worker.error_signal.connect(self._on_dl_error)
        self.downloader_worker.start()

    def _on_dl_progress(self, pct: float, msg: str) -> None:
        int_pct = max(0, min(100, int(round(pct * 100))))
        self.prog_bar_dl.setValue(int_pct)
        self.lbl_dl_status.setText(f"{int_pct}% - {msg}")

    def _on_dl_finished(self, saved_paths: dict) -> None:
        self.prog_bar_dl.setValue(100)
        self.lbl_dl_status.setText("✅ Modelos verificados e prontos no disco.")
        self._set_models_ui_busy(False)
        self.downloader_worker = None
        self._check_models_in_background()
        QMessageBox.information(
            self,
            "Download Concluído",
            "Os modelos foram baixados e verificados com sucesso!",
        )

    def _on_dl_error(self, err: str) -> None:
        self.lbl_dl_status.setText(f"❌ Erro no download: {err}")
        self._set_models_ui_busy(False)
        self.downloader_worker = None
        self._check_models_in_background()
        QMessageBox.warning(self, "Falha no Download", f"Ocorreu um erro durante o download dos modelos:\n{err}")

    # ─── Instalação de Dependências de Voz ────────────────────────────────────

    # ─── Instalação de Dependências de Voz (Individual e Lote) ─────────────────

    def _set_install_ui_busy(self, busy: bool, message: str = "") -> None:
        """Controla o estado visual dos botões e barra de progresso durante a instalação."""
        if hasattr(self, "btn_install_all"):
            self.btn_install_all.setEnabled(not busy)
        for btn in self.dep_action_btns.values():
            btn.setEnabled(not busy)

        self.prog_bar_install.setVisible(busy)
        if busy:
            self.prog_bar_install.setValue(0)
            self.lbl_install_status.setText(message)

    def _start_individual_install(self, target_key: str, display_name: str) -> None:
        """Inicia a instalação/atualização de uma única dependência selecionada."""
        dep_obj = next((d for d in getattr(self, "_last_checked_deps", []) if d.name == target_key), None)
        action_verb = "Atualizar" if (dep_obj and dep_obj.installed) else "Instalar"

        reply = QMessageBox.question(
            self,
            f"{action_verb} {display_name}",
            f"Deseja {action_verb.lower()} o componente '{display_name}' agora?\n\n"
            "A operação será executada no ambiente complementar (python_env).",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._set_install_ui_busy(True, f"Iniciando {action_verb.lower()} de {display_name}...")

        self._installer_worker = DependencyInstallerWorker(target=target_key, parent=self)
        self._installer_worker.progress_signal.connect(self._on_install_progress)
        self._installer_worker.finished_signal.connect(self._on_install_finished)
        self._installer_worker.error_signal.connect(self._on_install_error)
        self._installer_worker.start()

    def _start_smart_all_install(self) -> None:
        """Verifica o que falta no sistema e instala apenas o necessário automaticamente."""
        missing_installables = [d for d in getattr(self, "_last_checked_deps", []) if not d.installed and d.name in ("torch", "f5_tts", "index_tts")]
        available_updates = [d for d in getattr(self, "_last_checked_deps", []) if d.installed and getattr(d, "has_update", False) and d.name in ("torch", "f5_tts", "index_tts")]

        # Se já estiver tudo instalado e atualizado, faz uma reverificação online imediata
        if not missing_installables and not available_updates and getattr(self, "_last_checked_deps", []):
            self.lbl_install_status.setText("🔍 Verificando se há atualizações online para os componentes...")
            self._recheck_deps_in_background()
            return

        if missing_installables:
            prompt_title = "Baixar Dependências Ausentes"
            prompt_msg = (
                f"O KmellVox encontrou {len(missing_installables)} dependência(s) ausente(s).\n\n"
                "Deseja baixar e instalar os componentes ausentes agora?"
            )
        else:
            prompt_title = "Atualizar Dependências"
            prompt_msg = (
                f"O KmellVox encontrou {len(available_updates)} componente(s) com atualização disponível.\n\n"
                "Deseja atualizar agora para as versões mais recentes?"
            )

        reply = QMessageBox.question(
            self,
            prompt_title,
            prompt_msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._set_install_ui_busy(True, "Processando componentes necessários...")

        self._installer_worker = DependencyInstallerWorker(target="all", parent=self)
        self._installer_worker.progress_signal.connect(self._on_install_progress)
        self._installer_worker.finished_signal.connect(self._on_install_finished)
        self._installer_worker.error_signal.connect(self._on_install_error)
        self._installer_worker.start()

    def _start_voice_deps_install(self) -> None:
        """Compatibilidade para chamada legada."""
        self._start_smart_all_install()

    def _on_install_progress(self, pct: float, msg: str) -> None:
        int_pct = max(0, min(100, int(round(pct * 100))))
        self.prog_bar_install.setValue(int_pct)
        self.lbl_install_status.setText(f"{int_pct}% — {msg}")

    def _on_install_finished(self, result: dict) -> None:
        self._set_install_ui_busy(False)
        self.prog_bar_install.setValue(100)
        msg = result.get("message", "Instalado com sucesso.")
        self.lbl_install_status.setText(f"✅ {msg}")
        self._installer_worker = None
        self._recheck_deps_in_background()
        QMessageBox.information(
            self,
            "Operação Concluída",
            f"{msg}\n\nO ambiente está pronto para uso.",
        )

    def _on_install_error(self, err: str) -> None:
        self._set_install_ui_busy(False)
        self.lbl_install_status.setText(f"❌ Erro: {err[:200]}")
        self._installer_worker = None
        QMessageBox.warning(
            self,
            "Falha na Instalação",
            f"Ocorreu um erro durante a instalação:\n\n{err}",
        )

    def _apply_and_close(self) -> None:
        selected_prof = self._get_selected_profile_key()
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
