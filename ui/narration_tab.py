"""Aba de Geração de Narração em Áudio (PySide6).

Permite sintetizar áudio a partir de texto puro ou legendas SRT,
com suporte a clonagem de voz, presets, divisão de arquivos e gerenciamento de fila em lote.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.hardware import ModelProfile, detect_hardware
from core.narration import (
    NarrationEngine,
    NarrationJob,
    detect_text_format,
    list_preset_voices,
)


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
                outputs = self.engine.run(job, progress_callback=on_progress)
                self.job_finished_signal.emit(job.job_id, outputs)
            except Exception as e:
                self.job_error_signal.emit(job.job_id, str(e))

        self.queue_completed_signal.emit()

    def cancel(self) -> None:
        if self.engine:
            self.engine.cancel()


class NarrationTab(QWidget):
    """Widget completo da aba de Narração de Texto e SRT."""

    log_signal = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_source_file_path: Optional[str] = None
        self.queue_jobs: Dict[str, NarrationJob] = {}
        self.worker_thread: Optional[NarrationWorkerThread] = None

        self._init_ui()
        self._check_preset_voices()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Splitter horizontal: Formulário de Configuração (Esquerda) vs Fila de Narração (Direita)
        splitter = QSplitter(Qt.Horizontal)

        # -------------------------------------------------------------
        # 1. Coluna Esquerda: Entrada de Texto, Voz e Configurações
        # -------------------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Grupo: Entrada de Texto / SRT
        input_group = QGroupBox("Conteúdo de Origem (Texto ou Legenda SRT)")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(6)

        # Barra de botões do input
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

        # Caixa de texto
        self.txt_content = QPlainTextEdit()
        self.txt_content.setPlaceholderText(
            "Cole aqui o seu texto puro para narração contínua ou o conteúdo de um arquivo de legendas .SRT "
            "(com numeração e timestamps HH:MM:SS,mmm --> HH:MM:SS,mmm)..."
        )
        self.txt_content.textChanged.connect(self._on_text_changed)
        input_layout.addWidget(self.txt_content)

        left_layout.addWidget(input_group)

        # Grupo: Seleção de Voz (Clonagem vs Presets)
        voice_group = QGroupBox("Seleção de Voz e Clonagem")
        voice_layout = QVBoxLayout(voice_group)
        voice_layout.setSpacing(6)

        # Opção 1: Clonar Voz
        row_clone = QHBoxLayout()
        self.rb_voice_clone = QRadioButton("Clonar Voz (Upload de Áudio de Referência):")
        self.rb_voice_clone.setChecked(True)
        self.rb_voice_clone.toggled.connect(self._on_voice_mode_changed)
        row_clone.addWidget(self.rb_voice_clone)

        self.txt_ref_audio = QLineEdit()
        self.txt_ref_audio.setPlaceholderText("Selecione um áudio de referência (.wav, .mp3, .flac)...")
        row_clone.addWidget(self.txt_ref_audio)

        self.btn_browse_ref = QPushButton("Procurar...")
        self.btn_browse_ref.clicked.connect(self._browse_reference_audio)
        row_clone.addWidget(self.btn_browse_ref)
        voice_layout.addLayout(row_clone)

        # Opção 2: Voz do Modelo (Preset)
        self.row_preset_widget = QWidget()
        row_preset = QHBoxLayout(self.row_preset_widget)
        row_preset.setContentsMargins(0, 0, 0, 0)
        self.rb_voice_preset = QRadioButton("Voz Padrão do Modelo (Preset):")
        self.rb_voice_preset.toggled.connect(self._on_voice_mode_changed)
        row_preset.addWidget(self.rb_voice_preset)

        self.cb_preset_voices = QComboBox()
        self.cb_preset_voices.setEnabled(False)
        row_preset.addWidget(self.cb_preset_voices, stretch=1)
        voice_layout.addWidget(self.row_preset_widget)

        left_layout.addWidget(voice_group)

        # Grupo: Opções de Divisão SRT (Visível apenas para SRT)
        self.grp_srt_options = QGroupBox("Opções de Divisão de Áudio (Exclusivo para SRT)")
        srt_opt_layout = QVBoxLayout(self.grp_srt_options)
        self.rb_srt_split = QRadioButton("Gerar um áudio separado por trecho (ex: 001_trecho.mp3, 002_...)")
        self.rb_srt_single = QRadioButton("Juntar tudo em um único áudio no final (com pausas proporcionais do SRT)")
        self.rb_srt_single.setChecked(True)
        srt_opt_layout.addWidget(self.rb_srt_split)
        srt_opt_layout.addWidget(self.rb_srt_single)
        self.grp_srt_options.setVisible(False)  # Oculto por padrão até detectar SRT
        left_layout.addWidget(self.grp_srt_options)

        # Grupo: Pasta de Destino
        dest_group = QGroupBox("Destino dos Arquivos de Áudio (MP3)")
        dest_layout = QVBoxLayout(dest_group)
        dest_layout.setSpacing(6)

        # Checkbox Salvar na pasta de origem
        self.chk_save_source_dir = QCheckBox("Salvar na mesma pasta de origem do arquivo importado")
        self.chk_save_source_dir.setChecked(True)
        self.chk_save_source_dir.setEnabled(False)  # Desabilitado até importar um arquivo real
        self.chk_save_source_dir.toggled.connect(self._on_dest_mode_changed)
        dest_layout.addWidget(self.chk_save_source_dir)

        # Linha de pasta personalizada
        row_dest_folder = QHBoxLayout()
        row_dest_folder.addWidget(QLabel("Pasta de Destino:"))
        self.txt_dest_folder = QLineEdit()
        self.txt_dest_folder.setText(str(Path.home() / "Downloads"))
        self.txt_dest_folder.setEnabled(False)  # Desabilitado enquanto chk_save_source_dir estiver marcado
        row_dest_folder.addWidget(self.txt_dest_folder)

        self.btn_browse_dest = QPushButton("Procurar...")
        self.btn_browse_dest.setEnabled(False)
        self.btn_browse_dest.clicked.connect(self._browse_dest_folder)
        row_dest_folder.addWidget(self.btn_browse_dest)
        dest_layout.addLayout(row_dest_folder)

        # Checkbox subpasta Áudio
        self.chk_audio_subfolder = QCheckBox("Criar subpasta 'Áudio' dentro do diretório de destino")
        self.chk_audio_subfolder.setChecked(False)
        dest_layout.addWidget(self.chk_audio_subfolder)

        left_layout.addWidget(dest_group)

        # Botão Adicionar à Fila
        self.btn_add_to_queue = QPushButton("➕ Adicionar à Fila de Narração")
        self.btn_add_to_queue.setStyleSheet("background-color: #8257E5; color: white; font-weight: bold; padding: 8px;")
        self.btn_add_to_queue.clicked.connect(self._add_current_to_queue)
        left_layout.addWidget(self.btn_add_to_queue)

        splitter.addWidget(left_widget)

        # -------------------------------------------------------------
        # 2. Coluna Direita: Tabela de Fila e Controles de Execução
        # -------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Barra de ferramentas da fila
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
            "Origem", "Formato", "Modo Voz", "Status", "Progresso", "Ações"
        ])
        self.table_queue.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_queue.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table_queue.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_queue.setAlternatingRowColors(True)
        right_layout.addWidget(self.table_queue)

        # Painel de Progresso
        prog_box = QGroupBox("Status de Processamento")
        prog_box_layout = QVBoxLayout(prog_box)
        prog_box_layout.setSpacing(4)

        self.lbl_job_stage = QLabel("Etapa Atual: Ocioso")
        self.lbl_job_stage.setStyleSheet("color: #04D361; font-weight: 500;")
        prog_box_layout.addWidget(self.lbl_job_stage)

        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedHeight(20)
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

        splitter.setSizes([520, 560])
        main_layout.addWidget(splitter)

    def _check_preset_voices(self) -> None:
        """Verifica a presença de vozes pré-definidas e popula o combo ou oculta a opção."""
        hw = detect_hardware()
        presets = list_preset_voices(hw.model_profile)

        self.cb_preset_voices.clear()
        if presets:
            for p in presets:
                self.cb_preset_voices.addItem(p["label"], p["id"])
            self.row_preset_widget.setVisible(True)
        else:
            # Se não houver presets prontos, oculta opção e mantém somente clonagem
            self.row_preset_widget.setVisible(False)
            self.rb_voice_clone.setChecked(True)

    def _on_text_changed(self) -> None:
        """Atualiza a detecção de formato e visibilidade do grupo SRT em tempo real."""
        text = self.txt_content.toPlainText()
        fmt = detect_text_format(text)
        if fmt == "srt":
            self.lbl_format_detected.setText("⏱️ Formato detectado: Legenda (.srt)")
            self.lbl_format_detected.setStyleSheet("color: #FFA200; font-weight: bold;")
            self.grp_srt_options.setVisible(True)
        else:
            self.lbl_format_detected.setText("📄 Formato detectado: Texto Puro (.txt)")
            self.lbl_format_detected.setStyleSheet("color: #04D361; font-weight: bold;")
            self.grp_srt_options.setVisible(False)

    def _on_voice_mode_changed(self) -> None:
        """Alterna estado dos campos de clonagem vs preset."""
        is_clone = self.rb_voice_clone.isChecked()
        self.txt_ref_audio.setEnabled(is_clone)
        self.btn_browse_ref.setEnabled(is_clone)
        self.cb_preset_voices.setEnabled(not is_clone)

    def _on_dest_mode_changed(self) -> None:
        """Controla a habilitação do campo de pasta personalizada."""
        save_source = self.chk_save_source_dir.isChecked() and self.chk_save_source_dir.isEnabled()
        self.txt_dest_folder.setEnabled(not save_source)
        self.btn_browse_dest.setEnabled(not save_source)

    def _import_file(self) -> None:
        """Abre o diálogo de seleção de arquivo de texto ou legenda SRT."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Importar Arquivo para Narração",
            "",
            "Arquivos Suportados (*.txt *.srt);;Legendas SRT (*.srt);;Texto Puro (*.txt);;Todos (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            self.txt_content.setPlainText(content)
            self.current_source_file_path = path

            # Habilita e marca checkbox de salvar na mesma pasta
            self.chk_save_source_dir.setEnabled(True)
            self.chk_save_source_dir.setChecked(True)
            self._on_dest_mode_changed()

            self.log_signal.emit(f"Arquivo importado: {os.path.basename(path)} ({len(content)} caracteres)")
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Importar", f"Falha ao ler o arquivo selecionado:\n{e}")

    def _clear_input(self) -> None:
        """Limpa a caixa de texto e redefine o caminho de arquivo."""
        self.txt_content.clear()
        self.current_source_file_path = None
        self.chk_save_source_dir.setChecked(False)
        self.chk_save_source_dir.setEnabled(False)
        self._on_dest_mode_changed()

    def _browse_reference_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Áudio de Referência para Clonagem",
            "",
            "Áudios (*.wav *.mp3 *.flac *.ogg *.m4a);;Todos (*.*)",
        )
        if path:
            self.txt_ref_audio.setText(path)

    def _browse_dest_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Destino")
        if folder:
            self.txt_dest_folder.setText(folder)

    def _add_current_to_queue(self) -> None:
        """Valida o formulário atual e adiciona um NarrationJob à fila."""
        text = self.txt_content.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Texto Vazio", "Por favor, digite, cole ou importe um texto/SRT para narração.")
            return

        fmt = detect_text_format(text)
        is_clone = self.rb_voice_clone.isChecked()
        ref_audio = self.txt_ref_audio.text().strip() if is_clone else None

        if is_clone and (not ref_audio or not os.path.isfile(ref_audio)):
            QMessageBox.warning(
                self,
                "Áudio de Referência Ausente",
                "Para clonar voz, selecione um arquivo de áudio de referência válido (.wav, .mp3).",
            )
            return

        preset_id = self.cb_preset_voices.currentData() if not is_clone else None
        split_mode = "separado" if fmt == "srt" and self.rb_srt_split.isChecked() else "unico"

        save_source = self.chk_save_source_dir.isChecked() and self.chk_save_source_dir.isEnabled()
        dest_dir = self.txt_dest_folder.text().strip() or str(Path.home() / "Downloads")

        job_id = f"narr_job_{int(time.time() * 1000)}_{len(self.queue_jobs)}"
        job = NarrationJob(
            job_id=job_id,
            source_text=text,
            source_format=fmt,
            source_file_path=self.current_source_file_path,
            voice_mode="clone" if is_clone else "preset",
            reference_audio_path=ref_audio,
            preset_voice_id=preset_id,
            split_mode=split_mode,
            destination_folder=dest_dir,
            save_to_source_folder=save_source,
            create_audio_subfolder=self.chk_audio_subfolder.isChecked(),
        )

        self.queue_jobs[job_id] = job
        self._insert_job_row(job)
        self.log_signal.emit(f"Item adicionado à fila de narração: {job_id} ({fmt.upper()})")

    def _insert_job_row(self, job: NarrationJob) -> None:
        row = self.table_queue.rowCount()
        self.table_queue.insertRow(row)

        # 0. Origem
        origin_label = (
            os.path.basename(job.source_file_path)
            if job.source_file_path
            else f"Texto ({len(job.source_text)} caracteres)"
        )
        item_orig = QTableWidgetItem(origin_label)
        item_orig.setData(Qt.UserRole, job.job_id)
        item_orig.setToolTip(job.source_text[:200] + "...")
        self.table_queue.setItem(row, 0, item_orig)

        # 1. Formato
        item_fmt = QTableWidgetItem(job.source_format.upper())
        item_fmt.setTextAlignment(Qt.AlignCenter)
        self.table_queue.setItem(row, 1, item_fmt)

        # 2. Modo Voz
        voice_str = "Clonagem" if job.voice_mode == "clone" else "Preset"
        item_voice = QTableWidgetItem(voice_str)
        item_voice.setTextAlignment(Qt.AlignCenter)
        self.table_queue.setItem(row, 2, item_voice)

        # 3. Status
        item_status = QTableWidgetItem("⏳ Pendente")
        item_status.setTextAlignment(Qt.AlignCenter)
        self.table_queue.setItem(row, 3, item_status)

        # 4. Progresso
        prog = QProgressBar()
        prog.setRange(0, 100)
        prog.setValue(0)
        prog.setFixedWidth(110)
        self.table_queue.setCellWidget(row, 4, prog)

        # 5. Ações
        actions_widget = QWidget()
        act_layout = QHBoxLayout(actions_widget)
        act_layout.setContentsMargins(2, 2, 2, 2)
        btn_del = QPushButton("❌")
        btn_del.setToolTip("Remover da Fila")
        btn_del.setFixedSize(24, 24)
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
        pending_ids = [jid for jid, j in self.queue_jobs.items() if j.status != "Processando"]
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
            resolved_models_dir = str((Path(sys.executable).parent / "models").resolve())
        else:
            resolved_models_dir = str(Path("models").resolve())

        self.worker_thread = NarrationWorkerThread(
            jobs=pending,
            model_profile=hw.model_profile,
            models_dir=resolved_models_dir,
            parent=self,
        )
        self.worker_thread.progress_signal.connect(self._on_worker_progress)
        self.worker_thread.job_finished_signal.connect(self._on_job_finished)
        self.worker_thread.job_error_signal.connect(self._on_job_error)
        self.worker_thread.queue_completed_signal.connect(self._on_queue_completed)
        self.worker_thread.start()

    def _on_worker_progress(self, job_id: str, pct: float, msg: str) -> None:
        if job_id in self.queue_jobs:
            self.queue_jobs[job_id].status = "Processando"
            self.queue_jobs[job_id].progress = pct

        self.prog_bar.setValue(int(pct * 100))
        self.lbl_job_stage.setText(f"Processando: {msg}")

        for row in range(self.table_queue.rowCount()):
            item = self.table_queue.item(row, 0)
            if item and item.data(Qt.UserRole) == job_id:
                st_item = self.table_queue.item(row, 3)
                if st_item:
                    st_item.setText("⚙️ Processando")
                    st_item.setForeground(Qt.cyan)

                prog_w = self.table_queue.cellWidget(row, 4)
                if isinstance(prog_w, QProgressBar):
                    prog_w.setValue(int(pct * 100))
                break

    def _on_job_finished(self, job_id: str, outputs: list) -> None:
        if job_id in self.queue_jobs:
            self.queue_jobs[job_id].status = "Concluído"
            self.queue_jobs[job_id].output_files = outputs

        for row in range(self.table_queue.rowCount()):
            item = self.table_queue.item(row, 0)
            if item and item.data(Qt.UserRole) == job_id:
                st_item = self.table_queue.item(row, 3)
                if st_item:
                    st_item.setText("✅ Concluído")
                    st_item.setForeground(Qt.green)

                prog_w = self.table_queue.cellWidget(row, 4)
                if isinstance(prog_w, QProgressBar):
                    prog_w.setValue(100)
                break

        self.log_signal.emit(f"✅ Narração concluída: {len(outputs)} arquivo(s) gerado(s).")

    def _on_job_error(self, job_id: str, err: str) -> None:
        if job_id in self.queue_jobs:
            self.queue_jobs[job_id].status = "Erro"
            self.queue_jobs[job_id].status_message = err

        for row in range(self.table_queue.rowCount()):
            item = self.table_queue.item(row, 0)
            if item and item.data(Qt.UserRole) == job_id:
                st_item = self.table_queue.item(row, 3)
                if st_item:
                    st_item.setText("❌ Erro")
                    st_item.setForeground(Qt.red)
                    st_item.setToolTip(f"Erro:\n{err}")
                break

        self.lbl_job_stage.setText(f"❌ Erro na narração: {err}")
        self.log_signal.emit(f"❌ Erro na narração ({job_id}): {err}")

    def _on_queue_completed(self) -> None:
        self.btn_process_queue.setEnabled(True)
        self.btn_cancel_queue.setEnabled(False)
        self.prog_bar.setValue(100)
        self.lbl_job_stage.setText("Fila de narração finalizada com sucesso!")
        self.worker_thread = None
        self._update_queue_label()
        self.log_signal.emit("🏁 Fila de narração concluída.")

    def _cancel_queue_processing(self) -> None:
        if self.worker_thread:
            self.worker_thread.cancel()
            self.btn_cancel_queue.setEnabled(False)
            self.lbl_job_stage.setText("Cancelamento solicitado...")
            self.log_signal.emit("Cancelamento da fila de narração solicitado.")
