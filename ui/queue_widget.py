"""Widget de Gerenciamento da Fila de Processamento em Lote (PySide6)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class JobItem:
    """Representa um item na fila de dublagem."""
    job_id: str
    input_file: str
    output_file: str
    source_lang: str
    target_lang: str
    status: str = "Pendente"  # Pendente, Processando, Concluído, Erro, Cancelado
    progress: float = 0.0
    status_text: str = "Aguardando"
    enable_lipsync: bool = False
    use_indextts2: bool = False
    burn_subtitles: bool = False
    export_raw: bool = False


class QueueWidget(QWidget):
    """Widget de exibição e controle da fila de renderização."""

    job_removed = Signal(str)
    job_cancelled = Signal(str)
    queue_cleared = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.jobs: Dict[str, JobItem] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Barra de ferramentas da fila
        tools_layout = QHBoxLayout()
        self.lbl_queue_count = QLabel("Fila de Processamento (0 itens)")
        self.lbl_queue_count.setStyleSheet("font-weight: bold; color: #A8A8B3;")
        tools_layout.addWidget(self.lbl_queue_count)

        tools_layout.addStretch()

        btn_clear_completed = QPushButton("🧹 Limpar Concluídos")
        btn_clear_completed.setFixedHeight(26)
        btn_clear_completed.clicked.connect(self.clear_completed_jobs)
        tools_layout.addWidget(btn_clear_completed)

        btn_clear_all = QPushButton("🗑️ Limpar Tudo")
        btn_clear_all.setFixedHeight(26)
        btn_clear_all.clicked.connect(self.clear_all_jobs)
        tools_layout.addWidget(btn_clear_all)

        layout.addLayout(tools_layout)

        # Tabela da fila
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Vídeo de Entrada", "Idioma Alvo", "Status", "Progresso", "Ações"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

    def add_job(self, job: JobItem) -> None:
        """Adiciona um novo trabalho à fila."""
        self.jobs[job.job_id] = job
        row = self.table.rowCount()
        self.table.insertRow(row)

        # 0. Nome do arquivo
        file_name = os.path.basename(job.input_file)
        item_name = QTableWidgetItem(file_name)
        item_name.setToolTip(job.input_file)
        item_name.setData(Qt.UserRole, job.job_id)
        self.table.setItem(row, 0, item_name)

        # 1. Idioma
        lang_str = f"{job.source_lang.upper()} ➔ {job.target_lang.upper()}"
        item_lang = QTableWidgetItem(lang_str)
        item_lang.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 1, item_lang)

        # 2. Status
        item_status = QTableWidgetItem("⏳ Pendente")
        item_status.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 2, item_status)

        # 3. Barra de Progresso
        prog_bar = QProgressBar()
        prog_bar.setRange(0, 100)
        prog_bar.setValue(int(job.progress * 100))
        prog_bar.setFixedWidth(130)
        prog_bar.setTextVisible(True)
        prog_bar.setFormat("0%")
        self.table.setCellWidget(row, 3, prog_bar)

        # 4. Ações
        actions_widget = QWidget()
        act_layout = QHBoxLayout(actions_widget)
        act_layout.setContentsMargins(2, 2, 2, 2)
        act_layout.setSpacing(4)

        btn_cancel = QPushButton("❌")
        btn_cancel.setToolTip("Cancelar / Remover este item da fila")
        btn_cancel.setFixedSize(26, 26)
        btn_cancel.clicked.connect(lambda _, jid=job.job_id: self.cancel_or_remove_job(jid))
        act_layout.addWidget(btn_cancel)

        self.table.setCellWidget(row, 4, actions_widget)
        self._update_count_label()

    def update_job_progress(self, job_id: str, progress: float, status_text: str, status: Optional[str] = None) -> None:
        """Atualiza o progresso e status de um trabalho na tabela."""
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        job.progress = progress
        job.status_text = status_text
        if status:
            job.status = status

        status_icons = {
            "Pendente": "⏳ Pendente",
            "Processando": "⚙️ Processando",
            "Concluído": "✅ Concluído",
            "Erro": "❌ Erro",
            "Cancelado": "⏹️ Cancelado",
        }

        for row in range(self.table.rowCount()):
            item_name = self.table.item(row, 0)
            if item_name and item_name.data(Qt.UserRole) == job_id:
                # Atualiza texto e cor do status
                status_item = self.table.item(row, 2)
                if status_item:
                    display_status = status_icons.get(job.status, job.status)
                    status_item.setText(display_status)
                    if job.status == "Concluído":
                        status_item.setForeground(Qt.green)
                    elif job.status == "Erro":
                        status_item.setForeground(Qt.red)
                    elif job.status == "Processando":
                        status_item.setForeground(Qt.cyan)

                # Atualiza barra de progresso
                prog_widget = self.table.cellWidget(row, 3)
                if isinstance(prog_widget, QProgressBar):
                    pct = int(progress * 100)
                    prog_widget.setValue(pct)
                    prog_widget.setFormat(f"{pct}% - {status_text}")
                break

    def cancel_or_remove_job(self, job_id: str) -> None:
        """Cancela se estiver pendente ou remove o trabalho da fila."""
        if job_id not in self.jobs:
            return

        job = self.jobs[job_id]
        if job.status == "Processando":
            self.job_cancelled.emit(job_id)
            self.update_job_progress(job_id, job.progress, "Cancelado", status="Cancelado")
        else:
            self.remove_job(job_id)

    def remove_job(self, job_id: str) -> None:
        """Remove fisicamente a linha da tabela."""
        if job_id not in self.jobs:
            return

        del self.jobs[job_id]

        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.UserRole) == job_id:
                self.table.removeRow(row)
                break

        self.job_removed.emit(job_id)
        self._update_count_label()

    def clear_completed_jobs(self) -> None:
        """Remove todos os trabalhos já concluídos da tabela."""
        completed_ids = [jid for jid, j in self.jobs.items() if j.status in ("Concluído", "Cancelado")]
        for jid in completed_ids:
            self.remove_job(jid)

    def clear_all_jobs(self) -> None:
        """Limpa todos os trabalhos da fila que não estejam em processamento."""
        pending_or_finished = [jid for jid, j in self.jobs.items() if j.status != "Processando"]
        for jid in pending_or_finished:
            self.remove_job(jid)

    def get_pending_jobs(self) -> List[JobItem]:
        """Retorna a lista de itens com status Pendente na fila."""
        return [j for j in self.jobs.values() if j.status == "Pendente"]

    def _update_count_label(self) -> None:
        total = len(self.jobs)
        pending = len(self.get_pending_jobs())
        self.lbl_queue_count.setText(f"Fila de Processamento ({total} total, {pending} pendentes)")
