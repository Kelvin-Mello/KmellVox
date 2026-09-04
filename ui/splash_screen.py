"""Tela de Carregamento Moderna (Splash Screen) do KmellVox Studio.

Exibe animação fluida em loop contínuo e mensagens de status durante a
inicialização de modelos pesados, verificação de hardware e carregamento da interface.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class KmellVoxSplashScreen(QWidget):
    """Janela de Splash Screen moderna com design escuro, cantos arredondados e animação contínua."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(520, 280)

        # Variáveis de animação
        self._pulse_val = 0
        self._pulse_direction = 1

        self._build_ui()
        self._center_on_screen()

        # Timer para animação de pulso contínuo (60 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._step_animation)
        self._anim_timer.start(25)

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Card de Conteúdo Interno
        self.card = QFrame()
        self.card.setStyleSheet("""
            QFrame {
                background-color: #121214;
                border: 2px solid #8257E5;
                border-radius: 16px;
            }
        """)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(28, 24, 28, 24)
        card_layout.setSpacing(10)

        # Header: Título e Badge
        header_layout = QHBoxLayout()
        lbl_title = QLabel("KmellVox Studio")
        lbl_title.setStyleSheet("""
            QLabel {
                font-size: 26px;
                font-weight: 900;
                color: #FFFFFF;
                border: none;
                background: transparent;
            }
        """)
        header_layout.addWidget(lbl_title)

        header_layout.addStretch()

        badge_v2 = QLabel("v2.1 AI")
        badge_v2.setStyleSheet("""
            QLabel {
                background-color: #8257E533;
                color: #04D361;
                font-size: 11px;
                font-weight: bold;
                padding: 3px 8px;
                border-radius: 6px;
                border: 1px solid #04D36155;
            }
        """)
        header_layout.addWidget(badge_v2)
        card_layout.addLayout(header_layout)

        # Subtítulo
        lbl_sub = QLabel("Pipeline de Inteligência Artificial para Áudio, Dublagem e Narração")
        lbl_sub.setStyleSheet("color: #A1A1AA; font-size: 12px; border: none; background: transparent;")
        card_layout.addWidget(lbl_sub)

        card_layout.addStretch()

        # Barra de Progresso com Gradiente Animado
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #27272A;
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8257E5, stop:0.5 #04D361, stop:1 #8257E5);
                border-radius: 4px;
            }
        """)
        card_layout.addWidget(self.progress_bar)

        # Mensagem de Status
        status_row = QHBoxLayout()
        self.lbl_icon = QLabel("⚡")
        self.lbl_icon.setStyleSheet("font-size: 14px; border: none; background: transparent;")
        status_row.addWidget(self.lbl_icon)

        self.lbl_status = QLabel("Inicializando módulos de inteligência artificial...")
        self.lbl_status.setStyleSheet("color: #E1E1E6; font-size: 12px; font-weight: 500; border: none; background: transparent;")
        status_row.addWidget(self.lbl_status, stretch=1)

        self.lbl_percent = QLabel("0%")
        self.lbl_percent.setStyleSheet("color: #04D361; font-size: 12px; font-weight: bold; border: none; background: transparent;")
        status_row.addWidget(self.lbl_percent)
        card_layout.addLayout(status_row)

        main_layout.addWidget(self.card)

    def _center_on_screen(self) -> None:
        """Centraliza a Splash Screen na tela principal."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geom = screen.availableGeometry()
            x = (screen_geom.width() - self.width()) // 2 + screen_geom.x()
            y = (screen_geom.height() - self.height()) // 2 + screen_geom.y()
            self.move(x, y)

    def _step_animation(self) -> None:
        """Faz a barra pulsar dinamicamente em loop caso o progresso não seja fixo."""
        if self.progress_bar.value() < 100:
            self._pulse_val += self._pulse_direction * 2
            if self._pulse_val >= 95:
                self._pulse_direction = -1
            elif self._pulse_val <= 5:
                self._pulse_direction = 1
            self.progress_bar.setValue(self._pulse_val)

    def set_status(self, message: str, percent: int | None = None) -> None:
        """Atualiza a mensagem de status e opcionalmente o percentual da barra."""
        self.lbl_status.setText(message)
        if percent is not None:
            self.progress_bar.setValue(percent)
            self.lbl_percent.setText(f"{percent}%")
        QApplication.processEvents()

    def finish_and_show(self, main_window: QWidget) -> None:
        """Encerra a splash screen com transição suave e exibe a janela principal."""
        self._anim_timer.stop()
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()
        self.close()
