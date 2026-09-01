"""Módulo de Interface Gráfica (PySide6) do KmellVox."""

from .main_window import MainWindow
from .queue_widget import QueueWidget, JobItem
from .settings_dialog import SettingsDialog

__all__ = [
    "MainWindow",
    "QueueWidget",
    "JobItem",
    "SettingsDialog",
]
