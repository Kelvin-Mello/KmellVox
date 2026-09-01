"""Ponto de Entrada Principal da Aplicação KmellVox."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from core.hardware import detect_hardware
from core.pipeline import DubbingPipeline, PipelineConfig
from downloader.fetch_models import check_models_status, fetch_models_for_profile
from ui.main_window import MainWindow

# Configuração de Logging global
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("KmellVox")


def run_cli(args: argparse.Namespace) -> None:
    """Executa o pipeline via linha de comando."""
    logger.info("Iniciando KmellVox em modo CLI...")
    config = PipelineConfig(
        input_video=args.input,
        output_video=args.output or f"output/{Path(args.input).stem}_dubbed.mp4",
        source_language=args.source_lang,
        target_language=args.target_lang,
        hardware_profile=args.profile,
        enable_lipsync=not args.no_lipsync,
        burn_subtitles=args.burn_subtitles,
    )
    pipeline = DubbingPipeline(config)
    result = pipeline.run()
    logger.info("Vídeo gerado com sucesso em: %s", result)


def run_gui(first_run: bool = False) -> None:
    """Inicia a interface gráfica com PySide6."""
    logger.info("Iniciando KmellVox Studio (PySide6)...")
    app = QApplication(sys.argv)
    app.setApplicationName("KmellVox Studio")
    app.setOrganizationName("KmellVox")

    window = MainWindow()
    window.show()

    if first_run:
        # Abre automaticamente a tela de download de modelos na primeira execução após a inicialização da janela
        QTimer.singleShot(250, window.open_first_run_downloader)

    sys.exit(app.exec())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KmellVox - Pipeline com IA para Dublagem, Clonagem de Voz e Sincronia Labial."
    )
    parser.add_argument("--cli", action="store_true", help="Executa em modo linha de comando.")
    parser.add_argument("-i", "--input", help="Caminho do vídeo de entrada (modo CLI).")
    parser.add_argument("-o", "--output", help="Caminho do vídeo de saída (modo CLI).")
    parser.add_argument("--source-lang", default="auto", help="Idioma de origem do áudio.")
    parser.add_argument("--target-lang", default="pt", help="Idioma de destino.")
    parser.add_argument("--profile", default="auto", help="Perfil de hardware (auto, perfil_a, perfil_b, cpu).")
    parser.add_argument("--no-lipsync", action="store_true", help="Desativa a etapa de sincronia labial.")
    parser.add_argument("--burn-subtitles", action="store_true", help="Estampa legendas diretamente no vídeo.")
    parser.add_argument("--fetch-models", action="store_true", help="Baixa/atualiza os modelos necessários para o perfil detectado.")
    parser.add_argument("--check-models", action="store_true", help="Verifica a presença dos modelos no disco.")
    parser.add_argument("--first-run", action="store_true", help="Abre diretamente a tela de instalação de modelos após inicializar.")

    args = parser.parse_args()

    # Informações de inicialização
    hw = detect_hardware()
    logger.info("Hardware Inicializado: %s (VRAM: %.1f GB) | Perfil: %s",
                hw.device_name, hw.vram_total_gb, hw.profile.value)

    if args.check_models:
        statuses = check_models_status()
        print("\nStatus dos Modelos Locais:")
        for s in statuses:
            badge = "[INSTALADO]" if s["installed"] else "[AUSENTE]  "
            print(f"{badge:12} | {s['category']:14} | {s['name']}")
        return

    if args.fetch_models:
        logger.info("Iniciando download dos modelos para o perfil detectado...")
        fetch_models_for_profile()
        return

    if args.cli or args.input:
        if not args.input:
            parser.error("O parâmetro --input (-i) é obrigatório no modo CLI.")
        run_cli(args)
    else:
        run_gui(first_run=args.first_run)


if __name__ == "__main__":
    main()
