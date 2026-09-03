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

from core.safe_streams import ensure_safe_streams

# Garante que sys.stdout e sys.stderr nunca sejam None em aplicações GUI / PyInstaller
ensure_safe_streams()

# Configuração de Logging global (console + arquivo kmellvox.log)
_log_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
try:
    _app_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
    _log_file = _app_dir / "kmellvox.log"
    _log_handlers.append(logging.FileHandler(str(_log_file), encoding="utf-8", mode="a"))
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=_log_handlers,
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

    # Exibe a janela totalmente construída antes de qualquer operação pesada.
    # A detecção de hardware ocorrerá de forma diferida (após o primeiro render),
    # evitando o efeito de "piscar / abrir e fechar" durante a inicialização.
    window.show()

    if first_run:
        # Abre automaticamente a tela de configurações na primeira execução
        QTimer.singleShot(300, window._open_settings)

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
    parser.add_argument("--check-deps", action="store_true", help="Verifica e imprime o status de todas as dependências.")
    parser.add_argument("--first-run", action="store_true", help="Abre configurações na primeira execução.")

    args = parser.parse_args()

    # Informações de inicialização
    hw = detect_hardware()
    logger.info("Hardware Inicializado: %s (VRAM: %.1f GB) | Perfil: %s",
                hw.device_name, hw.vram_total_gb, hw.profile.value)

    if args.check_deps:
        from core.dependency_manager import check_all_dependencies
        deps = check_all_dependencies()
        _app_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
        with open(_app_dir / "deps_status.txt", "w", encoding="utf-8") as f:
            for d in deps:
                line = f"{d.name:15} | installed={str(d.installed):5} | {d.version} | {d.detail}\n"
                f.write(line)
                logger.info("DEP: %s", line.strip())
        return

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
        run_gui(first_run=getattr(args, "first_run", False))


if __name__ == "__main__":
    main()
