# -*- mode: python ; coding: utf-8 -*-
# PyInstaller Spec para KmellVox Studio (Windows x64)

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Diretório base do projeto (um nível acima de packaging/)
PROJECT_ROOT = Path(SPECPATH).parent.resolve()

datas = [
    (str(PROJECT_ROOT / 'config.yaml'), '.'),
    (str(PROJECT_ROOT / 'models' / '.gitkeep'), 'models'),
]

# Inclui binários do FFmpeg caso estejam empacotados localmente em tools/ffmpeg/bin/
ffmpeg_dir = PROJECT_ROOT / 'tools' / 'ffmpeg' / 'bin'
if ffmpeg_dir.exists():
    datas.append((str(ffmpeg_dir / '*'), 'tools/ffmpeg/bin'))

# Coleta de assets e pacotes dependentes
datas += collect_data_files('PySide6')
datas += collect_data_files('faster_whisper')
datas += collect_data_files('huggingface_hub')
datas += collect_data_files('tqdm')

hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'yaml',
    'faster_whisper',
    'huggingface_hub',
    'tqdm',
    'numpy',
    'soundfile',
    'psutil',
    'ffmpeg',
    # Módulos internos do KmellVox
    'core',
    'core.hardware',
    'core.audio_extract',
    'core.transcribe',
    'core.translate',
    'core.voice_clone',
    'core.lipsync',
    'core.assemble',
    'core.pipeline',
    'core.narration',
    'downloader',
    'downloader.fetch_models',
    'ui',
    'ui.main_window',
    'ui.queue_widget',
    'ui.settings_dialog',
    'ui.narration_tab',
]

# Adiciona submódulos dinâmicos de bibliotecas que utilizam lazy-loading
hiddenimports += collect_submodules('faster_whisper')
hiddenimports += collect_submodules('huggingface_hub')

# Exclusão explícita de testes e pesos brutos para manter o instalador leve
excludes = [
    'torch.testing',
    'IPython',
    'unittest',
    'tests',
    'models.whisper',
    'models.llm',
    'models.tts',
    'models.musetalk',
]

a = Analysis(
    [str(PROJECT_ROOT / 'main.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KmellVox',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KmellVox',
)
