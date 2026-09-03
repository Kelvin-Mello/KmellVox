# -*- mode: python ; coding: utf-8 -*-
# PyInstaller Spec para KmellVox Studio (Windows x64)

import os
import shutil
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# Diretório base do projeto (um nível acima de packaging/)
PROJECT_ROOT = Path(SPECPATH).parent.resolve()

# =============================================================================
# PROTEÇÃO DOS MODELOS: salva a pasta models/ antes que o --clean apague o dist
# Os modelos são grandes (20-30 GB) e nunca devem ser deletados pelo build.
# =============================================================================
DIST_DIR = PROJECT_ROOT / 'dist' / 'KmellVox'
MODELS_IN_DIST = DIST_DIR / 'models'
MODELS_BACKUP = PROJECT_ROOT / 'dist' / '_models_backup'

if MODELS_IN_DIST.exists() and not MODELS_BACKUP.exists():
    print(f"[SPEC] Fazendo backup de models/ antes do --clean: {MODELS_BACKUP}")
    shutil.copytree(str(MODELS_IN_DIST), str(MODELS_BACKUP))

datas = [
    (str(PROJECT_ROOT / 'config.yaml'), '.'),
    (str(PROJECT_ROOT / 'models' / '.gitkeep'), 'models'),
]

ui_assets = PROJECT_ROOT / 'ui' / 'assets'
if ui_assets.is_dir():
    datas.append((str(ui_assets), 'ui/assets'))


# Inclui binários do FFmpeg caso estejam empacotados localmente em tools/ffmpeg/bin/
binaries = []
ffmpeg_exe = PROJECT_ROOT / 'tools' / 'ffmpeg' / 'bin' / 'ffmpeg.exe'
ffprobe_exe = PROJECT_ROOT / 'tools' / 'ffmpeg' / 'bin' / 'ffprobe.exe'
if ffmpeg_exe.is_file():
    binaries.append((str(ffmpeg_exe), 'tools/ffmpeg/bin'))
if ffprobe_exe.is_file():
    binaries.append((str(ffprobe_exe), 'tools/ffmpeg/bin'))

# Coleta de assets e pacotes dependentes
datas += collect_data_files('PySide6')
datas += collect_data_files('faster_whisper')
datas += collect_data_files('huggingface_hub')
datas += collect_data_files('tqdm')
datas += collect_data_files('llama_cpp')

binaries += collect_dynamic_libs('llama_cpp')

hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'yaml',
    'faster_whisper',
    'llama_cpp',
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
    'core.safe_streams',
    'core.dependency_manager',
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
hiddenimports += collect_submodules('llama_cpp')
hiddenimports += collect_submodules('jinja2')

# Módulos da biblioteca padrão requeridos pelo ecossistema PyTorch e F5-TTS
hiddenimports += collect_submodules('unittest')
hiddenimports += [
    'timeit',
    'doctest',
    'difflib',
    'statistics',
    'calendar',
    'tarfile',
    'csv',
    'plistlib',
]

# Exclusão explícita de testes e pesos brutos para manter o instalador leve
excludes = [
    'torch.testing',
    'IPython',
    'tests',
    'models.whisper',
    'models.llm',
    'models.tts',
    'models.musetalk',
]

a = Analysis(
    [str(PROJECT_ROOT / 'main.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / 'packaging' / 'rthook_safe_streams.py')],
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

# =============================================================================
# RESTORE DOS MODELOS: recoloca a pasta models/ no dist após o COLLECT
# =============================================================================
MODELS_DEST = PROJECT_ROOT / 'dist' / 'KmellVox' / 'models'
if MODELS_BACKUP.exists():
    if MODELS_DEST.exists():
        shutil.rmtree(str(MODELS_DEST))
    print(f"[SPEC] Restaurando models/ do backup: {MODELS_BACKUP} -> {MODELS_DEST}")
    shutil.copytree(str(MODELS_BACKUP), str(MODELS_DEST))
    shutil.rmtree(str(MODELS_BACKUP))
    print("[SPEC] models/ restaurados com sucesso. Backup temporário removido.")
elif not MODELS_DEST.exists():
    print("[SPEC] Aviso: nenhum backup de models/ encontrado. Pasta models/ está vazia.")
    MODELS_DEST.mkdir(parents=True, exist_ok=True)
