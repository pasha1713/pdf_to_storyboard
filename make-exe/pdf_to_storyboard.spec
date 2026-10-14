# -*- mode: python ; coding: utf-8 -*-
#
# pdf_to_storyboard.spec  (ONEFILE)
# Build single .exe for Windows 7.
#
# Usage:
#     python -m PyInstaller --clean --noconfirm pdf_to_storyboard.spec
#
# Result: dist\pdf_to_storyboard.exe  (single file, no _internal folder)
#
from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs,
)

hiddenimports = [
    'gui',
    'utils',
    'transformations',
    'cv2',
    'fitz',
    'fitz.fitz',
    'fitz.utils',
] + collect_submodules('fitz') + collect_submodules('pymupdf')

datas = []
datas += collect_data_files('fitz')
datas += collect_data_files('pymupdf')

binaries = []
binaries += collect_dynamic_libs('fitz')
binaries += collect_dynamic_libs('cv2')

a = Analysis(
    ['pdf_to_storyboard.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'scipy', 'pandas',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pdf_to_storyboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # IMPORTANT: console needed for Perspectiva!
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
# NOTE: no COLLECT block — that is what makes it onefile.
