# PyInstaller spec for the Brileta sandbox.
#
# Builds a standalone, double-clickable distribution (onedir, plus a .app on
# macOS) that bundles the compiled C extension, wgpu's native backend, glfw's
# native lib, and every runtime asset. No uv/Python/toolchain needed to run.
#
# Build with:  uv run pyinstaller brileta.spec
# Output:      dist/Brileta/  (folder)  and  dist/Brileta.app  (macOS only)

import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# All runtime data assets live under assets/ and are resolved relative to the
# repo root via config.PROJECT_ROOT_PATH (= the brileta package's parent). In
# the bundle, brileta/ sits at _MEIPASS/brileta, so putting assets at
# _MEIPASS/assets makes those paths resolve unchanged.
datas = [("assets", "assets")]

# glfw ships its native lib as data next to the package; collect it plus the
# package data. glfw is PyInstaller-aware and finds it in the frozen layout.
binaries = collect_dynamic_libs("glfw")
datas += collect_data_files("glfw")

# wgpu's and rendercanvas's own PyInstaller hooks (auto-discovered) pull in the
# wgpu-native backend .so. miniaudio and the brileta C extension are ordinary
# imported extension modules, so Analysis collects them by following imports.
# _native is imported at top level in brileta.util.spatial; listed here too so
# the collection never depends on import ordering.
hiddenimports = ["brileta.util._native"]

a = Analysis(
    ["packaging/brileta_launcher.py"],
    # Repo root, so the brileta package is importable during analysis (the
    # launcher lives in packaging/, which is the only path added by default).
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Brileta",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,  # macOS: forward files/args opened via Finder
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Brileta",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Brileta.app",
        icon=None,
        bundle_identifier="com.brileta.sandbox",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleName": "Brileta",
            "CFBundleDisplayName": "Brileta",
        },
    )
