# PyInstaller spec for the desktop GUI.
#
# Build (run from this directory, on the target OS):
#   python build_gui.py            # recommended: also seeds .env/.env.example
#   # or raw:
#   pip install -e ".[gui,build]" && pyinstaller --clean mrag-gui.spec
# Output: dist/MultiModalRAG/MultiModalRAG.exe  (Windows)
#
# For a single-file exe instead of a folder, set ONEFILE = True below.
# Note: PyInstaller does NOT cross-compile - build on Windows for a Windows exe.

ONEFILE = False

import glob  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402

from PyInstaller.utils.hooks import (  # noqa: E402
    collect_all,
    collect_submodules,
    copy_metadata,
    get_package_paths,
)

hidden = collect_submodules("multimodal_rag")
hidden += ["google.genai", "google.genai.types", "openai", "anthropic"]

datas = []
binaries = []

# Bundle .env.example so a frozen first run can seed .env + .env.example next
# to the exe (see _seed_frozen_env in gui/app.py). Lands in _internal/
# (onedir) or the extraction root (onefile); both are covered at runtime.
datas += [(os.path.join(SPECPATH, ".env.example"), ".")]

# Local ML (local embeddings + reranker) resolves model/tokenizer classes
# dynamically (transformers Auto* factories, importlib.metadata versions), which
# static analysis misses. collect_all() bundles submodules + data files, and
# copy_metadata() bundles the .dist-info that importlib.metadata.version() needs
# (its PackageNotFoundError subclasses ModuleNotFoundError, so a missing
# metadata entry surfaces as a bogus "Could not import module 'PreTrainedModel'").
_ML_PACKAGES = [
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "accelerate",
    "fsspec",
    "filelock",
    "regex",
    "tqdm",
    "packaging",
]
for _pkg in _ML_PACKAGES:
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hidden += _h
    except Exception:
        pass
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        pass

# torchvision 0.29+ renamed its extension to `_C_stable.pyd`/`image_stable.pyd`,
# so the bundled hook's `hiddenimports=['torchvision._C']` matches nothing and
# `collect_dynamic_libs` skips `.pyd` (not in PY_DYLIB_PATTERNS). The result is a
# torchvision without its compiled ops -> `import torchvision` raises
# "operator torchvision::nms does not exist", which transformers then masks as
# "Could not import module 'PreTrainedModel'". Bundle the extensions + their DLLs.
_tv_base, _tv_dir = get_package_paths("torchvision")
for _pattern in ("*.pyd", "*.dll"):
    for _file in glob.glob(os.path.join(_tv_dir, _pattern)):
        binaries.append((_file, "torchvision"))
hidden += ["torchvision._C", "torchvision._C_stable", "torchvision.image_stable"]

a = Analysis(
    ["mrag_gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["mrag_runtime_hook.py"],
    excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        name="MultiModalRAG",
        debug=False,
        strip=False,
        upx=False,  # keep False: UPX compression raises AV false positives
        console=False,
        version="mrag_gui_version.txt",
        icon=None,  # put an .ico next to this spec and set icon="mrag.ico"
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="MultiModalRAG",
        debug=False,
        strip=False,
        upx=False,  # keep False: UPX compression raises AV false positives
        console=False,
        version="mrag_gui_version.txt",
        icon=None,  # put an .ico next to this spec and set icon="mrag.ico"
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="MultiModalRAG",
    )

# Seed first-run config next to the exe from the template (never real keys:
# this copies .env.example only). Runs on every build, including raw
# `pyinstaller mrag-gui.spec`, so build_gui.py is no longer required for it.
# .env.example is always refreshed; .env is created only if missing, so a
# rebuild never wipes the user's live keys.
_example = os.path.join(SPECPATH, ".env.example")
_dist = os.path.join(SPECPATH, "dist", "MultiModalRAG" if not ONEFILE else "")
if os.path.isfile(_example) and os.path.isdir(_dist):
    shutil.copyfile(_example, os.path.join(_dist, ".env.example"))
    _env_target = os.path.join(_dist, ".env")
    if not os.path.isfile(_env_target):
        shutil.copyfile(_example, _env_target)
