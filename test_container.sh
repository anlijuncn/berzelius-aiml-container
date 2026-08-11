#!/bin/bash
set -euo pipefail

# Run this script from a GPU compute node, e.g. after:
#   interactive --gpus=1 -t 0:10:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-${SCRIPT_DIR}/berzelius-aiml.sif}"

echo "======================================================================"
echo "GENERAL AI/ML CONTAINER TEST"
echo "======================================================================"
echo "Host      : $(hostname)"
echo "Container : ${CONTAINER}"
echo

if [[ ! -f "${CONTAINER}" ]]; then
    echo "[FAIL] Container not found: ${CONTAINER}"
    exit 1
fi

echo "[OK] Container exists"
echo

echo "======================================================================"
echo "[1] HOST GPU"
echo "======================================================================"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo

echo "======================================================================"
echo "[2] CONTAINER OS"
echo "======================================================================"
apptainer exec "${CONTAINER}" cat /etc/os-release
echo

echo "======================================================================"
echo "[3] PYTHON"
echo "======================================================================"
apptainer exec "${CONTAINER}" python - <<'PY'
import platform
import sys

print("Python version :", sys.version.replace("\n", " "))
print("Executable     :", sys.executable)
print("Platform       :", platform.platform())
PY
echo

echo "======================================================================"
echo "[4] PACKAGE IMPORTS"
echo "======================================================================"
apptainer exec "${CONTAINER}" python - <<'PY'
import importlib
import importlib.metadata as metadata

# import name -> distribution name
packages = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "imblearn": "imbalanced-learn",
    "optuna": "optuna",
    "shap": "shap",
    "umap": "umap-learn",
    "openTSNE": "openTSNE",
    "joblib": "joblib",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "torchmetrics": "torchmetrics",
    "lightning": "lightning",
    "tensorboard": "tensorboard",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "sentencepiece": "sentencepiece",
    "h5py": "h5py",
    "zarr": "zarr",
    "openpyxl": "openpyxl",
    "xlsxwriter": "xlsxwriter",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "nibabel": "nibabel",
    "nilearn": "nilearn",
    "networkx": "networkx",
    "neuromaps": "neuromaps",
    "brainspace": "brainspace",
    "Bio": "biopython",
    "anndata": "anndata",
    "scanpy": "scanpy",
    "jupyterlab": "jupyterlab",
    "notebook": "notebook",
    "IPython": "ipython",
    "ipywidgets": "ipywidgets",
    "tqdm": "tqdm",
    "rich": "rich",
    "yaml": "pyyaml",
    "requests": "requests",
    "click": "click",
    "dotenv": "python-dotenv",
    "cloudpickle": "cloudpickle",
    "pytest": "pytest",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "tabpfn": "tabpfn",
}

failed = []
for module_name, distribution_name in packages.items():
    try:
        importlib.import_module(module_name)
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            version = "version unavailable"
        print(f"[OK]   {distribution_name:<25} {version}")
    except Exception as exc:
        failed.append((distribution_name, module_name, str(exc)))
        print(f"[FAIL] {distribution_name:<25} {exc}")

print()
print(f"Packages tested : {len(packages)}")
print(f"Passed          : {len(packages) - len(failed)}")
print(f"Failed          : {len(failed)}")

if failed:
    print("\nFailed packages:")
    for distribution, module, error in failed:
        print(f"  - {distribution} (import {module}): {error}")
    raise SystemExit(1)
PY
echo

echo "======================================================================"
echo "[5] CLI TOOLS"
echo "======================================================================"
apptainer exec "${CONTAINER}" ruff --version
apptainer exec "${CONTAINER}" jupyter --version
echo

echo "======================================================================"
echo "[6] PINNED VERSIONS"
echo "======================================================================"
apptainer exec "${CONTAINER}" python - <<'PY'
import importlib.metadata as metadata

expected = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
    "tabpfn": "8.1.0",
}

failed = False
for package, expected_version in expected.items():
    actual = metadata.version(package)
    status = "OK" if actual.startswith(expected_version) else "FAIL"
    print(f"[{status}] {package:<15} expected={expected_version:<10} actual={actual}")
    failed |= status == "FAIL"

if failed:
    raise SystemExit("Pinned package version check failed.")
PY
echo

echo "======================================================================"
echo "[7] PYTHON DEPENDENCY CHECK"
echo "======================================================================"
apptainer exec "${CONTAINER}" python -m pip check
echo

echo "======================================================================"
echo "[8] CUDA / GPU"
echo "======================================================================"
apptainer exec --nv "${CONTAINER}" python - <<'PY'
import torch

print("PyTorch             :", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available      :", torch.cuda.is_available())
print("GPU count           :", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is NOT available inside the container.")

for i in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}")
    print("  Name       :", properties.name)
    print("  Capability :", torch.cuda.get_device_capability(i))
    print("  Memory     :", f"{properties.total_memory / 1024**3:.1f} GB")
PY
echo

echo "======================================================================"
echo "[9] REAL GPU COMPUTATION"
echo "======================================================================"
apptainer exec --nv "${CONTAINER}" python - <<'PY'
import time
import torch

device = torch.device("cuda:0")
print("Device:", device)
print("GPU   :", torch.cuda.get_device_name(0))

x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

_ = x @ y
torch.cuda.synchronize()

start = time.perf_counter()
z = x @ y
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print("Result shape :", tuple(z.shape))
print("Result device:", z.device)
print("Elapsed      :", f"{elapsed:.4f} seconds")

if z.device.type != "cuda":
    raise RuntimeError("Matrix multiplication did not execute on GPU.")

print("[OK] GPU computation passed")
PY
echo

echo "======================================================================"
echo "ALL CONTAINER TESTS PASSED"
echo "======================================================================"
