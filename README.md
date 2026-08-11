# General AI/ML Apptainer Container for Berzelius

A GPU **Apptainer** environment for general scientific AI/ML workloads on the **Berzelius HPC system at NSC (Linköping University)**.

The repository is designed for users who want to avoid maintaining large Conda environments directly on the project filesystem. A Conda/Mamba environment can contain tens or hundreds of thousands of small files; an Apptainer SIF packages the software environment into a single image file while keeping project code, data, and outputs outside the image.

## Repository contents

```text
.
├── berzelius-aiml.def       # Apptainer definition / build recipe
├── build_container.sh       # Safe build wrapper using node-local /tmp
├── test_container.sh        # CPU/package/CUDA/GPU smoke tests
└── README.md
```

The generated image is:

```text
berzelius-aiml.sif
```

---

## Why Apptainer on Berzelius?

NSC recommends container environments on Berzelius. Apptainer is available on both login and compute nodes and is the supported container runtime. NSC also recommends storing large image files in the project filesystem rather than `$HOME`. See the official [Berzelius Apptainer Guide](https://www.nsc.liu.se/support/systems/berzelius-software/berzelius-apptainer/).

This repository additionally keeps **build-time temporary files and the OCI cache on node-local `/tmp`**. This matters on projects with file-count/inode limits: an Apptainer build temporarily expands a full Linux root filesystem and package environment, which can create very large numbers of files. Keeping those transient files off `/proj` avoids unnecessary inode pressure.

---

## Software stack

### Base system

- Ubuntu 22.04 userspace
- NVIDIA CUDA 12.8.1 + cuDNN runtime
- Miniforge / Mamba
- Python 3.11

### Deep learning

- PyTorch 2.10.0 (`cu128`)
- torchvision 0.25.0
- torchaudio 2.10.0
- torchmetrics
- Lightning
- TensorBoard
- TabPFN 8.1.0

PyTorch publishes the `2.10.0 / 0.25.0 / 2.10.0` CUDA 12.8 wheel combination in its official [previous versions documentation](https://pytorch.org/get-started/previous-versions/).

### Scientific Python and classical ML

- NumPy, SciPy
- pandas, Polars, PyArrow
- scikit-learn, statsmodels
- XGBoost, LightGBM, CatBoost
- imbalanced-learn
- Optuna, SHAP
- UMAP, openTSNE

### Foundation-model / Hugging Face ecosystem

- transformers
- datasets
- accelerate
- huggingface-hub
- safetensors
- tokenizers
- sentencepiece

### Neuroscience / biomedical

- NiBabel
- Nilearn
- neuromaps
- BrainSpace
- NetworkX
- Biopython
- AnnData
- Scanpy

### Data, visualization, and development

- HDF5 (`h5py`), Zarr, Excel I/O
- Matplotlib, Seaborn, Plotly
- JupyterLab / Notebook / ipykernel / ipywidgets
- pytest, Ruff
- Pillow, headless OpenCV
- common utilities (`tqdm`, `rich`, `requests`, `PyYAML`, etc.)

---

## Prerequisites

On Berzelius, confirm that Apptainer is available:

```bash
apptainer --version
```

Put the repository somewhere under your project directory, for example:

```text
/proj/<project>/users/<username>/containers/berzelius-aiml/
```

NSC notes that Apptainer image files can be large and recommends storing them under `/proj` rather than the limited home directory.

---

## Build the container

Make the build script executable:

```bash
chmod +x build_container.sh
```

Then build:

```bash
./build_container.sh
```

The script automatically:

1. Finds `berzelius-aiml.def` relative to the repository directory.
2. Creates a unique build workspace under node-local `/tmp`.
3. Sets `APPTAINER_TMPDIR`, `APPTAINER_CACHEDIR`, and `TMPDIR` to that local workspace.
4. Runs `apptainer build --fakeroot`.
5. Cleans temporary files and cache on success, failure, `Ctrl+C`, or termination.
6. Writes the final image to the repository directory as `berzelius-aiml.sif`.

A typical startup summary looks like:

```text
Apptainer build configuration
Definition : .../berzelius-aiml.def
Output SIF : .../berzelius-aiml.sif
Cache      : /tmp/<user>_apptainer_build_<id>/cache
Build TMP  : /tmp/<user>_apptainer_build_<id>/tmp
```

### Rebuilding an existing image

By default the script refuses to overwrite an existing SIF. To rebuild deliberately:

```bash
OVERWRITE=1 ./build_container.sh
```

You can also override the input/output filenames:

```bash
DEF_FILE=my-image.def SIF_FILE=my-image.sif ./build_container.sh
```

---

## Why the build uses `/tmp`

Avoid placing `APPTAINER_TMPDIR` on `/proj` for this workflow. During an image build, the container root filesystem and package installations are temporarily expanded into many individual files. On a shared project filesystem this can be slow and can consume a large number of inodes.

The build wrapper instead uses:

```text
/tmp/<user>_apptainer_build_<id>/
├── tmp/
└── cache/
```

and deletes the whole workspace when the build exits.

Before a large build, you can check local temporary storage with:

```bash
df -h /tmp
```

---

## Test the container on a GPU

Make the test script executable:

```bash
chmod +x test_container.sh
```

Request an interactive GPU allocation. For example:

```bash
interactive --gpus=1 -t 0:10:00
```

Then run:

```bash
./test_container.sh
```

The test checks:

- host GPU and NVIDIA driver visibility
- container OS and Python executable
- imports and installed versions for the main packages
- pinned PyTorch / torchvision / torchaudio / TabPFN versions
- `pip check`
- CUDA availability through Apptainer `--nv`
- GPU name, compute capability, and memory
- a real 5000 × 5000 PyTorch CUDA matrix multiplication

A successful run ends with:

```text
ALL CONTAINER TESTS PASSED
```

To test another image:

```bash
CONTAINER=/path/to/another.sif ./test_container.sh
```

---

## Run Python inside the container

For CPU-only work:

```bash
apptainer exec berzelius-aiml.sif python analysis.py
```

For GPU work:

```bash
apptainer exec --nv berzelius-aiml.sif python train.py
```

The `--nv` option exposes the host NVIDIA devices and driver libraries to the container. This is the standard Berzelius Apptainer GPU workflow described by NSC.

You can also open an interactive shell:

```bash
apptainer shell --nv berzelius-aiml.sif
```

---

## Access project data

Berzelius/Apptainer normally exposes standard paths including the current working directory, `$HOME`, `/tmp`, and `/proj` inside the container. Therefore, if your code/data are under `/proj`, they can usually be accessed directly without copying them into the SIF.

Example:

```bash
cd /proj/<project>/users/<username>/my-study
apptainer exec --nv /path/to/berzelius-aiml.sif python train.py
```

For custom mappings, use an explicit bind:

```bash
apptainer exec --nv \
    -B /proj/<project>/data:/data \
    berzelius-aiml.sif \
    python train.py
```

See the NSC guide for details on default and custom bind mounts.

---

## Use in a Slurm batch job

A minimal GPU job can look like:

```bash
#!/bin/bash
#SBATCH -A <project-account>
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=aiml-test

CONTAINER=/proj/<project>/users/<username>/containers/berzelius-aiml/berzelius-aiml.sif

apptainer exec --nv "${CONTAINER}" python train.py
```

Replace the project account, image path, and Python script with your own values.

---

## Environment isolation

The definition sets:

```bash
PYTHONNOUSERSITE=True
```

This prevents Python inside the container from silently importing packages from a user-level site-packages directory on the host. NSC specifically recommends this pattern to avoid host/container package conflicts.

The resulting `.sif` is immutable for normal use. Keep code, datasets, checkpoints, and results outside the image and access them through `/proj` or explicit bind mounts.

---

## Package-management strategy

The build deliberately uses two package sources:

- **Mamba / conda-forge** for the scientific Python stack and compiled scientific packages.
- **pip** for the official PyTorch CUDA 12.8 wheels and selected fast-moving/PyPI packages.

All conda-forge packages are installed in **one Mamba transaction**. This avoids repeatedly solving an increasingly complex environment, which can make container builds unnecessarily slow.

After installation, the definition runs:

```bash
python -m pip check
```

to catch obvious Python dependency inconsistencies.

---

## Customization

Edit `berzelius-aiml.def` to add/remove packages. A useful rule is:

- put stable scientific packages available from conda-forge into the single `mamba install` block;
- put PyPI-only packages into the existing `pip install` block;
- keep core GPU versions explicit if reproducibility matters.

After changing the definition, rebuild:

```bash
OVERWRITE=1 ./build_container.sh
```

For a publication or long-term project, consider versioning the recipe and image, for example:

```text
berzelius-aiml-v1.def
berzelius-aiml-v1.sif
berzelius-aiml-v2.def
berzelius-aiml-v2.sif
```

The `.def` file should be committed to Git; the large `.sif` normally should not.

---

## Git recommendations

A minimal `.gitignore` entry is:

```gitignore
*.sif
```

You may also want to ignore large model checkpoints and local outputs:

```gitignore
*.sif
*.ckpt
*.pt
*.pth
__pycache__/
.ipynb_checkpoints/
```

---

## Notes and limitations

- The image is aimed at **general scientific Python, tabular ML, deep learning, neuroscience, and biomedical/omics analysis**.
- It intentionally does not bundle large standalone neuroimaging suites such as FreeSurfer, FSL, ANTs, or MRtrix.
- It does not include TensorFlow, JAX, RAPIDS, vLLM, or other large alternative GPU ecosystems by default.
- TabPFN model checkpoints are not baked into the image; keep model/data files outside the SIF.
- Building requires network access to the configured package/image registries.
- GPU operation depends on a compatible NVIDIA host driver; Apptainer provides the host driver stack through `--nv`.

---

## References

- NSC: [Berzelius Apptainer Guide](https://www.nsc.liu.se/support/systems/berzelius-software/berzelius-apptainer/)
- Apptainer: [User Guide](https://apptainer.org/docs/user/latest/)
- PyTorch: [Previous Versions](https://pytorch.org/get-started/previous-versions/)
- Miniforge: [GitHub repository](https://github.com/conda-forge/miniforge)
- NVIDIA CUDA Containers: [Docker Hub](https://hub.docker.com/r/nvidia/cuda)

---

## Disclaimer

This is a community example for building a reusable AI/ML environment on Berzelius. It is **not an official NSC repository**. Cluster configuration, available software, GPU drivers, and recommended practices can change, so consult the current NSC documentation when adapting the recipe.

