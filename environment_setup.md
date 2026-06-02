# Installation Instructions

This guide walks you through setting up the **Germinal** environment and installing all necessary dependencies.

---

## 1. Create and Activate Conda Environment

```bash
conda create --name germinal python=3.10
conda activate germinal
```

For Blackwell/CUDA 13 nodes, use the dedicated environment file instead:

```bash
micromamba create -f environment_blackwell.yaml
micromamba activate germinal
```

The Blackwell environment is intended for the PyRosetta-free path:

```bash
python validate_install.py --free
python run_germinal.py --free
```

We use `uv` to speed up the installation process. Subsequent `pip install` commands will use `uv` but feel free to skip this step and install with `pip` normally.
#### 1b. Install uv

```bash
pip install uv
```

---

## 2. Install Core Packages

```bash
uv pip install pandas matplotlib numpy biopython scipy seaborn tqdm ffmpeg py3dmol \
  chex dm-haiku dm-tree joblib ml-collections immutabledict optax cvxopt mdtraj colabfold ipsae
```

---

## 3. Install ColabDesign and scoring backends

> **Note:** Make sure you are in the **Germinal root directory** before running this.

```bash
# ColabDesign (editable install)
uv pip install -e colabdesign

# Open-source scoring backend (for scoring_backend: "opensource")
conda install -c conda-forge openmm pdbfixer freesasa biotite
uv pip install prodigy-prot

# Optional: PyRosetta backend (for scoring_backend: "pyrosetta")
uv pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

---

## 4. Install Torch, Chai, and IgLM

```bash
uv pip install iglm torchvision==0.21.* chai-lab==0.6.1 \
  torch==2.6.* torchaudio==2.6.* torchtyping==0.1.5 torch_geometric==2.6.*
```

> **Note:** ignore colabfold dependency errors

---

## 5. Install Project in Editable Mode

```bash
uv pip install -e .
```

---

## 6. Ensure Dependency Compatibility

To resolve version mismatches, install the following pinned versions:

```bash
uv pip install jax==0.5.3
uv pip install dm-haiku==0.0.13 
uv pip install hydra-core omegaconf
uv pip install "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install ablang2 --no-deps
uv pip install rotary_embedding_torch --no-deps 
```
