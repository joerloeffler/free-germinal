#!/bin/bash
#SBATCH --job-name=GER   # Job name
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1                    #  allocate only on one machine
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --partition=medium
#SBATCH --gres=shard:60
#SBATCH --nodelist=ai             # MAY BE CHANGED BY USER
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

 eval "$(/z/linux/bin/micromamba shell hook --shell bash)"
 micromamba activate /z/bio/biotools/miniconda/envs/germinal

 echo "HOST: $(hostname)"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo

echo "=== NVIDIA DRIVER / GPU ==="
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version,cuda_version --format=csv

echo
echo "=== CUDA TOOLS ==="
which nvcc || true
nvcc --version || true
which ptxas || true
ptxas --version || true

echo
echo "=== CUDA LIBRARIES ==="
ldconfig -p | egrep "libcuda|libcudart|libcublas|libcudnn|libnccl" || true

echo
echo "=== PYTHON/JAX/TORCH ==="
python - <<'PY'
import os, shutil, subprocess

print("which ptxas:", shutil.which("ptxas"))
print("which nvcc:", shutil.which("nvcc"))
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))

try:
    import jax
    print("jax:", jax.__version__)
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())
except Exception as e:
    print("JAX ERROR:", repr(e))

try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("torch cuda available:", torch.cuda.is_available())
except Exception as e:
    print("TORCH ERROR:", repr(e))
PY
