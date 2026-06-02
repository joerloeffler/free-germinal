#!/bin/bash
#SBATCH --job-name=GER-free-8ywa
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=48gb
#SBATCH --partition=medium
#SBATCH --gres=shard:60
#SBATCH --nodelist=ai
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

eval "$(/z/linux/bin/micromamba shell hook --shell bash)"
micromamba activate /z/bio/biotools/miniconda/envs/germinal

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_CLIENT_MEM_FRACTION=0.75
mkdir -p "$MPLCONFIGDIR"

echo "Running on $(hostname)"
echo "Working directory: $PWD"
echo "Checking free/open-source Germinal install"

python validate_install.py --free

python - <<'PY'
import importlib.util
from pathlib import Path

required = ["openmm", "pdbfixer", "freesasa", "biotite"]
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(
        "Missing free backend packages: "
        + ", ".join(missing)
        + "\nInstall with: micromamba install -n germinal -c conda-forge openmm pdbfixer freesasa biotite"
    )

if not Path("pdbs/8YWA.pdb").exists():
    raise SystemExit("Missing target PDB: pdbs/8YWA.pdb")
PY

echo "Starting one-trajectory 8YWA free-backend test"

python -u run_germinal.py --free \
  run=vhh \
  target=8ywa \
  filter/initial=vhh \
  filter/final=vhh \
  experiment_name=8ywa_vhh_free_test \
  max_trajectories=1 \
  max_hallucinated_trajectories=1 \
  max_passing_designs=1

echo "finished"
