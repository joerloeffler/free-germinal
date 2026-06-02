#!/usr/bin/env python
import os
import sys
import subprocess
from typing import List

def print_status(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}")

def try_import(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception as e:
        print_status("ERROR", f"Import failed: {pkg}: {e}")
        return False

def cmd_ok(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print_status("WARN", f"Command failed: {' '.join(cmd)} -> {e}")
        return False

def main() -> int:
    free_mode = "--free" in sys.argv
    required = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "yaml",
        "filelock",
        "omegaconf",
        "mdtraj",
        "Bio",
        "tqdm",
    ]
    print_status("INFO", "Checking Python imports")
    missing = [p for p in required if not try_import(p)]
    if missing:
        print_status("ERROR", f"Missing packages: {', '.join(missing)}")
        return 1

    print_status("INFO", "Checking GPU availability via nvidia-smi (optional)")
    if cmd_ok(["nvidia-smi"]):
        print_status("OK", "nvidia-smi detected")
    else:
        print_status("WARN", "nvidia-smi not available; GPU features may be disabled")

    print_status("INFO", "Checking JAX device")
    try:
        import jax
        device = jax.default_backend()
        print_status("OK", f"JAX backend: {device}")
    except Exception as e:
        print_status("WARN", f"JAX check failed: {e}")

    print_status("INFO", "Checking PyTorch CUDA")
    try:
        import torch
        print_status("OK", f"Torch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print_status("WARN", f"Torch check failed: {e}")

    print_status("INFO", "Checking local packages")
    for mod in ["colabdesign", "germinal"]:
        if try_import(mod):
            print_status("OK", f"Imported {mod}")
        else:
            return 2

    print_status("INFO", "Checking IgLM/Transformers compatibility (optional)")
    try:
        import transformers

        _ = transformers.GPT2LMHeadModel
        print_status("OK", "Transformers GPT2LMHeadModel available")
    except Exception as e:
        print_status(
            "WARN",
            "IgLM may fail in this environment. Prefer ablm_model=ablang "
            f"or pin compatible torch/transformers versions. Error: {e}",
        )

    print_status("INFO", "Checking open-source scoring backend packages")
    optional_oss = ["openmm", "pdbfixer", "freesasa", "biotite"]
    missing_oss = [p for p in optional_oss if not try_import(p)]
    if missing_oss:
        print_status(
            "WARN",
            "Open-source scoring backend missing packages: "
            + ", ".join(missing_oss)
            + ". Install environment.yml dependencies before using scoring_backend=opensource.",
        )
    else:
        print_status("OK", "Open-source scoring backend packages available")

    if free_mode:
        print_status("INFO", "--free set; skipping PyRosetta check")
    else:
        print_status("INFO", "Checking PyRosetta (optional)")
        try:
            import pyrosetta  # type: ignore
            print_status("OK", "PyRosetta available")
        except Exception:
            print_status(
                "WARN",
                "PyRosetta not found. Use --free/scoring_backend=opensource, or install a licensed build via: \n"
                "  bash scripts/install_env.sh --pyrosetta /path/to/pyrosetta.whl\n"
                "See README for details."
            )

    print_status("OK", "All checks passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
