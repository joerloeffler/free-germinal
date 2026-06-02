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

python validate_install.py --free
