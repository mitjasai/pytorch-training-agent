#!/bin/bash
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/slurm-%j.out

CONTAINER=/appl/local/laifs/containers/lumi-multitorch-latest.sif

mkdir -p logs slurm

module purge && module load Local-LAIF lumi-aif-singularity-bindings

if [[ ! -d .venv ]]; then
    singularity run $CONTAINER bash -c 'python3 -m venv .venv --system-site-packages && \
        .venv/bin/pip install -r requirements.txt'
fi

singularity run $CONTAINER .venv/bin/python3 src/agent.py
