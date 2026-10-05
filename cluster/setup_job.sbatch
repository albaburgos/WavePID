#!/bin/bash
#SBATCH --job-name=omsim_setup
#SBATCH --account=arguelles_delgado_lab
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
bash $HOME/IceCube/wavepid/cluster/setup_fasrc.sh
