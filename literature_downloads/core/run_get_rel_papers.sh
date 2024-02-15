#!/bin/bash

#SBATCH --partition=long
#SBATCH --mem=12G
#SBATCH --cpus-per-task=12
#SBATCH --job-name="getrelcore_log"
#SBATCH -o getrelcore_log.out
#SBATCH --mail-user=a.richard-bollans@kew.org
#SBATCH --mail-type=END,FAIL

set -euo pipefail # stop if fails
cd $SCRATCH/MedicinalPlantMining/literature_downloads/core
source $SCRATCH/apps/conda/bin/activate medplantmining # activate the conda env (instead of using conda run -n as this messes with stdout)
python -u get_relevant_papers_from_download.py # run python in conda env. See https://stackoverflow.com/questions/33178514/how-do-i-save-print-statements-when-running-a-program-in-slurm for logging.
