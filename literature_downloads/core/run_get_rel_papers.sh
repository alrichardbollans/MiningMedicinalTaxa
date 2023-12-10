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
conda run -n medplantmining python get_relevant_papers_from_download.py # run python in conda env

