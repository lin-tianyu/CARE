#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -n 72
#SBATCH --mem=600G
#SBATCH --gres=gpu:0
#SBATCH -t 100:0:00
#SBATCH --account=bodymaps
#SBATCH --mail-user=tlin67@jhu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate nnunet
cd /projects/bodymaps/Tianyu/recon/
python parse_low_quality_CTs.py --input_folder /projects/bodymaps/Data/AbdomenAtlasPro --workers 64