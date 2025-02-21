#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/Logifruit/$1/01-$2-$3.out"
python3 ../EX-Log.py $1 $2 $3 > $result