#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/$1/$2/01-$3-$4.out"
python3 ../EX-ILI.py $1 $2 $3 $4 > $result