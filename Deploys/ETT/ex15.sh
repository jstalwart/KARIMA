#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/$1/$2/15-LSTMGRU.out"
python3 ~/04-KARMA/EX15.py $1 $2 > $result
