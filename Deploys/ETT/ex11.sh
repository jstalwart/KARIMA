#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/$1/$2/11-GRUGRU.out"
python3 ~/04-KARMA/EX11.py $1 $2 > $result
git add ~/04-KARMA
git commit -m "Experiment 11 in dataset $1 pred_horizon $2 added"
git push origin master
