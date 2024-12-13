#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/$1/$2/09-GRUKAN.out"
python3 ~/04-KARMA/EX09.py $1 $2 > $result
<<<<<<< HEAD
=======
git add ~/04-KARMA
git commit -m "Experiment 9 in dataset $1 pred_horizon $2 added"
git push origin master
>>>>>>> c7df453f0aa978d41517c57e2b7c98308551bcd4
