#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/$1/$2/03-KANGRU.out"
python3 ~/04-KARMA/EX03.py $1 $2 > $result
<<<<<<< HEAD
=======
git add ~/04-KARMA
git commit -m "Experiment 3 in dataset $1 pred_horizon $2 added"
git push origin master
>>>>>>> c7df453f0aa978d41517c57e2b7c98308551bcd4
