#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/$1/$2/08-ELSTM.out"
python3 ~/04-KARMA/EX08.py $1 $2 > $result
<<<<<<< HEAD
=======
git add ~/04-KARMA
git commit -m "Experiment 8 in dataset $1 pred_horizon $2 added"
git push origin master
>>>>>>> c7df453f0aa978d41517c57e2b7c98308551bcd4
