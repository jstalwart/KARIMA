#!/bin/bash
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/$1/$2/01-$1-$2.out"
python3 ../EX-ILI.py $1 $2 $3 $4 > $result