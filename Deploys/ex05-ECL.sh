#!/bin/bash
source  ~/miniconda3/bin/activate KARMA
result="$HOME/04-KARMA/results/$1/$2/01-$3-$4.out"
python3 ../EX-ECL.py $1 $2 $3 $4 > $result