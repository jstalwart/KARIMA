#!/bin/bash
data="ETTh2"
pred_horizon=24

sbatch ./ETT/ex02.sh $data $pred_horizon
sbatch ./ETT/ex03.sh $data $pred_horizon
sbatch ./ETT/ex04.sh $data $pred_horizon
sbatch ./ETT/ex05.sh $data $pred_horizon
sbatch ./ETT/ex09.sh $data $pred_horizon
sbatch ./ETT/ex13.sh $data $pred_horizon
