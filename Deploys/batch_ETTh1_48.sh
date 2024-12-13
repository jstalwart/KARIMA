#!/bin/bash
data="ETTh1"
pred_horizon=48

sbatch ./ETT/ex01.sh $data $pred_horizon
sbatch ./ETT/ex02.sh $data $pred_horizon
sbatch ./ETT/ex03.sh $data $pred_horizon
sbatch ./ETT/ex04.sh $data $pred_horizon
sbatch ./ETT/ex05.sh $data $pred_horizon
sbatch ./ETT/ex06.sh $data $pred_horizon
sbatch ./ETT/ex07.sh $data $pred_horizon
sbatch ./ETT/ex08.sh $data $pred_horizon
sbatch ./ETT/ex09.sh $data $pred_horizon
sbatch ./ETT/ex10.sh $data $pred_horizon
sbatch ./ETT/ex11.sh $data $pred_horizon
sbatch ./ETT/ex12.sh $data $pred_horizon
sbatch ./ETT/ex13.sh $data $pred_horizon
sbatch ./ETT/ex14.sh $data $pred_horizon
sbatch ./ETT/ex15.sh $data $pred_horizon
sbatch ./ETT/ex16.sh $data $pred_horizon