#!/bin/bash
data="ECL"
pred_horizon=720

sbatch ./ex05-ECL.sh $data $pred_horizon KAN KAN
sbatch ./ex05-ECL.sh $data $pred_horizon KAN Elman
sbatch ./ex05-ECL.sh $data $pred_horizon KAN GRU
sbatch ./ex05-ECL.sh $data $pred_horizon KAN LSTM
sbatch ./ex05-ECL.sh $data $pred_horizon Elman KAN
sbatch ./ex05-ECL.sh $data $pred_horizon GRU KAN
sbatch ./ex05-ECL.sh $data $pred_horizon LSTM KAN