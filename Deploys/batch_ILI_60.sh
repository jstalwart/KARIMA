#!/bin/bash
data="ILI"
pred_horizon=60

sbatch ./ex04-ILI.sh $data $pred_horizon KAN KAN
sbatch ./ex04-ILI.sh $data $pred_horizon KAN Elman
sbatch ./ex04-ILI.sh $data $pred_horizon KAN GRU
sbatch ./ex04-ILI.sh $data $pred_horizon KAN LSTM
sbatch ./ex04-ILI.sh $data $pred_horizon Elman KAN
sbatch ./ex04-ILI.sh $data $pred_horizon LSTM KAN
sbatch ./ex04-ILI.sh $data $pred_horizon GRU KAN