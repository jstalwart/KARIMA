#!/bin/bash
data="ILI"

pred_horizon=24
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN KAN 
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN Elman
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN GRU
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN LSTM

pred_horizon=36
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN KAN 
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN Elman
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN GRU
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN LSTM

pred_horizon=48
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN KAN 
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN Elman
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN GRU
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN LSTM

pred_horizon=60
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN KAN 
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN Elman
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN GRU
sbatch ./ex04.1-ILI_rest.sh $data $pred_horizon KAN LSTM