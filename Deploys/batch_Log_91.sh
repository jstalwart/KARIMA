#!/bin/bash
pred_horizon=91

sbatch ./ex06-Log.sh $data $pred_horizon KAN Elman
sbatch ./ex06-Log.sh $data $pred_horizon KAN GRU
sbatch ./ex06-Log.sh $data $pred_horizon KAN LSTM
sbatch ./ex06-Log.sh $data $pred_horizon Elman KAN
sbatch ./ex06-Log.sh $data $pred_horizon LSTM KAN
sbatch ./ex06-Log.sh $data $pred_horizon GRU KAN