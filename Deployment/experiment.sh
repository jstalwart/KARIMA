#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1
#source  ~/miniconda3/bin/activate pyKAN-env

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|-d)
            dataset="$2"
            shift 2
            ;;
        --horizon|-h)
            pred="$2"
            shift 2
            ;;
        --autoregressive|-ar)
            ar="$2"
            shift 2
            ;;
        --error-regressive|-ma)
            ma="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if both arguments are provided
if [[ -z "$dataset" || -z "$pred" || -z "$ar" || -z "$ma" ]]; then
    echo "Usage: $0 --dataset <dataset> --horizon <horizon> --autoregressive <autoressive model> --error-regressive <error-regressive model>"
    exit 1
fi

result="$HOME/04-KARIMA/results/$dataset/$pred/$ar-$ma.out"

case "$dataset" in
    ETTh1|ETTh2|ETTm1)
        python ../EX-ETT.py $dataset $pred $ar $ma > $result
        ;;
    *)
        python ../EX-$dataset.py $dataset $pred $ar $ma > $result
        ;;
esac