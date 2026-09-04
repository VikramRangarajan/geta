#!/bin/bash
#
# GETA baseline runner (Slurm). One job = one paper baseline, selected via
# the GETA_JOB environment variable so every job is clearly labeled and gets
# its own log file and output directory (logs never get mixed across jobs).
#
# Submit with e.g.:
#   sbatch --job-name=geta-resnet20-cifar10 \
#          --output=logs/geta-resnet20-cifar10-%j.out \
#          --export=ALL,GETA_JOB=resnet20-cifar10 run.sh
# A100-80GB jobs (paper Table 6, ImageNet) override the partition:
#   sbatch --partition=a100-80gb --job-name=geta-vit-imagenet \
#          --output=logs/geta-vit-imagenet-%j.out \
#          --export=ALL,GETA_JOB=vit-imagenet run.sh
#
# Paper hyper-parameters are from Table 7 of geta.pdf.
# Valid GETA_JOB values: resnet20-cifar10, vgg7-cifar10, resnet56-cifar10,
#   simplevit-cifar10, vit-imagenet, deit-imagenet, swin-imagenet

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -q standby
#SBATCH -p a10
#SBATCH --constraint=J
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@300
#SBATCH --job-name=geta
#SBATCH --output=logs/%x-%j.out

max_restarts=400      # tweak this number to fit your needs
scontext=$(scontrol show job "${SLURM_JOB_ID}")
restarts=$(echo "${scontext}" | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
outfile=$(scontrol show job "${SLURM_JOB_ID}" | grep 'StdOut=' | cut -d= -f2)

##                                                          ##
##############################################################
##  Build a term-handler function to be executed            ##
##      when the job gets the SIGTERM                       ##

term_handler()
{
    echo "Executing term handler at $(date)"
    if [[ ${restarts} -lt ${max_restarts} ]];then
        # Copy the log file because it will be overwriten
        echo "Requeueing!"
        cp -v "${outfile}" "${outfile%.out}_${restarts}.out"
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    else
        echo "Your job is over the Maximun restarts limit"
        exit 1
    fi
}

## Call the function when the jobs recieves the SIGTERM     ##
trap 'term_handler' SIGTERM

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
export TRAINER_RESUME=1
fi

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
IMAGENET_ROOT="/scratch/gilbreth/wang4094/imagenet"

if [[ -z "${GETA_JOB:-}" ]]; then
    echo "ERROR: set GETA_JOB to one of: resnet20-cifar10, vgg7-cifar10,"
    echo "  resnet56-cifar10, simplevit-cifar10, vit-imagenet, deit-imagenet, swin-imagenet"
    exit 2
fi

case "${GETA_JOB}" in
    # Paper Table 2: ResNet20 on CIFAR10 (weight-only, SGD lr=0.1, 350 epochs)
    resnet20-cifar10)
        SCRIPT="test_scripts/Qtest_resnet56_ablation.py"
        ARGS="--model_name resnet20 --dataset cifar10 --batch_size 64 --epochs 350 \
            --lr 0.1 --lr_quant 1e-4 --weight_decay 1e-4 --sparsity 0.35 \
            --projection_start_step 0 --projection_periods 7 --projection_steps 35 \
            --pruning_start_step 35 --pruning_periods 5 --pruning_steps 30 \
            --variant sgd --bit_reduction 2 --min_bit_wt 4 --max_bit_wt 16 --seed 0"
        ;;
    # Paper Table 4: VGG7 on CIFAR10 (weight+activation, Adam lr=1e-3, 200 epochs)
    vgg7-cifar10)
        SCRIPT="test_scripts/Qtest_JPQ.py"
        ARGS="--model_name vgg7bn --dataset cifar10 --batch_size 64 --epochs 200 \
            --learning_rate 1e-3 --weight_decay 1e-4 --sparsity_level 0.7 \
            --projection_start_step 0 --projection_periods 5 --projection_steps 20 \
            --pruning_start_step 20 --pruning_periods 10 --pruning_steps 30 \
            --variant adam --bit_reduction 2 --min_bit 4 --max_bit 16 --seed 0"
        ;;
    # Paper Fig 4 ablation: ResNet56 CIFAR10 from scratch (Fig4a 94.61%, Fig4b sparsity sweep) — ResNet family schedule like Tab7 ResNet20
    resnet56-cifar10)
        SCRIPT="test_scripts/Qtest_resnet56_ablation.py"
        ARGS="--model_name resnet56 --dataset cifar10 --batch_size 64 --epochs 200 \
            --lr 0.1 --lr_quant 1e-4 --weight_decay 1e-4 --sparsity 0.5 \
            --projection_start_step 0 --projection_periods 7 --projection_steps 35 \
            --pruning_start_step 35 --pruning_periods 5 --pruning_steps 30 \
            --variant sgd --bit_reduction 2 --min_bit_wt 4 --max_bit_wt 16 --seed 0"
        ;;
    # Paper Table 6: SimpleViT on CIFAR10 (100 epochs)
    simplevit-cifar10)
        SCRIPT="test_scripts/Qtest_simpleViT.py"
        ARGS="--model_name simpleViT --dataset cifar10 --batch_size 64 --epochs 100 \
            --learning_rate 1e-3 --weight_decay 1e-4 --sparsity_level 0.3 \
            --projection_start_step 5 --projection_periods 5 --projection_steps 5 \
            --pruning_start_step 10 --pruning_periods 10 --pruning_steps 20 \
            --variant adam --bit_reduction 2 --min_bit 4 --max_bit 16 \
            --init_bit 16 --seed 0"
        ;;
    # Paper Table 6: ViT-Small / DeiT-Tiny / Swin-Tiny on ImageNet (100 epochs, A100)
    vit-imagenet|deit-imagenet|swin-imagenet)
        MODEL="${GETA_JOB%%-imagenet}"
        SCRIPT="test_scripts/Qtest_VIT_imagenet.py"
        ARGS="--model_name ${MODEL} --dataset imagenet --batch_size 64 --epochs 100 \
            --lr 1e-3 --weight_decay 1e-4 --sparsity_level 0.3 \
            --projection_start_step 5 --projection_periods 5 --projection_steps 5 \
            --pruning_start_step 10 --pruning_periods 10 --pruning_steps 20 \
            --variant adam --bit_reduction 2 --min_bit_wt 4 --max_bit_wt 16 \
            --init_bit 16 --seed 0 \
            --train_dir ${IMAGENET_ROOT}/train --test_dir ${IMAGENET_ROOT}/val"
        ;;
    *)
        echo "ERROR: unknown GETA_JOB='${GETA_JOB}'"
        exit 2
        ;;
esac

# Isolated dirs per job: nothing is written to the repo root, so concurrent
# jobs (and Slurm log files) can never be confused with each other.
OUT_DIR="${REPO_ROOT}/outputs/${GETA_JOB}"
DATA_DIR="${GETA_DATA_DIR:-${REPO_ROOT}/data}"
WORK_DIR="${OUT_DIR}/work"
mkdir -p "${OUT_DIR}" "${DATA_DIR}" "${WORK_DIR}" "${REPO_ROOT}/logs"
export GETA_OUTPUT_DIR="${OUT_DIR}" GETA_DATA_DIR="${DATA_DIR}"

echo "=== GETA job '${GETA_JOB}' (Slurm ${SLURM_JOB_ID:-local}) started at $(date) ==="
echo "script : ${SCRIPT}"
echo "out_dir: ${OUT_DIR}"
echo "workdir: ${WORK_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

cd "${WORK_DIR}"
# shellcheck disable=SC2086
pixi --no-progress run --environment geta --manifest-path "${REPO_ROOT}/pixi.toml" \
    python "${REPO_ROOT}/${SCRIPT}" ${ARGS} \
    --output_dir "${OUT_DIR}" --data_dir "${DATA_DIR}" &
wait $!
echo "=== GETA job '${GETA_JOB}' finished at $(date) ==="
