#!/bin/bash
#SBATCH --job-name=refalign_eval_track2
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=run_logs/slurm_track2_%j.log
#SBATCH --error=run_logs/slurm_track2_%j.err

echo "分配到的GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

PYTHON=/home/zdmaogroup/tyj2/.conda/envs/refalign/bin/python

cd /home/zdmaogroup/tyj2/IP2V/RefAlign
mkdir -p run_logs

export OMP_NUM_THREADS=4

TASKS=(
  "0,0,25"
  "1,25,50"
  "2,50,75"
  "3,75,100"
  "4,100,125"
  "5,125,150"
  "6,150,175"
  "7,175,200"
)

for idx in "${!TASKS[@]}"; do
  IFS=',' read -r GPU start_idx end_idx <<< "${TASKS[$idx]}"
  LOG_FILE="run_logs/track2_gpu${GPU}_start${start_idx}_end${end_idx}.log"
  echo "启动 GPU${GPU}: ${start_idx}~${end_idx}"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u examples/wanvideo/model_inference/Wan2.1-T2V-14B_subject_2.py \
    --start_id $start_idx --end_id $end_idx > $LOG_FILE 2>&1 &
  sleep 5
done

wait
echo "所有任务完成"
