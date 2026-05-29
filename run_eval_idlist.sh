#!/bin/bash
#SBATCH --job-name=refalign_idlist
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --output=run_logs/slurm_idlist_%j.log
#SBATCH --error=run_logs/slurm_idlist_%j.err

PYTHON=/home/zdmaogroup/tyj2/.conda/envs/refalign/bin/python

cd /home/zdmaogroup/tyj2/IP2V/RefAlign
mkdir -p run_logs

export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0 $PYTHON -u examples/wanvideo/model_inference/Wan2.1-T2V-14B_subject_eval_idlist.py \
  --id_list id005 > run_logs/idlist_gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PYTHON -u examples/wanvideo/model_inference/Wan2.1-T2V-14B_subject_eval_idlist.py \
  --id_list id040 > run_logs/idlist_gpu1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PYTHON -u examples/wanvideo/model_inference/Wan2.1-T2V-14B_subject_eval_idlist.py \
  --id_list id068 > run_logs/idlist_gpu2.log 2>&1 &

wait
echo "所有任务完成"
