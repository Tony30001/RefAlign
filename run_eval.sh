#!/bin/bash

CONDA_ENV="refalign"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
echo "conda: $CONDA_ENV"

LOG_DIR="./run_logs"
mkdir -p $LOG_DIR
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
  LOG_FILE="${LOG_DIR}/job_gpu${GPU}_start_idx${start_idx}_end_idx${end_idx}.log"

  echo "GPU: $GPU | ��־: $LOG_FILE"
  echo "start_idx=$start_idx end_idx=$end_idx"
 
  CUDA_VISIBLE_DEVICES=$GPU  python -u examples/wanvideo/model_inference/Wan2.1-T2V-14B_subject_eval.py --start_id $start_idx --end_id $end_idx > $LOG_FILE 2>&1 &

  # ���1�룬����������ͻ
  sleep 1
done