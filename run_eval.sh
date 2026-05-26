#!/bin/bash

CONDA_ENV="refalign"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
echo "conda: $CONDA_ENV"

LOG_DIR="./run_logs"
mkdir -p $LOG_DIR
TASKS=(
 "0,21,26"
 "1,46,51"
 "2,71,76"
 "3,96,101"
 "4,121,126"
 "5,146,151"
 "6,171,176"
 "7,196,201"
)
for idx in "${!TASKS[@]}"; do
  IFS=',' read -r GPU start_idx end_idx <<< "${TASKS[$idx]}"
  LOG_FILE="${LOG_DIR}/job_gpu${GPU}_start_idx${start_idx}_end_idx${end_idx}.log"

  echo "GPU: $GPU | »’÷æ: $LOG_FILE"
  echo "start_idx=$start_idx end_idx=$end_idx"
 
  CUDA_VISIBLE_DEVICES=$GPU  python -u examples/wanvideo/model_inference/Wan2.1-T2V-14B_subject_eval.py --start_id $start_idx --end_id $end_idx > $LOG_FILE 2>&1 &

  # º‰∏Ù1√Î£¨±‹√‚∆Ù∂Ø≥ÂÕª
  sleep 1
done