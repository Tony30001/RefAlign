import os
import json
from tqdm import tqdm

TEST_ROOT = "/home/zdmaogroup/tyj2/IP2V/vip200k_test_track_Facial/" 
VIDEO_SAVE_ROOT = "/home/zdmaogroup/tyj2/IP2V/RefAlign/generated_videos"
RESULT_JSON_PATH = f"/home/zdmaogroup/tyj2/IP2V/RefAlign/generated_videos/results.json"
all_results = []
for idx in tqdm(range(1, 201), desc="video"):
  id_index = f"id{idx:03d}"
  id_folder = os.path.join(TEST_ROOT, id_index)
  img_path = os.path.join(id_folder, "image.png")
  for p_idx in range(1, 2):
      prompt_file = os.path.join(id_folder, f"prompt{p_idx}.txt")
      if os.path.exists(prompt_file):
          with open(prompt_file, "r", encoding="utf-8") as f:
              prompt = f.read().strip()
  video_name = f"{id_index}_prompt1.mp4"
  video_path = os.path.join(VIDEO_SAVE_ROOT, video_name)
  result = {
              "id_index": id_index,
              "ref_img_path": img_path,
              "video_path": video_path,
              "prompt": prompt
           }
              
  all_results.append(result)

with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)