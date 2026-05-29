import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new6 import WanVideoPipeline, ModelConfig
import inspect, re, functools
from torchvision import transforms
from PIL import Image, ImageOps
from diffsynth.models.set_hypernet import set_hyper
from diffsynth.models.set_inputmlp import set_inputmlp
from diffsynth.models.set_dual_LoRA import set_unsharedLoRA
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('models/BiRefNet', trust_remote_code=True)
from torchvision import transforms
import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Inference by id list")
parser.add_argument("--id_list", type=str, required=True, help="逗号分隔的id，例如 id005,id040,id068")
args = parser.parse_args()

@torch.inference_mode()
def birefnet_mask_only(birefnet, pil_img: Image.Image, device="cuda", div=32):
    birefnet = birefnet.to(device).eval()
    img_rgb = pil_img.convert("RGB")
    W, H = img_rgb.size
    pad_h = (div - H % div) % div
    pad_w = (div - W % div) % div
    if pad_h or pad_w:
        padded = Image.new("RGB", (W + pad_w, H + pad_h), (0, 0, 0))
        padded.paste(img_rgb, (0, 0))
    else:
        padded = img_rgb
    x = transforms.ToTensor()(padded)
    x = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(x)
    x = x.unsqueeze(0).to(device)
    out = birefnet(x)
    if isinstance(out, (list, tuple)):
        logits = out[-1]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out[-1] if isinstance(out, dict) else out
    pred = logits.sigmoid()[0].detach().float().cpu().squeeze()
    mask = transforms.ToPILImage()(pred)
    mask = mask.resize(padded.size, Image.Resampling.BILINEAR)
    mask = mask.crop((0, 0, W, H))
    return mask

def apply_subject_mask(birefnet, pil_img, device="cuda", bg_color=(255, 255, 255), div=32):
    mask = birefnet_mask_only(birefnet, pil_img, device=device, div=div)
    img = pil_img.convert("RGB")
    bg = Image.new("RGB", img.size, bg_color)
    return Image.composite(img, bg, mask)

def short_resize_and_crop_pil(image, target_width, target_height):
    W, H = image.size
    img_ratio = W / H
    target_ratio = target_width / target_height
    if img_ratio > target_ratio:
        new_width = target_width
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * img_ratio)
    img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=(255, 255, 255))

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            "models/RefAlign-14B/model-00000.safetensors",
            "models/RefAlign-14B/model-00001.safetensors",
            "models/RefAlign-14B/model-00002.safetensors",
            "models/RefAlign-14B/model-00003.safetensors",
            "models/RefAlign-14B/model-00004.safetensors",
            "models/RefAlign-14B/model-00005.safetensors",
        ]),
        ModelConfig(path="models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan2.1-T2V-14B/Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path="models/Wan2.1-T2V-14B/umt5"),
    dinov3_model_id="models/dinov3",
)
pipe.enable_vram_management()

# ===================== 配置参数 =====================
DATASET_ROOT = "/home/zdmaogroup/tyj2/IP2V/RefAlign/data/IPVG2026-Test-Track1(3)/IPVG2026-Test-Track1"
EVAL_JSON    = os.path.join(DATASET_ROOT, "eval.json")
IMAGES_DIR   = os.path.join(DATASET_ROOT, "images")
VIDEO_SAVE_ROOT = "/home/zdmaogroup/tyj2/IP2V/RefAlign/generated_videos_track1_1280x720"
RESULT_JSON_PATH = os.path.join(VIDEO_SAVE_ROOT, f"results_idlist_{args.id_list.replace(',', '_')}.json")
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

os.makedirs(VIDEO_SAVE_ROOT, exist_ok=True)

# 解析 id_list
id_set = set(args.id_list.split(","))

# 加载 eval.json，过滤出指定 id
with open(EVAL_JSON, "r", encoding="utf-8") as f:
    all_samples = json.load(f)

samples = [s for s in all_samples if os.path.splitext(os.path.basename(s["img"]))[0] in id_set]
print(f"待处理样本：{[os.path.splitext(os.path.basename(s['img']))[0] for s in samples]}")

all_results = []

for sample in tqdm(samples, desc="生成视频"):
    img_filename = os.path.basename(sample["img"])
    img_path = os.path.join(IMAGES_DIR, img_filename)
    prompt = sample["prompt"]
    id_index = os.path.splitext(img_filename)[0]

    if not os.path.exists(img_path):
        print(f"警告：{img_path} 不存在，跳过")
        continue

    subject_image = Image.open(img_path).convert("RGB")
    subject_image = short_resize_and_crop_pil(subject_image, 1280, 720)
    subject_image = apply_subject_mask(birefnet, subject_image, device="cuda", bg_color=(255, 255, 255))

    video_path = os.path.join(VIDEO_SAVE_ROOT, f"{id_index}.mp4")

    try:
        video = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            subject_image=[subject_image],
            num_frames=81,
            height=720,
            width=1280,
            seed=42,
            tiled=True,
            cfg_scale=5.0
        )
        save_video(video, video_path, fps=16, quality=9)
        all_results.append({
            "id_index": id_index,
            "ref_img_path": img_path,
            "video_path": video_path,
            "prompt": prompt
        })
    except Exception as e:
        print(f"错误：{id_index} 生成失败，原因：{str(e)}")
        continue

with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print(f"\n处理完成！共生成 {len(all_results)} 个视频")
print(f"结果已保存至：{RESULT_JSON_PATH}")
