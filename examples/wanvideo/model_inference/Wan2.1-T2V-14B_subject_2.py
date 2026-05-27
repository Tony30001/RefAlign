import torch
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new6 import WanVideoPipeline, ModelConfig

# ===================== argparse =====================
parser = argparse.ArgumentParser(description="Track2 Batch Inference")
parser.add_argument("--start_id", type=int, default=0, help="起始索引 (包含)")
parser.add_argument("--end_id",   type=int, default=200, help="结束索引 (不包含)")
args = parser.parse_args()

# ===================== 路径配置 =====================
DATASET_ROOT  = "/home/zdmaogroup/tyj2/IP2V/RefAlign/data/IPVG2026-Test-Track2"
DATASET_JSON  = os.path.join(DATASET_ROOT, "eval.json")
VIDEO_SAVE_ROOT = "/home/zdmaogroup/tyj2/IP2V/RefAlign/output/track2"
RESULT_JSON_PATH = os.path.join(
    VIDEO_SAVE_ROOT,
    f"video_generation_results_{args.start_id}_{args.end_id}.json"
)
os.makedirs(VIDEO_SAVE_ROOT, exist_ok=True)

NEGATIVE_PROMPT = (
    "split-screen, multi-panel, collage, picture-in-picture, two scenes, multiple scenes, montage,"
    "duplicated subject, twin, clone, double subject, two people, extra person, extra body, multiple bodies,"
    "identity drift, face swap, wrong face, inconsistent face, inconsistent clothing, outfit change, age change, gender change,"
    "cutout, sticker, pasted, floating subject, halo, outline, edge artifacts, green screen, unnatural boundary,"
    "bad composition, off-center subject, cropped head, cropped body, out of frame,"
    "temporal flicker, frame-to-frame inconsistency, jitter, wobble, warping, melting, morphing, swimming textures,"
    "ghosting, motion smear, shimmering, crawling artifacts,"
    "text, subtitles, watermark, logo, caption,"
    "overexposure, oversaturated, lowres, blurry, jpeg artifacts, noisy, banding,"
    "bad anatomy, deformed body, disfigured face, extra fingers, fused fingers, missing fingers, malformed hands,"
    "messy background, crowded background, too many background people"
)

# ===================== 加载模型 =====================
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "models/BiRefNet", trust_remote_code=True
)

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

# ===================== 工具函数 =====================
def short_resize_and_crop_pil(image, target_width, target_height):
    W, H = image.size
    if W / H > target_width / target_height:
        new_w, new_h = target_width, int(target_width * H / W)
    else:
        new_w, new_h = int(target_height * W / H), target_height
    img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    dw, dh = target_width - new_w, target_height - new_h
    return ImageOps.expand(img, (dw//2, dh//2, dw-dw//2, dh-dh//2), fill=(255, 255, 255))


@torch.inference_mode()
def birefnet_mask_only(pil_img, device="cuda", div=32):
    birefnet.to(device).eval()
    img_rgb = pil_img.convert("RGB")
    W, H = img_rgb.size
    pad_h = (div - H % div) % div
    pad_w = (div - W % div) % div
    if pad_h or pad_w:
        padded = Image.new("RGB", (W + pad_w, H + pad_h), (0, 0, 0))
        padded.paste(img_rgb, (0, 0))
    else:
        padded = img_rgb
    x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
        transforms.ToTensor()(padded)
    ).unsqueeze(0).to(device)
    out = birefnet(x)
    logits = out[-1] if isinstance(out, (list, tuple)) else out
    pred = logits.sigmoid()[0].detach().float().cpu().squeeze()
    mask = transforms.ToPILImage()(pred)
    mask = mask.resize(padded.size, Image.Resampling.BILINEAR).crop((0, 0, W, H))
    return mask


def apply_subject_mask(pil_img, device="cuda", bg_color=(255, 255, 255)):
    mask = birefnet_mask_only(pil_img, device=device)
    bg = Image.new("RGB", pil_img.size, bg_color)
    return Image.composite(pil_img.convert("RGB"), bg, mask)


def build_prompt(global_prompt, temporal_captions, num_frames=81, fps=16):
    """把 temporal_captions 拼成时间轴 prompt 追加到 global_prompt 后面"""
    total_seconds = num_frames / fps
    parts = []
    for seg in temporal_captions:
        t_start = round(seg["start"] * total_seconds, 1)
        t_end   = round(seg["end"]   * total_seconds, 1)
        parts.append(f"{t_start}s-{t_end}s: {seg['description']}")
    timeline_str = " ".join(parts)
    return f"{global_prompt} Temporal sequence: {timeline_str}"


# ===================== 加载数据 =====================
with open(DATASET_JSON, "r", encoding="utf-8") as f:
    all_samples = json.load(f)

# dict 格式，key 为字符串 "1","2",...，取 start_id ~ end_id
keys = sorted(all_samples.keys(), key=lambda x: int(x))
selected_keys = [k for k in keys if args.start_id < int(k) <= args.end_id]
print(f"处理范围：{args.start_id} ~ {args.end_id}，共 {len(selected_keys)} 条")

# ===================== 批量推理 =====================
all_results = []

for key in tqdm(selected_keys, desc="generate videos"):
    sample = all_samples[key]
    video_id = f"id{int(key):03d}"

    # 读取参考图（支持多张）
    subject_images = []
    for rel_path in sample["img_paths"]:
        img_path = os.path.join(DATASET_ROOT, rel_path)
        if not os.path.exists(img_path):
            print(f"warning: {img_path} not found, skip")
            continue
        img = Image.open(img_path).convert("RGB")
        img = short_resize_and_crop_pil(img, 1280, 720)
        img = apply_subject_mask(img, device="cuda")
        subject_images.append(img)

    if not subject_images:
        print(f"warning: {video_id} no valid images, skip")
        continue

    # 构造 prompt（global + 时间线）
    prompt = build_prompt(
        sample["global_prompt"],
        sample.get("temporal_captions", []),
        num_frames=81, fps=16
    )

    video_path = os.path.join(VIDEO_SAVE_ROOT, f"{video_id}.mp4")

    try:
        video = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            subject_image=subject_images,
            seed=42,
            tiled=True,
            cfg_scale=5.0,
            height=720,
            width=1280,
            num_frames=81,
        )
        save_video(video, video_path, fps=16, quality=9)
        all_results.append({
            "id_index": video_id,
            "ref_img_paths": sample["img_paths"],
            "video_path": video_path,
            "prompt": prompt,
        })
    except Exception as e:
        print(f"error: {video_id} failed — {e}")
        continue

# ===================== 保存结果 =====================
with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print(f"\nFinished! total {len(all_results)} videos")
print(f"saved: {RESULT_JSON_PATH}")
