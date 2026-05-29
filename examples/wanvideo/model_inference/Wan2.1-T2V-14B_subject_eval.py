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
# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('models/BiRefNet', trust_remote_code=True)
from torchvision import transforms
import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Batch Inference from JSON")
parser.add_argument("--start_id", type=int, default=0, help="起始索引 (包含)")
parser.add_argument("--end_id", type=int, default=200, help="结束索引 (不包含)")
args = parser.parse_args()

@torch.inference_mode()
def birefnet_mask_only(birefnet, pil_img: Image.Image, device="cuda:5", div=32):
    """
    输入：PIL 任意尺寸
    输出：mask(PIL L)，尺寸 == 原图尺寸
    div：把输入 pad 到可被 div 整除（31/32 具体哪个更合适看模型，这里先用 32 更常见）
    """
    birefnet = birefnet.to(device).eval()

    img_rgb = pil_img.convert("RGB")
    W, H = img_rgb.size

    # --- pad 到 div 的整数倍 ---
    pad_h = (div - H % div) % div
    pad_w = (div - W % div) % div
    if pad_h or pad_w:
        padded = Image.new("RGB", (W + pad_w, H + pad_h), (0, 0, 0))
        padded.paste(img_rgb, (0, 0))
    else:
        padded = img_rgb

    x = transforms.ToTensor()(padded)
    x = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(x)
    x = x.unsqueeze(0).to(device)  # float32

    out = birefnet(x)
    if isinstance(out, (list, tuple)):
        logits = out[-1]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out[-1] if isinstance(out, dict) else out

    pred = logits.sigmoid()[0].detach().float().cpu().squeeze()
    mask = transforms.ToPILImage()(pred)

    # mask 先对齐到 padded 尺寸，再裁回原图尺寸
    mask = mask.resize(padded.size, Image.Resampling.BILINEAR)
    mask = mask.crop((0, 0, W, H))
    return mask

def apply_subject_mask(
    birefnet,
    pil_img: Image.Image,
    device="cuda:5",
    bg_color=(255, 255, 255),
    div=32,
):
    """
    输入：已 resize/crop 到目标尺寸的 PIL.Image (RGB)
    输出：背景被替换为 bg_color 的 PIL.Image (RGB)
    """
    mask = birefnet_mask_only(birefnet, pil_img, device=device, div=div)
    img = pil_img.convert("RGB")
    bg = Image.new("RGB", img.size, bg_color)
    out = Image.composite(img, bg, mask)  # mask 白=保留前景，黑=用背景
    return out



pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    # model_configs=[
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
    #     ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    # ],
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
    ModelConfig(path="models/Wan2.1-T2V-14B/Wan2.1_VAE.pth")],
    tokenizer_config=
    ModelConfig(path="models/Wan2.1-T2V-14B/umt5"),
    dinov3_model_id = "models/dinov3",
)
pipe.enable_vram_management()

def short_resize_and_crop_pil(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    按比例缩放后填充，输出指定大小的 PIL.Image。
    - 会保持原图纵横比；
    - 缩放后以白色背景居中填充到目标尺寸；
    """
    W, H = image.size
    img_ratio = W / H
    target_ratio = target_width / target_height

    # 等比例缩放
    if img_ratio > target_ratio:  # 图片更宽
        new_width = target_width
        new_height = int(new_width / img_ratio)
    else:  # 图片更高
        new_height = target_height
        new_width = int(new_height * img_ratio)

    # Resize
    img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 填充到目标尺寸
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - delta_w // 2,
        delta_h - delta_h // 2,
    )
    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
    return new_img

# Text-to-video
# video = pipe(
#     prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
# )
# save_video(video, "video_Wan2.1-T2V-1.3B.mp4", fps=15, quality=5)

def make_blank_subject(width=832, height=480, rgb=127):
    """生成中性灰空白图，尽量等价于“零特征”注入。"""
    img = Image.new("RGB", (width, height), (rgb, rgb, rgb))
    # 如果你仍想用自己的等比填充函数，也可以套一层：
    # img = short_resize_and_crop_pil(img, width, height)
    return img

# ===================== 1. 配置参数 =====================
DATASET_ROOT = "/home/zdmaogroup/tyj2/IP2V/RefAlign/data/IPVG2026-Test-Track1(3)/IPVG2026-Test-Track1"
EVAL_JSON    = os.path.join(DATASET_ROOT, "eval.json")
IMAGES_DIR   = os.path.join(DATASET_ROOT, "images")
VIDEO_SAVE_ROOT = "/home/zdmaogroup/tyj2/IP2V/RefAlign/generated_videos_track1_1280x720"
RESULT_JSON_PATH = f"/home/zdmaogroup/tyj2/IP2V/RefAlign/generated_videos_track1_1280x720/results_{args.start_id}_{args.end_id}.json"
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

os.makedirs(VIDEO_SAVE_ROOT, exist_ok=True)

# ===================== 2. 加载 eval.json =====================
with open(EVAL_JSON, "r", encoding="utf-8") as f:
    all_samples = json.load(f)

samples = all_samples[args.start_id:args.end_id]
print(f"处理样本：{args.start_id} ~ {args.end_id}，共 {len(samples)} 条")

# ===================== 3. 批量处理主逻辑 =====================
all_results = []

for sample in tqdm(samples, desc="批量生成视频"):
    img_filename = os.path.basename(sample["img"])  # e.g. id001.webp
    img_path = os.path.join(IMAGES_DIR, img_filename)
    prompt = sample["prompt"]
    id_index = os.path.splitext(img_filename)[0]    # e.g. id001

    if not os.path.exists(img_path):
        print(f"警告：{img_path} 不存在，跳过")
        continue

    # ===================== 读取并预处理参考图 =====================
    subject_image = Image.open(img_path).convert("RGB")
    subject_image = short_resize_and_crop_pil(subject_image, 1280, 720)
    subject_image = apply_subject_mask(birefnet, subject_image, device="cuda", bg_color=(255, 255, 255))

    # ===================== 生成视频 =====================
    video_name = f"{id_index}.mp4"
    video_path = os.path.join(VIDEO_SAVE_ROOT, video_name)

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

# ===================== 4. 保存结果 JSON =====================
with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print(f"\n处理完成！共生成 {len(all_results)} 个视频")
print(f"结果已保存至：{RESULT_JSON_PATH}")