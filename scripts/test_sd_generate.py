"""测试 SD 生成一张食谱图片"""
import torch, diffusers, time, json, gc
from pathlib import Path

# 加载食谱
recipes = json.load(open("src/data/recipes_real.json"))
sample = recipes[0]
print(f"生成图片: {sample['name']} ({sample['cuisine']})")

# 构造提示词
name = sample["name"]
cuisine = sample["cuisine"]
ings = [i["name"] for i in sample["ingredients"][:3]]
prompt = f"美食摄影, {name}, {cuisine}料理, 包含{','.join(ings)}, 精致摆盘, 餐厅灯光, 高分辨率, 写实风格"
neg = "cartoon, illustration, painting, text, watermark, ugly, blurry, low quality"
print(f"Prompt: {prompt[:150]}...")

# 加载 SD
t0 = time.time()
pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"模型加载: {time.time()-t0:.1f}s")

# 生成
t1 = time.time()
result = pipe(prompt, negative_prompt=neg, num_inference_steps=30, guidance_scale=7.5)
img = result.images[0]
out_dir = Path("src/data/images")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"test_{sample['recipe_id']}.jpg"
img.save(out_path)
print(f"生成耗时: {time.time()-t1:.1f}s")
print(f"保存到: {out_path.resolve()}")

# 清理
del pipe
gc.collect()
torch.cuda.empty_cache()
print("显存已清理")
