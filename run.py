from PIL import Image, ImageDraw

def prepare_batch_icl_inputs(a_path, aa_path, b_path, target_size=(512, 512)):
    """
    为 FLUX.1-Fill 准备 Batch=2 的 In-Context Learning 输入。
    target_size: 单张子图的尺寸 (W, H)。建议 512x512，拼接后为 1024x512。
    """
    # 1. 加载并统一尺寸
    img_A = Image.open(a_path).convert("RGB").resize(target_size)
    img_A_prime = Image.open(aa_path).convert("RGB").resize(target_size)
    img_B = Image.open(b_path).convert("RGB").resize(target_size)

    w, h = target_size
    batch_w, batch_h = w * 2, h

    # ==========================================
    # 2. 构建 Image Batch (拼图)
    # ==========================================
    # Batch 0: [ A | A' ]
    image_0 = Image.new("RGB", (batch_w, batch_h))
    image_0.paste(img_A, (0, 0))
    image_0.paste(img_A_prime, (w, 0))

    # Batch 1: [ B | B ]  <- 右侧放 B 的复制品作为结构先验
    image_1 = Image.new("RGB", (batch_w, batch_h))
    image_1.paste(img_B, (0, 0))
    image_1.paste(img_B, (w, 0))

    # ==========================================
    # 3. 构建 Mask Batch
    # ==========================================
    # Batch 0 Mask: 全黑 (0)，代表完全保留，不重绘
    mask_0 = Image.new("L", (batch_w, batch_h), 0)

    # Batch 1 Mask: 左黑右白，代表保留左侧，重绘右侧
    mask_1 = Image.new("L", (batch_w, batch_h), 0)
    draw = ImageDraw.Draw(mask_1)
    # 画一个白色的矩形覆盖右半边
    draw.rectangle([w, 0, batch_w, batch_h], fill=255)

    # ==========================================
    # 4. 组装为 Pipeline 可接收的格式
    # ==========================================
    images = [image_0, image_1]
    masks = [mask_0, mask_1]

    # 为了方便 Debug，你可以选择把拼接好的图保存下来看一眼
    image_0.save("batch_0_context.png")
    image_1.save("batch_1_target.png")
    mask_1.save("batch_1_mask.png")

    return images, masks

def prepare_2x2_icl_inputs(a_path, aa_path, b_path, target_size=(512, 512)):
    """
    为 FLUX.1-Fill 准备 2x2 四方格的 In-Context Learning 输入。
    """
    w, h = target_size
    grid_w, grid_h = w * 2, h * 2

    # 1. 加载图像
    img_A = Image.open(a_path).convert("RGB").resize(target_size)
    img_A_prime = Image.open(aa_path).convert("RGB").resize(target_size)
    img_B = Image.open(b_path).convert("RGB").resize(target_size)

    # 2. 构建 Image Grid
    # [ A (左上) | A' (右上) ]
    # [ B (左下) | B' (右下, 用 B 垫底以稳固结构) ]
    image_grid = Image.new("RGB", (grid_w, grid_h))
    image_grid.paste(img_A, (0, 0))              
    image_grid.paste(img_A_prime, (w, 0))        
    image_grid.paste(img_B, (0, h))              
    image_grid.paste(img_B, (w, h))              

    # 3. 构建 Mask Grid
    # 前三个象限全黑 (0)，只有右下角 B' 为纯白 (255)
    mask_grid = Image.new("L", (grid_w, grid_h), 0)
    draw = ImageDraw.Draw(mask_grid)
    draw.rectangle([w, h, grid_w, grid_h], fill=255)

    # 保存预览
    image_grid.save("grid_image.png")
    mask_grid.save("grid_mask.png")

    return image_grid, mask_grid

# 使用示例：
image, mask = prepare_2x2_icl_inputs("a.png", "aa.png", "b.png")
# pipeline(prompt="", image=images, mask_image=masks, ...)

import torch
import torchvision.transforms as T
import numpy as np
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from attn_proc.vanillia import VanilliaFluxAttnProcessor
from attn_proc.sac import SACFluxAttnProcessor

pipe = FluxFillPipeline.from_pretrained("/home/frain/Documents/FLUX.1-Fill", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

num_inference_steps = 50
prompt=[
"In the masked region on the bottom-right, generate the exact same oil painting scene, but replace the three red apples with three oranges. Strictly preserve the original oil painting style, the lighting, the plate, the blue patterned cloth, and all other background details perfectly."
]
out_width = 1024
out_height = 1024

attn_processor = VanilliaFluxAttnProcessor(pipe, prompt, out_width, out_height)

image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    width=out_width,
    height=out_height,
    guidance_scale=30,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images

image[0].save(f"out.png")
to_tensor = T.ToTensor()
save_dict = {
    "image": torch.stack([to_tensor(img) for img in image]),
    "attention_map": attn_processor.attention_store,
    "out_width": out_width,
    "out_height": out_height,
}
torch.save(save_dict, "controller_attention_store.pt")
