import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import FluxFillPipeline
import mod_attn

def prepare_batch_icl_inputs(a_path, aa_path, b_path, target_size=(512, 512)):
    """
    为 FLUX.1-Fill 准备 Batch=2 的 In-Context Learning 输入。
    【完美解耦新布局】
    Batch 0: [ A | B ] (纯参考：左边是原图A(真苹果)，右边是原图B(油画苹果))
    Batch 1: [ A' | B ] (待生成：左边是目标图A'(真橘子)，右边是B的复制品，用于重绘为油画橘子)
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
    # Batch 0: [ A | B ]
    image_0 = Image.new("RGB", (batch_w, batch_h))
    image_0.paste(img_A, (0, 0))
    image_0.paste(img_B, (w, 0))

    # Batch 1: [ A' | B ]  <- 右侧放 B 的复制品作为生成底板
    image_1 = Image.new("RGB", (batch_w, batch_h))
    image_1.paste(img_A_prime, (0, 0))
    image_1.paste(img_B, (w, 0))

    # ==========================================
    # 3. 构建 Mask Batch
    # ==========================================
    # Batch 0 Mask: 全黑 (0)，代表完全保留，不重绘
    mask_0 = Image.new("L", (batch_w, batch_h), 0)

    # Batch 1 Mask: 左黑右白，代表保留左侧A'，重绘右侧为B'
    mask_1 = Image.new("L", (batch_w, batch_h), 0)
    draw = ImageDraw.Draw(mask_1)
    # 画一个白色的矩形覆盖右半边
    draw.rectangle([w, 0, batch_w, batch_h], fill=255)

    # ==========================================
    # 4. 组装为 Pipeline 可接收的格式
    # ==========================================
    images = [image_0, image_1]
    masks = [mask_0, mask_1]

    # 为了方便 Debug，保存下来看一眼
    image_0.save("batch_0_context.png")
    image_1.save("batch_1_target.png")
    mask_1.save("batch_1_mask.png")

    return images, masks

# ==========================================
# 运行推理部分
# ==========================================

# 1. 准备输入 (注意这里接收的是列表 images 和 masks)
images, masks = prepare_batch_icl_inputs("a.png", "aa.png", "b.png")

# 2. 加载模型
pipe = FluxFillPipeline.from_pretrained("/home/frain/Documents/FLUX.1-Fill", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

num_inference_steps = 50

# 提示词稍微泛化一点也是可以的，因为主要的引导力现在来自图像上下文
prompt=[
"In the masked region on the right, generate the exact same oil painting scene, but replace the three red apples with three oranges. Strictly preserve the original oil painting style, the lighting, the plate, the blue patterned cloth, and all other background details perfectly."
]*2

# 3. 注册你的自定义 Attention
mod_attn.register_attention_control(pipe, mod_attn.TwoBatchFluxAttnProcessor, prompt)

# 4. 执行管道
output_images = pipe(
    prompt=prompt,
    image=images,       # 传入 Batch 图像列表
    mask_image=masks,   # 传入 Batch 遮罩列表
    width=1024,
    height=512,
    guidance_scale=30,  # 稍微偏高的 cfg scale 有助于压制伪影
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42)
).images

# 5. 保存结果
output_images[0].save("out0.png")
output_images[1].save("out1.png")