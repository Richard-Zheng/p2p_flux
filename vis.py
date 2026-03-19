import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_flux_icl_attention(
    original_image_path,
    attention_store,
    target_height,
    target_width,
    output_dir="attention_maps",
    batch_idx=0,
    step=16
):
    """
    可视化 FLUX.1-Fill 的 Latent-to-Latent 注意力图，支持任意 16 的倍数的宽高。
    
    参数:
    original_image_path: 原图路径
    attention_store: 形状为 (batch_size, seq_len, seq_len) 的注意力张量
    target_height: 生成/处理时设定的高度 (必须是 16 的倍数)
    target_width: 生成/处理时设定的宽度 (必须是 16 的倍数)
    output_dir: 图像保存目录
    batch_idx: 需要可视化的 batch 索引
    step: 遍历 token 的步长
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载原图，并强制 resize 到与 Pipeline 中相同的尺寸
    # 这非常重要，否则红框在原图上的物理位置会产生偏移
    img = Image.open(original_image_path).convert("RGB")
    if img.size != (target_width, target_height):
        img = img.resize((target_width, target_height), Image.LANCZOS)
    img_array = np.array(img)

    # 2. 动态计算 Token 网格的形状
    pixels_per_token = 16  # 8 (VAE) * 2 (Patch)
    h_tok = target_height // pixels_per_token
    w_tok = target_width // pixels_per_token
    total_tokens = h_tok * w_tok

    attn_matrix = attention_store[batch_idx]
    
    # 确保张量在 CPU 上并转为 numpy
    if torch.is_tensor(attn_matrix):
        attn_matrix = attn_matrix.detach().cpu().float().numpy()

    # 校验张量尺寸是否匹配
    expected_shape = (total_tokens, total_tokens)
    if attn_matrix.shape != expected_shape:
        raise ValueError(f"注意力矩阵形状 {attn_matrix.shape} 与推导出的形状 {expected_shape} 不匹配！")

    count = 0
    for token_idx in range(0, total_tokens, step):
        # 3. 计算二维 Token 坐标 (注意宽度使用的是 w_tok)
        y_tok = token_idx // w_tok
        x_tok = token_idx % w_tok

        # 计算在原图中的左上角像素坐标
        y_pixel = y_tok * pixels_per_token
        x_pixel = x_tok * pixels_per_token

        # 4. 提取并 Reshape 注意力图为实际的高宽比例
        attn_map = attn_matrix[token_idx]
        attn_map_2d = attn_map.reshape((h_tok, w_tok))

        # 可视化部分
        fig, axes = plt.subplots(1, 2, figsize=(12, 6 * (target_height / target_width)))

        # --- 左图 ---
        axes[0].imshow(img_array)
        axes[0].set_title(f"Image (Token Index: {token_idx})\nPos: x={x_tok}, y={y_tok}")
        axes[0].axis('off')

        rect = patches.Rectangle(
            (x_pixel, y_pixel),
            pixels_per_token, pixels_per_token,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[0].add_patch(rect)

        # --- 右图 ---
        # 加入一个非常小的偏置然后取 log，增强较暗区域的对比度，方便观察长距离上下文
        log_attn_map_2d = np.log(attn_map_2d + 1e-6)
        im = axes[1].imshow(log_attn_map_2d, cmap='inferno')
        axes[1].set_title(f"Attention Map ({w_tok}x{h_tok})")
        axes[1].axis('off')
        
        fig.colorbar(im, ax=axes[1], fraction=0.046 * (target_height / target_width), pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"attn_map_token_{token_idx:04d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        count += 1

    print(f"可视化完成。共生成 {count} 张对比图。")

loaded_attention_store = torch.load("controller_attention_store.pt", weights_only=True)
# --- 结合你的代码示例进行调用 ---
visualize_flux_icl_attention(
    original_image_path="flux-fill-dev.png",
    attention_store=loaded_attention_store, # 确保传入正确的属性
    target_height=1632,  # 传入你 Pipeline 中设置的 height
    target_width=1232,   # 传入你 Pipeline 中设置的 width
    batch_idx=0,
    step=16
)