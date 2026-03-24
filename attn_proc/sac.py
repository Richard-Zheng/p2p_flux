import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import itertools
import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from diffusers.models.embeddings import apply_rotary_emb
from enum import IntEnum


class TransType(IntEnum):
    DOUBLE = 0
    SINGLE = 1


# copied from diffusers.models.transformers.transformer_flux._get_projections
def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


# copied from diffusers.models.transformers.transformer_flux._get_fused_projections
def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


# copied from diffusers.models.transformers.transformer_flux._get_qkv_projections
def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class SACFluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, pipe, prompt, out_width, out_height):
        super().__init__()
        self.pipe = pipe
        self.prompt = prompt
        self.block_idx = [0, 0]
        self.step_idx = 0
        self.batch_size = len(prompt)
        self.out_width = out_width
        self.out_height = out_height

        print("Injecting [SAC] Attention Processor into the pipeline...")
        for i, tblock in enumerate(pipe.transformer.transformer_blocks):
            tblock.attn.set_processor(self)
        for i, tblock in enumerate(pipe.transformer.single_transformer_blocks):
            tblock.attn.set_processor(self)
        self.attention_store = None
        self.num_text = None

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block_type = TransType.SINGLE if encoder_hidden_states is None else TransType.DOUBLE
        if block_type == TransType.DOUBLE:
            num_blocks = len(self.pipe.transformer.transformer_blocks)
        elif block_type == TransType.SINGLE:
            num_blocks = len(self.pipe.transformer.single_transformer_blocks)

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # 1. 正常应用 RoPE (因为 SAC 利用的就是同构的相对坐标，所以必须先加 RoPE)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # q k v shape: (batch_size, seq_len, heads, head_dim) [1, 4608, 24, 128]
        # to use attn.get_attention_scores() we need to reshape to (batch_size * heads, seq_len, head_dim)
        batch_size, seq_len, heads, head_dim = query.shape
        query = query.permute(0, 2, 1, 3).reshape(-1, query.shape[1], query.shape[3])
        key = key.permute(0, 2, 1, 3).reshape(-1, key.shape[1], key.shape[3])
        value = value.permute(0, 2, 1, 3).reshape(-1, value.shape[1], value.shape[3])

        # 2. 计算原生的 pre-softmax 注意力分数 (Similarity Matrix)
        scale = attn.head_dim**-0.5
        sim = torch.bmm(query, key.transpose(-1, -2)) * scale

        # =====================================================================
        # 🌟 核心魔改区：Self-Attention Cloning (SAC) 矩阵手术
        # =====================================================================
        inject_threshold = 30  # 注入步数
        
        if self.step_idx < inject_threshold:
            import math
            
            # 🔥 修复：动态计算文本 Token 数量
            if self.num_text is None and block_type == TransType.DOUBLE:
                # Double 块：文本和图像分开
                num_txt = encoder_hidden_states.shape[1]
                self.num_text = encoder_hidden_states.shape[1]
            else:
                # Single 块：文本和图像已经拼接在 query 里了
                # 图像长度固定为 4096 (1024x1024 打包后)，多出来的都是文本
                num_txt = self.num_text
            assert self.num_text is not None

            S_img = seq_len - num_txt # 确保 S_img 永远是 4096
            
            # 将图像部分的注意力矩阵重塑为 5D: (batch*heads, Q_y, Q_x, K_y, K_x)
            grid_size = int(math.sqrt(S_img)) # 这里会完美得到 64
            mid = grid_size // 2              # 32

            # 只截取图像与图像算注意力的部分 (右下角的 4096 x 4096)
            img_sim = sim[:, num_txt:, num_txt:].view(-1, grid_size, grid_size, grid_size, grid_size)

            # --- 矩阵克隆开始 ---
            # 目标: B' (右下, y:mid~end, x:mid~end)

            # 动作 1: 语义与结构克隆 (SAC 真正的灵魂)
            # 将 B' -> A' 的注意力分数直接改成与 B -> A 相同
            # 这样 B' 在尝试画橘子时，会完美套用原版苹果的结构骨架！
            clone_b_to_a = img_sim[:, mid:, :mid, :mid, :mid].clone()
            native_b_prime_to_a_prime = img_sim[:, mid:, mid:, :mid, mid:].clone()
            max_native_b_prime_to_a_prime = native_b_prime_to_a_prime.amax(dim=-1, keepdim=True)
            max_b_to_a = clone_b_to_a.amax(dim=-1, keepdim=True)
            # 🌟 动态对齐 (Logit Alignment)
            tau = 1.6
            aligned_b_prime_to_a_prime = clone_b_to_a - max_b_to_a + max_native_b_prime_to_a_prime + tau
            img_sim[:, mid:, mid:, :mid, mid:] = aligned_b_prime_to_a_prime

            # 动作 2: 阻断旧语义源 (防止画出红苹果)
            # 严禁 B' 看 A (左上的原苹果)，逼迫它只能顺着动作1去 A' 吸取橘子特征
            # img_sim[:, mid:, mid:, :mid, :mid] = -10000.0

            # 动作 3: B'结构需要和B一致
            # 直接把 B -> B 的分数克隆到 B' -> B 上，确保它们的结构感完全一致！
            # 但是 RoPE 导致的空间错位让它们的注意力分数大小不在同一水平，必须动态对齐一下才能安全克隆！
            # clone_b_to_b = img_sim[:, mid:, :mid, mid:, :mid].clone()
            # native_b_prime_to_b = img_sim[:, mid:, mid:, mid:, :mid].clone()
            # max_native_b_prime_to_b = native_b_prime_to_b.amax(dim=-1, keepdim=True)
            # max_b_to_b = clone_b_to_b.amax(dim=-1, keepdim=True)
            # # 🌟 动态对齐 (Logit Alignment)
            # # 将干净分数的峰值，强行平移到原生噪声分数的峰值水平
            # # tau = 0.0 表示 1:1 绝对公平竞争；tau = 1.0 表示让克隆特征有 e^1 倍的优势
            # tau = 1.0
            # aligned_b_to_b = clone_b_to_b - max_b_to_b + max_native_b_prime_to_b + tau
            # img_sim[:, mid:, mid:, mid:, :mid] = aligned_b_to_b

            # ⚠️ 极其关键的修复：什么都不做！
            # 绝对不要动 B' -> B'：让它自己平滑噪声，生成完美的油画笔触！
            # 绝对不要动 B' -> B：让它利用原生 RoPE，自然地与左边的桌布和背景融合！
            # --- 矩阵克隆结束 ---

            # 将手术后的注意力矩阵展平放回原处
            sim[:, num_txt:, num_txt:] = img_sim.view(-1, S_img, S_img)

        # =====================================================================
        # 继续正常的 Softmax 和 Value 聚合
        # =====================================================================
        attn.upcast_attention = True
        attn.upcast_softmax = True

        # Softmax 会自动处理我们塞进去的 -10000.0，使其概率归零
        attention_probs = sim.softmax(dim=-1)

        # =====================================================================
        # 🌟 Store Image-to-Image Attention Maps
        # =====================================================================
        # Extract L2L part, shape: (batch_size * heads, S_img, S_img)
        l2l_attn = attention_probs[:, self.num_text:, self.num_text:]
        
        # Reshape to separate batch and heads: (batch_size, heads, S_img, S_img)
        # l2l_attn = l2l_attn.reshape(self.batch_size, heads, S_img, S_img)
        
        # Average across all heads to reduce memory: (batch_size, S_img, S_img)
        l2l_attn = l2l_attn.mean(dim=0)
        
        # Accumulate the attention maps
        if self.attention_store is None:
            self.attention_store = l2l_attn
        else:
            self.attention_store += l2l_attn
        # =====================================================================

        hidden_states = torch.bmm(attention_probs, value)
        
        # ... 后面的维度还原代码保持原样 ...
        hidden_states = hidden_states.view(batch_size, attn.heads, seq_len, attn.head_dim).permute(0, 2, 1, 3)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        ret = None
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            ret = hidden_states, encoder_hidden_states
        else:
            ret = hidden_states

        print(f"完成第 {self.step_idx} 步，第 {self.block_idx[block_type]} 个 {block_type.name} 块的注意力计算。")
        self.block_idx[block_type] += 1
        if block_type is TransType.SINGLE and self.block_idx[block_type] >= num_blocks:
            self.block_idx = [0, 0]
            self.step_idx += 1
        return ret