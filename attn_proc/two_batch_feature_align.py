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


class FeatureAlignFluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, pipe, prompt):
        super().__init__()
        self.pipe = pipe
        self.prompt = prompt
        self.block_idx = [0, 0]
        self.step_idx = 0
        self.batch_size = len(prompt)

        for i, tblock in enumerate(pipe.transformer.transformer_blocks):
            tblock.attn.set_processor(self)
        for i, tblock in enumerate(pipe.transformer.single_transformer_blocks):
            tblock.attn.set_processor(self)

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

        # =====================================================================
        # 🌟 终极魔改区：无 RoPE 语义路由与动态对齐代数
        # 必须在 apply_rotary_emb 之前执行，利用纯语义特征进行空间对齐！
        # =====================================================================
        
        start_inject = 0   # 前 5 步让初始噪声稳定
        end_inject = 35    # 后 15 步交还给模型自己画油画质感
        
        if start_inject <= self.step_idx < end_inject:
            # 1. 剥离文本 Token，只拿图像 Token
            num_text = encoder_key.shape[1] if attn.added_kv_proj_dim is not None else 0
            img_q = query[:, num_text:, :, :].clone()
            img_k = key[:, num_text:, :, :].clone()
            img_v = value[:, num_text:, :, :].clone()

            # 序列长度 S (1024x512 打包后通常是 2048)，mid 就是 1024
            S = img_q.shape[1]
            mid = S // 2

            # 2. 提取特征并转换形状为 (heads, sequence, head_dim) 以便计算 bmm
            # B: 待生成的底图 (油画苹果)
            q_B = img_q[1, :mid, :, :].permute(2, 0, 1)      # (heads, mid, dim)
            v_B = img_v[1, :mid, :, :].permute(2, 0, 1)      
            
            # A & A': 参考图 (照片苹果 & 照片橘子)
            k_A = img_k[0, :mid, :, :].permute(2, 0, 1)      
            v_A = img_v[0, :mid, :, :].permute(2, 0, 1)      
            v_A_prime = img_v[0, mid:, :, :].permute(2, 0, 1) 

            # 3. 🧠 核心魔法 1：无 RoPE 语义寻路
            # 让油画苹果(q_B) 去照片苹果(k_A) 里找对应的部位
            scale = attn.head_dim ** -0.5
            sim = torch.bmm(q_B, k_A.transpose(1, 2)) * scale
            attn_weights = torch.softmax(sim, dim=-1)  # 形状: (heads, mid, mid)

            # 4. 🧠 核心魔法 2：特征重组 (Feature Reshuffling)
            # 用注意力权重把 A 和 A' "吸"过来，使得它们的像素排列强行扭曲成 B 的形状！
            aligned_v_A = torch.bmm(attn_weights, v_A)             # 变成 B 形状的照片苹果
            aligned_v_A_prime = torch.bmm(attn_weights, v_A_prime) # 变成 B 形状的照片橘子

            # 5. 🧠 核心魔法 3：安全的对齐任务代数
            # 因为现在大家形状完全一致，直接相减不会有任何空间错位污染！
            lambda_shift = 1.0  # 橘子特征强度
            delta_v_aligned = aligned_v_A_prime - aligned_v_A
            
            # 施加到油画苹果上，诞生"油画橘子"的特征
            v_syn_h = v_B + lambda_shift * delta_v_aligned
            
            # 6. 覆写到 Value 张量 B' 的位置
            v_syn = v_syn_h.permute(1, 2, 0) # 还原形状: (mid, heads, dim)
            value[1, num_text + mid:, :, :] = v_syn

            # (可选) 也可以把 K 同步覆写，但通常修改 V 已经足够主导生成内容
            # k_syn_h = k_B + lambda_shift * (torch.bmm(attn_weights, k_A_prime.permute(2,0,1)) - torch.bmm(attn_weights, k_A.permute(2,0,1)))
            # key[1, num_text + mid:, :, :] = k_syn_h.permute(1, 2, 0)

        # =====================================================================
        # 恢复代码，继续执行 RoPE 和正常的 Self-Attention
        # =====================================================================
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            
        # ... 后续原封不动的 Attention 计算 ...

        # q k v shape: (batch_size, seq_len, heads, head_dim) [1, 4608, 24, 128]
        # to use attn.get_attention_scores() we need to reshape to (batch_size * heads, seq_len, head_dim)
        batch_size, seq_len, heads, head_dim = query.shape
        query = query.permute(0, 2, 1, 3).reshape(-1, query.shape[1], query.shape[3])
        key = key.permute(0, 2, 1, 3).reshape(-1, key.shape[1], key.shape[3])
        value = value.permute(0, 2, 1, 3).reshape(-1, value.shape[1], value.shape[3])

        # the original dispatch_attention_fn uses torch.nn.functional.scaled_dot_product_attention
        # which is highly optimized to be numerically stable in float16 or bfloat16 by using online softmax(?)
        # we don't have that. So we upcast to float32 here to be safe.
        attn.upcast_attention = True
        attn.upcast_softmax = True

        attn.scale = attn.head_dim**-0.5

        attention_probs = attn.get_attention_scores(
            query,
            key,
            attention_mask=attention_mask,
        )

        hidden_states = torch.bmm(attention_probs, value)
        # hidden_states shape: (batch_size * heads, seq_len, head_dim)
        # reshape back to (batch_size, seq_len, heads, head_dim)
        hidden_states = hidden_states.view(batch_size, attn.heads, seq_len, attn.head_dim).permute(0, 2, 1, 3)
        # flatten heads and head_dim
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