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


class TwoBatchFluxAttnProcessor:
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

        if image_rotary_emb is not None:
            # 应用 RoPE 之后，此时坐标已经完美对齐！
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # =====================================================================
        # 🌟 核心魔改区：跨批次特征注入 (Cross-Batch Injection)
        # 当前形状: (batch_size=2, seq_len, heads, head_dim)
        # =====================================================================
        
        # 1. 将 Batch 0 和 Batch 1 沿着 batch 维度拆开
        q_0, q_1 = query.chunk(2, dim=0)  # (1, seq_len, heads, head_dim)
        k_0, k_1 = key.chunk(2, dim=0)
        v_0, v_1 = value.chunk(2, dim=0)

        # 2. 策略控制：决定在哪些去噪步 (Step) 执行特征注入
        # 建议只在前中期的结构/语义形成阶段注入，后期的像素细化阶段让 Batch 1 自己画
        inject_threshold = 30  # 假设总步数是 50 步
        
        if self.step_idx < inject_threshold:
            # 将 Batch 0 的 K 和 V 沿着 seq_len 维度 (dim=1) 拼接到 Batch 1 后面
            # 这样 Batch 1 的 Query 在寻找线索时，既能看到自己，也能看到 Batch 0
            k_1_ext = torch.cat([k_1, k_0], dim=1)  # (1, 2 * seq_len, heads, head_dim)
            v_1_ext = torch.cat([v_1, v_0], dim=1)
        else:
            k_1_ext = k_1
            v_1_ext = v_1

        # =====================================================================
        # 重新排列维度以适配 Attention 计算
        # 形状变换: (1, seq_len, heads, head_dim) -> (heads, seq_len, head_dim)
        # =====================================================================
        
        # Batch 0 保持原样
        q_0 = q_0.permute(0, 2, 1, 3).reshape(-1, q_0.shape[1], q_0.shape[3])
        k_0 = k_0.permute(0, 2, 1, 3).reshape(-1, k_0.shape[1], k_0.shape[3])
        v_0 = v_0.permute(0, 2, 1, 3).reshape(-1, v_0.shape[1], v_0.shape[3])

        # Batch 1 使用扩充后的 KV
        q_1 = q_1.permute(0, 2, 1, 3).reshape(-1, q_1.shape[1], q_1.shape[3])
        k_1_ext = k_1_ext.permute(0, 2, 1, 3).reshape(-1, k_1_ext.shape[1], k_1_ext.shape[3])
        v_1_ext = v_1_ext.permute(0, 2, 1, 3).reshape(-1, v_1_ext.shape[1], v_1_ext.shape[3])

        attn.upcast_attention = True
        attn.upcast_softmax = True
        attn.scale = attn.head_dim**-0.5

        # =====================================================================
        # 分别计算 Attention，避免填充 (Padding) 的麻烦
        # =====================================================================
        
        # 计算 Batch 0
        attention_probs_0 = attn.get_attention_scores(q_0, k_0, attention_mask=None)
        hidden_states_0 = torch.bmm(attention_probs_0, v_0)

        # 计算 Batch 1
        attention_probs_1 = attn.get_attention_scores(q_1, k_1_ext, attention_mask=None)
        hidden_states_1 = torch.bmm(attention_probs_1, v_1_ext)

        # 将 Batch 0 和 Batch 1 沿着 Batch*Heads 维度 (dim=0) 拼回完整的张量
        hidden_states = torch.cat([hidden_states_0, hidden_states_1], dim=0)

        # =====================================================================
        # 恢复维度并输出 (与原代码保持一致)
        # =====================================================================
        
        # hidden_states shape: (batch_size * heads, seq_len, head_dim)
        # reshape back to (batch_size, seq_len, heads, head_dim)
        batch_size = query.shape[0] # 这里的 batch_size 是 2
        seq_len = query.shape[1]
        
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