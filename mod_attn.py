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


class VanilliaFluxAttnProcessor:
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
        trans_type = TransType.SINGLE if encoder_hidden_states is None else TransType.DOUBLE
        if trans_type == TransType.DOUBLE:
            num_blocks = len(self.pipe.transformer.transformer_blocks)
        elif trans_type == TransType.SINGLE:
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
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

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

            self.after_call()
            ret = hidden_states, encoder_hidden_states
        else:
            ret = hidden_states

        print(f"完成第 {self.step_idx} 步，第 {self.block_idx[trans_type]} 个 {trans_type.name} 块的注意力计算。")
        self.block_idx[trans_type] += 1
        if trans_type is TransType.SINGLE and self.block_idx[trans_type] >= num_blocks:
            self.block_idx = [0, 0]
            self.step_idx += 1
        return ret


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
        trans_type = TransType.SINGLE if encoder_hidden_states is None else TransType.DOUBLE
        if trans_type == TransType.DOUBLE:
            num_blocks = len(self.pipe.transformer.transformer_blocks)
        elif trans_type == TransType.SINGLE:
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

            self.after_call()
            ret = hidden_states, encoder_hidden_states
        else:
            ret = hidden_states

        print(f"完成第 {self.step_idx} 步，第 {self.block_idx[trans_type]} 个 {trans_type.name} 块的注意力计算。")
        self.block_idx[trans_type] += 1
        if trans_type is TransType.SINGLE and self.block_idx[trans_type] >= num_blocks:
            self.block_idx = [0, 0]
            self.step_idx += 1
        return ret


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
        trans_type = TransType.SINGLE if encoder_hidden_states is None else TransType.DOUBLE
        if trans_type == TransType.DOUBLE:
            num_blocks = len(self.pipe.transformer.transformer_blocks)
        elif trans_type == TransType.SINGLE:
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

            self.after_call()
            ret = hidden_states, encoder_hidden_states
        else:
            ret = hidden_states

        print(f"完成第 {self.step_idx} 步，第 {self.block_idx[trans_type]} 个 {trans_type.name} 块的注意力计算。")
        self.block_idx[trans_type] += 1
        if trans_type is TransType.SINGLE and self.block_idx[trans_type] >= num_blocks:
            self.block_idx = [0, 0]
            self.step_idx += 1
        return ret


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
        trans_type = TransType.SINGLE if encoder_hidden_states is None else TransType.DOUBLE
        if trans_type == TransType.DOUBLE:
            num_blocks = len(self.pipe.transformer.transformer_blocks)
        elif trans_type == TransType.SINGLE:
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
            if self.num_text is None and trans_type == TransType.DOUBLE:
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

        print(f"完成第 {self.step_idx} 步，第 {self.block_idx[trans_type]} 个 {trans_type.name} 块的注意力计算。")
        self.block_idx[trans_type] += 1
        if trans_type is TransType.SINGLE and self.block_idx[trans_type] >= num_blocks:
            self.block_idx = [0, 0]
            self.step_idx += 1
        return ret