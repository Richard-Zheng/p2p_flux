import abc
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from diffusers.models.embeddings import apply_rotary_emb


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


class P2PFluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, attn_map_callback):
        super().__init__()
        self.attn_map_callback = attn_map_callback

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        query_reshaped = query.permute(0, 2, 1, 3).reshape(-1, query.shape[1], query.shape[3])
        key_reshaped = key.permute(0, 2, 1, 3).reshape(-1, key.shape[1], key.shape[3])
        value_reshaped = value.permute(0, 2, 1, 3).reshape(-1, value.shape[1], value.shape[3])

        # the original dispatch_attention_fn uses torch.nn.functional.scaled_dot_product_attention
        # which is highly optimized to be numerically stable in float16 or bfloat16 by using online softmax(?)
        # we don't have that. So we upcast to float32 here to be safe.
        attn.upcast_attention = True
        attn.upcast_softmax = True

        attn.scale = attn.head_dim**-0.5

        attention_probs = attn.get_attention_scores(
            query_reshaped,
            key_reshaped,
            attention_mask=attention_mask,
        )

        # one-liner
        self.attn_map_callback(attention_probs)

        hidden_states = torch.bmm(attention_probs, value_reshaped)
        # hidden_states shape: (batch_size * heads, seq_len, head_dim)
        # reshape back to (batch_size, seq_len, heads, head_dim)
        hidden_states = hidden_states.view(batch_size, attn.heads, seq_len, attn.head_dim).permute(0, 2, 1, 3)
        # flatten heads and head_dim
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class AttentionControl(abc.ABC):
    def register_attention_control(self, pipe):
        for i, tblock in enumerate(pipe.transformer.transformer_blocks):
            attn_map_callback = lambda probs, idx=i: self.on_attn_map(probs, is_single=False, index=idx)
            attn_proc = P2PFluxAttnProcessor(attn_map_callback=attn_map_callback)
            tblock.attn.set_processor(attn_proc)
        for i, tblock in enumerate(pipe.transformer.single_transformer_blocks):
            attn_map_callback = lambda probs, idx=i: self.on_attn_map(probs, is_single=True, index=idx)
            attn_proc = P2PFluxAttnProcessor(attn_map_callback=attn_map_callback)
            tblock.attn.set_processor(attn_proc)

    @abc.abstractmethod
    def on_attn_map(self, attn, is_single: bool, index):
        raise NotImplementedError

    def __init__(self, num_att_layers, num_single_att_layers, num_inference_steps):
        self.num_att_layers = num_att_layers
        self.num_single_att_layers = num_single_att_layers
        self.num_inference_steps = num_inference_steps


class EmptyControl(AttentionControl):
    def on_attn_map(self, attn, is_single: bool, index):
        if is_single:
            print(f'single attn[{index}] processed, total {self.num_single_att_layers}')
        else:
            print(f'attn[{index}] processed, total {self.num_att_layers}')
        return attn


class AttentionStore(AttentionControl):
    def on_attn_map(self, attn, is_single: bool, index):
        if not is_single:
            # store attention maps for MM-DiT layers only
            # attn shape: (batch_size * heads, prompt_seq+latent_seq, prompt_seq+latent_seq)
            # here we assume batch_size=1 during inference
            # prompt_seq 512 for 77 tokens, latent_seq 4096 for 64x64 latent image
            # extract prompt->latent attention maps only
            heads, seq_len, _ = attn.shape
            prompt_seq_len = 512
            latent_seq_len = seq_len - prompt_seq_len
            prompt_to_latent_attn = attn[:, :prompt_seq_len, prompt_seq_len:]
            # now the shape is (batch_size * heads, prompt_seq, latent_seq) softmaxed over latent_seq
            # average over heads and self.num_att_layers and self.num_inference_steps
            prompt_to_latent_attn = prompt_to_latent_attn.mean(dim=0) / self.num_att_layers / self.num_inference_steps
            # now add to step_store, take average over self.num_att_layers
            if self.attention_store is None:
                self.attention_store = prompt_to_latent_attn
            else:
                self.attention_store += prompt_to_latent_attn
            del prompt_to_latent_attn
        return attn

    def __init__(self, num_att_layers, num_single_att_layers, num_inference_steps):
        super().__init__(num_att_layers, num_single_att_layers, num_inference_steps)
        self.attention_store = None

