import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import itertools
import torch
from diffusers.models.transformers.transformer_flux import FluxAttention
from diffusers.models.embeddings import apply_rotary_emb


# copied from examples/community/pipeline_prompt2prompt.py
def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    """
    Updates the alpha tensor based on the temporal bounds for cross-attention replacement.

    Args:
        alpha: The tensor storing attention weights modification values.
        bounds: A float (threshold) or a tuple (start, end) indicating the time steps to modify.
        prompt_ind: Verify the prompt index to begin the modification.
        word_inds: Indices of the words (tokens) to modify. If None, all words are modified.

    Returns:
        The updated alpha tensor.
    """
    if isinstance(bounds, float):
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_word_inds(text: str, target_word: Union[str, int], tokenizer) -> np.ndarray:
    words = text.split(" ")
    # strip the </s> end-of-sentence token
    tokens = [tokenizer.decode([item]) for item in tokenizer.encode(text)][:-1]

    # word_ends[i] = the index where i+1 word ends (not counting spaces)
    word_ends = list(itertools.accumulate(len(w) for w in words))
    token_ends = list(itertools.accumulate(len(t) for t in tokens))

    if isinstance(target_word, str):
        target_word_indices = [i for i, w in enumerate(words) if w == target_word]
    elif isinstance(target_word, int):
        target_word_indices = [target_word]
    else:
        raise TypeError("target_word must be either a string or an integer.")

    valid_ranges = []
    for idx in target_word_indices:
        start_char = 0 if idx == 0 else word_ends[idx - 1]
        end_char = word_ends[idx]
        valid_ranges.append((start_char, end_char))

    out = []
    for tok_idx, tok_end in enumerate(token_ends):
        if len(tokens[tok_idx]) == 0:
            if any(start <= tok_end <= end for start, end in valid_ranges):
                out.append(tok_idx)
        elif any(start < tok_end <= end for start, end in valid_ranges):
            out.append(tok_idx)

    return np.array(out)


def get_replacement_mapper(x: str, y: str, tokenizer, max_len=512):
    """
    终极版：基于“补集思想”的无指针实现。
    保证未替换部分（包含 BOS/EOS/Pad）形成严格的平移对角线 1-to-1 映射。
    """
    words_x, words_y = x.split(' '), y.split(' ')
    if len(words_x) != len(words_y):
        raise ValueError("Prompt A and B must have the same number of words for Replacement.")
        
    mapper = np.zeros((max_len, max_len))
    
    # 记录哪些 token 索引属于“被替换的词”，它们将被特殊处理
    replaced_src_toks = set()
    replaced_tgt_toks = set()
    
    # 1. 优先处理发生替换的单词区块
    for word_idx, (w_x, w_y) in enumerate(zip(words_x, words_y)):
        if w_x != w_y:
            src_toks = get_word_inds(x, word_idx, tokenizer)
            tgt_toks = get_word_inds(y, word_idx, tokenizer)
            
            # 过滤超长截断
            src_toks = [t for t in src_toks if t < max_len]
            tgt_toks = [t for t in tgt_toks if t < max_len]
            
            if src_toks and tgt_toks:
                # 均分注意力权重
                ratio = 1.0 / len(tgt_toks)
                mapper[np.ix_(src_toks, tgt_toks)] = ratio
                
                # 将这些 token 索引加入“已消耗”集合
                replaced_src_toks.update(src_toks)
                replaced_tgt_toks.update(tgt_toks)

    # 2. 处理剩余的所有 Token（未替换的词、BOS、EOS、Padding）
    # 巧妙之处：提取出所有没被修改的 token 索引
    unreplaced_src = [i for i in range(max_len) if i not in replaced_src_toks]
    unreplaced_tgt = [i for i in range(max_len) if i not in replaced_tgt_toks]
    
    # 既然它们都没被修改，那它们绝对是一一对应的！直接拉链(zip)配对即可。
    # 这会在矩阵上画出一条完美的、根据替换情况自动断开并平移的对角线 1.0
    for s, t in zip(unreplaced_src, unreplaced_tgt):
        mapper[s, t] = 1.0

    return torch.from_numpy(mapper).float()

def get_replacement_mapper_multi_prompts(prompts, tokenizer, max_len=512):
    """对多个 prompt 进行 Replace 映射矩阵生成的包装函数"""
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        # 逐个调用底层的 _ 函数
        mapper = get_replacement_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers) # 堆叠打包返回

# copied from examples/community/pipeline_prompt2prompt.py
def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
    tokenizer,
    max_num_words=512,
):
    """
    Computes the alpha tensor controlling cross-attention replacement over time and words.

    Args:
        prompts: A list of text prompts.
        num_steps: The number of inference steps.
        cross_replace_steps: A dictionary or float defining the time steps for cross-attention replacement.
                             If a dictionary, keys are words and values are (start, end) tuples.
        tokenizer: The tokenizer for encoding prompts.
        max_num_words: The maximum number of words (tokens) to consider.

    Returns:
        A tensor of shape (num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) containing the alpha values.
    """
    if not isinstance(cross_replace_steps, dict):
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


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

        # one-liner
        self.attn_map_callback(attention_probs)

        hidden_states = torch.bmm(attention_probs, value)
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

    def __init__(self, prompts, pipeline, num_inference_steps):
        self.prompts = prompts
        self.pipeline = pipeline
        self.num_inference_steps = num_inference_steps
        self.batch_size = len(prompts) if prompts is not None else 1


class EmptyControl(AttentionControl):
    def on_attn_map(self, attn, is_single: bool, index):
        if is_single:
            print(f'single attn[{index}] processed, total {len(self.pipeline.transformer.single_transformer_blocks)}')
        else:
            print(f'attn[{index}] processed, total {len(self.pipeline.transformer.transformer_blocks)}')
        return attn


class AttentionStore(AttentionControl):
    def on_attn_map(self, attn, is_single: bool, index):
        if not is_single:
            # attn shape: (batch_size * heads, prompt_seq+latent_seq, prompt_seq+latent_seq)
            # first we verify every line sums to 1
            assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), rtol=0.01), f'Attention map rows sums to {attn.sum(dim=-1)}'

            # store attention maps for MM-DiT layers only
            # prompt_seq 512 tokens, latent_seq 4096 for 64x64 latent image
            # extract prompt->latent attention maps only
            batch_heads, seq_len, _ = attn.shape
            assert batch_heads % self.batch_size == 0, f'batch_heads {batch_heads} not divisible by batch_size {self.batch_size}'
            heads = batch_heads // self.batch_size
            prompt_seq_len = 512
            latent_seq_len = seq_len - prompt_seq_len
            prompt_to_latent_attn = attn[:, :prompt_seq_len, prompt_seq_len:]
            # now the shape is (batch_size * heads, prompt_seq, latent_seq) softmaxed over latent_seq
            assert prompt_to_latent_attn.shape == (self.batch_size * heads, prompt_seq_len, latent_seq_len)

            num_att_layers = len(self.pipeline.transformer.transformer_blocks)
            # average over heads and num_att_layers and self.num_inference_steps
            prompt_to_latent_attn = prompt_to_latent_attn.reshape(self.batch_size, heads, prompt_seq_len, latent_seq_len)
            prompt_to_latent_attn = prompt_to_latent_attn.mean(dim=1) / num_att_layers / self.num_inference_steps
            # now add to step_store, take average over num_att_layers
            if self.attention_store is None:
                self.attention_store = prompt_to_latent_attn
            else:
                self.attention_store += prompt_to_latent_attn
            del prompt_to_latent_attn
        return attn

    def __init__(self, prompts, pipeline, num_inference_steps):
        super().__init__(prompts, pipeline, num_inference_steps)
        self.attention_store = None


class AttentionControlEdit(AttentionControl):
    @abc.abstractmethod
    def replace_p2l_attention(self, attn_base, att_replace):
        raise NotImplementedError

    @abc.abstractmethod
    def replace_l2p_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def on_attn_map(self, attn, is_single: bool, index):
        if not is_single:
            batch_heads, seq_len, _ = attn.shape
            assert batch_heads % self.batch_size == 0, f'batch_heads {batch_heads} not divisible by batch_size {self.batch_size}'
            heads = batch_heads // self.batch_size
            prompt_seq_len = 512
            latent_seq_len = seq_len - prompt_seq_len

            prompt_to_latent_attn = attn[:, :prompt_seq_len, prompt_seq_len:]
            assert prompt_to_latent_attn.shape == (self.batch_size * heads, prompt_seq_len, latent_seq_len)
            prompt_to_latent_attn = prompt_to_latent_attn.reshape(self.batch_size, heads, prompt_seq_len, latent_seq_len)
            prompt_to_latent_attn = self.replace_p2l_attention(prompt_to_latent_attn[0], prompt_to_latent_attn[1:])
            prompt_to_latent_attn = prompt_to_latent_attn.reshape(batch_heads, prompt_seq_len, latent_seq_len)
            attn[:, :prompt_seq_len, prompt_seq_len:] = prompt_to_latent_attn

            latent_to_prompt_attn = attn[:, prompt_seq_len:, :prompt_seq_len]
            assert latent_to_prompt_attn.shape == (self.batch_size * heads, latent_seq_len, prompt_seq_len)
            latent_to_prompt_attn = latent_to_prompt_attn.reshape(self.batch_size, heads, latent_seq_len, prompt_seq_len)
            latent_to_prompt_attn = self.replace_l2p_attention(latent_to_prompt_attn[0], latent_to_prompt_attn[1:])
            latent_to_prompt_attn = latent_to_prompt_attn.reshape(batch_heads, latent_seq_len, prompt_seq_len)
            attn[:, prompt_seq_len:, :prompt_seq_len] = latent_to_prompt_attn

        return attn

    def __init__(self, prompts, pipeline, num_inference_steps):
        super().__init__(prompts, pipeline, num_inference_steps)


class AttentionReplace(AttentionControlEdit):
    def replace_p2l_attention(self, attn_base, att_replace):
        return torch.einsum('hpl,bpn->bhnl', attn_base, self.replacement_mapper)

    def replace_l2p_attention(self, attn_base, att_replace):
        return torch.einsum('hlp,bpn->bhln', attn_base, self.replacement_mapper)

    def __init__(self, prompts, pipeline, num_inference_steps):
        super().__init__(prompts, pipeline, num_inference_steps)
        self.replacement_mapper = get_replacement_mapper_multi_prompts(prompts, pipeline.tokenizer).to('cuda')