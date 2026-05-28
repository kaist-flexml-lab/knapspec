# utils.py
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import transformers
import math
import triton

@dataclass
class Env:
    model: PreTrainedModel
    tok: PreTrainedTokenizerBase
    device: str
    eos_id: Optional[int]
    pad_id: Optional[int]

@dataclass
class GenerationResult:
    text: str
    num_output_tokens: int
    output_ids: Optional[List[int]] = None
    # Speculative Decoding fields
    acceptance_rate: float = None
    tokens_per_layer: float = None
    draft_time: float = None
    verify_time: float = None
    optimization_time: float = None
    total_time: float = None
    total_accepted_length: int = None
    total_steps: int = None # speculation step
    avg_best_tpt: float = None

# ------------------------
# Attention mask helpers
# ------------------------

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """Create causal attention mask for decoder."""
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask

def _make_branch_parallel_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int,
    branch_len: int,
    num_branches: Optional[int] = None,
):
    bsz, tgt_len = input_ids_shape
    if num_branches is None:
        assert tgt_len % branch_len == 0
        num_branches = tgt_len // branch_len

    total_src_len = past_key_values_length + tgt_len

    mask = torch.full(
        (tgt_len, total_src_len),
        torch.finfo(dtype).min,
        device=device,
    )
    mask = mask.to(dtype)

    for b in range(num_branches):
        for t in range(branch_len):
            q_idx = b * branch_len + t

            if past_key_values_length > 0:
                mask[q_idx, :past_key_values_length] = 0.0

            branch_start = past_key_values_length + b * branch_len
            branch_end = branch_start + t + 1
            mask[q_idx, branch_start:branch_end] = 0.0

    mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, total_src_len)
    return mask

# ------------------------
# KV Cache helpers
# ------------------------

def crop_kv_cache(kv_cache, target_length: int):
    """Crop KV cache to target_length"""
    if kv_cache is None:
        return None
    
    cropped_cache = []
    for layer_cache in kv_cache:
        if layer_cache is None:
            cropped_cache.append(None)
        else:
            key, value = layer_cache
            cropped_key = key[:, :, :target_length, :]
            cropped_value = value[:, :, :target_length, :]
            cropped_cache.append((cropped_key, cropped_value))
    
    return tuple(cropped_cache)


# ------------------------
# Sampling helpers
# ------------------------

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    if top_k > 0:
        logits = transformers.generation.logits_process.TopKLogitsWarper(
            top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep
        )(None, logits)

    if 0 <= top_p <= 1.0:
        logits = transformers.generation.logits_process.TopPLogitsWarper(
            top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep
        )(None, logits)

    return logits


# ------------------------
# Decode helpers
# ------------------------

def decode_next_token(
    logits: torch.Tensor,
    token_idx: int = None,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if token_idx:
        logits = logits[:, -1, :]

    if not sample:
        next_token = logits.argmax(dim=-1)
        if not token_idx:
            logits = logits.squeeze(dim=0)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return next_token, probabilities
    else:
        if not token_idx:
            logits = logits.squeeze(dim=0)
        filtered_logits = top_k_top_p_filtering(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if not token_idx:
            next_token = next_token.transpose(1, 0)
        return next_token, probabilities


# ------------------------
# Forward passes
# ------------------------

def forward(model, input_ids: torch.Tensor, past_kv_cache=None):
    if input_ids.dim() == 1:
        input_ids = input_ids.view(1, -1)
    
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # Handle past key values
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_kv_cache is not None:
        past_key_values_length = past_kv_cache[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    # Position IDs
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # Attention mask
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Embedding
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    for idx, decoder_layer in enumerate(model.model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_divided(model, input_ids: torch.Tensor, past_kv_cache=None,):
    if input_ids.dim() == 1:
        input_ids = input_ids.view(1, -1)
    
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # Handle past key values
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_kv_cache is not None:
        past_key_values_length = past_kv_cache[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    # Position IDs
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Embedding
    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

    for idx, decoder_layer in enumerate(model.model.layers): 
        normed_hidden = decoder_layer.input_layernorm(hidden_states)
        attn_output, _ = decoder_layer.self_attn(
            hidden_states=normed_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
        )
        hidden_states = hidden_states + attn_output

        normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_output = decoder_layer.mlp(normed_hidden)
        hidden_states = hidden_states + mlp_output

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache

def forward_draft(model, input_ids: torch.Tensor, skip_set: List[int], past_kv_cache=None):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # Handle past key values
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_kv_cache is not None:
        max_cache_length = 0
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                cache_length = layer_cache[0].shape[2]
                max_cache_length = max(max_cache_length, cache_length)
        past_key_values_length = max_cache_length
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    # Position IDs
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Embedding
    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    for i, decoder_layer in enumerate(model.model.layers):
        if skip_set[i] == 1:
            continue
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_verify(model, verify_input: torch.Tensor, draft_tokens: List[int], past_kv_cache=None, clasp=None, optimization_phase=False):
    device = verify_input.device
    batch_size, seq_length = verify_input.shape
    seq_length_with_past = seq_length
    draft_past_key_values_length = 0
    full_past_key_values_length = 0
    prompt_length = seq_length - len(draft_tokens)

    # Handle past key values
    if past_kv_cache is not None and past_kv_cache[0] is not None:
        draft_past_key_values_length = past_kv_cache[0][0].shape[2]
        
        if len(past_kv_cache) == len(model.model.layers):
            full_past_key_values_length = past_kv_cache[-1][0].shape[2]
        else:
            full_past_key_values_length = 0
        
        seq_length_with_past = seq_length + draft_past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    
    inputs_embeds = model.model.embed_tokens(verify_input)
    
    # Position IDs
    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length + full_past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask for full verification
    attention_mask = verify_input.new_ones(
        (batch_size, seq_length + full_past_key_values_length),
        dtype=torch.bool,
    )
    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,
    )

    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    layer_hidden = hidden_states[:, prompt_length-1:, :]
    if optimization_phase:
        clasp.cached_hidden_states[0].append(layer_hidden.detach())

    # Run through all layers and collect hidden_states for CLASP
    for idx, decoder_layer in enumerate(model.model.layers):
        hidden_states= decoder_layer(
            hidden_states,
            attention_mask=full_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
        
        # Store layer output for CLASP (verification tokens only)
        if optimization_phase:
            layer_hidden = hidden_states[:, prompt_length-1:, :]
            clasp.cached_hidden_states[idx+1].append(layer_hidden.detach())

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_draft_multi(model, input_ids: torch.Tensor, skip_set: List[int], past_kv_cache=None):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_kv_cache is not None:
        max_cache_length = 0
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                cache_length = layer_cache[0].shape[2]
                max_cache_length = max(max_cache_length, cache_length)
        past_key_values_length = max_cache_length
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    # Position IDs
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # Embedding
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    # Decoder layers with skipping using skip_set directly
    for i, decoder_layer in enumerate(model.model.layers):
        if skip_set[i] == 1:  # 1 means skip this layer
            continue
            
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_kv_cache = past_kv_cache.to_legacy_cache()    
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_verify_multi(
    model, 
    verify_input: torch.Tensor, 
    past_kv_cache=None, 
    clasp=None, 
    optimization_phase=False,
):
    device = verify_input.device
    batch_size, seq_length = verify_input.shape
    full_past_key_values_length = 0

    # Handle past key values
    if past_kv_cache is not None and past_kv_cache[0] is not None:        
        if len(past_kv_cache) == len(model.model.layers):
            full_past_key_values_length = past_kv_cache[-1][0].shape[2]
        else:
            full_past_key_values_length = 0
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    
    inputs_embeds = model.model.embed_tokens(verify_input)
    
    # Position IDs
    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length + full_past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask for full verification
    attention_mask = verify_input.new_ones(
        (batch_size, seq_length + full_past_key_values_length),
        dtype=torch.bool,
    )
    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,
    )

    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds    
    if optimization_phase:
        if clasp.is_prefill_stage:
            layer_hidden = hidden_states[:, -64:, :]
        else:
            layer_hidden = hidden_states
        clasp.cached_hidden_states[0].append(layer_hidden.detach())

    # Run through all layers and collect hidden_states for CLASP
    for idx, decoder_layer in enumerate(model.model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=full_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
        
        # Store layer output for CLASP (verification tokens only)
        if optimization_phase:
            if clasp.is_prefill_stage:
                layer_hidden = hidden_states[:, -64:, :]
            else:
                layer_hidden = hidden_states
            clasp.cached_hidden_states[idx+1].append(layer_hidden.detach())

    past_kv_cache = past_kv_cache.to_legacy_cache()    
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_draft_divided(
    model,
    input_ids: torch.Tensor,
    skip_set: List[int],
    past_kv_cache: Optional[list] = None,
):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    past_key_values_length = 0
    if past_kv_cache is not None:
        # Find maximum cache length across all layers
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                cache_len = layer_cache[0].shape[2]  # key tensor: [batch, heads, seq_len, head_dim]
                past_key_values_length = max(past_key_values_length, cache_len)

    seq_with_past = seq_length + past_key_values_length

    # Convert legacy cache to DynamicCache for processing
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    position_ids = torch.arange(
        past_key_values_length, 
        past_key_values_length + seq_length, 
        dtype=torch.long, 
        device=device
    ).unsqueeze(0)  # [1, seq_length]

    attention_mask_bool = input_ids.new_ones((batch_size, seq_with_past), dtype=torch.bool)
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model, 
        attention_mask_bool, 
        (batch_size, seq_length), 
        inputs_embeds, 
        past_key_values_length
    )

    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    for idx, decoder_layer in enumerate(model.model.layers):
        attn_skip = bool(skip_set[2 * idx + 0])
        mlp_skip = bool(skip_set[2 * idx + 1])

        if not attn_skip:
            normed_hidden = decoder_layer.input_layernorm(hidden_states)
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_kv_cache,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = hidden_states + attn_output
        else:
            pass

        if not mlp_skip:
            normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
            mlp_output = decoder_layer.mlp(normed_hidden)
            hidden_states = hidden_states + mlp_output
        else:
            pass

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_verify_divided(
    model, 
    verify_input: torch.Tensor, 
    draft_tokens: List[int], 
    past_kv_cache=None, 
    claspd=None, 
    optimization_phase=False,
):
    device = verify_input.device
    batch_size, seq_length = verify_input.shape
    full_past_key_values_length = 0
    prompt_length = seq_length - len(draft_tokens)

    # Handle past key values
    if past_kv_cache is not None and past_kv_cache[0] is not None:        
        if len(past_kv_cache) == len(model.model.layers):
            full_past_key_values_length = past_kv_cache[-1][0].shape[2]
        else:
            full_past_key_values_length = 0
    
    # Convert to DynamicCache
    if past_kv_cache is None:
        past_kv_cache = transformers.cache_utils.DynamicCache()
    else:
        past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    
    inputs_embeds = model.model.embed_tokens(verify_input)
    
    # Position IDs
    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length + full_past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask for full verification
    attention_mask = verify_input.new_ones(
        (batch_size, seq_length + full_past_key_values_length),
        dtype=torch.bool,
    )
    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,
    )

    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    
    # Store initial embedding hidden states (index 0)
    if optimization_phase:
        layer_hidden = hidden_states[:, prompt_length-1:, :]  # [batch, num_verify_tokens, hidden]
        claspd.cached_hidden_states[0].append(layer_hidden.detach())

    # Run through all layers with attention/MLP division
    for idx, decoder_layer in enumerate(model.model.layers):
        normed_hidden = decoder_layer.input_layernorm(hidden_states)
        attn_output, _ = decoder_layer.self_attn(
            hidden_states=normed_hidden,
            attention_mask=full_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
        )
        hidden_states = hidden_states + attn_output
        
        # Store hidden states after attention (index 2*idx + 1)
        if optimization_phase:
            layer_hidden = hidden_states[:, prompt_length-1:, :]  # [batch, num_verify_tokens, hidden]
            claspd.cached_hidden_states[2*idx + 1].append(layer_hidden.detach())

        normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_output = decoder_layer.mlp(normed_hidden)
        hidden_states = hidden_states + mlp_output
        
        # Store hidden states after MLP (index 2*idx + 2)
        if optimization_phase:
            layer_hidden = hidden_states[:, prompt_length-1:, :]  # [batch, num_verify_tokens, hidden]
            claspd.cached_hidden_states[2*idx + 2].append(layer_hidden.detach())
    
    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache



def forward_draft_divided_multi(
    model,
    input_ids: torch.Tensor,
    skip_set: List[int],
    past_kv_cache: Optional[list] = None,
):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    past_key_values_length = 0
    if past_kv_cache is not None:
        for layer_cache in past_kv_cache:
            if layer_cache is not None:
                cache_len = layer_cache[0].shape[2]
                past_key_values_length = max(past_key_values_length, cache_len)

    seq_with_past = seq_length + past_key_values_length

    # Convert legacy cache to DynamicCache for processing
    past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)

    # ---- Prepare position IDs ----
    position_ids = torch.arange(
        past_key_values_length, 
        past_key_values_length + seq_length, 
        dtype=torch.long, 
        device=device
    ).unsqueeze(0)  # [1, seq_length]

    # ---- Prepare attention mask ----
    attention_mask_bool = input_ids.new_ones(
        (batch_size, seq_with_past), 
        dtype=torch.bool
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model, 
        attention_mask_bool, 
        (batch_size, seq_length), 
        inputs_embeds, 
        past_key_values_length
    )

    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    for idx, decoder_layer in enumerate(model.model.layers):
        attn_skip = bool(skip_set[2 * idx + 0])
        mlp_skip = bool(skip_set[2 * idx + 1])

        if not attn_skip:
            normed_hidden = decoder_layer.input_layernorm(hidden_states)
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_kv_cache,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = hidden_states + attn_output
        else:
            pass

        if not mlp_skip:
            normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
            mlp_output = decoder_layer.mlp(normed_hidden)
            hidden_states = hidden_states + mlp_output
        else:
            pass

    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache


def forward_verify_divided_multi(
    model, 
    verify_input: torch.Tensor, 
    past_kv_cache=None, 
    claspd=None, 
    optimization_phase=False,
):
    device = verify_input.device
    batch_size, seq_length = verify_input.shape
    full_past_key_values_length = 0

    # Handle past key values
    if past_kv_cache is not None and past_kv_cache[0] is not None:
        if len(past_kv_cache) == len(model.model.layers):
            full_past_key_values_length = past_kv_cache[-1][0].shape[2]
        else:
            full_past_key_values_length = 0
    
    # Convert to DynamicCache
    past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_kv_cache)
    
    inputs_embeds = model.model.embed_tokens(verify_input)
    
    # Position IDs
    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length + full_past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Attention mask for full verification
    attention_mask = verify_input.new_ones(
        (batch_size, seq_length + full_past_key_values_length),
        dtype=torch.bool,
    )
    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,
    )

    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    if optimization_phase:
        if claspd.is_prefill_stage:
            layer_hidden = hidden_states[:, -64:, :]
        else:
            layer_hidden = hidden_states
        claspd.cached_hidden_states[0].append(layer_hidden.detach())

    # Run through all layers with attention/MLP division
    for idx, decoder_layer in enumerate(model.model.layers):
        normed_hidden = decoder_layer.input_layernorm(hidden_states)
        attn_output, _ = decoder_layer.self_attn(
            hidden_states=normed_hidden,
            attention_mask=full_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
        )
        hidden_states = hidden_states + attn_output
        
        # Store hidden states after attention (index 2*idx + 1)
        if optimization_phase:
            if claspd.is_prefill_stage:
                layer_hidden = hidden_states[:, -64:, :]
            else:
                layer_hidden = hidden_states
            claspd.cached_hidden_states[2*idx+1].append(layer_hidden.detach())

        normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_output = decoder_layer.mlp(normed_hidden)
        hidden_states = hidden_states + mlp_output
        
        # Store hidden states after MLP (index 2*idx + 2)
        if optimization_phase:
            if claspd.is_prefill_stage:
                layer_hidden = hidden_states[:, -64:, :]
            else:
                layer_hidden = hidden_states
            claspd.cached_hidden_states[2*idx+2].append(layer_hidden.detach())
    
    past_kv_cache = past_kv_cache.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, past_kv_cache
# ------------------------
# DEL Forward functions
# ------------------------

def forward_early_DEL(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
    DEL,
) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]]]:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    # Check if past_key_values is already DynamicCache or needs conversion
    if past_key_values is None:
         past_key_values = transformers.cache_utils.DynamicCache()
    elif not isinstance(past_key_values, transformers.cache_utils.DynamicCache):
         past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

    for idx, decoder_layer in enumerate(model.model.layers[:exit_layer]):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
        if idx in DEL.eligible_exit_layers:
            DEL.cached_hidden_states[idx].append(hidden_states)

    past_key_values = past_key_values.to_legacy_cache()

    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = hidden_states
    else:
        exit_query_cache = torch.cat([exit_query_cache, hidden_states], dim=1)

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    
    return logits, past_key_values, exit_query_cache


def forward_remainder_DEL(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
    DEL,
) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]]]:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    num_tokens_to_generate: int = 1
    seq_length_with_past = seq_length
    draft_past_key_values_length: int = 0
    full_past_key_values_length: int = 0

    if past_key_values is not None and past_key_values[0] is not None:
        # it's okay to use the first layer because the draft model necessairly computes it
        draft_past_key_values_length = past_key_values[0][0].shape[2]
        # the total sequence length is the past key values since that includes the draft tokens

        # the last layer should not have been skipped, we can get this to check how many of the tokens have gone through full
        # verification
        if len(past_key_values) == len(model.model.layers):
            full_past_key_values_length = past_key_values[-1][0].shape[2]
        else:
            # we have not done a full pass yet so the history is 0
            full_past_key_values_length = 0
    
        seq_length_with_past = num_tokens_to_generate + draft_past_key_values_length
    
    # Check if past_key_values is already DynamicCache or needs conversion
    if past_key_values is None:
         past_key_values = transformers.cache_utils.DynamicCache()
    elif not isinstance(past_key_values, transformers.cache_utils.DynamicCache):
         past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    inputs_embeds = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length_with_past,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    early_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, num_tokens_to_generate),
        inputs_embeds,
        draft_past_key_values_length,
    )

    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,  # we have no past for the full model
    )
    
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

    next_decoder_cache = []
    hidden_states = inputs_embeds
    # TODO simplify
    full_hidden_states: Optional[torch.FloatTensor] = None
    
    for idx, decoder_layer in enumerate(model.model.layers):
        is_early_exit = idx < exit_layer
        
        if is_early_exit:
            # early hidden states: B x num_gen x C
            early_hidden_states = hidden_states[:, -num_tokens_to_generate:]
            
            # Recalculate position embeddings for early exit if needed, or slice existing ones if possible?
            # Since model.model.rotary_emb returns (cos, sin)
            cos, sin = position_embeddings
            early_pos_embeddings = (cos[:, -num_tokens_to_generate:, :], sin[:, -num_tokens_to_generate:, :])

            hidden_states = decoder_layer(
                early_hidden_states,
                attention_mask=early_attention_mask,
                position_embeddings=early_pos_embeddings,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
            if idx in DEL.eligible_exit_layers:
                DEL.cached_hidden_states[idx].append(hidden_states)
        else:
            if full_hidden_states is None and exit_query_cache is not None:
                # first time seeing the full hidden states, we need to rely on the
                # query cache
                # only use if exit query cache exists, if not this is our first call
                full_hidden_states = torch.cat(
                    [exit_query_cache, hidden_states[:, -num_tokens_to_generate:]],
                    dim=1,
                )
            else:
                # we already have seen the fully hidden states we can re-use them now
                full_hidden_states = hidden_states
                
            hidden_states = decoder_layer(
                full_hidden_states,
                attention_mask=full_attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
            if idx in DEL.eligible_exit_layers:
                DEL.cached_hidden_states[idx].append(hidden_states)

    past_key_values = past_key_values.to_legacy_cache()
    DEL.cached_hidden_states[len(model.model.layers)-1].append(hidden_states)

    return None, past_key_values, exit_query_cache
