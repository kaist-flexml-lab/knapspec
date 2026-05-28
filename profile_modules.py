# profile_modules.py
import torch
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# ==========================================
# 1. Mask Functions (User Provided)
# ==========================================
def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(input_shape, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length)
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    return combined_attention_mask

# ==========================================
# 2. Precise Profiling Logic (Manual Sync)
# ==========================================
def profile_model(
    model,
    tokenizer=None,
    seq_lengths: list = None,
    layer_idx: int = 0,
    warmup_steps: int = 5,
    measure_steps: int = 10 
):
    device = next(model.parameters()).device
    print(f"[Profiling] Profiling model on device: {device}...")
    
    config = model.config
    num_layers = config.num_hidden_layers
    target_layer = model.model.layers[layer_idx]
    rotary_emb = model.model.rotary_emb
    
    # Dimension Extraction
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    
    # [수정 부분] Qwen3-4B처럼 공식이 안 맞는 경우를 위해 실제 head_dim 속성을 우선 참조합니다.
    head_dim = getattr(config, "head_dim", getattr(target_layer.self_attn, "head_dim", config.hidden_size // num_heads))

    print(f"[Profiling] Detected: heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    if seq_lengths is None:
         seq_lengths = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000]

    attn_times = []
    mlp_times = []

    print("[Profiling] Starting measurement loop...")
    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        
        # 1. Setup Inputs
        hidden_states = torch.randn(1, 1, config.hidden_size, device=device, dtype=torch.float16)
        position_ids = torch.tensor([[seq_len]], device=device, dtype=torch.long)
        cos, sin = rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)
        
        # 2. Mask
        raw_attention_mask = torch.ones((1, 1 + seq_len), dtype=torch.bool, device=device)
        attention_mask = _prepare_decoder_attention_mask(
            model=model, attention_mask=raw_attention_mask, input_shape=(1, 1),
            inputs_embeds=hidden_states, past_key_values_length=seq_len
        )
        
        legacy_list = []
        for i in range(num_layers):
            if i == layer_idx:
                pk = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
                pv = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
                legacy_list.append((pk, pv))
            else:
                 # Minimal dummy for other layers to satisfy DynamicCache format
                 legacy_list.append((
                     torch.empty(1, num_kv_heads, 0, head_dim, device=device, dtype=torch.float16), 
                     torch.empty(1, num_kv_heads, 0, head_dim, device=device, dtype=torch.float16)
                 ))
        
        legacy_snap = tuple(legacy_list)

        # --- Attn Measurement ---
        temp_attn = []
        for i in range(warmup_steps + measure_steps):
            dc = DynamicCache.from_legacy_cache(legacy_snap)
            normed = target_layer.input_layernorm(hidden_states)
            
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.inference_mode():
                outputs = target_layer.self_attn(
                    hidden_states=normed, 
                    attention_mask=attention_mask, 
                    position_embeddings=position_embeddings, 
                    past_key_values=dc, 
                    use_cache=True
                )
                _ = hidden_states + outputs[0]
            end.record()
            torch.cuda.synchronize()
            
            if i >= warmup_steps:
                temp_attn.append(start.elapsed_time(end))

        # --- MLP Measurement ---
        temp_mlp = []
        for i in range(warmup_steps + measure_steps):
            normed = target_layer.post_attention_layernorm(hidden_states)
            
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.inference_mode():
                out = target_layer.mlp(normed)
                _ = hidden_states + out
            end.record()
            torch.cuda.synchronize()
            
            if i >= warmup_steps:
                temp_mlp.append(start.elapsed_time(end))

        t_attn, t_mlp = np.mean(temp_attn), np.mean(temp_mlp)
        attn_times.append(t_attn)
        mlp_times.append(t_mlp)

    # Linear Regression Fitting
    c_1 = np.mean(mlp_times)
    slope, intercept = np.polyfit(np.array(seq_lengths), np.array(attn_times), 1)
    
    print(f"[Profiling] Done. c_1={c_1:.6f}, c_2={slope:.6e}, c_3={intercept:.6f}")
    
    # [Added for Plotting]
    print("seq_lengths =", seq_lengths)
    print("attn_times =", attn_times)
    print("mlp_times =", mlp_times)
    
    return c_1, slope, intercept