# knapspec.py

from typing import List, Optional
import torch
import torch.nn.functional as F
from utils import crop_kv_cache, _make_branch_parallel_causal_mask
import transformers
import time

class Knapspec:
    """
    Dynamic programming to select a skip set under a skip budget M.
    - L: number of decoder layers
    - M: max number of skips allowed
    - optimize() takes the per-layer hidden states for a single absolute token
      position and updates self.skip_set in place (1 = skip, 0 = keep).
    """

    def __init__(self, L: int, M: int, model, device: Optional[torch.device] = None, coefficients: Optional[tuple] = None, sim_threshold: float = 0.5):
        self.L = int(L)
        self.M = int(M)
        self.model = model
        self.layers = model.model.layers
        self.device = device or next(model.parameters()).device
        self.is_prefill_stage = True
        self.cached_hidden_states = [[] for i in range(2 * self.L + 1)]
        self.skip_set: List[int] = [0] * (2 * self.L)
        self.total_skip = 0

        # Stats tracking
        self.sum_best_tpt = 0.0
        self.sum_skip = 0
        self.sum_attn_skip = 0
        self.sum_mlp_skip = 0
        self.optimize_count = 0
        
        self.c_1, self.c_2, self.c_3 = coefficients
        self.sim_threshold = sim_threshold

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        # x = x.float()
        return F.normalize(x, p=2, dim=-1)

    def clear_cached_hidden_states(self):
        """Reset cached hidden states for all layers (used at the start of each SD round)."""
        self.cached_hidden_states = [[] for i in range(2 * self.L + 1)]

    def TPT(self, l, d, alpha, skip_set, t_attn, t_mlp):
        L = len(skip_set) // 2
        a = alpha[l]
        generated_tokens = (d + 1) if a == 1.0 else (a**(d + 1) - 1.0) / (a - 1.0)

        n_attn = sum(1 - skip_set[2*i + 0] for i in range(L))
        n_mlp  = sum(1 - skip_set[2*i + 1] for i in range(L))

        loaded_time = t_attn * (n_attn * d + L) + t_mlp * (n_mlp * d + L)
        return generated_tokens / loaded_time if loaded_time > 0 else 0.0

    def optimize_tpt(self, alpha, skip_sets, t_attn, t_mlp,d_max=18):
        max_skip = len(skip_sets) - 1
        max_tpt = -1
        best_l, best_d = -1, -1
        for l in range(1, max_skip + 1):
            for d in range(d_max):
                tpt = self.TPT(l, d, alpha, skip_sets[l], t_attn, t_mlp)
                if tpt > max_tpt:
                    max_tpt = tpt
                    best_l, best_d = l, d
        return best_l, best_d, max_tpt

    def _apply_layer_single(self, idx: int, x: torch.Tensor, past_key_values=None, branch_len=None, num_branches=None) -> torch.Tensor:
        layer_idx = idx // 2  # Which layer (0, 1, 2, ...)
        is_attention = (idx % 2 == 0)  # Even = attention, Odd = MLP
            
        decoder_layer = self.layers[layer_idx]
        device = self.device
        
        x = x.to(device)
        batch_size, seq_length = x.shape[0], x.shape[1]
        hidden_states = x
        
        if is_attention:
            # --- Apply Attention Block ---
            # Handle past key values for attention
            seq_length_with_past = seq_length
            past_key_values_length = 0
            
            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
                seq_length_with_past = seq_length + past_key_values_length
        
            # Convert to DynamicCache
            past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

            if branch_len is not None and num_branches is not None:
                pos_one = torch.arange(
                    past_key_values_length,
                    past_key_values_length + branch_len,
                    dtype=torch.long,
                    device=device,
                )
                pos_all = pos_one.repeat(num_branches)
                position_ids = pos_all.unsqueeze(0).expand(batch_size, -1)
                
            else:
                seq_length_with_past = seq_length + past_key_values_length
                position_ids = torch.arange(
                    past_key_values_length,
                    seq_length_with_past,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0).expand(batch_size, -1)

            attention_mask = _make_branch_parallel_causal_mask(
                input_ids_shape=(batch_size, seq_length),
                dtype=hidden_states.dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                branch_len=branch_len,
                num_branches=num_branches,
            )
            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            # Apply input layer norm
            normed_hidden = decoder_layer.input_layernorm(hidden_states)
            
            # Self-attention forward pass
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_kv_cache,
                output_attentions=False,
                use_cache=True,
            )
            
            # Residual connection
            hidden_states = hidden_states + attn_output
            
        else:
            # --- Apply MLP Block ---
            # Apply post-attention layer norm
            normed_hidden = decoder_layer.post_attention_layernorm(hidden_states)
            
            # MLP forward pass
            mlp_output = decoder_layer.mlp(normed_hidden)
            
            # Residual connection
            hidden_states = hidden_states + mlp_output
        
        return hidden_states.squeeze(1)

    @torch.inference_mode()
    def optimize(self, past_key_values=None) -> None:
        optimize_start_time = time.perf_counter()
        
        past_key_values_length = past_key_values[0][0].shape[2]
        t_mlp = self.c_1
        t_attn = self.c_2 * past_key_values_length + self.c_3
        
        if t_attn > t_mlp:
            w_attn, w_mlp = min(int(round(t_attn / t_mlp)), 5), int(1.0)
        else:
            w_attn, w_mlp = int(1.0), min(int(round(t_mlp / t_attn)), 5)
        
        max_time_saved = (w_attn + w_mlp) * self.L
        budget = max_time_saved // 2 

        xs: List[torch.Tensor] = []
        if self.cached_hidden_states:
            for layer_idx in range(2 * self.L + 1):
                if self.cached_hidden_states[layer_idx]:
                    xs.append(torch.cat(self.cached_hidden_states[layer_idx], dim=1).to(self.device))

        opt_num = xs[0].shape[1]
        optimization_cache = crop_kv_cache(past_key_values, past_key_values_length - opt_num)

        g: List[List[Optional[torch.Tensor]]] = [[None] * (budget + 1) for _ in range(2 * self.L + 1)]
        parent: List[List[Optional[tuple]]] = [[None] * (budget + 1) for _ in range(2 * self.L + 1)]
        best_sims = torch.full((2 * self.L + 1, budget + 1), -float('inf'), device=self.device)
        
        sim_threshold = self.sim_threshold
        
        # --- Optimization: Start from block 2 (xs[2]) directly ---
        g[2][0] = xs[2]
        best_sims[2][0] = 0.0 # Reset baseline similarity at the start of DP
        
        total_forward_branches = 0
        entries_per_layer = {} # Track g entries count

        # Start DP loop from i=3 (Processing Block 2)
        for i in range(3, 2 * self.L + 1):
            block_idx = i - 1
            is_attn = (block_idx % 2 == 0)
            w = w_attn if is_attn else w_mlp
            xi_norm = self._norm(xs[i].squeeze(0))

            # Find valid states from the previous layer
            prev_js = [j for j, state in enumerate(g[i-1]) if state is not None]
            
            # g entry count tracking
            entries_per_layer[i-1] = len(prev_js)
            
            if not prev_js: continue

            # --- Vectorized Parallel Processing ---
            G_input = torch.cat([g[i-1][j] for j in prev_js], dim=1)
            num_branches = len(prev_js)
            total_forward_branches += num_branches
            
            # 1. Forward Execute (Mandatory for the last layer blocks)
            applied = self._apply_layer_single(block_idx, G_input, optimization_cache, branch_len=opt_num, num_branches=num_branches)
            G_exec_batch = applied.reshape(num_branches, opt_num, -1)
            sims_exec = torch.einsum('ntd,td->n', self._norm(G_exec_batch), xi_norm)
            
            for idx, j in enumerate(prev_js):
                s_exec = sims_exec[idx]
                if (s_exec / opt_num) >= sim_threshold:
                    if s_exec > best_sims[i, j]:
                        best_sims[i, j] = s_exec
                        g[i][j] = G_exec_batch[idx:idx+1]
                        parent[i][j] = (j, False)

            # 2. Forward Skip (Only if NOT the last layer blocks)
            if block_idx < (2 * self.L - 2):
                G_skip_batch = G_input.reshape(num_branches, opt_num, -1)
                sims_skip = torch.einsum('ntd,td->n', self._norm(G_skip_batch), xi_norm)

                for idx, j in enumerate(prev_js):
                    target_j = j + w
                    if target_j <= budget:
                        s_skip = sims_skip[idx]
                        if (s_skip / opt_num) >= sim_threshold:
                            if s_skip > best_sims[i, target_j]:
                                best_sims[i, target_j] = s_skip
                                g[i][target_j] = G_skip_batch[idx:idx+1]
                                parent[i][target_j] = (j, True)

        # 3. Final Path Selection
        valid_js = [j for j, s in enumerate(g[-1]) if s is not None]
        entries_per_layer[2 * self.L] = len(valid_js)
        
        if not valid_js:
            print("[Warning] All paths pruned by threshold. Falling back to default.")
            self.skip_set = [0] * (2 * self.L)
            self.clear_cached_hidden_states()
            return

        hidden_stack = torch.cat([g[-1][j] for j in valid_js], dim=0)
        normed = self.model.model.norm(hidden_stack)
        top1_tokens = torch.argmax(self.model.lm_head(normed), dim=-1)
        teacher_top1 = torch.argmax(self.model.lm_head(self.model.model.norm(xs[-1])), dim=-1).squeeze(0)
        alpha_list = ((top1_tokens == teacher_top1.unsqueeze(0)).sum(dim=-1).float() / opt_num).tolist()

        all_skip_sets = []
        for j_start in valid_js:
            mask = [0] * (2 * self.L)
            curr_j = j_start
            for i in range(2 * self.L, 2, -1):
                p_info = parent[i][curr_j]
                if p_info is None: break
                prev_j, skipped = p_info
                if skipped: mask[i-1] = 1
                curr_j = prev_j
            all_skip_sets.append(mask)

        # Final TPT optimization
        best_idx, best_d, best_tpt = self.optimize_tpt(alpha_list, all_skip_sets, t_attn, t_mlp)
        best_alpha = alpha_list[best_idx]
        best_j = valid_js[best_idx]
        
        best_cos = best_sims[2 * self.L, best_j].item()
        self.skip_set = all_skip_sets[best_idx]

        # Housekeeping
        self.is_prefill_stage = False
        self.clear_cached_hidden_states()
        self.total_skip += sum(self.skip_set)

        # Accumulate stats
        self.sum_best_tpt += best_tpt
        self.sum_skip += sum(self.skip_set)
        self.sum_attn_skip += sum(self.skip_set[::2])
        self.sum_mlp_skip += sum(self.skip_set[1::2])
        self.optimize_count += 1
        
        optimize_end_time = time.perf_counter()
        opt_duration = optimize_end_time - optimize_start_time

        print(f"[Knapspec] New skip set {self.skip_set}")
        print(f"skips {sum(self.skip_set)}", 
              f"attn_skips {sum(self.skip_set[::2])}", 
              f"mlp_skips {sum(self.skip_set[1::2])}", 
              f"best_cos_avg {best_cos/opt_num:.4f}",
              f"best_alpha {best_alpha:.4f}",
              f"best_tpt {best_tpt:.4f}",
              f"past_len {past_key_values_length}")
              
        print(f"[OPT STATS] Duration: {opt_duration:.4f}s")
        print(f"[OPT STATS] Total Forward Branches: {total_forward_branches}")
        print(f"[OPT STATS] g[-1] entries: {len(valid_js)}")
        # Optional: Print all layers entries if needed, or just average
        avg_entries = sum(entries_per_layer.values()) / len(entries_per_layer) if entries_per_layer else 0
        print(f"[OPT STATS] Avg g entries per layer: {avg_entries:.2f}")
        # print(f"[OPT STATS] Per layer entries: {entries_per_layer}")