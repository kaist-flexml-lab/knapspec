# clasp.py

from typing import List, Optional
import torch
import torch.nn.functional as F
from utils import _prepare_decoder_attention_mask, crop_kv_cache, _make_branch_parallel_causal_mask
import transformers
import math

class CLaSp:
    def __init__(self, L: int, M: int, model, device: Optional[torch.device] = None, coefficients=None):
        self.L = int(L)
        self.M = int(M)
        self.model = model
        self.layers = model.model.layers
        self.device = device or next(model.parameters()).device
        self.is_prefill_stage = True
        self.cached_hidden_states = [[] for i in range(self.L + 1)]
        self.skip_set: List[int] = [0] * self.L
        
        self.coefficients = coefficients
        if coefficients:
             self.c_1, self.c_2, self.c_3 = coefficients
        else:
             self.c_1 = 0
             self.c_2 = 0
             self.c_3 = 0
        
        self.best_tpt_list = []

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def clear_cached_hidden_states(self):
        self.cached_hidden_states = [[] for i in range(self.L + 1)]

    @torch.inference_mode()
    def _apply_layer_single(self, idx: int, x: torch.Tensor, past_key_values=None, num_branches=None) -> torch.Tensor:
        """Parallel forward pass for multiple DP candidates (branches)."""
        decoder_layer = self.layers[idx]
        device = self.device
        
        batch_size, seq_length = x.shape[0], x.shape[1]
        branch_len = 1 
        
        # Handle case where past_key_values might be None (initial step)
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            past_kv_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)
        else:
            past_kv_cache = transformers.cache_utils.DynamicCache()
        
        seq_length_with_past = seq_length + past_key_values_length
    
        pos_one = torch.tensor([past_key_values_length], dtype=torch.long, device=device)
        pos_all = pos_one.repeat(num_branches)
        position_ids = pos_all.unsqueeze(0).expand(batch_size, -1)
        
        attention_mask = _make_branch_parallel_causal_mask(
            input_ids_shape=(batch_size, seq_length),
            dtype=x.dtype,
            device=device,
            past_key_values_length=past_key_values_length,
            branch_len=branch_len,
            num_branches=num_branches,
        )
        
        position_embeddings = self.model.model.rotary_emb(x, position_ids)
        
        outputs = decoder_layer(
            x,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_kv_cache,
            output_attentions=False,
            use_cache=True,
        )
    
        # Ensure the output is 3D [batch, seq, hidden] to avoid IndexError in DP table
        hidden_out = outputs[0]
        if hidden_out.ndim == 2:
            hidden_out = hidden_out.unsqueeze(0)
            
        return hidden_out

    @torch.inference_mode()
    def optimize(self, number_of_matches: int, past_key_values=None, alpha: float = 0.0, gamma: int = 4) -> None:
        past_key_values_length = past_key_values[0][0].shape[2]
        
        if self.coefficients and alpha >= 0:
            # Calculate TPT
            # expected_tokens = sum_{k=0}^{gamma} alpha^k = (1 - alpha^(gamma+1)) / (1 - alpha)
            if alpha >= 1.0:
                expected_tokens = gamma + 1.0
            elif alpha <= 0.0:
                expected_tokens = 1.0
            else:
                expected_tokens = (1 - alpha**(gamma + 1)) / (1 - alpha)
            
            # Cost = (Verify_L + Draft_Ops) * Time_Per_Layer
            # Verify_L = L
            # Draft_Ops = gamma * (L - sum(skip_set))
            skip_count = sum(self.skip_set)
            layer_ops = self.L + gamma * (self.L - skip_count)
            
            t_mlp = self.c_1
            t_attn = self.c_2 * past_key_values_length + self.c_3

            time_cost = layer_ops * (t_attn + t_mlp)
            
            if time_cost > 0:
                tpt = expected_tokens / time_cost
                self.best_tpt_list.append(tpt)
        
        xs: List[torch.Tensor] = []
        if self.cached_hidden_states:
            for layer_idx in range(self.L + 1):
                if len(self.cached_hidden_states[layer_idx]) > 0:
                    layer_hidden = self.cached_hidden_states[layer_idx][0]
                    x = layer_hidden[:, number_of_matches:number_of_matches+1, :]
                    xs.append(x.to(self.device))

        g: List[List[Optional[torch.Tensor]]] = [[None] * (self.M + 1) for _ in range(self.L + 1)]
        g[0][0] = xs[0]

        optimization_cache = crop_kv_cache(past_key_values, past_key_values_length-1)
        
        for i in range(1, self.L + 1):
            g[i][0] = xs[i]
            ell = min(i - 1, self.M)
            
            if ell > 0:
                G_list = []
                S_list = []
                
                for j in range(1, ell + 1):
                    G_list.append(g[i - 1][j])
                    S_list.append(g[i - 1][j - 1])
                
                # --- PARALLEL STEP ---
                G_cat = torch.cat(G_list, dim=1) # [1, ell, D]
                applied_states = self._apply_layer_single(i - 1, G_cat, optimization_cache, num_branches=ell)
                
                # --- SIMILARITY STEP ---
                xi = xs[i] # [1, 1, D]
                sim_no = torch.sum(self._norm(applied_states) * self._norm(xi), dim=-1).squeeze(0) # [ell]
                
                S_cat = torch.cat(S_list, dim=1) # [1, ell, D]
                sim_sk = torch.sum(self._norm(S_cat) * self._norm(xi), dim=-1).squeeze(0) # [ell]

                choose_no = sim_no > sim_sk

                for j in range(1, ell + 1):
                    if bool(choose_no[j - 1]):
                        # slice ensures result stays 3D [1, 1, D]
                        g[i][j] = applied_states[:, j-1:j, :] 
                    else:
                        g[i][j] = S_list[j - 1]

            if i <= self.M:
                g[i][i] = g[i - 1][i - 1]

        # Backtrack
        mask = [0] * self.L
        i, j = self.L, self.M
        while i > 0 and j > 0:
            cur, diag = g[i][j], g[i - 1][j - 1]
            if (cur is not None) and (diag is not None) and (cur is diag):
                mask[i - 1] = 1
                i -= 1
                j -= 1
            else:
                i -= 1

        self.skip_set = mask
        self.is_prefill_stage = False
        self.clear_cached_hidden_states()
        print(f"[CLASP] New skip set: {self.skip_set}", "skips:", sum(self.skip_set))