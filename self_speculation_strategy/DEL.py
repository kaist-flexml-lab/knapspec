import torch
import itertools
from typing import List, Optional, Tuple

from utils import (
    decode_next_token,
)


class DEL:
    def __init__(self, model, gamma_max, eligible_exit_layers=None, omega=1.0, gamma_min=0, coefficients=None):
        self.model = model
        self.token_match_counts = torch.zeros((len(model.model.layers), 2))
        self.confidence_sums = torch.zeros((len(model.model.layers), 2))
        self.average_confidence = torch.zeros((len(model.model.layers), 2))
        self.is_prefill_stage = True
        self.omega = omega
        if eligible_exit_layers is None:
            self.eligible_exit_layers = list(range(0, len(self.model.model.layers)-1))
        else:
            self.eligible_exit_layers = eligible_exit_layers
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.cached_hidden_states = [[] for i in range(len(self.model.model.layers))]
        self.current_exit_layer = self.eligible_exit_layers[0]+1
        self.current_gamma = 1
        
        self.c_1, self.c_2, self.c_3 = coefficients
        
        self.best_tpt_list = []

    def clear_cached_hidden_states(self):
        """Reset cached hidden states for all layers (used at the start of each SD round)."""
        self.cached_hidden_states = [[] for i in range(len(self.model.model.layers))]
    
    def remove_old_hidden_states(self, input_length):
        """
        Trim outdated cached hidden states based on consumed input length.
        Applies a 32-token prefill adjustment on first use.
        """
        if self.is_prefill_stage:
            input_length = max(0, input_length - 32)
            self.is_prefill_stage = False
        if input_length == 0:
            return 0
        
        for i in self.eligible_exit_layers:
            rmln = input_length
            if self.cached_hidden_states[i]:
                new_hidden_states = []
                for j in range(len(self.cached_hidden_states[i])):
                    if rmln == 0:
                        new_hidden_states.append(self.cached_hidden_states[i][j])
                    elif rmln >= self.cached_hidden_states[i][j].shape[1]:
                        rmln -= self.cached_hidden_states[i][j].shape[1]
                    else:
                        new_hidden_states.append(self.cached_hidden_states[i][j][:, rmln:, :])
                        rmln = 0
                self.cached_hidden_states[i] = new_hidden_states
        return input_length

    def update_acceptance_and_confidence_stats(self, tokens, shadow_confidence_scores, exit_layer, context_length=0):
        """
        Update acceptance counts (α_ℓ) and confidence score statistics (τ_ℓ) 
        using shadow tokens from each candidate draft layer.
        """
        # All but the last row are predictions from candidate draft layers.
        layers_tokens = tokens[:-1, :]
        
        # Last row is the target model’s predictions (full-depth layer).
        correct_tokens = tokens[-1, :].unsqueeze(0).expand_as(layers_tokens)
        
        # Boolean table indicating where layer predictions match the target model
        token_match_table = layers_tokens == correct_tokens

        # Split context tokens if present
        if context_length > 0:
            context_token_match_table = token_match_table[:, :context_length]
            token_match_table = token_match_table[:, context_length:]
            context_shadow_confidence_scores = shadow_confidence_scores[:, :context_length]
            shadow_confidence_scores = shadow_confidence_scores[:, context_length:]
        

        # Clip tokens after the first mismatch at the current exit layer
        if self.current_gamma > 0:
            current_can_layer = exit_layer - 1 - self.eligible_exit_layers[0]
            current_can_layer_false_indices = (token_match_table[current_can_layer, :] == False).nonzero(as_tuple=True)[0]
            if len(current_can_layer_false_indices) > 0:
                token_match_table = token_match_table[:, :current_can_layer_false_indices[0].item()+1]
                shadow_confidence_scores = shadow_confidence_scores[:, :current_can_layer_false_indices[0].item()+1]
        
        # Restore context if it was separated
        if context_length > 0:
            token_match_table = torch.cat((context_token_match_table, token_match_table), dim=1)
            shadow_confidence_scores = torch.cat((context_shadow_confidence_scores, shadow_confidence_scores), dim=1)
        
        # Sum confidence scores over matched and mismatched tokens
        true_shadow_confidence_scores = (shadow_confidence_scores * token_match_table.float()).sum(dim=1)
        false_shadow_confidence_scores = (shadow_confidence_scores * (~token_match_table).float()).sum(dim=1)
        confidence_score_sums_update = torch.stack((true_shadow_confidence_scores, false_shadow_confidence_scores), dim=1)
        
        # Apply exponential moving average update for confidence scores
        self.confidence_sums[self.eligible_exit_layers, :] = (
            self.omega * self.confidence_sums[self.eligible_exit_layers, :] + confidence_score_sums_update.to(self.confidence_sums)
        )
        
        # Count number of matches/mismatches per layer
        true_counts = token_match_table.sum(dim=1)
        false_counts = token_match_table.shape[1]-true_counts
        match_counts_update = torch.stack((true_counts, false_counts), dim=1)
        
        # Update moving average for token matches
        self.token_match_counts[self.eligible_exit_layers, :] = (
            self.omega * self.token_match_counts[self.eligible_exit_layers, :] + match_counts_update.to(self.token_match_counts)
        )
        
        # Derive average confidence per layer over matched/mismatched tokens
        self.average_confidence = self.confidence_sums / self.token_match_counts
    
    def select_optimal_exit_and_gamma(self, current_length):
        """
        Compute TPL(ℓ, γ) for all candidate exit layers and speculation lengths,
        and select the configuration that maximizes it.
        """
        
        # Estimate α_ℓ = accepted / total tokens for each candidate layer
        probs = self.token_match_counts[self.eligible_exit_layers, 0] / torch.sum(self.token_match_counts[self.eligible_exit_layers, :], dim=1)
        probs = torch.where(torch.isnan(probs), torch.tensor(float('-inf')), probs)
        
        # Compute expected number of accepted tokens per speculation length using α_ℓ
        gamma_probs = torch.zeros((len(self.eligible_exit_layers), self.gamma_max))
        for i in range(self.gamma_max):
            if i == 0:
                gamma_probs[:, i] = 1
            else:
                gamma_probs[:, i] = 1 + probs * gamma_probs[:, i-1]
        
        # Cost model: each round costs γ * draft_cost + verification_cost
        draft_cost = torch.tensor(self.eligible_exit_layers) + 1
        verify_cost = len(self.model.model.layers) - draft_cost
        layers_costs = torch.zeros((len(self.eligible_exit_layers), self.gamma_max))
        for i in range(self.gamma_max):
            layers_costs[:, i] = verify_cost + (i+1) * draft_cost
        
        # Compute benefit = expected tokens / total cost = TPL
        tpl = gamma_probs / layers_costs
        
        # Enforce minimum speculation length
        if self.gamma_min > 0:
            tpl[:, :self.gamma_min] = float('-inf')
            
        # Select (layer, γ) pair that maximizes TPL
        max_tpl = torch.max(tpl).item()
        del_layer, del_gamma = divmod(torch.argmax(tpl).item(), tpl.shape[1])
        
        # Log Best TPT
        t_mlp = self.c_1
        t_attn = self.c_2 * current_length + self.c_3
        
        time_per_layer = t_attn + t_mlp
        if time_per_layer > 0:
            best_tpt = max_tpl / time_per_layer
            self.best_tpt_list.append(best_tpt)

        # Convert back to global layer index
        self.current_exit_layer = del_layer + self.eligible_exit_layers[0] + 1
        self.current_gamma = del_gamma

    def run(self, exit_layer, input_length, sample, current_length=None):
        """
        Finalize verification of the current speculative decoding round:
        - Compute shadow logits
        - Derive token match and confidence stats
        - Update optimal exit layer and speculation length (E, γ)
        """
        
        # Compute number of new tokens generated
        output_length = self.cached_hidden_states[-1][0].shape[1] - input_length
        total_length = current_length
        
        # Remove past tokens (prefill or already processed)
        trim_length = self.remove_old_hidden_states(input_length)
        context_length = input_length - trim_length

        # Flatten cached hidden states from all layers
        flattened_hiddens = list(itertools.chain.from_iterable(self.cached_hidden_states))
        if not flattened_hiddens:
            return None, None
        
        # Send all to LM-head (on correct device)
        lm_head_device = self.model.lm_head.weight.device
        flattened_hiddens = [h.to(lm_head_device) for h in flattened_hiddens]
        hiddens = torch.cat(flattened_hiddens, dim=1)
        hiddens = self.model.model.norm(hiddens)
        
        # Compute logits and extract segment of interest
        logits = self.model.lm_head(hiddens)
        return_logits = logits[:, -(input_length+output_length):, :]
        logits = torch.cat(
            (logits[:, 0:-(input_length + output_length), :], logits[:, -(context_length + output_length):, :]),
            dim=1
        )
        
        # Greedy decode shadow tokens and extract their confidence scores
        tokens, shadow_confidence_scores = decode_next_token(logits=logits, sample=False)
        shadow_confidence_scores = shadow_confidence_scores[torch.arange(shadow_confidence_scores.size(0)), tokens.squeeze(0)]
        
        # Only return generated tokens (exclude context) if not sampling
        return_tokens = tokens[:, -output_length:] if not sample else None
        
        # Reshape to [#layers + 1, T] format
        tokens = tokens.view(len(self.eligible_exit_layers)+1, -1)
        shadow_confidence_scores = shadow_confidence_scores.view(len(self.eligible_exit_layers)+1, -1)
        
        # Update statistics (α_ℓ, τ_ℓ) and select next config
        self.update_acceptance_and_confidence_stats(tokens, shadow_confidence_scores[:-1, :], exit_layer, context_length)
        self.select_optimal_exit_and_gamma(current_length=total_length)
        
        # Clear cache for next round
        self.clear_cached_hidden_states()
        
        return return_logits, return_tokens
