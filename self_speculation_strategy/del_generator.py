# del_generator.py
import torch
import time
from typing import List, Optional, Tuple, NamedTuple

import transformers
from self_speculation_strategy.DEL import DEL
from utils import (
    crop_kv_cache,
    decode_next_token,
    forward_early_DEL,
    forward_remainder_DEL,
    GenerationResult,
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

class DELGenerator:
    def __init__(self, env, **kwargs):
        self.env = env
        self.model = env.model
        self.tok = env.tok
        self.device = env.device
        self.args = kwargs
        # DEL Module Definition
        self.coefficients= kwargs.get("coefficients", None)
        self.DEL = DEL(self.model, gamma_max=18, omega=0.95, coefficients=self.coefficients)

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> GenerationResult:
        input_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        input_ids_list = input_ids[0].tolist()
        
        # Configuration
        max_steps = len(input_ids_list) + max_new_tokens
        sample = kwargs.get("sample", False)
        temperature = kwargs.get("temperature", 0.0)
        top_k = kwargs.get("top_k", 50)
        top_p = kwargs.get("top_p", 1.0)
        reference_output_ids = kwargs.get("reference_output_ids", None)

        self.DEL = DEL(self.model, gamma_max=18, omega=0.95, coefficients=self.coefficients)

        eos_token_ids = [self.tok.eos_token_id]

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        total_layers = 0
        total_tokens = 0
        
        output_ids = []
        past_key_values = None
        
        t0 = time.time()

        while len(output_ids) < max_new_tokens:
            num_speculations = min(
                10,
                max_new_tokens - len(output_ids) - 1
            )
            prev_len = len(output_ids)
            
            # Pass teacher_output_ids to single_step_speculation
            current_input_ids, output_ids, past_key_values, number_of_matches, specs = self.single_step_speculation(
                input_ids=input_ids, 
                full_input_ids_list=input_ids_list + output_ids[:len(output_ids)-len(output_ids)], 
                output_ids=output_ids,
                num_speculations=num_speculations,
                past_key_values=past_key_values,
                eos_token_ids=eos_token_ids,
                calls=calls,
                exit_layer=self.DEL.current_exit_layer,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                reference_output_ids=reference_output_ids
            )
            
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += specs
            total_tokens += (number_of_matches+1)
            total_layers += self.DEL.current_exit_layer * specs + len(self.model.model.layers)
            
            # Check progress
            if len(output_ids) == prev_len:
                break
                
            input_ids = current_input_ids
            
            eos_found = False
            for eos_token_id in eos_token_ids:
                if eos_token_id in output_ids:
                    # Truncate
                    idx = output_ids.index(eos_token_id)
                    output_ids = output_ids[:idx]
                    eos_found = True
                    break
            if eos_found:
                break
        
        total_time = time.time() - t0
        text = self.tok.decode(output_ids, skip_special_tokens=True)

        print("exit layer: ", self.DEL.current_exit_layer)
        
        avg_best_tpt = sum(self.DEL.best_tpt_list) / len(self.DEL.best_tpt_list) if self.DEL.best_tpt_list else None

        return GenerationResult(
            text=text,
            num_output_tokens=len(output_ids),
            output_ids=output_ids,
            acceptance_rate=total_draft_matches / total_generations if total_generations > 0 else 0,
            tokens_per_layer=total_tokens / total_layers if total_layers > 0 else 0,
            total_time=total_time,
            avg_best_tpt=avg_best_tpt,
        )

    def single_step_speculation(
        self,
        input_ids: torch.Tensor,
        full_input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_ids: List[int],
        calls: int,
        exit_layer: int,
        sample: bool = False,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        reference_output_ids: Optional[List[int]] = None,
    ):
        # input_ids: [1, seq_len] (either prompt or last accepted)
        prompt_length = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids = []
        
        draft_probabilities = [] if sample else None
        exit_query_cache = None

        d_max = 1 if self.DEL.is_prefill_stage else num_speculations
        
        for di in range(d_max):
            draft_logits, past_key_values, exit_query_cache = forward_early_DEL(
                self.model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
                self.DEL,
            )

            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            draft_next_token_item = draft_next_token.item()
            draft_output_ids.append(draft_next_token_item)
            
            if sample:
                draft_probabilities.append(draft_next_prob)
                
            draft_input_ids = torch.tensor([[draft_next_token_item]]).to(draft_input_ids)
            
            if draft_next_token_item in eos_token_ids:
                break

            # DEL Confidence check
            draft_token_confidence_score = draft_next_prob[0, draft_next_token_item].item()
            avg_confidence_stats = self.DEL.average_confidence[exit_layer - 1]
            tau_threshold_for_exit_layer = ((avg_confidence_stats[0] + avg_confidence_stats[1]) / 2).item()

            if (not self.DEL.is_prefill_stage) and draft_token_confidence_score < tau_threshold_for_exit_layer:
                break
                
        # Concat inputs for verification
        if len(draft_output_ids) > 0:
            draft_output_ids_tensor = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
            prefill_token_ids = torch.cat([input_ids, draft_output_ids_tensor], dim=-1)
        else:
             draft_output_ids_tensor = torch.tensor([[]], dtype=torch.long).to(input_ids)
             prefill_token_ids = input_ids

        # Verify
        _, past_key_values, exit_query_cache = forward_remainder_DEL(
            self.model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer,
            exit_query_cache,
            self.DEL,
        )
        
        current_length = past_key_values[0][0].shape[2]
        logits, del_tokens = self.DEL.run(exit_layer, prompt_length-1, sample, current_length=current_length)
        
        verification_logits = logits[:, prompt_length - 1 :, :]
        
        
        if reference_output_ids is not None:
             # Force verified tokens to match reference (teacher)
            current_idx = len(output_ids)
            verified_tokens_list = []
            
            check_len = len(draft_output_ids) + 1
            for i in range(check_len):
                if current_idx + i < len(reference_output_ids):
                    verified_tokens_list.append(reference_output_ids[current_idx + i])
                else:
                    break
            
            verified_tokens = torch.tensor([verified_tokens_list], device=prefill_token_ids.device)
            verified_probabilities = None # We don't use probabilities when teacher forcing
        else:
            if sample:
                verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            else:
                verified_tokens = del_tokens
                verified_probabilities = None

        verified_tokens = verified_tokens.to(prefill_token_ids)
        
        if len(draft_output_ids) > 0:
            # Robust comparison logic from claspd
            if verified_tokens.shape[1] > draft_output_ids_tensor.shape[1]:
                verified_comparison = verified_tokens[:, :-1]
                compare_draft = draft_output_ids_tensor
            else:
                min_len = min(verified_tokens.shape[1], draft_output_ids_tensor.shape[1])
                verified_comparison = verified_tokens[:, :min_len]
                compare_draft = draft_output_ids_tensor[:, :min_len]

            verified = compare_draft == verified_comparison

            if not sample:
                number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
            else:
                number_of_matches = 0
                rand = torch.rand_like(compare_draft, dtype=torch.float)
                for i in range(compare_draft.numel()):
                    val = verified_probabilities[i, compare_draft[0, i]].item()
                    d_val = draft_probabilities[i][0, compare_draft[0, i]].item()
                    if rand[0, i] < min(1, val / d_val):
                        number_of_matches += 1
                    else:
                        # Resample
                        diff = verified_probabilities[i, :] - draft_probabilities[i]
                        verified_tokens[0][number_of_matches] = torch.multinomial(max_fn(diff), num_samples=1).item()
                        break
        else:
            number_of_matches = 0
            
        # Extend output_ids with matched draft tokens
        if len(draft_output_ids) > 0:
            output_ids.extend(draft_output_ids_tensor[0, : number_of_matches].tolist())

        # Prepare next input_ids
        if number_of_matches < verified_tokens.shape[1]:
            # Has bonus token
            bonus_token = verified_tokens[:, number_of_matches : number_of_matches + 1]
            output_ids.extend(bonus_token[0].tolist())
            next_input_ids = bonus_token
        else:
            # No bonus token, use last accepted token as next input
            if len(output_ids) > 0:
                next_input_ids = torch.tensor([[output_ids[-1]]], device=input_ids.device)
            else:
                # Fallback (should be rare/impossible if prompt exists)
                next_input_ids = input_ids

        # Crop Cache
        past_key_values = crop_kv_cache(
            past_key_values, len(full_input_ids_list) + len(output_ids) - 1
        )
        
        return (
            next_input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            len(draft_output_ids),
        )

