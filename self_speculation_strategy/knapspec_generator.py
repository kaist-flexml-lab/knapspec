# knapspec_generator.py
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from utils import (
    Env,
    forward,
    forward_draft_divided_multi,
    forward_verify_divided_multi,
    crop_kv_cache,
    decode_next_token,
    GenerationResult,
)
from .knapspec import Knapspec

class KnapspecGenerator:
    def __init__(
        self,
        env: Env,
        gamma: int = 4,
        skip_budget_M: int = 8,
        optimize_interval: int = 64,
        coefficients: Optional[tuple] = None,
        sim_threshold: float = 0.5,
    ):
        self.env = env
        self.model = env.model
        self.gamma = int(gamma)
        self.optimize_interval = int(optimize_interval)
        self.coefficients = coefficients
        self.sim_threshold = float(sim_threshold)

        self.L = len(self.model.model.layers)
        self.M = int(skip_budget_M)
        self.step_count = 0
        self.threshold = 0.7
        
        # Timing statistics
        self.total_draft_time = 0.0
        self.total_verify_time = 0.0
        self.total_optimization_time = 0.0
        self.total_accepted_length = 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample: bool = False,
        reference_output_ids: Optional[List[int]] = None,
    ) -> GenerationResult:
        self.knapspec_model = Knapspec(L=self.L, M=int(self.M), model=self.model, coefficients=self.coefficients, sim_threshold=self.sim_threshold)
        
        enc = self.env.tok(prompt, return_tensors="pt")
        input_ids_list = enc["input_ids"][0].tolist()
        input_ids = torch.tensor([input_ids_list], device=self.env.model.device)

        eos_token_ids: List[int] = []
        if self.env.eos_id is not None:
            eos_token_ids.append(self.env.eos_id)

        self.step_count = 0
        self.total_accepted_length = 0
        output_ids: List[int] = []
        past_key_values = None

        total_accepted = 0
        total_drafted = 0
        total_tokens = 0
        total_layers = 0
        
        # Reset timing statistics
        self.total_draft_time = 0.0
        self.total_verify_time = 0.0
        self.total_optimization_time = 0.0

        total_start_time = time.perf_counter()
        while len(output_ids) < max_new_tokens:
            num_speculations = min(self.gamma, max_new_tokens - len(output_ids) - 1)
            prev_len = len(output_ids)
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_drafted,
            ) = self.single_step_speculation(
                input_ids=input_ids,
                input_ids_list=input_ids_list,
                output_ids=output_ids,
                num_speculations=num_speculations,
                past_key_values=past_key_values,
                eos_token_ids=eos_token_ids,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                reference_output_ids=reference_output_ids,
            )

            # Check for zero progress
            if len(output_ids) == prev_len:
                break

            total_accepted += number_of_matches
            total_drafted += num_drafted
            total_tokens += (number_of_matches + 1)  # accepted + correction/bonus
            skip_count = sum(self.knapspec_model.skip_set)
            total_layers += (2 * self.L - skip_count) * num_drafted + 2 * self.L
            self.step_count += 1

            # Stop if EOS appeared in committed output
            stop = False
            for eid in eos_token_ids:
                if eid in output_ids:
                    output_ids = output_ids[: output_ids.index(eid)]
                    stop = True
                    break
            if stop:
                break

        total_end_time = time.perf_counter()
        total_time = total_end_time - total_start_time

        text = self.env.tok.decode(output_ids, skip_special_tokens=True) if output_ids else ""
        acc_rate = (total_accepted / total_drafted) if total_drafted > 0 else None
        tpl = (total_tokens / total_layers) if total_layers > 0 else None
        avg_best_tpt = self.knapspec_model.sum_best_tpt / self.knapspec_model.optimize_count if self.knapspec_model.optimize_count > 0 else None
        if self.knapspec_model.optimize_count > 0:
            print("Average Skip_num", self.knapspec_model.total_skip / self.knapspec_model.optimize_count)

        # Print timing breakdown
        if self.total_draft_time + self.total_verify_time + self.total_optimization_time > 0:
            total_measured = self.total_draft_time + self.total_verify_time + self.total_optimization_time
            draft_pct = (self.total_draft_time / total_measured) * 100
            verify_pct = (self.total_verify_time / total_measured) * 100
            opt_pct = (self.total_optimization_time / total_measured) * 100
            
            print(f"[TIMING] Draft: {self.total_draft_time:.3f}s ({draft_pct:.1f}%)")
            print(f"[TIMING] Verify: {self.total_verify_time:.3f}s ({verify_pct:.1f}%)")
            print(f"[TIMING] Optimization: {self.total_optimization_time:.3f}s ({opt_pct:.1f}%)")
            print(f"[TIMING] D+V+O measured: {total_measured:.3f}s / Total: {total_time:.3f}s")

        return GenerationResult(
            text=text,
            num_output_tokens=len(output_ids),
            output_ids=output_ids,
            acceptance_rate=acc_rate,
            tokens_per_layer=tpl,
            draft_time=self.total_draft_time,
            verify_time=self.total_verify_time,
            optimization_time=self.total_optimization_time,
            total_time=total_time,
            total_accepted_length=self.total_accepted_length,
            total_steps=self.step_count,
            avg_best_tpt=avg_best_tpt
        )

    def single_step_speculation(
        self,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_ids: List[int],
        sample: bool = False,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        reference_output_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[int], Optional[List[Tuple[torch.Tensor, torch.Tensor]]], int, int]:
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        draft_cache = past_key_values

        if self.step_count == 0:
            logits, past_key_values = forward(
                self.model,
                input_ids,
                past_key_values,
            )
            next_token, probabilities = decode_next_token(
                logits,
                token_idx = -1,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_token_id = int(next_token.item() if isinstance(next_token, torch.Tensor) else next_token)
            
            output_ids.append(next_token_id)
            input_ids = torch.tensor([[next_token_id]], device=input_ids.device)
            self.knapspec_model.is_prefill_stage = False
            return (
                input_ids,
                output_ids,
                past_key_values,
                0,
                0,
            )

        # accumulate_steps = 10
        accumulate_steps = 5
        accumulate_phase = (1 < (self.step_count % self.optimize_interval) <= (1 + accumulate_steps))
        optimization_phase = ((self.step_count % self.optimize_interval) == (1 + accumulate_steps))

        # Draft phase
        draft_start = time.perf_counter()
        for i in range(num_speculations):
            next_logits, draft_cache = forward_draft_divided_multi(
                self.model,
                draft_input_ids,
                self.knapspec_model.skip_set,
                draft_cache,
            )
            next_token, next_prob = decode_next_token(
                next_logits,
                token_idx = -1,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_id = int(next_token.item() if isinstance(next_token, torch.Tensor) else next_token)
            draft_output_ids.append(next_id)
            draft_input_ids = torch.tensor([[next_id]], device=draft_input_ids.device)
            if next_id in eos_token_ids:
                break

            draft_token_confidence_score = next_prob[0, next_token].item()
            if (not self.knapspec_model.is_prefill_stage) and draft_token_confidence_score < self.threshold:
                break

        draft_end = time.perf_counter()
        self.total_draft_time += (draft_end - draft_start)

        # Verify phase
        verify_start = time.perf_counter()
        
        # Prepare tokens for verification
        draft_tensor = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        prefill_token_ids = torch.cat([input_ids, draft_tensor], dim=-1)
        
        # Run verification forward pass (always needed for state updates)
        verification_logits, past_key_values = forward_verify_divided_multi(
            self.model, prefill_token_ids, past_key_values, self.knapspec_model, accumulate_phase
        )

        # Determine verified tokens
        if reference_output_ids is not None:
             # Use reference to get ground truth tokens
            current_idx = len(output_ids)
            verified_tokens_list = []
            
            check_len = len(draft_output_ids) + 1
            for i in range(check_len):
                if current_idx + i < len(reference_output_ids):
                    verified_tokens_list.append(reference_output_ids[current_idx + i])
                else:
                    break
            
            verified_tokens = torch.tensor([verified_tokens_list], device=prefill_token_ids.device)

        else:
            # Standard decoding from logits
            # Get logits for the drafted positions + one extra token
            prompt_length = input_ids.shape[1]  # Original input length
            verification_logits = verification_logits[:, prompt_length - 1:, :]  # [1, T_d + 1, V]

            # Decode verified tokens from the verification logits
            verified_tokens, verified_probabilities = decode_next_token(
                logits=verification_logits,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        verified_tokens = verified_tokens.to(input_ids.device)

        # Compare draft vs verified
        draft_tensor_1d = torch.tensor(draft_output_ids, device=self.model.device).unsqueeze(0)  # [1, T_d]
        
        # Handle potential shape mismatch (e.g. at the end of teacher forcing sequence)
        if verified_tokens.shape[1] > draft_tensor_1d.shape[1]:
            verified_comparison = verified_tokens[:, :-1]
        else:
            min_len = min(verified_tokens.shape[1], draft_tensor_1d.shape[1])
            verified_comparison = verified_tokens[:, :min_len]
            draft_tensor_1d = draft_tensor_1d[:, :min_len]
        
        verified = draft_tensor_1d == verified_comparison

        # Count number of matches (consecutive from the beginning)
        if not sample:
            number_of_matches = ((~verified).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            for i in range(draft_tensor_1d.numel()):
                if bool(verified[0, i].item()):
                    number_of_matches += 1
                else:
                    break

        accepted_tokens = draft_output_ids[:number_of_matches]  # Matched draft tokens

        # Get the additional token (next token after matches) if available
        if number_of_matches < verified_tokens.shape[1]:
            additional_token = int(verified_tokens[0, number_of_matches].item())
            accepted_tokens.append(additional_token)

        if not accepted_tokens:
             return (
                 input_ids,
                 output_ids,
                 past_key_values,
                 number_of_matches,
                 len(draft_output_ids),
            )

        # State update
        new_token_id = accepted_tokens[-1]
        input_ids = torch.tensor([[new_token_id]], device=input_ids.device)
        output_ids.extend(accepted_tokens)

        # Crop cache to match the verified length
        new_len = len(input_ids_list) + len(output_ids) - 1
        past_key_values = crop_kv_cache(past_key_values, new_len)
        
        verify_end = time.perf_counter()
        self.total_verify_time += (verify_end - verify_start)
        self.total_accepted_length += len(accepted_tokens)


        if accumulate_phase:
            for i in range(2*self.L+1):
                hidden = self.knapspec_model.cached_hidden_states[i][-1]
                cropped = hidden[:, :number_of_matches + 1, :].clone()
                self.knapspec_model.cached_hidden_states[i][-1] = cropped

        # Optimization phase with timing
        if optimization_phase:
            opt_start = time.perf_counter()
            self.knapspec_model.optimize(past_key_values=past_key_values)
            opt_end = time.perf_counter()
            self.total_optimization_time += (opt_end - opt_start)

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            len(draft_output_ids),
        )
