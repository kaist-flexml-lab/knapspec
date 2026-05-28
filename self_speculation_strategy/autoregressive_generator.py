# autoregressive_generator.py
from dataclasses import dataclass
from typing import List, Optional
import torch
import time

from utils import (
    Env,
    forward,
    forward_divided,
    decode_next_token,
    GenerationResult
)

class AutoregressiveGenerator:
    def __init__(self, env: Env, coefficients: Optional[tuple] = None):
        self.env = env
        self.model = env.model
        self.coefficients = coefficients
        if coefficients:
             self.c_1, self.c_2, self.c_3 = coefficients
        else:
             self.c_1 = 0
             self.c_2 = 0
             self.c_3 = 0
        
        self.best_tpt_list = []

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
        """Generate text autoregressively."""
        # Tokenize prompt to get input_ids
        enc = self.env.tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"][0].tolist()  # Convert to List[int] like Meta
        initial_len = len(input_ids)
        
        # EOS token IDs
        eos_token_ids = []
        if self.env.eos_id is not None:
            eos_token_ids.append(self.env.eos_id)
        
        # Main generation loop (like Meta's generate_token_ids)
        past_key_values = None
        input_ids_tensor = torch.tensor([input_ids]).to(self.env.model.device)
        output_ids: List[int] = []
        
        self.best_tpt_list = []
        L = len(self.model.model.layers)

        for step in range(max_new_tokens):
            logits, past_key_values = forward(
                self.model,
                input_ids_tensor,
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
            
            # Convert to int if tensor
            next_token_id = next_token.item()
                
            # Check for EOS
            if next_token_id in eos_token_ids:
                break
            
            output_ids.append(next_token_id)
            
            # TPT Logging every 50 tokens
            if self.coefficients and (step + 1) % 50 == 0:
                current_length = initial_len + len(output_ids)
                t_mlp = self.c_1
                t_attn = self.c_2 * current_length + self.c_3
                
                cost = L * (t_attn + t_mlp)
                if cost > 0:
                    tpt = 1.0 / cost
                    self.best_tpt_list.append(tpt)

            
            # Update input_ids for next iteration (single token)
            input_ids_tensor = torch.tensor([[next_token_id]]).to(self.env.model.device)
            
            if step % 1000 == 0:
                print(f"\r[AR] Generated {step} tokens...", end="", flush=True)
        
        # Decode generated tokens
        generated_text = self.env.tok.decode(output_ids, skip_special_tokens=True) if output_ids else ""
        
        avg_best_tpt = sum(self.best_tpt_list) / len(self.best_tpt_list) if self.best_tpt_list else None

        return GenerationResult(
            text=generated_text,
            num_output_tokens=len(output_ids),
            output_ids=output_ids,
            avg_best_tpt=avg_best_tpt,
        )
