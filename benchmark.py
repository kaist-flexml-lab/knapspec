# benchmark.py
import random
import argparse
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from utils import Env
from self_speculation_strategy.autoregressive_generator import AutoregressiveGenerator
from self_speculation_strategy.clasp_generator import ClaspGenerator
from self_speculation_strategy.knapspec_generator import KnapspecGenerator
from self_speculation_strategy.del_generator import DELGenerator

from profile_modules import profile_model

from data import get_dataset, DatasetFormat, get_dataset_info, EvaluationExample, get_valid_dataset_formats

# ------------------------
# Records
# ------------------------

@dataclass
class SampleRecord:
    idx: int
    sample_id: str
    prompt: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    elapsed_sec: float
    tokens_per_sec: float
    text: str
    # Add optional fields for speculative decoding metrics
    acceptance_rate: float = None
    tokens_per_layer: float = None
    # Timing metrics
    draft_time: float = None
    verify_time: float = None
    optimization_time: float = None
    total_time: float = None
    total_accepted_length: int = None
    total_steps: int = None
    # Optimization avg metrics
    avg_best_tpt: float = None
    avg_skip: float = None
    avg_attn_skip: float = None
    avg_mlp_skip: float = None


# ------------------------
# Main
# ------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF repo_id or local path")
    ap.add_argument("--strategy", type=str, choices=["autoregressive", "clasp", "knapspec", "del"], default="autoregressive")
    ap.add_argument("--compare-strategies", action="store_true", help="Run comparison: AR first, then others using AR as reference")
    ap.add_argument("--dataset", type=str, required=True, 
                   choices=get_valid_dataset_formats(),
                   help="Dataset format to use")
    ap.add_argument("--data-path", type=str, help="Path to custom dataset file (for custom_jsonl / spec_bench))")
    ap.add_argument("--n-shot", type=int, default=0, help="Number of few-shot examples")
    ap.add_argument("--template", type=str, help="Template to apply to prompts")
    ap.add_argument("--num-samples", type=int, default=None, help="Number of samples to run (default: 100 if end-idx not set)")
    ap.add_argument("--start-idx", type=int, default=0, help="Start index for dataset slicing")
    ap.add_argument("--end-idx", type=int, default=None, help="End index for dataset slicing (exclusive)")
    ap.add_argument("--max-length", type=int, default=1024, help="max_new_tokens for generation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="bench_out")
    
    # Generation parameters
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    ap.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    ap.add_argument("--sample", action="store_true", help="Use sampling instead of greedy")
    
    # CLASP specific parameters
    ap.add_argument("--gamma", type=int, default=4, help="Draft length (gamma) for CLASP/SWIFT")
    ap.add_argument("--skip-budget", type=int, default=8, help="Skip budget (M layers) for CLASP/SWIFT")
    ap.add_argument("--optimize-interval", type=int, default=64, help="Optimization interval")
    ap.add_argument("--enable-thinking", action="store_true", help="Enable thinking process in chat template (e.g. for Qwen3)")
    
    ap.add_argument("--sim-threshold", type=float, default=0.5, help="Cosine similarity threshold for pruning in Claspdmodified")
    
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Simple device/dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if device == "cuda" else None
    # if device == "cuda":
    #     dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # else:
    #     dtype = torch.float32
    dtype = torch.float16

    print(f"[ENV] device={device}, device_map={device_map}, dtype={dtype}")
    print(f"[CFG] strategy={args.strategy}, start_idx={args.start_idx}, end_idx={args.end_idx}, num_samples={args.num_samples}")
    print(f"[GEN] temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, sample={args.sample}")

    # GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load model/tokenizer
    print(f"[MODEL] id='{args.model}'")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    ).eval()

    # Check memory status after model loading
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU] Memory allocated: {allocated:.2f} GB")
        print(f"[GPU] Memory reserved: {reserved:.2f} GB")

    # Dataset loading
    print(f"[DATA] Loading {args.dataset} dataset...")
    dataset_info = get_dataset_info(args.dataset)
    print(f"[DATA] {dataset_info['description']} - {dataset_info['task_type']}")
    
    evaluation_examples = get_dataset(
        dataset_format=args.dataset,
        num_samples=None,  # Load all available first, then slice
        random_shuffle=False,
        seed=args.seed,
        data_path=args.data_path,
        n_shot=args.n_shot,
        template=args.template
    )

    # ------------------------
    # Slicing Logic
    # ------------------------
    start_idx = args.start_idx
    if args.end_idx is not None:
        end_idx = args.end_idx
    elif args.num_samples is not None:
        end_idx = start_idx + args.num_samples
    else:
        end_idx = len(evaluation_examples)

    # Clamp end_idx
    end_idx = min(end_idx, len(evaluation_examples))
    
    if start_idx >= len(evaluation_examples):
        print(f"[WARN] start_idx {start_idx} is out of bounds for dataset size {len(evaluation_examples)}. Returning empty.")
        evaluation_examples = []
    else:
        evaluation_examples = evaluation_examples[start_idx:end_idx]
        print(f"[DATA] Sliced to {len(evaluation_examples)} examples (original indices {start_idx} to {end_idx})")
    
    print(f"[DATA] Loaded {len(evaluation_examples)} examples")
    
    # Env and generator
    env = Env(
        model=model,
        tok=tok,
        device=device,
        eos_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
    )

    # --- Profiling ---
    coefficients = None
    # Check if main strategy or any compared strategy needs coefficients
    strategies_using_coeffs = ["autoregressive", "clasp", "knapspec", "del"] 
    needs_profiling = args.strategy in strategies_using_coeffs
    
    # If comparison mode, check if any of the hardcoded comparison strategies need profiling.
    if args.compare_strategies:
         # Hardcoded list in benchmark.py: ["autoregressive", "del", "clasp", "knapspec"]
         # All of them need coefficients except maybe older clasp versions, but let's be safe.
         needs_profiling = True


    if needs_profiling:
        print("Running profiling to determine C1, C2, C3...")
        c1, c2, c3 = profile_model(env.model)
        args.coefficients = (c1, c2, c3)
        print(f"Profiling complete: C1={c1}, C2={c2}, C3={c3}")


    # Strategy selection
    if args.compare_strategies:
        strategies = ["autoregressive", "del", "clasp", "knapspec"]
        # strategies = ["autoregressive", "knapspec"]
        print(f"[CFG] Running comparison mode with strategies: {strategies}")
    else:
        strategies = [args.strategy]

    # Store reference tokens: sample_idx -> output_ids
    reference_outputs_map = {}

    model_tag = args.model.replace("/", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")

    for current_strategy in strategies:
        print(f"\n{'='*40}")
        print(f"Running Strategy: {current_strategy}")
        print(f"{'='*40}\n")
        
        # Instantiate generator for current strategy
        if current_strategy == "autoregressive":
            generator = AutoregressiveGenerator(env, coefficients=args.coefficients)
        elif current_strategy == "clasp":
            generator = ClaspGenerator(
                env, 
                gamma=args.gamma, 
                skip_budget_M=args.skip_budget,
                optimize_interval=args.optimize_interval,
                coefficients=args.coefficients,
            )
        elif current_strategy == "knapspec":
            generator = KnapspecGenerator(
                env, 
                gamma=args.gamma, 
                skip_budget_M=args.skip_budget,
                optimize_interval=args.optimize_interval,
                coefficients=args.coefficients,
                sim_threshold=args.sim_threshold
            )
        elif current_strategy == "del":
            generator = DELGenerator(env, **vars(args))
        else:
            raise NotImplementedError(f"'{current_strategy}' strategy is not implemented yet.")

        records: List[SampleRecord] = []
        
        # Run samples
        for i, example in enumerate(evaluation_examples):
            # Clear memory before each sample
            if device == "cuda":
                torch.cuda.empty_cache()
                
            prompt = example.input

            # Apply chat template if enable_thinking is requested
            if args.enable_thinking:
                messages = [{"role": "user", "content": prompt}]
                # Assume tokenizer supports enable_thinking or is patched
                prompt = tok.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    enable_thinking=True
                )
            
            # Count prompt tokens
            enc = tok(prompt, return_tensors="pt")
            prompt_len = enc["input_ids"].shape[1]

            print(f"[{current_strategy}] SAMPLE {i + 1}/{len(evaluation_examples)}")
            
            # Use reference tokens if available
            reference_ids = reference_outputs_map.get(i) if args.compare_strategies else None
            
            # No reference for AR itself
            if current_strategy == "autoregressive":
                reference_ids = None

            t0 = time.perf_counter()
            with torch.no_grad():
                # Pass reference_output_ids only if generator supports it (all detailed above do)
                # But need to check if kwargs supported or updated generator method
                # We updated AR, Clasp, Claspdparallel, Claspdmodified
                # Others might fail if we pass kwarg.
                # Assuming user only runs supported ones in comparison.
                
                if current_strategy in ["autoregressive", "del", "clasp", "knapspec"]:
                    res = generator.generate(
                        prompt=prompt,
                        max_new_tokens=args.max_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        sample=args.sample,
                        reference_output_ids=reference_ids
                    )
                else:
                    # Fallback for others not yet updated
                    res = generator.generate(
                        prompt=prompt,
                        max_new_tokens=args.max_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        sample=args.sample,
                    )
            dt = time.perf_counter() - t0
            
            # Cache AR outputs for subsequent strategies
            if current_strategy == "autoregressive" and args.compare_strategies:
                reference_outputs_map[i] = res.output_ids

            out_tokens = int(res.num_output_tokens)
            total_tokens = int(prompt_len + out_tokens)
            tps = (out_tokens / dt) if dt > 0 else 0.0

            # Sample ID
            rid = str(example.get("id", i)) if isinstance(example, dict) else str(i)

            # Extract additional metrics
            acceptance_rate = getattr(res, 'acceptance_rate', None)
            tokens_per_layer = getattr(res, 'tokens_per_layer', None)
            
            # Extract timing metrics
            draft_time = getattr(res, 'draft_time', None)
            verify_time = getattr(res, 'verify_time', None)
            optimization_time = getattr(res, 'optimization_time', None)
            total_time = getattr(res, 'total_time', None)
            total_accepted_length = getattr(res, 'total_accepted_length', None)
            total_steps = getattr(res, 'total_steps', None)
            avg_best_tpt = getattr(res, 'avg_best_tpt', None)

            # Stats Retrieval if available
            avg_skip = None
            avg_attn_skip = None
            avg_mlp_skip = None
            
            # Check if generator has the stats (CLASP variants)
            # We access the internal strategy object if present (e.g. generator.claspdparallel)
            if hasattr(generator, "knapspec_model") and hasattr(generator.knapspec_model, "optimize_count") and generator.knapspec_model.optimize_count > 0:
                opt = generator.knapspec_model
                avg_skip = opt.sum_skip / opt.optimize_count
                avg_attn_skip = opt.sum_attn_skip / opt.optimize_count
                avg_mlp_skip = opt.sum_mlp_skip / opt.optimize_count

            records.append(
                SampleRecord(
                    idx=i,
                    sample_id=rid,
                    prompt=prompt,
                    prompt_tokens=int(prompt_len),
                    output_tokens=out_tokens,
                    total_tokens=total_tokens,
                    elapsed_sec=float(dt),
                    tokens_per_sec=float(tps),
                    text=res.text,
                    acceptance_rate=float(acceptance_rate) if acceptance_rate is not None else None,
                    tokens_per_layer=float(tokens_per_layer) if tokens_per_layer is not None else None,
                    draft_time=float(draft_time) if draft_time is not None else None,
                    verify_time=float(verify_time) if verify_time is not None else None,
                    optimization_time=float(optimization_time) if optimization_time is not None else None,
                    total_time=float(total_time) if total_time is not None else None,
                    total_accepted_length=float(total_accepted_length) if total_accepted_length is not None else None,
                    total_steps=float(total_steps) if total_steps is not None else None,
                    avg_best_tpt=avg_best_tpt,
                    avg_skip=avg_skip,
                    avg_attn_skip=avg_attn_skip,
                    avg_mlp_skip=avg_mlp_skip,
                )
            )

            # Progress output
            if acceptance_rate is not None:
                print(f"[METRICS] {out_tokens} tokens, {tps:.1f} tok/s, acc_rate: {acceptance_rate:.3f}")
                if draft_time is not None and verify_time is not None and optimization_time is not None:
                    total_measured = draft_time + verify_time + optimization_time
                    if total_measured > 0:
                        draft_pct = (draft_time / total_measured) * 100
                        verify_pct = (verify_time / total_measured) * 100
                        opt_pct = (optimization_time / total_measured) * 100
                        print(f"[TIMING] D:{draft_pct:.1f}% V:{verify_pct:.1f}% O:{opt_pct:.1f}%")
            else:
                print(f"[METRICS] {out_tokens} tokens, {tps:.1f} tok/s")
        
        # Save results for this strategy
        out_path = Path(args.out_dir) / f"results_{ts}_{current_strategy}_{model_tag}_{args.dataset}_{start_idx}.json"
        
        # Summary calculation for this strategy
        n = max(len(records), 1)
        avg_tps = sum(r.tokens_per_sec for r in records) / n
        
        total_out_tokens = sum(r.output_tokens for r in records)
        total_elapsed = sum(r.elapsed_sec for r in records)
        avg_tps_micro = total_out_tokens / max(total_elapsed, 1e-12)
        
        avg_lat = sum(r.elapsed_sec for r in records) / n
        avg_out = sum(r.output_tokens for r in records) / n
        
        acceptance_rates = [r.acceptance_rate for r in records if r.acceptance_rate is not None]
        avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else None
        
        tokens_per_layers = [r.tokens_per_layer for r in records if r.tokens_per_layer is not None]
        avg_tokens_per_layer = sum(tokens_per_layers) / len(tokens_per_layers) if tokens_per_layers else None
        
        draft_times = [r.draft_time for r in records if r.draft_time is not None]
        verify_times = [r.verify_time for r in records if r.verify_time is not None]
        opt_times = [r.optimization_time for r in records if r.optimization_time is not None]
        
        avg_draft_time = sum(draft_times) / len(draft_times) if draft_times else None
        avg_verify_time = sum(verify_times) / len(verify_times) if verify_times else None
        avg_opt_time = sum(opt_times) / len(opt_times) if opt_times else None
        
        total_accepted_length_list = [r.total_accepted_length for r in records if r.total_accepted_length is not None]
        total_steps_list = [r.total_steps for r in records if r.total_steps is not None]
        avg_accepted_length = sum(total_accepted_length_list) / sum(total_steps_list) if total_steps_list else None

        payload: Dict[str, object] = {
            "meta": {
                "model": args.model,
                "strategy": current_strategy,
                "dataset": args.dataset,
                "num_samples": len(records),
                "max_new_tokens": args.max_length,
                "seed": args.seed,
                "dtype": str(dtype),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "sample": args.sample,
                "compare_strategies_mode": args.compare_strategies,
            },
            "summary": {
                "avg_tokens_per_sec": avg_tps,
                "avg_tokens_per_sec_micro": avg_tps_micro,
                "avg_latency_sec": avg_lat,
                "avg_output_tokens": avg_out,
            },
            "samples": [
                {
                    "idx": r.idx,
                    "sample_id": r.sample_id,
                    "prompt": r.prompt,
                    "prompt_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "total_tokens": r.total_tokens,
                    "elapsed_sec": r.elapsed_sec,
                    "tokens_per_sec": r.tokens_per_sec,
                    "text": r.text,
                    "acceptance_rate": r.acceptance_rate,
                    "tokens_per_layer": r.tokens_per_layer,
                    "draft_time": r.draft_time,
                    "verify_time": r.verify_time,
                    "optimization_time": r.optimization_time,
                    "total_time": r.total_time,
                    "avg_best_tpt": r.avg_best_tpt,
                    "avg_skip": r.avg_skip,
                    "avg_attn_skip": r.avg_attn_skip,
                    "avg_mlp_skip": r.avg_mlp_skip,
                }
                for r in records
            ],
        }

        if avg_acceptance_rate is not None:
            payload["summary"]["avg_acceptance_rate"] = avg_acceptance_rate
        if avg_tokens_per_layer is not None:
            payload["summary"]["avg_tokens_per_layer"] = avg_tokens_per_layer
        
        if avg_draft_time is not None:
            payload["summary"]["avg_draft_time"] = avg_draft_time
        if avg_verify_time is not None:
            payload["summary"]["avg_verify_time"] = avg_verify_time
        if avg_opt_time is not None:
            payload["summary"]["avg_optimization_time"] = avg_opt_time
        
        # Calculate overall averages for new metrics
        avg_best_tpt_list = [r.avg_best_tpt for r in records if r.avg_best_tpt is not None]
        avg_skip_list = [r.avg_skip for r in records if r.avg_skip is not None]
        avg_attn_skip_list = [r.avg_attn_skip for r in records if r.avg_attn_skip is not None]
        avg_mlp_skip_list = [r.avg_mlp_skip for r in records if r.avg_mlp_skip is not None]
        
        if avg_best_tpt_list:
            payload["summary"]["avg_best_tpt_overall"] = sum(avg_best_tpt_list) / len(avg_best_tpt_list)
        if avg_skip_list:
            payload["summary"]["avg_skip_overall"] = sum(avg_skip_list) / len(avg_skip_list)
        if avg_attn_skip_list:
            payload["summary"]["avg_attn_skip_overall"] = sum(avg_attn_skip_list) / len(avg_attn_skip_list)
        if avg_mlp_skip_list:
            payload["summary"]["avg_mlp_skip_overall"] = sum(avg_mlp_skip_list) / len(avg_mlp_skip_list)


        if avg_accepted_length is not None:
            payload["summary"]["avg_accepted_length"] = avg_accepted_length

        if current_strategy in ["clasp", "knapspec"]:
            payload["meta"]["gamma"] = args.gamma
            payload["meta"]["skip_budget"] = args.skip_budget
            payload["meta"]["optimize_interval"] = args.optimize_interval

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[SAVE] {out_path}")
        
        summary_msg = (
            f"[{current_strategy} SUMMARY] {avg_tps:.1f} tok/s avg, "
            f"{avg_tps_micro:.1f} tok/s avg(micro), "
            f"{avg_out:.1f} tokens avg"
        )
        if avg_acceptance_rate is not None:
            summary_msg += f", {avg_acceptance_rate:.3f} acc rate"
        if avg_tokens_per_layer is not None:
            summary_msg += f", {avg_tokens_per_layer:.4f} tpl"
        print(summary_msg)

        if avg_draft_time is not None and avg_verify_time is not None and avg_opt_time is not None:
            avg_total_measured = avg_draft_time + avg_verify_time + avg_opt_time
            if avg_total_measured > 0:
                d_pct = (avg_draft_time / avg_total_measured) * 100
                v_pct = (avg_verify_time / avg_total_measured) * 100
                o_pct = (avg_opt_time / avg_total_measured) * 100
                print(f"[TIMING AVG] Draft: {avg_draft_time:.4f}s ({d_pct:.1f}%) | "
                      f"Verify: {avg_verify_time:.4f}s ({v_pct:.1f}%) | "
                      f"Optimize: {avg_opt_time:.4f}s ({o_pct:.1f}%)")

    print(f"[DONE ALL]")
    print()

if __name__ == "__main__":
    main()
