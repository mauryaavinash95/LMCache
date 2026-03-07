
# lmsys/longchat-7b-v1.5-32k
import argparse
import os
import random
import sys
import time
import pandas as pd
import torch
import json
from dataclasses import dataclass
from vllm import LLM, SamplingParams

# Global output filename
OUTPUT_FILE = None

@dataclass
class RequestStats:
    prompt_id: int
    request_start: float
    ttft: float
    request_end: float
    successful: bool

def write_resp(text: str):
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, "a") as resp_file:
            resp_file.write(text)
    else:
        sys.stdout.write(text)

def relative_time(df, start_time):
    if "request_start" in df.columns and not df["request_start"].eq(0).all():
        min_start = df["request_start"].min()
        if min_start > 0:
            df["request_start"] = df["request_start"] - min_start
            df["request_end"] = df["request_end"] - min_start
            df["ttft_time"] = df["request_start"] + df["ttft"]
    else:
        df["request_start"] = 0
        df["request_end"] = df["request_end"]
        df["ttft_time"] = df["ttft"]

def trimmed_mean(series: pd.Series, trim_fraction: float) -> float:
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    if trim_fraction <= 0:
        return float(s.mean())
    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError("--trim-fraction must be in [0, 0.5).")
    s = s.sort_values()
    n = len(s)
    k = int(n * trim_fraction)
    if n - 2 * k <= 0:
        return float(s.mean())
    return float(s.iloc[k : n - k].mean())

def run_offline_batch(llm, prompts, output_len, desc="Benchmark"):
    sampling_params = SamplingParams(
        max_tokens=output_len,
        temperature=0.0,
        ignore_eos=True
    )
    
    write_resp(f"\n--- Starting {desc} with {len(prompts)} prompts ---\n")
    
    torch.cuda.nvtx.range_push(f"VLLM_Offline_{desc}")
    
    start_wall = time.time()
    
    # CRITICAL FIX 1: Wrap list[int] in dicts for vLLM API compliance
    # vLLM expects: [{"prompt_token_ids": [1, 2, ...]}, ...]
    final_inputs = []
    if len(prompts) > 0 and isinstance(prompts[0], list):
        for p in prompts:
            final_inputs.append({"prompt_token_ids": p})
    else:
        final_inputs = prompts 
    
    # CRITICAL FIX 2: Use 'prompts' argument, not 'prompt_token_ids'
    request_outputs = llm.generate(prompts=final_inputs, sampling_params=sampling_params)

    end_wall = time.time()
    
    torch.cuda.nvtx.range_pop()
    
    stats_list = []
    for i, output in enumerate(request_outputs):
        ttft = 0.0
        req_start = start_wall
        req_end = end_wall
        
        if hasattr(output, 'metrics') and output.metrics:
            m = output.metrics
            if m.first_token_time and m.arrival_time:
                ttft = m.first_token_time - m.arrival_time
                req_start = m.arrival_time
                req_end = m.finished_time if m.finished_time else end_wall
        
        stats_list.append(RequestStats(
            prompt_id=i,
            request_start=req_start,
            ttft=ttft,
            request_end=req_end,
            successful=not output.finished 
        ))
        stats_list[-1].successful = True 

    return stats_list

def main(args):
    random.seed(args.shuffle_seed)

    print(f"Initializing vLLM Engine with model: {args.model}")
    
    # CRITICAL FIX 3: Parse JSON string to Dict
    kv_config_dict = None
    if args.kv_transfer_config:
        try:
            kv_config_dict = json.loads(args.kv_transfer_config)
            print(f"Parsed kv_transfer_config: {kv_config_dict}")
        except json.JSONDecodeError as e:
            print(f"Error parsing kv_transfer_config: {e}")
            sys.exit(1)

    # Force max_model_len to be safe
    forced_max_len = None 
    if args.document_length > 30000:
        forced_max_len = args.document_length + 2048
        if forced_max_len > 32768:
             forced_max_len = 32768

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_inflight_requests, 
        kv_transfer_config=kv_config_dict, # Pass the Dict!
        max_model_len=forced_max_len,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,            
        disable_custom_all_reduce=True,
        trust_remote_code=True
    )

    # --- Pre-Warmup ---
    # Safe range for Llama-2 vocab (32k)
    pre_warmup_prompts = [[100] * 100 for _ in range(5)]
    run_offline_batch(llm, pre_warmup_prompts, args.output_len, desc="Pre-Warmup")

    # --- Generate Base Document ---
    print(f"Generating base document of length {args.document_length} tokens...")
    # Token 500 is a safe generic token
    base_doc = [500] * args.document_length

    # --- 1. Prepare WARMUP Prompts ---
    warmup_prompts = []
    for i in range(args.num_documents):
        # Range 1000+ is safe
        unique_prompt = [1000 + i] + base_doc[:-1]
        warmup_prompts.append(unique_prompt)

    # --- 2. Prepare QUERY Prompts (GUARANTEED MISSES) ---
    print(f"Generating {args.repeat_count * args.num_documents} UNIQUE prompts for Query Round (Force Miss)...")
    
    query_prompts = []
    total_query_reqs = args.num_documents * args.repeat_count
    
    for i in range(total_query_reqs):
        # FIXED RANGE: 5000 to 30000 is safe for Llama-2/Mistral (Vocab size ~32000)
        random_prefix = random.randint(5000, 30000)
        
        unique_prompt = [random_prefix] + base_doc[:-1]
        query_prompts.append(unique_prompt)

    # --- Run Warmup ---
    write_resp("------warm up round------\n")
    warmup_start_time = time.time()
    warmup_stats = run_offline_batch(llm, warmup_prompts, args.output_len, desc="Warmup_Round")
    warmup_end_time = time.time()

    if args.sleep_time_after_warmup > 0:
        write_resp(f"Sleeping for {args.sleep_time_after_warmup} seconds...\n")
        time.sleep(args.sleep_time_after_warmup)

    # --- Run Query ---
    write_resp("------query round------\n")
    benchmark_start_time = time.time()
    benchmark_stats = run_offline_batch(llm, query_prompts, args.output_len, desc="Query_Round")
    benchmark_end_time = time.time()

    # --- Process Results ---
    warmup_df = pd.DataFrame([s.__dict__ for s in warmup_stats])
    relative_time(warmup_df, warmup_start_time)
    
    benchmark_df = pd.DataFrame([s.__dict__ for s in benchmark_stats])
    relative_time(benchmark_df, benchmark_start_time)

    warmup_df.to_csv("warmup_round.csv", index=False)
    benchmark_df.to_csv("query_round.csv", index=False)

    # Print Summary
    warmup_mean_ttft = trimmed_mean(
        warmup_df.query("successful == True")["ttft"], args.trim_fraction
    )
    query_mean_ttft = trimmed_mean(
        benchmark_df.query("successful == True")["ttft"], args.trim_fraction
    )
    
    CSI = "\x1b["
    RESET = CSI + "0m"
    print(f"Warmup round mean TTFT: {warmup_mean_ttft:.3f}s")
    print(f"Warmup round time: {warmup_end_time - warmup_start_time:.3f}s")
    print(f"{CSI}36;1m\n=== BENCHMARK RESULTS ==={RESET}")
    print(f"{CSI}32mQuery round mean TTFT: {query_mean_ttft:.3f}s{RESET}")
    print(f"{CSI}33mQuery round time: {benchmark_end_time - benchmark_start_time:.3f}s{RESET}")
    
    if args.json_output:
        summary = {
            "query_ttft_per_prompt": query_mean_ttft,
            "query_round_time_per_prompt": (benchmark_end_time - benchmark_start_time) / len(benchmark_df),
        }
        print(json.dumps(summary))

def create_argument_parser():
    parser = argparse.ArgumentParser(description="Offline Long Doc QA Benchmark")

    parser.add_argument("--document-length", type=int, default=20000)
    parser.add_argument("--num-documents", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=100)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--repeat-mode", type=str, default="random")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-inflight-requests", type=int, default=2)
    parser.add_argument("--sleep-time-after-warmup", type=float, default=0.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--hit-miss-ratio", type=str, default=None)
    parser.add_argument("--trim-fraction", type=float, default=0.0)
    parser.add_argument("--json-output", action="store_true")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--kv-transfer-config", type=str, default=None)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")

    return parser

if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    OUTPUT_FILE = args.output
    main(args)