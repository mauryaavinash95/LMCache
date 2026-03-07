import argparse
import os
import random
import sys
import time
import pandas as pd
import json
import torch
import multiprocessing
from dataclasses import dataclass
from vllm import LLM, SamplingParams

# --- Helper Classes & Functions ---

@dataclass
class RequestStats:
    prompt_id: int
    ttft: float
    duration: float
    successful: bool

def run_vllm_instance(
    gpu_ids: str,
    instance_role: str, # 'producer' or 'consumer'
    model_name: str,
    prompts: list,
    output_len: int,
    kv_config: str,
    tp_size: int,
    barrier: multiprocessing.Barrier = None
):
    """
    Runs a distinct vLLM engine on specific GPUs.
    """
    # 1. Set Environment Variables for this Process
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # 2. Parse KV Config and Inject Role
    try:
        config = json.loads(kv_config)
    except:
        config = {"kv_connector": "LMCacheConnectorV1"}
    
    # Role specific configs
    if instance_role == "producer":
        config["kv_role"] = "kv_producer"
        desc = "PREFILL_NODE (GPU 0,1)"
    else:
        config["kv_role"] = "kv_consumer"
        desc = "DECODE_NODE (GPU 2,3)"
    
    print(f"\n[{desc}] Initializing vLLM on {gpu_ids} with role {config['kv_role']}...")

    # 3. Initialize Engine
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        kv_transfer_config=config,
        max_model_len=32768,
        enable_chunked_prefill=True, # Often helpful for prefill node
        trust_remote_code=True,
        enforce_eager=True # Recommended for accurate timing benchmarks
    )

    sampling_params = SamplingParams(
        max_tokens=output_len,
        temperature=0.0,
        ignore_eos=True
    )

    # 4. Synchronization (Optional)
    # If we want them to start at the exact same time (unlikely for this logic)
    if barrier:
        barrier.wait()

    # 5. Execution
    print(f"[{desc}] Starting processing {len(prompts)} prompts...")
    
    # NVTX Marker for Profiling
    torch.cuda.nvtx.range_push(f"Bench_{instance_role}")
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    torch.cuda.nvtx.range_pop()

    # 6. Calculate Stats
    stats = []
    for i, o in enumerate(outputs):
        # TTFT is only available if metrics are enabled, else we estimate
        ttft = o.metrics.first_token_time - o.metrics.arrival_time if o.metrics.first_token_time else 0
        stats.append({
            "role": instance_role,
            "prompt_id": i,
            "ttft": ttft,
            "duration": end_time - start_time
        })

    print(f"[{desc}] Finished. Total Time: {end_time - start_time:.2f}s")
    
    # Save partial results to disk (simple IPC)
    df = pd.DataFrame(stats)
    df.to_csv(f"results_{instance_role}.csv", index=False)


def main(args):
    # --- Data Generation (Same as original) ---
    print("Generating Datasets...")
    # 1. Warmup / Prefill Prompts (The Contexts)
    context_prompts = [
        str(i) + " " + " ".join(["hi"] * args.document_length)
        for i in range(args.num_documents)
    ]
    
    # 2. Query / Decode Prompts (Context + Question)
    # We append a small dummy question to ensure it's not identical (though LMCache handles prefix matches)
    query_prompts = [p + " \n Q: What is the meaning of hi?" for p in context_prompts]

    # --- Disaggregated Execution ---
    
    # Process 1: The Prefill Engine (Produces KV Cache)
    # We map this to the first half of GPUs (e.g., 0,1)
    p1 = multiprocessing.Process(
        target=run_vllm_instance,
        args=("0,1", "producer", args.model, context_prompts, 1, args.kv_transfer_config, 2, None)
    )

    # Process 2: The Decode Engine (Consumes KV Cache)
    # We map this to the second half of GPUs (e.g., 2,3)
    p2 = multiprocessing.Process(
        target=run_vllm_instance,
        args=("2,3", "consumer", args.model, query_prompts, args.output_len, args.kv_transfer_config, 2, None)
    )

    print("\n=== PHASE 1: PREFILL (Storing to LMCache) ===")
    p1.start()
    p1.join() # We wait for Prefill to finish completely before starting Decode
    
    print("\n=== PHASE 2: DECODE (Retrieving from LMCache) ===")
    p2.start()
    p2.join()

    print("\n=== BENCHMARK COMPLETE ===")
    # Load and display results
    try:
        df_cons = pd.read_csv("results_consumer.csv")
        print(f"Decode Node Mean TTFT: {df_cons['ttft'].mean():.4f} s")
    except:
        print("Could not read results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-length", type=int, default=20000)
    parser.add_argument("--num-documents", type=int, default=4)
    parser.add_argument("--output-len", type=int, default=100)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--kv-transfer-config", type=str, default='{"kv_connector": "LMCacheConnectorV1"}')
    args = parser.parse_args()
    
    # Set start method to spawn to avoid CUDA context issues
    multiprocessing.set_start_method('spawn')
    main(args)