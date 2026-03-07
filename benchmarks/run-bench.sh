#!/bin/bash
MODEL="lmsys/longchat-7b-v1.5-32k"
MOONCAKE_DIR="/vast/users/amaurya/scratch/mooncake_data"
mkdir -p $MOONCAKE_DIR

# Get Real IP
MY_IP=$(hostname -I | awk '{print $1}')
echo ">>> Detected Node IP: $MY_IP"

# --- 1. START MOONCAKE MASTER ---
function start_mooncake() {
    pkill -9 mooncake_master || true
    echo ">>> Starting Mooncake Master..."
    
    # We bind to 0.0.0.0 to accept connections from any interface (Localhost or LAN IP)
    # We explicitly set the metadata port to 8080
    nohup mooncake_master \
        --port 50051 \
        --rpc_address 0.0.0.0 \
        --enable_http_metadata_server=1 \
        --http_metadata_server_port=8080 \
        --http_metadata_server_host=0.0.0.0 \touch
        --v=4 \
        > "$MOONCAKE_DIR/master.log" 2>&1 &
    
    SERVER_PID=$!
    sleep 5
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "CRITICAL: Mooncake Master failed to start."
        tail -n 20 "$MOONCAKE_DIR/master.log"
        exit 1
    fi
}

# --- 2. VERIFY CONNECTION (THE DIAGNOSTIC STEP) ---
function verify_connection() {
    echo ">>> Verifying Client Connectivity..."
    
    # We write a temporary python script to test the connection explicitly.
    # This mimics what LMCache does internally.
    cat <<EOF > verify_mooncake.py
import sys
try:
    import mooncake
    print(f"Mooncake Version: {mooncake.__version__ if hasattr(mooncake, '__version__') else 'Unknown'}")
except ImportError:
    print("CRITICAL: Cannot import mooncake package.")
    sys.exit(1)

# Configuration matching what we will feed LMCache
config = {
    "local_hostname": "$MY_IP",
    "metadata_server": "http://$MY_IP:8080",
    "master_server_address": "$MY_IP:50051",
    "storage_root_dir": "$MOONCAKE_DIR",
    "protocol": "tcp",
    "device_name": "eth0", # Fallback, might need adjustment if using RDMA
}

print(f"Attempting connection to {config['metadata_server']}...")

try:
    # Attempt to initialize the client (This is where LMCache failed silently)
    # Note: Exact class instantiation depends on version, checking basic binding first
    from mooncake import MooncakeClient
    client = MooncakeClient(
        config["metadata_server"],
        config["master_server_address"],
        config["local_hostname"],
        config["protocol"],
        config.get("device_name", "")
    )
    print("SUCCESS: MooncakeClient instantiated and connected!")
except Exception as e:
    print("-" * 60)
    print("CONNECTION FAILED WITH ERROR:")
    print(e)
    import traceback
    traceback.print_exc()
    print("-" * 60)
    sys.exit(1)
EOF

    # Run the verification
    python3 verify_mooncake.py
    if [ $? -ne 0 ]; then
        echo ">>> ABORTING: Mooncake Client check failed. See error above."
        pkill -9 mooncake_master
        exit 1
    fi
}

# --- 3. GENERATE CONFIG ---
function generate_lmcache_config() {
    cat <<EOF > ./mooncake_config.yaml
chunk_size: $1
local_cpu: false
max_local_cpu_size: 5.0
remote_url: "mooncakestore://$MY_IP:50051"
remote_serde: "naive"
extra_config:
  local_hostname: "$MY_IP"
  metadata_server: "http://$MY_IP:8080"
  master_server_address: "$MY_IP:50051"
  storage_root_dir: "$MOONCAKE_DIR"
  protocol: "tcp"
  local_buffer_size: 0
  save_chunk_meta: false
EOF
}

# --- 4. RUN ---
function run_benchmark() {
    local chunk_size=1024
    
    start_mooncake
    verify_connection
    generate_lmcache_config $chunk_size

    export LMCACHE_CONFIG_FILE="./mooncake_config.yaml"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export PYTHONHASHSEED=0 
    
    # Attempt to fix the libtinfo warning which might indicate underlying library rot
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64

    echo ">>> Starting vLLM..."
    cmd="CUDA_VISIBLE_DEVICES=0 nsys profile \
        --trace=cuda,nvtx,osrt \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --output=vllm_mooncake_VERIFIED_${MODEL//\//-}_chunk_${chunk_size} \
        --force-overwrite=true \
        python3 ~/dl-io/LMCache/benchmarks/long_doc_qa/long_doc_qa_offline.py \
        --model $MODEL \
        --num-documents 50 \
        --document-length 10000 \
        --output-len 1 \
        --repeat-count 1 \
        --repeat-mode tile \
        --max-inflight-requests 400 \
        --tensor-parallel-size 1 \
        --enable-chunked-prefill \
        --max-num-batched-tokens 32768 \
        --kv-transfer-config '{\"kv_connector\": \"LMCacheConnectorV1\", \"kv_role\": \"kv_both\"}'"

    eval $cmd
    pkill -9 mooncake_master
}

run_benchmark