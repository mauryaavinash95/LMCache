import sys
try:
    import mooncake
    print(f"Mooncake Version: {mooncake.__version__ if hasattr(mooncake, '__version__') else 'Unknown'}")
except ImportError:
    print("CRITICAL: Cannot import mooncake package.")
    sys.exit(1)

# Configuration matching what we will feed LMCache
config = {
    "local_hostname": "10.17.6.201",
    "metadata_server": "http://10.17.6.201:8080",
    "master_server_address": "10.17.6.201:50051",
    "storage_root_dir": "/vast/users/amaurya/scratch/mooncake_data",
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
