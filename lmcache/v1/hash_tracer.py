# SPDX-License-Identifier: Apache-2.0
"""Per-request hash lifecycle tracer.

Tracks which KV cache chunk hashes flow through each tier
(vLLM GPU, CPU memory, SSD) during a single request's
lifecycle.  Each hash is mapped to the fastest tier where
it exists: ``"gpu"`` > ``"cpu"`` > ``"disk"`` > ``null``.
"""

# Standard
import json
import os
import threading
from typing import Dict, List, Optional

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Module-level JSONL output path.  Set via the
# ``KVSTREAM_HASH_TRACE_FILE`` env-var or by calling
# :func:`set_trace_file`.  Defaults to ``/tmp/hash_trace.jsonl``.
_TRACE_FILE: str = os.environ.get(
    "KVSTREAM_HASH_TRACE_FILE", "/tmp/hash_trace.jsonl"
)
_TRACE_LOCK = threading.Lock()


def set_trace_file(path: str) -> None:
    """Override the JSONL output path at runtime.

    Args:
        path: Absolute path to the output JSONL file.
    """
    global _TRACE_FILE  # noqa: PLW0603
    _TRACE_FILE = path


# Backend name -> canonical tier label, ordered by speed.
_BACKEND_TO_TIER: Dict[str, str] = {
    "LocalCPUBackend": "cpu",
}
# Disk backends: any name containing "Disk" or "KVStream"
# gets mapped to "disk" (handled in _best_tier()).

_TIER_PRIORITY = {"gpu": 0, "cpu": 1, "disk": 2}


def _best_tier(backend_names: List[str]) -> Optional[str]:
    """Return the fastest tier label from a list of backend names.

    Args:
        backend_names: Backend class names where the hash exists.

    Returns:
        ``"cpu"`` or ``"disk"``, whichever is fastest, or
        ``None`` if the list is empty.
    """
    best: Optional[str] = None
    best_pri = 999
    for bn in backend_names:
        tier = _BACKEND_TO_TIER.get(bn)
        if tier is None:
            # Any disk-like backend
            tier = "disk"
        pri = _TIER_PRIORITY.get(tier, 999)
        if pri < best_pri:
            best = tier
            best_pri = pri
    return best


class RequestHashTracer:
    """Lightweight per-request hash-to-tier map.

    For each chunk hash in the request (ordered by prefix
    position), records which tier it exists in at lookup time.
    The output is a flat list of ``{hex_hash: tier}`` dicts.

    Attributes:
        req_id: The vLLM / LMCache request identifier.
        total_tokens: Total tokens in the request.
        chunk_size: Number of tokens per LMCache chunk.
        all_hashes: Full ordered hash sequence for the request.
        gpu_prefix_count: Number of leading chunks in vLLM GPU.
        tier_presence: Per-hash backend existence (probed).
    """

    def __init__(self, req_id: str, chunk_size: int = 128) -> None:
        self.req_id = req_id
        self.total_tokens: int = 0
        self.chunk_size = chunk_size
        self.all_hashes: List[int] = []
        self.gpu_prefix_count: int = 0
        # Maps chunk_hash -> list of backend names where it exists.
        self.tier_presence: Dict[int, List[str]] = {}

    def set_all_hashes(
        self,
        hashes: List[int],
        total_tokens: int,
    ) -> None:
        """Set the full hash sequence and total token count.

        Args:
            hashes: Ordered list of chunk_hash ints for the request.
            total_tokens: Total number of tokens in the request.
        """
        self.all_hashes = hashes
        self.total_tokens = total_tokens

    def set_vllm_gpu_prefix(
        self,
        num_vllm_cached_tokens: int,
    ) -> None:
        """Mark the first N chunks as served from vLLM GPU cache.

        Args:
            num_vllm_cached_tokens: Number of tokens already in
                vLLM's GPU KV cache.
        """
        self.gpu_prefix_count = num_vllm_cached_tokens // self.chunk_size

    def set_tier_presence(
        self,
        tier_presence: Dict[int, List[str]],
    ) -> None:
        """Set the per-hash tier existence map.

        This records which backends each hash exists in,
        independent of prefix-chain contiguity.  Populated
        by probing every backend for every hash in the request.

        Args:
            tier_presence: Mapping from chunk_hash to list of
                backend names where it currently exists.
        """
        self.tier_presence = tier_presence

    def to_dict(self) -> Dict:
        """Return the tracer state as a compact dictionary.

        Each hash is mapped to its fastest existing tier:
        ``"gpu"`` for the vLLM prefix, ``"cpu"`` / ``"disk"``
        from backend probing, or ``null`` if absent everywhere.

        Returns:
            Dictionary with request metadata and ordered hash
            list.
        """
        hashes_list: List[Dict[str, Optional[str]]] = []
        for idx, h in enumerate(self.all_hashes):
            if idx < self.gpu_prefix_count:
                tier: Optional[str] = "gpu"
            else:
                backends = self.tier_presence.get(h, [])
                tier = _best_tier(backends)
            hashes_list.append({hex(h): tier})

        return {
            "req_id": self.req_id,
            "total_tokens": self.total_tokens,
            "chunk_size": self.chunk_size,
            "total_chunks": len(self.all_hashes),
            "hashes": hashes_list,
        }

    def emit(self) -> str:
        """Serialize to JSON, log, and append to the JSONL file.

        The output file path is controlled by the
        ``KVSTREAM_HASH_TRACE_FILE`` environment variable
        (default ``/tmp/hash_trace.jsonl``).

        Returns:
            The JSON string that was emitted.
        """
        json_str = json.dumps(self.to_dict())
        logger.info("HASH_TRACE %s", json_str)

        with _TRACE_LOCK:
            with open(_TRACE_FILE, "a") as f:
                f.write(json_str + "\n")

        return json_str
