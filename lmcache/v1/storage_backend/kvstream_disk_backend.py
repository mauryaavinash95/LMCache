# SPDX-License-Identifier: Apache-2.0
"""KVStream disk backend for LMCache — P-tier weighted layer-striped storage.

Uses the KVStream library (io_uring-based async I/O engine) to perform
high-performance disk reads and writes for KV cache data, replacing the
Python open()/write()/read() calls used by LocalDiskBackend.

Architecture
~~~~~~~~~~~~
KVStream supports **P independent storage tiers** (e.g. local NVMe,
parallel filesystem, scratch), each with its own io_uring engine and
tuning parameters.  KV chunks in ``KV_2LTD`` format
``[kv_size, num_layers, num_tokens, hidden_dim]`` are layer-split across
tiers according to a configurable ratio string (e.g. ``"0.6:0.4"``).
Each tier stores a contiguous K-block and V-block for its layer range in
a per-chunk file.  The split is an implementation detail — LMCache sees
a single ``StorageBackendInterface`` and never needs to know about tiers.

When only one tier is configured (``kvstream_split_ratios = "1.0"``), the
code path is functionally equivalent to the pre-split version: one engine,
one directory, two I/O ops per chunk (K-block + V-block).

Writes are drain-based: ``batched_submit_put_task`` submits SQEs on the
calling thread and returns immediately.  A background drain thread polls
for completed writes every 50 ms (configurable via
``kvstream_drain_poll_interval_s``), freeing CPU memory buffers as soon
as the kernel finishes writing them.

Reads use ``wait_one`` / ``wait_all`` with ``IOQueue.READ`` so they never
block on outstanding writes.

Activated when ``config.kvstream_enable`` is ``True`` and
``config.local_disk`` points to a directory.
"""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

# ======================================================================== #
#  Tier data model (private to this module)                                 #
# ======================================================================== #


@dataclass
class _TierConfig:
    """Static configuration for one storage tier."""

    index: int
    path: str
    ratio: float
    layer_start: int  # inclusive
    layer_end: int  # exclusive
    read_chunk_size_kb: int
    read_queue_depth: int
    write_queue_depth: int
    write_chunk_size_kb: int
    max_fds: int
    max_retries: int
    try_odirect: bool

    @property
    def num_layers(self) -> int:
        """Number of layers assigned to this tier."""
        return self.layer_end - self.layer_start


@dataclass
class _TierState:
    """Runtime state for one storage tier: config + engine handle."""

    config: _TierConfig
    engine: Any  # kvstream_core.KVStream — typed as Any to avoid import-time dep


@dataclass
class _TierSlice:
    """Describes one tier's portion of a stored chunk on disk.

    Fields are designed for Phase-2 slab-file compatibility: explicit
    file offsets instead of implicit computation.
    """

    tier_index: int
    path: str  # full file path on this tier
    layer_start: int
    layer_end: int
    size: int  # total bytes on this tier (K + V combined)
    k_file_offset: int  # byte offset of the K-layers block within the file
    v_file_offset: int  # byte offset of the V-layers block within the file


@dataclass
class _TieredChunkMeta:
    """Per-key metadata for a chunk spread across P tiers.

    Stored in ``self.dict`` (the cache policy's mutable mapping).
    Implements ``pin``/``unpin``/``can_evict`` so cache policies work
    unchanged.
    """

    slices: list[_TierSlice]
    total_size: int  # sum of all slice sizes
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    cached_positions: Optional[torch.Tensor] = None
    pin_count: int = 0

    # -- Compat interface expected by cache policies ---------------------

    @property
    def size(self) -> int:
        """Alias for ``total_size`` — matches ``DiskCacheMetadata.size``."""
        return self.total_size

    def pin(self) -> bool:
        """Increment pin count to prevent eviction."""
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        """Decrement pin count."""
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        """Return whether this entry is pinned."""
        return self.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """Return whether this entry can be evicted."""
        return not self.is_pinned


@dataclass
class _InflightGroup:
    """Tracks all sub-write I/O ops for one logical chunk across P tiers.

    A single chunk produces ``2 * P`` I/O hashes (K-block + V-block per
    tier).  The group is resolved when ``remaining`` reaches zero.
    """

    key: CacheEngineKey
    memory_obj: MemoryObj
    remaining: int  # starts at 2*P, decremented on each completion
    total_size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    cached_positions: Optional[torch.Tensor]
    on_complete_callback: Optional[Callable[[CacheEngineKey], None]]
    io_hashes: list[str] = field(default_factory=list)
    failed: bool = False


# ======================================================================== #
#  Helper: parse tier ratios                                                #
# ======================================================================== #


def _parse_ratios(ratios_str: str) -> list[float]:
    """Parse a colon-separated ratio string and validate.

    Args:
        ratios_str: E.g. ``"0.6:0.4"`` or ``"0.2:0.7:0.1"``.

    Returns:
        List of floats that sum to 1.0.

    Raises:
        ValueError: If ratios are invalid.
    """
    parts = ratios_str.strip().split(":")
    ratios = [float(p) for p in parts]
    if any(r < 0 for r in ratios):
        raise ValueError(
            f"kvstream_split_ratios contains negative value: {ratios_str}"
        )
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"kvstream_split_ratios must sum to 1.0, got {total}: {ratios_str}"
        )
    return ratios


def _compute_layer_boundaries(
    ratios: list[float], num_layers: int
) -> list[int]:
    """Compute layer boundaries from ratios.

    Returns a list of ``len(ratios) + 1`` boundary values where
    ``boundaries[i]`` is the start layer and ``boundaries[i+1]`` is the
    end layer for tier ``i``.

    Args:
        ratios: Per-tier ratios (must sum to 1.0).
        num_layers: Total number of model layers.

    Returns:
        Monotonically non-decreasing list starting at 0 and ending at
        ``num_layers``.
    """
    cumulative = [0.0]
    for r in ratios:
        cumulative.append(cumulative[-1] + r)
    boundaries = [round(c * num_layers) for c in cumulative]
    # Clamp end
    boundaries[-1] = num_layers
    return boundaries


# ======================================================================== #
#  Main backend class                                                       #
# ======================================================================== #


class KVStreamDiskBackend(StorageBackendInterface):
    """Disk backend using KVStream (io_uring) for async I/O.

    Supports P-tier weighted layer-striping across heterogeneous
    storage tiers (e.g. local NVMe + parallel filesystem).

    Activated when ``config.kvstream_enable`` is ``True``.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ) -> None:
        """Initialize the KVStream disk backend.

        Args:
            config: LMCache engine configuration.
            loop: The asyncio event loop (kept for read-path
                ``asyncio.to_thread`` usage only).
            local_cpu_backend: CPU memory allocator backend.
            dst_device: Target device string (e.g. ``"cuda:0"``).
            lmcache_worker: Optional cache controller worker for
                ADMIT/EVICT messages.
            metadata: Optional LMCache metadata (provides model geometry
                for layer-split computation).
        """
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        # -- Cache policy & metadata dict --------------------------------
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device
        self.local_cpu_backend = local_cpu_backend
        self.disk_lock = threading.Lock()

        # -- Primary directory -------------------------------------------
        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info("Created KVStream disk cache directory: %s", self.path)

        # Event loop — only used for the read path (asyncio.to_thread)
        self.loop = loop

        # -- Capacity tracking -------------------------------------------
        self.max_cache_size: int = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size: float = 0.0
        self.usage: int = 0
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        # -- In-flight put-task tracking ---------------------------------
        self.put_tasks_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        # -- Key ordering for cache recency (suffix -> prefix) -----------
        self.keys_in_request: List[CacheEngineKey] = []

        # -- Unique I/O hash counter (avoids collisions across ops) ------
        self._hash_counter: int = 0
        self._counter_lock = threading.Lock()

        # -- Extra config ------------------------------------------------
        extra = config.extra_config or {}

        # -- Global defaults for engine params ---------------------------
        global_read_chunk_kb: int = int(
            extra.get("kvstream_read_chunk_size_kb", 4096)
        )
        global_read_qd: int = int(
            extra.get("kvstream_read_queue_depth", 64)
        )
        global_write_qd: int = int(
            extra.get("kvstream_write_queue_depth", 4)
        )
        global_write_chunk_kb: int = int(
            extra.get("kvstream_write_chunk_size_kb", 256)
        )
        global_max_fds: int = int(extra.get("kvstream_max_fds", 4096))
        global_max_retries: int = int(
            extra.get("kvstream_max_retries", 10)
        )
        global_try_odirect: bool = bool(
            extra.get("kvstream_try_odirect", True)
        )

        # -- Import kvstream_core ----------------------------------------
        try:
            from kvstream import kvstream_core

            self.kvstream_core = kvstream_core
        except ImportError:
            raise ImportError(
                "kvstream_core is not installed. "
                "Build and install the kvstream package before using "
                "KVStreamDiskBackend."
            )

        # -- Tier configuration ------------------------------------------
        ratios_str: str = str(extra.get("kvstream_split_ratios", "1.0"))
        ratios = _parse_ratios(ratios_str)
        num_tiers = len(ratios)

        # Determine num_layers from metadata.  metadata may be None in
        # some test/scheduler contexts; fall back to kv_shape if available.
        if metadata is not None:
            self._num_layers: int = metadata.kv_shape[0]
            self._chunk_size: int = metadata.chunk_size
            kv_size: int = 1 if metadata.use_mla else 2
            hidden_dim: int = metadata.kv_shape[3] * metadata.kv_shape[4]
            dtype_size: int = metadata.kv_dtype.itemsize
        else:
            # Fallback: cannot compute layer geometry without metadata.
            # Only valid for single-tier (no split) mode.
            if num_tiers > 1:
                raise ValueError(
                    "KVStream multi-tier split requires LMCacheMetadata "
                    "to compute layer boundaries. metadata is None."
                )
            self._num_layers = 0
            self._chunk_size = 0
            kv_size = 2
            hidden_dim = 0
            dtype_size = 0

        boundaries = _compute_layer_boundaries(ratios, self._num_layers)

        # Per-layer byte size: T * D * dtype_size (for one KV-component,
        # one layer).  The K-block and V-block each contain
        # num_layers * per_layer_bytes bytes.
        self._per_layer_bytes: int = (
            self._chunk_size * hidden_dim * dtype_size
        )
        self._kv_block_bytes: int = self._num_layers * self._per_layer_bytes
        self._kv_size: int = kv_size

        # Validate O_DIRECT alignment
        if self._per_layer_bytes > 0 and self._per_layer_bytes % 4096 != 0:
            logger.warning(
                "per_layer_bytes=%d is not 4KB-aligned; O_DIRECT may "
                "fall back to buffered I/O for split writes.",
                self._per_layer_bytes,
            )

        # Build tier configs and engines
        self._tiers: list[_TierState] = []
        for i in range(num_tiers):
            tier_path: str
            if i == 0:
                tier_path = self.path
            else:
                tier_key = f"kvstream_tier_{i}_path"
                tier_path = str(extra.get(tier_key, ""))
                if not tier_path:
                    raise ValueError(
                        f"kvstream_split_ratios specifies {num_tiers} tiers "
                        f"but {tier_key} is not set."
                    )

            if not os.path.exists(tier_path):
                os.makedirs(tier_path)
                logger.info(
                    "Created KVStream tier %d directory: %s", i, tier_path
                )

            tc = _TierConfig(
                index=i,
                path=tier_path,
                ratio=ratios[i],
                layer_start=boundaries[i],
                layer_end=boundaries[i + 1],
                read_chunk_size_kb=int(
                    extra.get(
                        f"kvstream_tier_{i}_read_chunk_size_kb",
                        global_read_chunk_kb,
                    )
                ),
                read_queue_depth=int(
                    extra.get(
                        f"kvstream_tier_{i}_read_queue_depth",
                        global_read_qd,
                    )
                ),
                write_queue_depth=int(
                    extra.get(
                        f"kvstream_tier_{i}_write_queue_depth",
                        global_write_qd,
                    )
                ),
                write_chunk_size_kb=int(
                    extra.get(
                        f"kvstream_tier_{i}_write_chunk_size_kb",
                        global_write_chunk_kb,
                    )
                ),
                max_fds=int(
                    extra.get(
                        f"kvstream_tier_{i}_max_fds", global_max_fds
                    )
                ),
                max_retries=int(
                    extra.get(
                        f"kvstream_tier_{i}_max_retries",
                        global_max_retries,
                    )
                ),
                try_odirect=bool(
                    extra.get(
                        f"kvstream_tier_{i}_try_odirect",
                        global_try_odirect,
                    )
                ),
            )

            engine = kvstream_core.KVStream(
                read_chunk_size_kb=tc.read_chunk_size_kb,
                read_queue_depth=tc.read_queue_depth,
                write_queue_depth=tc.write_queue_depth,
                write_chunk_size_kb=tc.write_chunk_size_kb,
                max_fds_open=tc.max_fds,
                try_using_odirect=tc.try_odirect,
                max_retries=tc.max_retries,
            )

            self._tiers.append(_TierState(config=tc, engine=engine))

            logger.info(
                "KVStream tier %d initialized: path=%s, layers=[%d,%d), "
                "ratio=%.2f, read_chunk_kb=%d, read_qd=%d, write_qd=%d, "
                "write_chunk_kb=%d, max_fds=%d, try_odirect=%s",
                i,
                tc.path,
                tc.layer_start,
                tc.layer_end,
                tc.ratio,
                tc.read_chunk_size_kb,
                tc.read_queue_depth,
                tc.write_queue_depth,
                tc.write_chunk_size_kb,
                tc.max_fds,
                tc.try_odirect,
            )

        # Convenience: keep a reference to the single engine for
        # backwards-compat logging.  ``self.engine`` is NOT used in
        # data-path code (always go through ``self._tiers``).
        self.engine = self._tiers[0].engine

        # -- Inflight writes (grouped drain-based bookkeeping) -----------
        # Maps io_hash -> _InflightGroup (many-to-one: 2*P hashes per
        # group).
        self._hash_to_group: dict[str, _InflightGroup] = {}
        # Serializes _drain_completed() calls.
        self._drain_lock = threading.Lock()

        # -- Overlapped load tracking ------------------------------------
        # Maps synthetic group_hash -> list of (tier, io_hash) for the
        # overlapped read path (submit_batch_load / wait_one_load).
        self._overlapped_load_refs: dict[
            str, list[tuple[_TierState, str]]
        ] = {}

        # -- Batched message sender (controller ADMIT/EVICT msgs) --------
        self.batched_msg_sender: Optional[BatchedMessageSender] = None
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning(
                "KVStreamDiskBackend: controller message sender "
                "not initialized"
            )

        # -- Background drain thread -------------------------------------
        self._drain_stop = threading.Event()
        drain_interval: float = float(
            extra.get("kvstream_drain_poll_interval_s", 0.05)
        )
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            args=(drain_interval,),
            name="kvstream-drain",
            daemon=True,
        )
        self._drain_thread.start()
        logger.info(
            "KVStream background drain thread started "
            "(poll interval=%.3fs)",
            drain_interval,
        )

        # -- Deferred write support --------------------------------------
        self._deferred_writes_enabled: bool = bool(
            extra.get("kvstream_deferred_writes", True)
        )
        # Each entry: (group, list of (engine, io_hash, raw_slice, path,
        #               file_offset) tuples)
        self._deferred_queue: list[
            tuple[_InflightGroup, list[tuple[Any, str, torch.Tensor, str, int]]]
        ] = []
        if self._deferred_writes_enabled:
            logger.info("KVStream deferred writes enabled")

    # ------------------------------------------------------------------ #
    #  String / helpers                                                    #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        """Return human-readable backend name."""
        return "KVStreamDiskBackend"

    def _key_to_tier_path(
        self, key: CacheEngineKey, tier: _TierState
    ) -> str:
        """Convert a cache key to a file path on the given tier.

        Args:
            key: The cache engine key.
            tier: The target storage tier.

        Returns:
            Absolute path for the cached file on this tier.
        """
        return os.path.join(
            tier.config.path,
            key.to_string().replace("/", "-") + ".bin",
        )

    def _next_io_hash(self, key: CacheEngineKey, op: str) -> str:
        """Generate a globally unique hash for a KVStream I/O operation.

        Args:
            key: The cache engine key.
            op: Operation prefix (e.g. ``"save_k0"``, ``"load_v1"``).

        Returns:
            A unique hash string.
        """
        with self._counter_lock:
            self._hash_counter += 1
            return f"{op}:{key.to_string()}:{self._hash_counter}"

    def _build_tier_slices(
        self, key: CacheEngineKey
    ) -> list[_TierSlice]:
        """Build the list of ``_TierSlice`` for a key across all tiers.

        In Phase 1, each tier gets its own file.  K-block is stored at
        file offset 0, V-block immediately after.

        Args:
            key: The cache engine key.

        Returns:
            List of ``_TierSlice``, one per tier.
        """
        slices: list[_TierSlice] = []
        for tier in self._tiers:
            tc = tier.config
            tier_layer_bytes = tc.num_layers * self._per_layer_bytes
            tier_size = self._kv_size * tier_layer_bytes  # K + V
            slices.append(
                _TierSlice(
                    tier_index=tc.index,
                    path=self._key_to_tier_path(key, tier),
                    layer_start=tc.layer_start,
                    layer_end=tc.layer_end,
                    size=tier_size,
                    k_file_offset=0,
                    v_file_offset=tier_layer_bytes,
                )
            )
        return slices

    # ------------------------------------------------------------------ #
    #  Contains / existence checks                                         #
    # ------------------------------------------------------------------ #

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check whether *key* exists in the disk cache.

        Args:
            key: The cache engine key.
            pin: If ``True``, pin the entry to prevent eviction.

        Returns:
            ``True`` if the key exists, ``False`` otherwise.
        """
        with self.disk_lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether *key* has an in-flight put task.

        Args:
            key: The cache engine key.

        Returns:
            ``True`` if a save for this key is currently pending.
        """
        with self.put_tasks_lock:
            return key in self.put_tasks

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check contiguous prefix of *keys* present in the cache.

        Stops at the first miss (prefix-match semantics).

        Args:
            lookup_id: Opaque lookup identifier (for logging/tracing).
            keys: Ordered list of cache keys to check.
            pin: If ``True``, pin every hit key.

        Returns:
            The count of contiguous hits starting from index 0.
        """
        num_hit_counts = 0
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

    def touch_cache(self) -> None:
        """Update cache recency for keys accumulated during lookup."""
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    # ------------------------------------------------------------------ #
    #  Pin / unpin / remove                                                #
    # ------------------------------------------------------------------ #

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin *key* to prevent eviction.

        Args:
            key: The cache engine key.

        Returns:
            ``True`` if the key was found and pinned, ``False`` otherwise.
        """
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            return False

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin *key*, allowing eviction when appropriate.

        Args:
            key: The cache engine key.

        Returns:
            ``True`` if the key was found and unpinned, ``False``
            otherwise.
        """
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove *key* from the disk cache.

        Deletes files on all tiers that stored portions of this chunk.

        Args:
            key: The cache engine key.
            force: If ``True`` (default, external removal), acquires the
                disk lock and notifies the cache policy.  If ``False``
                (internal eviction), assumes the caller already holds
                the lock.

        Returns:
            ``True`` if the key was removed, ``False`` if not found.
        """
        if force:
            self.disk_lock.acquire()

        meta: Optional[_TieredChunkMeta] = self.dict.pop(key, None)
        if not meta:
            if force:
                self.disk_lock.release()
            return False

        total_size = meta.total_size
        self.usage -= total_size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # Delete the file on every tier
        for tier_slice in meta.slices:
            try:
                os.remove(tier_slice.path)
            except FileNotFoundError:
                logger.warning(
                    "KVStream: file already removed: %s", tier_slice.path
                )

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.EVICT,
                key=key.chunk_hash,
            )

        return True

    # ------------------------------------------------------------------ #
    #  Insert key (metadata registration after successful write)           #
    # ------------------------------------------------------------------ #

    def _insert_tiered_key(
        self,
        key: CacheEngineKey,
        slices: list[_TierSlice],
        total_size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
    ) -> None:
        """Register a successfully written key with tiered metadata.

        If the key already exists (duplicate write), the cache policy is
        updated for recency but no new metadata entry is created.

        Args:
            key: The cache engine key.
            slices: Per-tier slice descriptors.
            total_size: Combined physical size across all tiers.
            shape: Logical tensor shape.
            dtype: Tensor data type.
            fmt: Memory format enum.
            cached_positions: Optional position tensor.
        """
        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = _TieredChunkMeta(
                    slices=slices,
                    total_size=total_size,
                    shape=shape,
                    dtype=dtype,
                    fmt=fmt,
                    cached_positions=cached_positions,
                )

        if self.batched_msg_sender is not None and not has_stored:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.ADMIT,
                key=key.chunk_hash,
            )

    # ------------------------------------------------------------------ #
    #  Put-task tracking helpers                                           #
    # ------------------------------------------------------------------ #

    def _insert_put_task(self, key: CacheEngineKey) -> None:
        """Register *key* as having an in-flight put task.

        Args:
            key: The cache engine key.
        """
        with self.put_tasks_lock:
            self.put_tasks.append(key)

    def _remove_put_task(self, key: CacheEngineKey) -> None:
        """Remove *key* from the in-flight put task list.

        Args:
            key: The cache engine key.
        """
        with self.put_tasks_lock:
            try:
                self.put_tasks.remove(key)
            except ValueError:
                logger.warning(
                    "KVStream: put task for %s not found during removal",
                    key,
                )

    # ------------------------------------------------------------------ #
    #  Background drain thread                                             #
    # ------------------------------------------------------------------ #

    def _drain_loop(self, interval: float) -> None:
        """Background thread: periodically drain completed writes.

        Args:
            interval: Seconds between drain polls.
        """
        while not self._drain_stop.is_set():
            try:
                self._drain_completed()
            except Exception:
                logger.exception("KVStream drain loop error")
            self._drain_stop.wait(interval)

    # ------------------------------------------------------------------ #
    #  Drain-based write completion bookkeeping (multi-engine)             #
    # ------------------------------------------------------------------ #

    def _drain_completed(self) -> None:
        """Poll **all** tier engines for completed and failed writes.

        Uses grouped completion: each logical chunk has ``2 * P`` I/O
        hashes.  The buffer is released and the key is registered only
        when **all** sub-ops for a group complete.  If **any** sub-op
        fails, the entire group is marked failed.

        Thread-safe: serialized via ``_drain_lock``.
        """
        with self._drain_lock:
            # Collect completions and failures across all tier engines
            all_completed: list[str] = []
            all_failed: list[str] = []
            for tier in self._tiers:
                all_completed.extend(tier.engine.drain_completed())
                all_failed.extend(tier.engine.drain_failed())

            # -- Process failures first (mark groups failed) ----
            for io_hash in all_failed:
                group = self._hash_to_group.pop(io_hash, None)
                if group is None:
                    logger.warning(
                        "KVStream: drained failed hash %s with no "
                        "inflight group",
                        io_hash,
                    )
                    continue
                group.failed = True
                group.remaining -= 1
                if group.remaining == 0:
                    self._resolve_failed_group(group)

            # -- Process completions ----
            for io_hash in all_completed:
                group = self._hash_to_group.pop(io_hash, None)
                if group is None:
                    logger.warning(
                        "KVStream: drained completed hash %s with no "
                        "inflight group",
                        io_hash,
                    )
                    continue
                group.remaining -= 1
                if group.remaining == 0:
                    if group.failed:
                        self._resolve_failed_group(group)
                    else:
                        self._resolve_completed_group(group)

    def _resolve_completed_group(self, group: _InflightGroup) -> None:
        """Finalize a successfully completed write group.

        Args:
            group: The completed inflight group.
        """
        self.usage += group.total_size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # Release buffer before registering key (matches
        # LocalDiskBackend ordering for mem-leak test compatibility).
        group.memory_obj.ref_count_down()

        # Build _TierSlice list from tier configs
        slices = self._build_tier_slices(group.key)

        self._insert_tiered_key(
            group.key,
            slices,
            group.total_size,
            group.shape,
            group.dtype,
            group.fmt,
            cached_positions=group.cached_positions,
        )

        self._remove_put_task(group.key)

        if group.on_complete_callback is not None:
            try:
                group.on_complete_callback(group.key)
            except Exception as e:
                logger.warning(
                    "on_complete_callback failed for key %s: %s",
                    group.key,
                    e,
                )

    def _resolve_failed_group(self, group: _InflightGroup) -> None:
        """Finalize a failed write group.

        Releases the buffer, removes the put task, rolls back capacity,
        and cleans up any partially written files.

        Args:
            group: The failed inflight group.
        """
        logger.error(
            "KVStream: write failed for key %s", group.key
        )

        group.memory_obj.ref_count_down()
        self._remove_put_task(group.key)

        # Roll back the pre-reserved capacity
        with self.disk_lock:
            self.current_cache_size -= group.total_size

        # Clean up any partially written files (best-effort)
        for tier in self._tiers:
            path = self._key_to_tier_path(group.key, tier)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        # Remove any remaining hashes for this group still in the map
        for h in group.io_hashes:
            self._hash_to_group.pop(h, None)

    # ------------------------------------------------------------------ #
    #  Put (write) path — drain-based, P-tier layer-striped              #
    # ------------------------------------------------------------------ #

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[
            Callable[[CacheEngineKey], None]
        ] = None,
    ) -> None:
        """Submit a batch of KV chunks for async disk write.

        Each chunk is layer-split across P tiers, producing ``2 * P``
        I/O operations (K-block + V-block per tier).  All ops for a
        chunk are grouped; the buffer is released only when all ops
        complete.

        Args:
            keys: Cache keys for the KV chunks.
            objs: Memory objects containing the KV data.
            transfer_spec: Unused (interface compatibility).
            on_complete_callback: Optional callback invoked per key
                after that key's disk write completes.
        """
        # Step 1: drain previously completed/failed writes
        self._drain_completed()

        num_tiers = len(self._tiers)

        # Step 2-3: process each key
        for key, memory_obj in zip(keys, objs, strict=False):
            assert memory_obj.tensor is not None

            # Skip repeated save
            if self.exists_in_put_tasks(key):
                logger.debug(
                    "Put task for %s is already in progress.", key
                )
                continue

            self._insert_put_task(key)

            # Eviction loop
            required_size = memory_obj.get_physical_size()
            evict_success = True
            with self.disk_lock:
                while (
                    self.current_cache_size + required_size
                    > self.max_cache_size
                ):
                    evict_keys = (
                        self.cache_policy.get_evict_candidates(
                            self.dict, num_candidates=1
                        )
                    )
                    if not evict_keys:
                        logger.warning(
                            "KVStream: no eviction candidates. "
                            "Disk space under pressure."
                        )
                        evict_success = False
                        break

                    for evict_key in evict_keys:
                        self.current_cache_size -= (
                            self.dict[evict_key].size
                        )

                    self.batched_remove(evict_keys, force=False)

                if evict_success:
                    self.current_cache_size += required_size

            if not evict_success:
                self._remove_put_task(key)
                continue

            self.cache_policy.update_on_put(key)

            # ref_count_up BEFORE returning — the buffer must stay
            # alive while async writes are in-flight.
            memory_obj.ref_count_up()

            # Collect metadata
            raw_tensor = memory_obj.raw_tensor
            assert raw_tensor is not None
            total_size = memory_obj.get_physical_size()
            shape = memory_obj.metadata.shape
            dtype = memory_obj.metadata.dtype
            fmt = memory_obj.metadata.fmt
            cached_positions = memory_obj.metadata.cached_positions

            # Build the inflight group for this chunk
            group = _InflightGroup(
                key=key,
                memory_obj=memory_obj,
                remaining=self._kv_size * num_tiers,
                total_size=total_size,
                shape=shape,
                dtype=dtype,
                fmt=fmt,
                cached_positions=cached_positions,
                on_complete_callback=on_complete_callback,
            )

            # Build per-tier I/O descriptors
            save_ops: list[tuple[Any, str, torch.Tensor, str, int]] = []
            for tier in self._tiers:
                tc = tier.config
                tier_layer_bytes = tc.num_layers * self._per_layer_bytes
                path = self._key_to_tier_path(key, tier)

                # For each KV component (K=0, V=1 for standard;
                # just K=0 for MLA)
                for kv_idx in range(self._kv_size):
                    kv_label = "k" if kv_idx == 0 else "v"
                    io_hash = self._next_io_hash(
                        key, f"save_{kv_label}{tc.index}"
                    )
                    group.io_hashes.append(io_hash)

                    # Byte offset within the raw buffer:
                    #   kv_idx * kv_block_bytes + layer_start * per_layer
                    buf_offset = (
                        kv_idx * self._kv_block_bytes
                        + tc.layer_start * self._per_layer_bytes
                    )
                    raw_slice = raw_tensor[
                        buf_offset : buf_offset + tier_layer_bytes
                    ]

                    # File offset: K at 0, V at tier_layer_bytes
                    file_offset = kv_idx * tier_layer_bytes

                    save_ops.append(
                        (tier.engine, io_hash, raw_slice, path, file_offset)
                    )

            if self._deferred_writes_enabled:
                self._deferred_queue.append((group, save_ops))
                # Register hashes in the group map so drain can find
                # them if the drain thread runs before flush.
                for h in group.io_hashes:
                    self._hash_to_group[h] = group
            else:
                # Submit immediately
                for engine, io_hash, raw_slice, path, foff in save_ops:
                    engine.save(io_hash, raw_slice, path, foff)
                for h in group.io_hashes:
                    self._hash_to_group[h] = group

    def flush_deferred_writes(self) -> int:
        """Submit all deferred writes to the io_uring engines.

        Called after all reads for the current step complete, so writes
        never contend with reads on the NVMe.

        Returns:
            Number of logical chunks flushed.
        """
        if not self._deferred_queue:
            return 0

        self._drain_completed()

        n = 0
        for group, save_ops in self._deferred_queue:
            for engine, io_hash, raw_slice, path, foff in save_ops:
                engine.save(io_hash, raw_slice, path, foff)
            # Hashes are already registered in _hash_to_group during
            # batched_submit_put_task.
            n += 1

        self._deferred_queue.clear()
        logger.debug("KVStream: flushed %d deferred writes", n)
        return n

    # ------------------------------------------------------------------ #
    #  Get (read) path — P-tier layer-striped                             #
    # ------------------------------------------------------------------ #

    def _submit_tiered_loads(
        self,
        key: CacheEngineKey,
        meta: _TieredChunkMeta,
        raw_tensor: torch.Tensor,
    ) -> list[tuple[_TierState, str]]:
        """Submit read ops for all tier slices of a single chunk.

        Args:
            key: The cache engine key.
            meta: Tiered metadata for this key.
            raw_tensor: Destination buffer (flat uint8).

        Returns:
            List of ``(tier, io_hash)`` tuples for waiting.
        """
        load_refs: list[tuple[_TierState, str]] = []

        for tier_slice in meta.slices:
            tier = self._tiers[tier_slice.tier_index]
            tier_layer_bytes = (
                (tier_slice.layer_end - tier_slice.layer_start)
                * self._per_layer_bytes
            )

            for kv_idx in range(self._kv_size):
                kv_label = "k" if kv_idx == 0 else "v"
                io_hash = self._next_io_hash(
                    key, f"load_{kv_label}{tier_slice.tier_index}"
                )

                # Buffer offset within the raw tensor
                buf_offset = (
                    kv_idx * self._kv_block_bytes
                    + tier_slice.layer_start * self._per_layer_bytes
                )
                raw_slice = raw_tensor[
                    buf_offset : buf_offset + tier_layer_bytes
                ]

                # File offset
                file_offset = kv_idx * tier_layer_bytes

                tier.engine.load(
                    io_hash, raw_slice, tier_slice.path, file_offset
                )
                load_refs.append((tier, io_hash))

        return load_refs

    def get_blocking(
        self, key: CacheEngineKey
    ) -> Optional[MemoryObj]:
        """Blocking read of a KV chunk from disk via KVStream.

        Submits ``2 * P`` load ops across tiers and blocks until all
        complete.

        Args:
            key: The cache engine key.

        Returns:
            A ``MemoryObj`` with the loaded KV data, or ``None`` if the
            key does not exist.
        """
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None

        self.cache_policy.update_on_hit(key, self.dict)

        meta: _TieredChunkMeta = self.dict[key]
        shape = meta.shape
        dtype = meta.dtype
        fmt = meta.fmt
        assert dtype is not None
        assert shape is not None

        self.disk_lock.release()

        # Allocate destination memory
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, (
            "Memory allocation failed during KVStream disk load."
        )

        raw_tensor = memory_obj.raw_tensor
        assert raw_tensor is not None

        start_time = time.time()

        # Submit loads for all tiers
        load_refs = self._submit_tiered_loads(key, meta, raw_tensor)

        # Wait for all loads across all tiers
        for tier, io_hash in load_refs:
            tier.engine.wait_one(
                io_hash, self.kvstream_core.IOQueue.READ
            )

        elapsed = time.time() - start_time

        size = memory_obj.get_physical_size()
        if elapsed > 0:
            logger.debug(
                "KVStream blocking load: %d bytes, %.2f MB/s",
                size,
                size / elapsed / 1e6,
            )

        # Recover cached_positions metadata
        disk_meta = self.dict.get(key, None)
        if disk_meta is not None:
            memory_obj.metadata.cached_positions = (
                disk_meta.cached_positions
            )

        return memory_obj

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Batched blocking read using io_uring batch submission.

        Submits all reads across all tiers, then waits for completion
        on every tier engine.

        Args:
            keys: Ordered list of cache keys to load.

        Returns:
            List of ``MemoryObj`` (or ``None`` for missing keys), in
            the same order as *keys*.
        """
        mem_objs: List[Optional[MemoryObj]] = []
        valid_indices: list[int] = []
        # Track which tiers have outstanding reads
        tiers_with_reads: set[int] = set()

        for i, key in enumerate(keys):
            with self.disk_lock:
                if key not in self.dict:
                    mem_objs.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.dict)
                meta: _TieredChunkMeta = self.dict[key]
                shape = meta.shape
                dtype = meta.dtype
                fmt = meta.fmt

            assert dtype is not None
            assert shape is not None

            memory_obj = self.local_cpu_backend.allocate(
                shape, dtype, fmt
            )
            assert memory_obj is not None, (
                "Memory allocation failed during KVStream batched "
                "disk load."
            )

            raw_tensor = memory_obj.raw_tensor
            assert raw_tensor is not None

            # Submit loads for all tier slices
            load_refs = self._submit_tiered_loads(
                key, meta, raw_tensor
            )
            for tier, _hash in load_refs:
                tiers_with_reads.add(tier.config.index)

            mem_objs.append(memory_obj)
            valid_indices.append(i)

        if valid_indices:
            start_time = time.time()

            # Wait for all reads on every tier that has outstanding ops
            for tier_idx in tiers_with_reads:
                self._tiers[tier_idx].engine.wait_all(
                    self.kvstream_core.IOQueue.READ
                )

            elapsed = time.time() - start_time

            total_bytes = sum(
                mem_objs[i].get_physical_size()  # type: ignore[union-attr]
                for i in valid_indices
            )
            if elapsed > 0:
                logger.debug(
                    "KVStream batched_get_blocking: %d entries, "
                    "%d bytes, %.2f MB/s",
                    len(valid_indices),
                    total_bytes,
                    total_bytes / elapsed / 1e6,
                )

            # Recover cached_positions metadata
            for i in valid_indices:
                key = keys[i]
                disk_meta = self.dict.get(key, None)
                if disk_meta is not None:
                    mem_objs[i].metadata.cached_positions = (  # type: ignore[union-attr]
                        disk_meta.cached_positions
                    )

        return mem_objs

    # ------------------------------------------------------------------ #
    #  Overlapped read path (Level 2)                                      #
    # ------------------------------------------------------------------ #

    def submit_batch_load(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[tuple[str, CacheEngineKey, MemoryObj]]]:
        """Submit all reads without waiting (overlapped path).

        Pre-allocates ``MemoryObj`` instances, submits all tier loads,
        and returns immediately.  The returned ``io_hash`` is a
        synthetic group hash that can be passed to ``wait_one_load``.

        Args:
            keys: Ordered list of cache keys to load.

        Returns:
            A list with one entry per key: ``(group_hash, key,
            memory_obj)`` for valid keys, or ``None`` for missing keys.
        """
        results: List[
            Optional[tuple[str, CacheEngineKey, MemoryObj]]
        ] = []
        # Track (group_hash -> list of (tier, io_hash)) for wait_one

        for key in keys:
            with self.disk_lock:
                if key not in self.dict:
                    results.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.dict)
                meta: _TieredChunkMeta = self.dict[key]
                shape = meta.shape
                dtype = meta.dtype
                fmt = meta.fmt

            assert dtype is not None
            assert shape is not None

            memory_obj = self.local_cpu_backend.allocate(
                shape, dtype, fmt
            )
            assert memory_obj is not None, (
                "Memory allocation failed during KVStream overlapped "
                "load."
            )

            raw_tensor = memory_obj.raw_tensor
            assert raw_tensor is not None

            load_refs = self._submit_tiered_loads(
                key, meta, raw_tensor
            )

            # Create a synthetic group hash for the caller
            group_hash = self._next_io_hash(key, "load_group")
            self._overlapped_load_refs[group_hash] = load_refs

            results.append((group_hash, key, memory_obj))

        return results

    def wait_one_load(
        self,
        io_hash: str,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """Block until all read ops for one chunk complete.

        Args:
            io_hash: The synthetic group hash from ``submit_batch_load``.
            key: The cache engine key (used to recover metadata).
            memory_obj: The pre-allocated ``MemoryObj`` receiving data.
        """
        load_refs = self._overlapped_load_refs.pop(io_hash, [])
        for tier, sub_hash in load_refs:
            tier.engine.wait_one(
                sub_hash, self.kvstream_core.IOQueue.READ
            )

        # Recover cached_positions metadata
        disk_meta = self.dict.get(key, None)
        if disk_meta is not None:
            memory_obj.metadata.cached_positions = (
                disk_meta.cached_positions
            )

    def _sync_batch_load(
        self,
        io_hashes: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
    ) -> list[MemoryObj]:
        """Block until all KVStream reads complete (worker thread).

        Args:
            io_hashes: Synthetic group hashes (one per chunk).
            keys: Cache keys (one per chunk).
            memory_objs: Pre-allocated MemoryObjs to receive data.

        Returns:
            The same ``memory_objs`` list, now populated with data.
        """
        start_time = time.time()

        # Wait for all reads on all tiers
        for tier in self._tiers:
            tier.engine.wait_all(self.kvstream_core.IOQueue.READ)

        elapsed = time.time() - start_time

        total_bytes = sum(m.get_physical_size() for m in memory_objs)
        if elapsed > 0:
            logger.debug(
                "KVStream batch load: %d entries, %d bytes, %.2f MB/s",
                len(memory_objs),
                total_bytes,
                total_bytes / elapsed / 1e6,
            )

        # Bookkeeping: recover metadata, unpin disk entries
        for key, mem_obj in zip(keys, memory_objs, strict=False):
            disk_meta = self.dict.get(key, None)
            if disk_meta is not None:
                mem_obj.metadata.cached_positions = (
                    disk_meta.cached_positions
                )
                with self.disk_lock:
                    disk_meta.unpin()

        # Clean up overlapped refs
        for h in io_hashes:
            self._overlapped_load_refs.pop(h, None)

        return memory_objs

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Async batched read of KV chunks from disk.

        Pre-allocates all ``MemoryObj`` instances, submits tiered loads,
        then awaits completion on a background thread.

        Args:
            lookup_id: Opaque lookup identifier for logging/tracing.
            keys: Ordered list of cache keys to load.
            transfer_spec: Unused (interface compatibility).

        Returns:
            List of ``MemoryObj`` instances populated with loaded data.
        """
        mem_objs: list[MemoryObj] = []
        io_hashes: list[str] = []

        logger.debug(
            "lookup_id: %s; KVStream prefetching %d keys from disk.",
            lookup_id,
            len(keys),
        )

        for key in keys:
            self.disk_lock.acquire()
            assert key in self.dict, (
                f"Key {key} not found in KVStream disk cache after "
                f"pinning"
            )

            meta: _TieredChunkMeta = self.dict[key]
            shape = meta.shape
            dtype = meta.dtype
            fmt = meta.fmt

            assert dtype is not None
            assert shape is not None

            _alloc_t0 = time.time()
            memory_obj = self.local_cpu_backend.allocate(
                shape, dtype, fmt
            )
            _alloc_elapsed = time.time() - _alloc_t0
            if _alloc_elapsed > 0.01:
                logger.warning(
                    "D2H allocate for KVStream disk read took %.3fs "
                    "(key=%s)",
                    _alloc_elapsed,
                    key,
                )
            assert memory_obj is not None, (
                "Memory allocation failed during async KVStream "
                "disk load."
            )

            meta.pin()
            self.cache_policy.update_on_hit(key, self.dict)

            self.disk_lock.release()

            memory_obj.pin()
            mem_objs.append(memory_obj)

            raw_tensor = memory_obj.raw_tensor
            assert raw_tensor is not None

            load_refs = self._submit_tiered_loads(
                key, meta, raw_tensor
            )

            # Create a synthetic group hash
            group_hash = self._next_io_hash(key, "load_group")
            self._overlapped_load_refs[group_hash] = load_refs
            io_hashes.append(group_hash)

        # Wait for completion on a background thread
        return await asyncio.to_thread(
            self._sync_batch_load, io_hashes, keys, mem_objs
        )

    # ------------------------------------------------------------------ #
    #  Allocator / lifecycle                                               #
    # ------------------------------------------------------------------ #

    def get_allocator_backend(self) -> LocalCPUBackend:
        """Return the CPU memory allocator backend used for reads.

        Returns:
            The ``LocalCPUBackend`` instance.
        """
        return self.local_cpu_backend

    def close(self) -> None:
        """Shut down all KVStream tier engines and flush pending work.

        1. Flushes deferred writes.
        2. Stops the background drain thread.
        3. Calls ``engine.shutdown()`` on every tier.
        4. Final ``_drain_completed()`` for bookkeeping.
        5. Closes the batched message sender.
        """
        # Flush any deferred writes
        n_flushed = self.flush_deferred_writes()
        if n_flushed:
            logger.info(
                "KVStream: flushed %d deferred writes during close",
                n_flushed,
            )

        # Stop background drain thread
        self._drain_stop.set()
        self._drain_thread.join(timeout=2.0)
        if self._drain_thread.is_alive():
            logger.warning(
                "KVStream drain thread did not stop within 2s"
            )

        # Shutdown all tier engines
        for tier in self._tiers:
            tier.engine.shutdown()

        self._drain_completed()

        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()

        logger.info(
            "KVStreamDiskBackend closed (%d tiers).", len(self._tiers)
        )
