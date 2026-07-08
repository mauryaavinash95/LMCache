# SPDX-License-Identifier: Apache-2.0
"""KVStream block-replicated disk backend for LMCache.

Simplified block-level replicated storage across NVMe + PFS.  Each
chunk is treated as a single opaque blob (all layers, K+V combined)
and replicated in full to both tiers.  Reads use dual-cursor
front/back partitioning with dynamic work-stealing: NVMe reads from
the front of the work list, PFS from the back, advancing toward each
other.  The first tier to complete a block wins; the other tier's
in-flight read is harmlessly discarded (same buffer, identical data).

Activated when ``kvstream_placement = "block_replicated"`` and
``config.kvstream_enable`` is ``True``.
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
#  Module-level helpers                                                      #
# ======================================================================== #


def _union_busy_us(intervals: list[tuple[float, float]]) -> float:
    """Return the total wall-time (us) covered by the union of intervals.

    Overlapping/adjacent intervals are merged so shared time is counted
    once.  Used to turn per-op (submission_us, completion_us) samples into
    a device "busy time" and, via inclusion-exclusion, read/write overlap.

    Args:
        intervals: List of (start_us, end_us) pairs.

    Returns:
        Union busy time in microseconds.
    """
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    total += cur_e - cur_s
    return total


# ======================================================================== #
#  Data model (private to this module)                                      #
# ======================================================================== #


@dataclass
class _BlockMeta:
    """Per-key metadata for a (possibly partially) replicated chunk.

    Stored in ``self.dict`` (the cache policy's mutable mapping).
    Implements ``pin``/``unpin``/``can_evict`` for cache policy
    compatibility.

    ``tiers`` holds the tier indices that physically store this chunk
    (e.g. ``[0]`` for NVMe-only, ``[1]`` for PFS-only, ``[0, 1]`` for a
    replicated/band chunk).  ``tier_paths`` is index-aligned with
    ``tiers`` (one path per held tier).  Under full replication
    (``delta_band >= 1.0``) every chunk holds all tiers, reproducing the
    original behaviour.
    """

    tier_paths: list[str]  # paths for held tiers, index-aligned with `tiers`
    tiers: list[int]  # tier indices that physically hold this chunk
    chunk_bytes: int  # total raw tensor bytes
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    cached_positions: Optional[torch.Tensor] = None
    pin_count: int = 0

    @property
    def size(self) -> int:
        """Alias matching ``DiskCacheMetadata.size``."""
        return self.chunk_bytes

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
class _WriteGroup:
    """Tracks in-flight writes for one chunk across both tiers."""

    key: CacheEngineKey
    memory_obj: MemoryObj
    remaining: int  # starts at 2 (one save per tier)
    meta: _BlockMeta
    on_complete_callback: Optional[Callable[[CacheEngineKey], None]]
    io_hashes: list[str] = field(default_factory=list)
    failed: bool = False


# ======================================================================== #
#  Main backend class                                                       #
# ======================================================================== #


class KVStreamBlockReplicatedBackend(StorageBackendInterface):
    """Block-replicated disk backend using KVStream (io_uring).

    Replicates full chunks to both NVMe and PFS.  Reads use
    dual-cursor front/back partitioning with dynamic work-stealing.

    Requires exactly 2 tiers (tier 0 = NVMe, tier 1 = PFS).
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
        """Initialize the block-replicated KVStream backend.

        Args:
            config: LMCache engine configuration.
            loop: The asyncio event loop.
            local_cpu_backend: CPU memory allocator backend.
            dst_device: Target device string (e.g. ``"cuda:0"``).
            lmcache_worker: Optional cache controller worker.
            metadata: LMCache metadata (required — provides model
                geometry for chunk size computation).
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
            logger.info(
                "Created KVStream disk cache directory: %s", self.path
            )

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

        # -- Unique I/O hash counter -------------------------------------
        self._hash_counter: int = 0
        self._counter_lock = threading.Lock()

        # -- Extra config ------------------------------------------------
        extra = config.extra_config or {}

        # -- Global defaults for engine params ---------------------------
        global_read_chunk_kb: int = int(
            extra.get("kvstream_read_chunk_size_kb", 4096)
        )
        global_read_qd: int = int(
            extra.get("kvstream_read_queue_depth", 128)
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
                "KVStreamBlockReplicatedBackend."
            )

        # -- Per-tier locking mode ---------------------------------------
        locking_str: str = str(
            extra.get("kvstream_tier_locking", "none")
        )
        locking_modes = [
            m.strip().lower() for m in locking_str.split(":")
        ]
        while len(locking_modes) < 2:
            locking_modes.append(locking_modes[-1])
        for lm in locking_modes:
            if lm not in ("rw", "none"):
                raise ValueError(
                    f"kvstream_tier_locking: unknown mode '{lm}'. "
                    "Valid values: 'rw', 'none'"
                )

        # -- Model geometry (required) -----------------------------------
        if metadata is None:
            raise ValueError(
                "KVStreamBlockReplicatedBackend requires "
                "LMCacheMetadata for chunk size computation."
            )
        self._num_layers: int = metadata.kv_shape[0]
        self._chunk_size: int = metadata.chunk_size
        kv_size: int = 1 if metadata.use_mla else 2
        hidden_dim: int = metadata.kv_shape[3] * metadata.kv_shape[4]
        dtype_size: int = metadata.kv_dtype.itemsize

        self._per_layer_bytes: int = (
            self._chunk_size * hidden_dim * dtype_size
        )
        self._kv_block_bytes: int = (
            self._num_layers * self._per_layer_bytes
        )
        self._kv_size: int = kv_size
        # Total bytes for one chunk (entire raw_tensor blob)
        self._chunk_bytes: int = self._kv_size * self._kv_block_bytes

        # -- Tier 1 (PFS) path ------------------------------------------
        # Empty path => single-tier mode (NVMe only). Used by ablation
        # variants that isolate the contribution of multi-path replication.
        pfs_path: str = str(extra.get("kvstream_tier_1_path", ""))
        self._num_tiers: int = 2 if pfs_path else 1
        if pfs_path and not os.path.exists(pfs_path):
            os.makedirs(pfs_path)
            logger.info(
                "Created KVStream PFS tier directory: %s", pfs_path
            )

        # -- Build tier engines (1 or 2 depending on PFS availability) ---
        if self._num_tiers == 2:
            self._tier_paths: list[str] = [self.path, pfs_path]
        else:
            self._tier_paths = [self.path]
        self._tier_locking: list[str] = locking_modes[:self._num_tiers]
        self._tier_engines: list[Any] = []

        for i in range(self._num_tiers):
            read_chunk_kb = int(
                extra.get(
                    f"kvstream_tier_{i}_read_chunk_size_kb",
                    global_read_chunk_kb,
                )
            )
            read_qd = int(
                extra.get(
                    f"kvstream_tier_{i}_read_queue_depth",
                    global_read_qd,
                )
            )
            write_qd = int(
                extra.get(
                    f"kvstream_tier_{i}_write_queue_depth",
                    global_write_qd,
                )
            )
            write_chunk_kb = int(
                extra.get(
                    f"kvstream_tier_{i}_write_chunk_size_kb",
                    global_write_chunk_kb,
                )
            )
            max_fds = int(
                extra.get(f"kvstream_tier_{i}_max_fds", global_max_fds)
            )
            max_retries = int(
                extra.get(
                    f"kvstream_tier_{i}_max_retries", global_max_retries
                )
            )
            try_odirect = bool(
                extra.get(
                    f"kvstream_tier_{i}_try_odirect", global_try_odirect
                )
            )

            engine = kvstream_core.KVStream(
                read_chunk_size_kb=read_chunk_kb,
                read_queue_depth=read_qd,
                write_queue_depth=write_qd,
                write_chunk_size_kb=write_chunk_kb,
                max_fds_open=max_fds,
                try_using_odirect=try_odirect,
                max_retries=max_retries,
            )
            self._tier_engines.append(engine)

            tier_label = "NVMe" if i == 0 else "PFS"
            logger.info(
                "KVStream %s tier %d: path=%s, "
                "read_chunk_kb=%d, read_qd=%d, write_qd=%d, "
                "write_chunk_kb=%d, max_fds=%d, "
                "try_odirect=%s, locking=%s",
                tier_label,
                i,
                self._tier_paths[i],
                read_chunk_kb,
                read_qd,
                write_qd,
                write_chunk_kb,
                max_fds,
                try_odirect,
                self._tier_locking[i],
            )

        # Convenience alias
        self.engine = self._tier_engines[0]

        # -- Work-stealing config ----------------------------------------
        self._steal_batch: int = int(
            extra.get("kvstream_steal_batch_size", 4)
        )

        # -- Static partition mode (no work stealing) --------------------
        # When steal_batch < 0, the two tiers operate on disjoint, fixed
        # index ranges defined by kvstream_split_ratios. This isolates
        # the contribution of adaptive work-stealing for ablation.
        self._static_partition: bool = self._steal_batch < 0
        ratios_str: str = str(extra.get("kvstream_split_ratios", ""))
        if ratios_str:
            try:
                parsed = [float(x) for x in ratios_str.split(":")]
            except ValueError as e:
                raise ValueError(
                    f"kvstream_split_ratios must be colon-separated "
                    f"floats, got '{ratios_str}'"
                ) from e
            if abs(sum(parsed) - 1.0) > 1e-6:
                raise ValueError(
                    f"kvstream_split_ratios must sum to 1.0, "
                    f"got {sum(parsed)}: {ratios_str}"
                )
            self._split_ratios: list[float] = parsed
        elif self._num_tiers == 1:
            self._split_ratios = [1.0]
        else:
            # Default 3:1 from Table II NVMe:PFS read bandwidth ratio.
            self._split_ratios = [0.75, 0.25]

        if (self._static_partition
                and self._num_tiers != len(self._split_ratios)):
            raise ValueError(
                f"kvstream_split_ratios length "
                f"({len(self._split_ratios)}) must match num_tiers "
                f"({self._num_tiers})"
            )

        # -- Partial (delta-band) replication --------------------------------
        # delta_band is the fraction of each prefix that is replicated on
        # BOTH tiers (the stealable "band" straddling the NVMe:PFS pivot).
        #   delta_band >= 1.0 -> full replication (default; original behaviour)
        #   delta_band == 0.0 -> single placement (front->NVMe, back->PFS)
        #   0 < delta_band < 1 -> band replication around the pivot
        # The pivot is split_ratios[0] (NVMe's share of the prefix).
        # Placement uses a per-chunk fractional prefix position supplied by
        # the caller (placement_hints); if absent, we fall back to full
        # replication so correctness never depends on the hint.
        self._delta_band: float = float(
            extra.get("kvstream_delta_band", 1.0)
        )
        if self._delta_band < 0.0:
            self._delta_band = 0.0
        if self._delta_band > 1.0:
            self._delta_band = 1.0
        self._placement_pivot: float = (
            self._split_ratios[0] if self._num_tiers > 1 else 1.0
        )
        # Invert exclusive placement: front-of-prefix -> PFS, tail -> NVMe.
        # Motivated by CPU caching absorbing the hot front chunks, so the
        # disk-miss stream skews to the tail; putting the tail on the fast
        # tier (NVMe) aligns disk demand with bandwidth. Band still on both.
        self._placement_invert: bool = bool(
            extra.get("kvstream_placement_invert", False)
        )
        # Warn once if hints are missing while band placement is requested.
        self._placement_hint_warned: bool = False

        # -- Per-op timeline recording (overlap analysis) --------------------
        self._timeline_enabled: bool = bool(
            extra.get("kvstream_timeline_enabled", False)
        )
        if self._timeline_enabled:
            # Degrade gracefully if kvstream_core predates the timeline API
            # (avoids a hard crash when the native module isn't rebuilt).
            if all(
                hasattr(e, "set_timeline_enabled") for e in self._tier_engines
            ):
                for engine in self._tier_engines:
                    engine.set_timeline_enabled(True)
            else:
                logger.warning(
                    "kvstream_core lacks set_timeline_enabled; rebuild "
                    "kvstream to enable timeline recording. Disabling."
                )
                self._timeline_enabled = False

        # Opt in to receiving per-chunk placement hints from the storage
        # manager (used by delta-band placement).
        self.supports_placement_hints: bool = True

        # -- Per-step measurement accumulators (reset each step) -------------
        # Host-buffer allocate (incl. CPU-cache eviction) wait on the read
        # path, and write-amplification counters for delta-band placement.
        self._read_alloc_wait_us: float = 0.0
        self._chunks_written: int = 0     # logical chunks written
        self._tier_writes: int = 0        # physical (per-tier) writes

        # Cursor boundaries (re-computed per call in submit_batch_load).
        # In dynamic mode they cover the full range; in static mode they
        # encode the partition pivot.
        self._tier_0_end: int = 0
        self._tier_1_start: int = 0

        # -- Inflight write tracking (grouped drain) ---------------------
        self._hash_to_write_group: dict[str, _WriteGroup] = {}
        self._drain_lock = threading.Lock()

        # -- Overlapped read tracking (per-step) -------------------------
        # Populated by submit_batch_load, consumed by wait_any_load.
        # work_items: ordered list of (group_hash, key, memory_obj, meta)
        self._work_items: list[
            tuple[str, CacheEngineKey, MemoryObj, _BlockMeta]
        ] = []
        # io_hash -> (work_items_index, tier_index)
        self._io_hash_to_block: dict[str, tuple[int, int]] = {}
        # Completed block indices (first-wins semantics)
        self._done_blocks: set[int] = set()
        # Per-tier inflight read count
        self._tier_inflight: list[int] = [0] * self._num_tiers
        # Dual cursors
        self._nvme_cursor: int = 0
        self._pfs_cursor: int = -1
        # Total blocks pending (not yet done) for current step
        self._pending_count: int = 0

        # -- Batched message sender --------------------------------------
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
                "KVStreamBlockReplicatedBackend: controller message "
                "sender not initialized"
            )

        # -- Background drain thread -------------------------------------
        self._drain_stop = threading.Event()
        drain_interval: float = float(
            extra.get("kvstream_drain_poll_interval_s", 0.05)
        )
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            args=(drain_interval,),
            name="kvstream-block-drain",
            daemon=True,
        )
        self._drain_thread.start()

        # -- Deferred write support --------------------------------------
        self._deferred_writes_enabled: bool = bool(
            extra.get("kvstream_deferred_writes", True)
        )
        # Each entry: (_WriteGroup, list of (engine, io_hash, tensor, path))
        self._deferred_queue: list[
            tuple[_WriteGroup, list[tuple[Any, str, torch.Tensor, str]]]
        ] = []

        logger.info(
            "KVStreamBlockReplicatedBackend initialized: "
            "placement=block_replicated, num_tiers=%d, "
            "chunk_bytes=%d, num_layers=%d, kv_size=%d, "
            "steal_batch=%d, deferred_writes=%s, "
            "static_partition=%s, split_ratios=%s, "
            "delta_band=%.3f, placement_pivot=%.3f, "
            "placement_invert=%s, "
            "timeline_enabled=%s, drain_interval=%.3fs",
            self._num_tiers,
            self._chunk_bytes,
            self._num_layers,
            self._kv_size,
            self._steal_batch,
            self._deferred_writes_enabled,
            self._static_partition,
            self._split_ratios,
            self._delta_band,
            self._placement_pivot,
            self._placement_invert,
            self._timeline_enabled,
            drain_interval,
        )

    # ------------------------------------------------------------------ #
    #  String / helpers                                                    #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        """Return backend name (matches KVStreamDiskBackend for compat)."""
        return "KVStreamDiskBackend"

    def _key_to_path(self, key: CacheEngineKey, tier_idx: int) -> str:
        """Convert a cache key to a file path on the given tier.

        Args:
            key: The cache engine key.
            tier_idx: Tier index (0 = NVMe, 1 = PFS).

        Returns:
            Absolute path for the cached file on this tier.
        """
        return os.path.join(
            self._tier_paths[tier_idx],
            key.to_string().replace("/", "-") + ".bin",
        )

    def _next_io_hash(self, key: CacheEngineKey, op: str) -> str:
        """Generate a globally unique hash for a KVStream I/O operation.

        Args:
            key: The cache engine key.
            op: Operation prefix (e.g. ``"save_t0"``, ``"load_t1"``).

        Returns:
            A unique hash string.
        """
        with self._counter_lock:
            self._hash_counter += 1
            return f"{op}:{key.to_string()}:{self._hash_counter}"

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

        Args:
            lookup_id: Opaque lookup identifier.
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

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin *key* to prevent eviction.

        Args:
            key: The cache engine key.

        Returns:
            ``True`` if the key exists and was pinned.
        """
        with self.disk_lock:
            if key not in self.dict:
                return False
            self.dict[key].pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin *key* to allow eviction.

        Args:
            key: The cache engine key.

        Returns:
            ``True`` if the key exists and was unpinned.
        """
        with self.disk_lock:
            if key not in self.dict:
                return False
            self.dict[key].unpin()
            return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove *key* from the cache (no-op — no evictions).

        Args:
            key: The cache engine key.
            force: Whether this is a forced removal.

        Returns:
            ``True`` if the key existed and was removed.
        """
        with self.disk_lock:
            meta = self.dict.pop(key, None)
            if meta is None:
                return False
        return True

    def get_allocator_backend(self) -> LocalCPUBackend:
        """Return the CPU allocator backend used for read buffers.

        Returns:
            The ``LocalCPUBackend`` instance.
        """
        return self.local_cpu_backend

    def get_blocking(
        self, key: CacheEngineKey
    ) -> Optional[MemoryObj]:
        """Blocking single-key read (fallback path).

        Args:
            key: The cache engine key.

        Returns:
            ``MemoryObj`` with loaded data, or ``None`` if not found.
        """
        with self.disk_lock:
            if key not in self.dict:
                return None
            self.cache_policy.update_on_hit(key, self.dict)
            meta: _BlockMeta = self.dict[key]

        memory_obj = self.local_cpu_backend.allocate(
            meta.shape, meta.dtype, meta.fmt
        )
        assert memory_obj is not None
        raw_tensor = memory_obj.raw_tensor
        assert raw_tensor is not None

        # Read from the first tier that physically holds this chunk.
        ti0 = meta.tiers[0]
        io_hash = self._next_io_hash(key, f"get_blocking_t{ti0}")
        io_queue_read = self.kvstream_core.IOQueue.READ
        self._tier_engines[ti0].load(
            io_hash, raw_tensor, meta.tier_paths[0], 0
        )
        self._tier_engines[ti0].wait_one(io_hash, io_queue_read)

        if meta.cached_positions is not None:
            memory_obj.metadata.cached_positions = (
                meta.cached_positions
            )

        return memory_obj

    # ------------------------------------------------------------------ #
    #  Put-task tracking helpers                                            #
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
    #  Write completion bookkeeping                                        #
    # ------------------------------------------------------------------ #

    @_lmcache_nvtx_annotate
    def _drain_completed(self) -> None:
        """Poll both tier engines for completed and failed writes.

        Each chunk has 2 I/O hashes (one per tier).  The buffer is
        released and the key is registered only when both complete.
        """
        with self._drain_lock:
            all_completed: list[str] = []
            all_failed: list[str] = []
            for engine in self._tier_engines:
                all_completed.extend(engine.drain_completed())
                all_failed.extend(engine.drain_failed())

            for io_hash in all_failed:
                group = self._hash_to_write_group.pop(io_hash, None)
                if group is None:
                    continue
                group.failed = True
                group.remaining -= 1
                if group.remaining == 0:
                    self._resolve_failed_write(group)

            for io_hash in all_completed:
                group = self._hash_to_write_group.pop(io_hash, None)
                if group is None:
                    continue
                group.remaining -= 1
                if group.remaining == 0:
                    if group.failed:
                        self._resolve_failed_write(group)
                    else:
                        self._resolve_completed_write(group)

    def _resolve_completed_write(self, group: _WriteGroup) -> None:
        """Finalize a successfully completed write group.

        Args:
            group: The completed write group.
        """
        self.usage += group.meta.chunk_bytes
        self.stats_monitor.update_local_storage_usage(self.usage)

        # Release buffer before registering key
        group.memory_obj.ref_count_down()

        has_stored = False
        with self.disk_lock:
            if group.key in self.dict:
                self.cache_policy.update_on_hit(group.key, self.dict)
                has_stored = True
            else:
                self.dict[group.key] = group.meta

        self._remove_put_task(group.key)

        if self.batched_msg_sender is not None and not has_stored:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.ADMIT,
                key=group.key.chunk_hash,
            )

        if group.on_complete_callback is not None:
            try:
                group.on_complete_callback(group.key)
            except Exception as e:
                logger.warning(
                    "on_complete_callback failed for key %s: %s",
                    group.key,
                    e,
                )

    def _resolve_failed_write(self, group: _WriteGroup) -> None:
        """Finalize a failed write group.

        Args:
            group: The failed write group.
        """
        logger.error(
            "KVStream: write failed for key %s", group.key
        )
        group.memory_obj.ref_count_down()
        self._remove_put_task(group.key)

        with self.disk_lock:
            self.current_cache_size -= group.meta.chunk_bytes

        # Best-effort cleanup of partial files
        for path in group.meta.tier_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        # Remove any remaining hashes for this group
        for h in group.io_hashes:
            self._hash_to_write_group.pop(h, None)

    # ------------------------------------------------------------------ #
    #  Write path                                                          #
    # ------------------------------------------------------------------ #

    def _placement_tiers(self, hint: Optional[float]) -> list[int]:
        """Decide which tier indices physically store a chunk.

        Implements delta-band partial replication: front-of-prefix
        chunks go to NVMe (tier 0), tail-of-prefix chunks to PFS
        (tier 1), and a band of width ``delta_band`` straddling the
        pivot is replicated on both (the stealable set).

        Args:
            hint: Fractional position of the chunk within its prefix in
                ``[0, 1]`` (0 = first chunk, 1 = last), or ``None`` if
                unknown.

        Returns:
            Sorted list of tier indices that should store this chunk.
        """
        # Single-tier deployments always use tier 0.
        if self._num_tiers == 1:
            return [0]
        # Full replication (default) or missing hint -> replicate on all.
        if self._delta_band >= 1.0 or hint is None:
            if (
                hint is None
                and self._delta_band < 1.0
                and not self._placement_hint_warned
            ):
                logger.warning(
                    "KVStream delta_band=%.3f requested but placement hint "
                    "is missing; falling back to full replication.",
                    self._delta_band,
                )
                self._placement_hint_warned = True
            return list(range(self._num_tiers))
        # Band placement around the pivot.
        lo = self._placement_pivot - self._delta_band / 2.0
        hi = self._placement_pivot + self._delta_band / 2.0
        p = 0.0 if hint < 0.0 else (1.0 if hint > 1.0 else hint)
        # Exclusive tiers for the front and tail regions. Default: front on
        # NVMe (tier 0), tail on PFS (tier 1). Inverted: front on PFS, tail
        # on NVMe (align the disk-miss-heavy tail with the fast tier).
        front_tier, tail_tier = (
            (1, 0) if self._placement_invert else (0, 1)
        )
        if p < lo:
            return [front_tier]
        if p >= hi:
            return [tail_tier]
        return [0, 1]  # replicated band (stealable)

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
        placement_hints: Optional[Sequence[float]] = None,
    ) -> None:
        """Submit a batch of KV chunks for async disk write.

        Each chunk is written to the tier(s) chosen by delta-band
        placement (``_placement_tiers``): NVMe-only, PFS-only, or both.
        Under full replication (``delta_band >= 1.0``, the default) every
        chunk is written to all tiers, reproducing the original
        behaviour.  Writes are optionally deferred until the next
        ``flush_deferred_writes()`` call.

        Args:
            keys: Cache keys for the KV chunks.
            objs: Memory objects containing the KV data.
            transfer_spec: Unused (interface compatibility).
            on_complete_callback: Optional callback invoked per key
                after that key's disk write completes on all held tiers.
            placement_hints: Optional per-chunk fractional prefix
                positions in ``[0, 1]`` (index-aligned with ``keys``),
                used to drive delta-band placement.  ``None`` => full
                replication (safe default).
        """
        self._drain_completed()

        for idx, (key, memory_obj) in enumerate(
            zip(keys, objs, strict=False)
        ):
            assert memory_obj.tensor is not None

            if self.exists_in_put_tasks(key):
                logger.debug(
                    "Put task for %s is already in progress.", key
                )
                continue

            self._insert_put_task(key)

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
            memory_obj.ref_count_up()

            raw_tensor = memory_obj.raw_tensor
            assert raw_tensor is not None
            shape = memory_obj.metadata.shape
            dtype = memory_obj.metadata.dtype
            fmt = memory_obj.metadata.fmt
            cached_positions = memory_obj.metadata.cached_positions

            # Decide physical placement (delta-band partial replication).
            hint: Optional[float] = None
            if placement_hints is not None and idx < len(placement_hints):
                hint = placement_hints[idx]
            held_tiers = self._placement_tiers(hint)
            self._chunks_written += 1
            self._tier_writes += len(held_tiers)

            # File paths for the held tiers (index-aligned with held_tiers)
            tier_paths = [
                self._key_to_path(key, ti) for ti in held_tiers
            ]
            meta = _BlockMeta(
                tier_paths=tier_paths,
                tiers=held_tiers,
                chunk_bytes=required_size,
                shape=shape,
                dtype=dtype,
                fmt=fmt,
                cached_positions=cached_positions,
            )

            group = _WriteGroup(
                key=key,
                memory_obj=memory_obj,
                remaining=len(held_tiers),  # one save per held tier
                meta=meta,
                on_complete_callback=on_complete_callback,
            )

            # Build save ops: one per held tier, full blob
            save_ops: list[tuple[Any, str, torch.Tensor, str]] = []
            for ti, path in zip(held_tiers, tier_paths):
                io_hash = self._next_io_hash(key, f"save_t{ti}")
                group.io_hashes.append(io_hash)
                save_ops.append((
                    self._tier_engines[ti],
                    io_hash,
                    raw_tensor,
                    path,
                ))

            if self._deferred_writes_enabled:
                self._deferred_queue.append((group, save_ops))
                for h in group.io_hashes:
                    self._hash_to_write_group[h] = group
            else:
                for engine, io_hash, tensor, path in save_ops:
                    torch.cuda.nvtx.range_push("kvs_save_imm")
                    engine.save(io_hash, tensor, path, 0)
                    torch.cuda.nvtx.range_pop()
                for h in group.io_hashes:
                    self._hash_to_write_group[h] = group

    @_lmcache_nvtx_annotate
    def flush_deferred_writes(self) -> int:
        """Submit all deferred writes to the io_uring engines.

        Called at the start of the next step, before reads begin.

        Returns:
            Number of logical chunks flushed.
        """
        if not self._deferred_queue:
            self._wait_rw_locked_tiers()
            return 0

        self._drain_completed()

        t0 = time.perf_counter()
        n = 0
        total_bytes = 0
        for group, save_ops in self._deferred_queue:
            for engine, io_hash, tensor, path in save_ops:
                torch.cuda.nvtx.range_push("kvs_save_flush")
                engine.save(io_hash, tensor, path, 0)
                torch.cuda.nvtx.range_pop()
                total_bytes += tensor.nbytes
            n += 1

        self._deferred_queue.clear()
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        logger.debug(
            "KVStream: flushed %d deferred writes, "
            "%.2f MB in %.2f ms",
            n,
            total_bytes / 1e6,
            elapsed_ms,
        )

        self._wait_rw_locked_tiers()
        return n

    def _wait_rw_locked_tiers(self) -> None:
        """Block until writes complete on rw-locked tiers."""
        io_queue_write = self.kvstream_core.IOQueue.WRITE
        for ti in range(self._num_tiers):
            if self._tier_locking[ti] != "rw":
                continue
            torch.cuda.nvtx.range_push(f"kvs_rw_wait_t{ti}")
            self._tier_engines[ti].wait_all(io_queue_write)
            torch.cuda.nvtx.range_pop()
            self._drain_completed()

    # ------------------------------------------------------------------ #
    #  Read path                                                           #
    # ------------------------------------------------------------------ #

    @_lmcache_nvtx_annotate
    def submit_batch_load(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[tuple[str, CacheEngineKey, MemoryObj]]]:
        """Submit all reads without waiting (overlapped path).

        Pre-allocates ``MemoryObj`` instances, sets up dual-cursor
        work distribution, and submits initial read batches to both
        NVMe and PFS engines.

        Args:
            keys: Ordered list of cache keys to load.

        Returns:
            A list with one entry per key: ``(group_hash, key,
            memory_obj)`` for valid keys, or ``None`` for missing.
        """
        results: List[
            Optional[tuple[str, CacheEngineKey, MemoryObj]]
        ] = []

        # Accumulate new work items (may be called multiple times
        # per step, once per request).
        base_idx = len(self._work_items)

        for key in keys:
            with self.disk_lock:
                if key not in self.dict:
                    results.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.dict)
                meta: _BlockMeta = self.dict[key]

            # Time the host-buffer allocate (includes CPU-cache eviction
            # wait / write-back-pressure stalls) on the read path.
            _t_alloc = time.perf_counter()
            memory_obj = self.local_cpu_backend.allocate(
                meta.shape, meta.dtype, meta.fmt
            )
            self._read_alloc_wait_us += (
                time.perf_counter() - _t_alloc
            ) * 1e6
            assert memory_obj is not None, (
                "Memory allocation failed during KVStream "
                "overlapped load."
            )
            raw_tensor = memory_obj.raw_tensor
            assert raw_tensor is not None

            group_hash = self._next_io_hash(key, "load_group")
            self._work_items.append(
                (group_hash, key, memory_obj, meta)
            )
            self.local_cpu_backend.submit_put_task(key, memory_obj)
            results.append((group_hash, key, memory_obj))

        # Submit new work items only if this call actually added any.
        total = len(self._work_items)
        new_items_added = total > base_idx

        if new_items_added:
            # Compute cursor boundaries.
            # Dynamic mode (work-stealing): tier 0 covers [0, total),
            # tier 1 covers (-, total) going backward — boundaries are
            # effectively the full range.
            # Static mode: tier 0 owns [0, n_pivot), tier 1 owns
            # [n_pivot, total). Cursors stop at the pivot.
            if self._static_partition and self._num_tiers > 1:
                n_pivot = int(total * self._split_ratios[0])
                self._tier_0_end = n_pivot
                self._tier_1_start = n_pivot
            else:
                self._tier_0_end = total
                self._tier_1_start = 0

            if self._num_tiers > 1:
                # Extend PFS cursor to cover newly added items (never
                # move it backward — earlier items may already be done
                # or in-flight from a prior call this step).
                self._pfs_cursor = max(self._pfs_cursor, total - 1)

            # Submit initial batch from each active tier. In static
            # mode, _steal_batch is negative; use its magnitude as the
            # initial batch size (no actual stealing will occur).
            steal = max(1, abs(self._steal_batch))
            for _ in range(steal):
                if not self._submit_next_block(0):
                    break
            if self._num_tiers > 1:
                for _ in range(steal):
                    if not self._submit_next_block(1):
                        break

        self._pending_count = total - len(self._done_blocks)

        return results

    def _item_on_tier(self, idx: int, tier_idx: int) -> bool:
        """Return whether work-item ``idx`` is physically stored on
        ``tier_idx`` (delta-band membership check).

        Args:
            idx: Work-item index.
            tier_idx: Storage tier index.

        Returns:
            ``True`` if the chunk at ``idx`` resides on ``tier_idx``.
        """
        return tier_idx in self._work_items[idx][3].tiers

    def _next_cursor_index(self, tier_idx: int) -> Optional[int]:
        """Advance the tier's cursor, skipping done and non-resident blocks.

        A tier only reads chunks it physically holds (delta-band
        membership): the NVMe cursor skips PFS-only chunks and vice
        versa.  Chunks replicated on both tiers form the stealable band
        where the two cursors meet.

        Args:
            tier_idx: 0 = NVMe (front, ascending),
                      1 = PFS (back, descending).

        Returns:
            Work item index, or ``None`` if exhausted, out of the
            tier's static-partition range, or the tier is inactive.
        """
        # Tier 1 has no work in single-tier mode.
        if tier_idx == 1 and self._num_tiers == 1:
            return None
        while True:
            if tier_idx == 0:
                idx = self._nvme_cursor
                # Stop at end-of-list OR at static-partition pivot.
                # In dynamic mode _tier_0_end == len(self._work_items).
                if (idx >= self._tier_0_end
                        or idx >= len(self._work_items)):
                    return None
                self._nvme_cursor += 1
            else:
                idx = self._pfs_cursor
                # Stop at start-of-list OR at static-partition pivot.
                # In dynamic mode _tier_1_start == 0.
                if idx < self._tier_1_start or idx < 0:
                    return None
                self._pfs_cursor -= 1
            # Skip already-served blocks and blocks not on this tier.
            if idx in self._done_blocks:
                continue
            if not self._item_on_tier(idx, tier_idx):
                continue
            return idx

    def _submit_next_block(self, tier_idx: int) -> bool:
        """Submit the next block from this tier's cursor.

        Args:
            tier_idx: 0 = NVMe, 1 = PFS.

        Returns:
            ``True`` if a block was submitted, ``False`` if cursor
            exhausted.
        """
        idx = self._next_cursor_index(tier_idx)
        if idx is None:
            return False

        group_hash, key, memory_obj, meta = self._work_items[idx]
        io_hash = self._next_io_hash(key, f"load_t{tier_idx}")
        # Path is deterministic from (key, tier); membership already
        # verified by _next_cursor_index so this tier holds the chunk.
        path = self._key_to_path(key, tier_idx)
        raw_tensor = memory_obj.raw_tensor
        assert raw_tensor is not None

        torch.cuda.nvtx.range_push(
            f"kvs_load_t{tier_idx}_{key.chunk_hash_hex[:8]}"
        )
        self._tier_engines[tier_idx].load(
            io_hash, raw_tensor, path, 0
        )
        torch.cuda.nvtx.range_pop()

        self._io_hash_to_block[io_hash] = (idx, tier_idx)
        self._tier_inflight[tier_idx] += 1
        return True

    @_lmcache_nvtx_annotate
    def wait_any_load(
        self,
    ) -> list[tuple[str, CacheEngineKey, MemoryObj]]:
        """Block until at least one chunk completes all I/O.

        Uses completion-driven reaping with work-stealing: when a
        tier finishes a block, the next block from that tier's cursor
        is immediately submitted.

        Returns:
            List of ``(group_hash, key, memory_obj)`` for every
            chunk that completed.
        """
        ready: list[tuple[str, CacheEngineKey, MemoryObj]] = []
        io_queue_read = self.kvstream_core.IOQueue.READ

        while not ready:
            # Phase 1: non-blocking drain from each active engine
            for ti in range(self._num_tiers):
                newly_done = self._tier_engines[ti].drain_completed_queue(
                    io_queue_read
                )
                if newly_done:
                    self._process_read_completions(newly_done, ready)

            if ready:
                break

            # Phase 2: block on a tier with inflight > 0
            blocked = False
            for ti in range(self._num_tiers):
                if self._tier_inflight[ti] > 0:
                    torch.cuda.nvtx.range_push(f"kvs_wait_any_t{ti}")
                    completed = self._tier_engines[
                        ti
                    ].wait_any_completed(io_queue_read)
                    torch.cuda.nvtx.range_pop()
                    self._process_read_completions(completed, ready)
                    blocked = True
                    break

            if not blocked:
                logger.error(
                    "wait_any_load: no tier has inflight reads but "
                    "%d blocks remain — breaking to avoid hang",
                    self._pending_count,
                )
                break

        # Clean up when all blocks are done
        if not self._pending_count or len(self._done_blocks) == len(
            self._work_items
        ):
            self._reset_read_state()

        return ready

    def _process_read_completions(
        self,
        completed_hashes: list[str],
        ready: list[tuple[str, CacheEngineKey, MemoryObj]],
    ) -> None:
        """Process completed read hashes and replenish cursors.

        For each completed hash, marks the block done if it's the
        first tier to finish, recovers metadata, and submits the
        next block from the completing tier's cursor.

        Args:
            completed_hashes: io_hashes that just completed.
            ready: Output list — ready chunks are appended here.
        """
        for io_hash in completed_hashes:
            block_info = self._io_hash_to_block.pop(io_hash, None)
            if block_info is None:
                continue
            block_idx, tier_idx = block_info
            self._tier_inflight[tier_idx] -= 1

            if block_idx not in self._done_blocks:
                # First tier to finish this block — it's ready
                self._done_blocks.add(block_idx)
                self._pending_count -= 1

                group_hash, key, memory_obj, meta = (
                    self._work_items[block_idx]
                )

                # Recover cached_positions
                disk_meta: Optional[_BlockMeta] = self.dict.get(
                    key, None
                )
                if disk_meta is not None:
                    memory_obj.metadata.cached_positions = (
                        disk_meta.cached_positions
                    )

                ready.append((group_hash, key, memory_obj))
            # else: other tier already completed this block — discard

            # Replenish: submit next block from this tier's cursor
            self._submit_next_block(tier_idx)

    def _reset_read_state(self) -> None:
        """Reset per-step read tracking state."""
        self._work_items.clear()
        self._io_hash_to_block.clear()
        self._done_blocks.clear()
        self._tier_inflight = [0] * self._num_tiers
        self._nvme_cursor = 0
        self._pfs_cursor = -1
        self._pending_count = 0
        # Cursor boundaries are re-computed in submit_batch_load.
        self._tier_0_end = 0
        self._tier_1_start = 0

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def get_tier_read_stats(self) -> list[dict[str, Any]]:
        """Snapshot per-tier read statistics (no reset).

        Returns:
            List of per-tier dicts with read stats from the C++
            io_uring engines.  Does NOT reset counters.
        """
        IOQueue = self.kvstream_core.IOQueue
        results: list[dict[str, Any]] = []
        for i, engine in enumerate(self._tier_engines):
            rs = engine.get_stats(IOQueue.READ)
            results.append({
                "tier": i,
                "read_bytes": rs.total_bytes_completed,
                "read_bw_mb_s": round(rs.bandwidth_mb_s(), 1),
                "read_elapsed_ms": round(
                    rs.elapsed_us() / 1e3, 2
                ),
            })
        return results

    def get_tier_stats_and_reset(self) -> list[dict[str, Any]]:
        """Get per-tier I/O stats and reset counters.

        Returns:
            List of stat dicts, one per tier.
        """
        IOQueue = self.kvstream_core.IOQueue
        results: list[dict[str, Any]] = []
        for i, engine in enumerate(self._tier_engines):
            rs = engine.get_stats(IOQueue.READ)
            ws = engine.get_stats(IOQueue.WRITE)
            results.append({
                "tier": i,
                "read_bytes": rs.total_bytes_completed,
                "read_ops": rs.total_ops_completed,
                "read_bw_mb_s": round(rs.bandwidth_mb_s(), 1),
                "read_elapsed_ms": round(rs.elapsed_us() / 1e3, 2),
                "read_retries": rs.total_retries,
                "write_bytes": ws.total_bytes_completed,
                "write_ops": ws.total_ops_completed,
                "write_bw_mb_s": round(ws.bandwidth_mb_s(), 1),
                "write_elapsed_ms": round(ws.elapsed_us() / 1e3, 2),
            })
            engine.reset_stats(IOQueue.READ)
            engine.reset_stats(IOQueue.WRITE)
        return results

    def get_write_backlog_stats(self) -> dict[str, Any]:
        """Snapshot deferred-write backlog and pinning pressure.

        Used by Phase-1 instrumentation to test the write-back-pressure
        hypothesis: slow (2x replicated) deferred flushes keep CPU
        chunks pinned (``ref_count_up`` held until *both* tier writes
        complete), which can stall the next step's store ``allocate()``.

        Returns:
            Dict with:
              - ``deferred_queue_len``: chunks queued but not yet
                submitted to io_uring.
              - ``inflight_write_groups``: chunks whose writes have been
                submitted but not yet completed on all tiers (each keeps
                its CPU ``memory_obj`` pinned).
              - ``pinned_mb``: aggregate bytes of CPU chunks pinned by
                in-flight writes.
              - ``put_tasks_len``: number of registered (not-yet-final)
                put tasks.
              - ``disk_fill``: current_cache_size / max_cache_size for
                the disk tier (1.0 => disk-tier eviction pressure).
              - ``disk_used_gb`` / ``disk_max_gb``: raw disk-tier sizes.
        """
        # Each in-flight chunk contributes ``_num_tiers`` io_hashes to
        # the map; divide to count logical chunks.
        n_io_hashes = len(self._hash_to_write_group)
        inflight_groups = (
            n_io_hashes // self._num_tiers if self._num_tiers else n_io_hashes
        )
        # Sum pinned bytes over the *unique* in-flight write groups.
        seen: set[int] = set()
        pinned_bytes = 0
        for group in self._hash_to_write_group.values():
            gid = id(group)
            if gid in seen:
                continue
            seen.add(gid)
            pinned_bytes += group.meta.chunk_bytes
        with self.put_tasks_lock:
            put_tasks_len = len(self.put_tasks)
        disk_fill = (
            self.current_cache_size / self.max_cache_size
            if self.max_cache_size > 0
            else 0.0
        )
        return {
            "deferred_queue_len": len(self._deferred_queue),
            "inflight_write_groups": inflight_groups,
            "pinned_mb": round(pinned_bytes / 1e6, 1),
            "put_tasks_len": put_tasks_len,
            "disk_fill": round(disk_fill, 4),
            "disk_used_gb": round(self.current_cache_size / 1024**3, 3),
            "disk_max_gb": round(self.max_cache_size / 1024**3, 3),
        }

    def get_io_timeline_and_reset(self) -> dict[str, Any]:
        """Compute overlap-aware read/write timing and reset accumulators.

        Drains each tier engine's per-op (submission_us, completion_us)
        timeline and derives, per tier, the read-busy time, write-busy
        time, and their temporal overlap (read/write contention window)
        via inclusion-exclusion.  Also returns the host-buffer allocate
        wait accrued on the read path and the effective write
        amplification (physical tier-writes / logical chunks) since the
        last call.

        This is the primary overlap-aware instrumentation: it reveals how
        much of a step's read window ran concurrently with (spilled)
        writes, which sequential phase timers cannot capture.

        Returns:
            Dict with ``per_tier`` list and step-level counters.  Empty
            ``per_tier`` when timeline recording is disabled.
        """
        per_tier: list[dict[str, Any]] = []
        if self._timeline_enabled:
            IOQueue = self.kvstream_core.IOQueue
            for i, engine in enumerate(self._tier_engines):
                reads = engine.drain_timeline(IOQueue.READ)
                writes = engine.drain_timeline(IOQueue.WRITE)
                ur = _union_busy_us(reads)
                uw = _union_busy_us(writes)
                uall = _union_busy_us(list(reads) + list(writes))
                overlap = max(0.0, ur + uw - uall)
                per_tier.append({
                    "tier": i,
                    "read_busy_ms": round(ur / 1e3, 2),
                    "write_busy_ms": round(uw / 1e3, 2),
                    "rw_overlap_ms": round(overlap / 1e3, 2),
                    "n_read_ops": len(reads),
                    "n_write_ops": len(writes),
                })

        waf = (
            self._tier_writes / self._chunks_written
            if self._chunks_written > 0
            else 0.0
        )
        result = {
            "per_tier": per_tier,
            "read_alloc_wait_ms": round(self._read_alloc_wait_us / 1e3, 2),
            "chunks_written": self._chunks_written,
            "tier_writes": self._tier_writes,
            "write_amplification": round(waf, 3),
        }

        # Reset step-level accumulators.
        self._read_alloc_wait_us = 0.0
        self._chunks_written = 0
        self._tier_writes = 0
        return result

    # ------------------------------------------------------------------ #
    #  Close / lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Shut down engines and flush pending work."""
        n_flushed = self.flush_deferred_writes()
        if n_flushed:
            logger.info(
                "KVStream: flushed %d deferred writes during close",
                n_flushed,
            )

        self._drain_stop.set()
        self._drain_thread.join(timeout=2.0)
        if self._drain_thread.is_alive():
            logger.warning(
                "KVStream drain thread did not stop within 2s"
            )

        for engine in self._tier_engines:
            engine.shutdown()

        self._drain_completed()

        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()

        logger.info(
            "KVStreamBlockReplicatedBackend closed (%d tier(s)).",
            self._num_tiers,
        )
