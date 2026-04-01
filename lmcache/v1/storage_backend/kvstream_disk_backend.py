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
    locking: str  # "rw" or "none"
    slab_size_mb: int  # 0 = file-per-chunk, >0 = slab aggregation

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
    """Tracks all sub-write I/O ops for one logical chunk.

    In layer_stripe mode a chunk produces ``2 * P`` I/O hashes
    (K-block + V-block per tier).  In whole_chunk mode a chunk goes
    to one tier, producing ``2`` hashes (K + V).  The group is
    resolved when ``remaining`` reaches zero.
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
    # Pre-built tier slices for slab mode.  When set,
    # _resolve_completed_group uses these instead of calling
    # _build_tier_slices (which cannot reconstruct slab offsets).
    tier_slices: Optional[list[_TierSlice]] = None


@dataclass
class _WorkStealState:
    """Per-step work-stealing state for replicated sub-chunk reads.

    NVMe steals from the front (index 0 upward), PFS steals from the
    back (index M-1 downward).  When cursors cross, both tiers may
    read the same item — this is safe because the data is identical
    and writes to the same buffer region.

    Each "item" corresponds to one chunk's replicated sub-chunk
    (the layer range that exists on BOTH NVMe and PFS).  An item
    requires ``kv_size`` sub-hash completions (K + V) from ANY
    single tier before it is considered done.
    """

    # (group_hash, key, meta, raw_tensor, layer_start, layer_end)
    items: list[tuple]

    nvme_cursor: int = 0    # next item for NVMe (front, increasing)
    pfs_cursor: int = -1    # next item for PFS (back, decreasing)

    nvme_inflight: int = 0
    pfs_inflight: int = 0
    max_inflight_per_tier: int = 2

    # sub_hash -> item index
    sub_to_item: dict[str, int] = field(default_factory=dict)
    # item index -> count of completed sub-hashes (need kv_size)
    item_completions: dict[int, int] = field(default_factory=dict)
    # items fully resolved (first tier to complete kv_size wins)
    done_items: set[int] = field(default_factory=set)
    # Track which items each tier has already submitted reads for,
    # to avoid double-submission when NVMe re-steals PFS items.
    nvme_submitted: set[int] = field(default_factory=set)
    pfs_submitted: set[int] = field(default_factory=set)


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

        # Per-tier locking mode: "rw" = wait for writes to complete
        # before reads start (prevents NVMe read/write contention);
        # "none" = no synchronisation.  Default is "none" for all tiers.
        locking_str: str = str(
            extra.get("kvstream_tier_locking", "none")
        )
        locking_modes = [
            m.strip().lower()
            for m in locking_str.split(":")
        ]
        # Extend to num_tiers if fewer values given (last value repeats)
        while len(locking_modes) < num_tiers:
            locking_modes.append(locking_modes[-1])
        for lm in locking_modes:
            if lm not in ("rw", "none"):
                raise ValueError(
                    f"kvstream_tier_locking: unknown mode '{lm}'. "
                    "Valid values: 'rw', 'none'"
                )

        # Per-tier slab aggregation: 0 = file-per-chunk (default),
        # >0 = slab files of that size in MB.  Colon-separated for
        # per-tier control, e.g. "0:512" = file-per-chunk on tier 0,
        # 512 MB slabs on tier 1.
        slab_str: str = str(
            extra.get("kvstream_slab_size_mb", "0")
        )
        slab_sizes = [int(s.strip()) for s in slab_str.split(":")]
        while len(slab_sizes) < num_tiers:
            slab_sizes.append(slab_sizes[-1])

        # Chunk placement mode: "layer_stripe" (default) splits each
        # chunk's layers across tiers; "whole_chunk" places each chunk
        # entirely on one tier using weighted round-robin.
        placement_str: str = str(
            extra.get("kvstream_placement", "layer_stripe")
        ).strip().lower()
        if placement_str not in (
            "layer_stripe", "whole_chunk", "replicated_chunks",
        ):
            raise ValueError(
                f"kvstream_placement: unknown mode '{placement_str}'. "
                "Valid values: 'layer_stripe', 'whole_chunk', "
                "'replicated_chunks'"
            )
        self._whole_chunk_placement: bool = (
            placement_str == "whole_chunk"
        )
        self._replicated_chunks: bool = (
            placement_str == "replicated_chunks"
        )

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

        if self._whole_chunk_placement:
            # In whole_chunk mode every tier handles all layers;
            # ratios control chunk-to-tier assignment, not layer splits.
            boundaries = [0] + [self._num_layers] * num_tiers
        else:
            # Both layer_stripe and replicated_chunks use the same
            # layer boundaries derived from split_ratios.
            boundaries = _compute_layer_boundaries(
                ratios, self._num_layers
            )

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

        # Build tier configs and engines.
        # Tiers with zero assigned layers (e.g. from split_ratios "1.0:0.0")
        # are silently dropped to avoid null-pointer I/O submissions and
        # leaked inflight groups.
        self._tiers: list[_TierState] = []
        for i in range(num_tiers):
            if (
                not self._whole_chunk_placement
                and boundaries[i] == boundaries[i + 1]
            ):
                logger.info(
                    "KVStream: skipping tier %d (zero layers assigned)", i
                )
                continue

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

            # Re-index surviving tiers sequentially.
            tier_index = len(self._tiers)

            tc = _TierConfig(
                index=tier_index,
                path=tier_path,
                ratio=ratios[i],
                layer_start=(
                    0 if self._whole_chunk_placement
                    else boundaries[i]
                ),
                layer_end=(
                    self._num_layers if self._whole_chunk_placement
                    else boundaries[i + 1]
                ),
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
                locking=locking_modes[i],
                slab_size_mb=slab_sizes[i],
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
                "write_chunk_kb=%d, max_fds=%d, try_odirect=%s, "
                "locking=%s, slab_size_mb=%d",
                tier_index,
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
                tc.locking,
                tc.slab_size_mb,
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

        # -- Completion-driven (wait_any) tracking -------------------------
        # sub_hash -> group_hash: maps individual per-tier I/O hashes
        # back to the chunk-level group hash.
        self._sub_to_group: dict[str, str] = {}
        # sub_hash -> tier index: identifies which tier engine owns
        # a sub-hash so we can decrement the right pending counter.
        self._sub_to_tier_idx: dict[str, int] = {}
        # group_hash -> count of sub-hashes still pending.
        self._group_pending_count: dict[str, int] = {}
        # group_hash -> (key, memory_obj) for chunks awaiting completion.
        self._group_meta: dict[
            str, tuple[CacheEngineKey, MemoryObj]
        ] = {}
        # Per-tier count of sub-hashes still pending across all
        # in-flight chunks.  Used to select which tier engine to
        # block on in wait_any_load().
        self._tier_pending_subs: list[int] = [
            0 for _ in range(len(self._tiers))
        ]

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

        # -- Slab file aggregation ---------------------------------------
        # Per-tier slab sizes are already parsed into each _TierConfig.
        # If ANY tier has slab_size_mb > 0, allocate per-tier bump
        # allocator state (unused entries for file-per-chunk tiers are
        # harmless and simplify indexing).
        actual_num_tiers = len(self._tiers)
        self._any_slab_enabled: bool = any(
            t.config.slab_size_mb > 0 for t in self._tiers
        )
        if self._any_slab_enabled:
            # Per-tier bump allocator: current slab number and write offset
            self._slab_counters: list[int] = [
                0 for _ in range(actual_num_tiers)
            ]
            self._slab_offsets: list[int] = [
                0 for _ in range(actual_num_tiers)
            ]
            slab_tiers = [
                f"tier{t.config.index}={t.config.slab_size_mb}MB"
                for t in self._tiers
                if t.config.slab_size_mb > 0
            ]
            logger.info(
                "KVStream slab aggregation enabled: %s",
                ", ".join(slab_tiers),
            )

        # -- Whole-chunk placement state --------------------------------
        if self._whole_chunk_placement:
            # Build a weighted round-robin pattern from ratios.
            # E.g. ratios [0.5, 0.5] → pattern [0, 1] (length 2)
            #      ratios [0.7, 0.3] → pattern [0,0,0,0,0,0,0, 1,1,1]
            self._wc_pattern: list[int] = []
            for tier in self._tiers:
                count = max(1, round(tier.config.ratio * 100))
                self._wc_pattern.extend(
                    [tier.config.index] * count
                )
            self._wc_counter: int = 0
            logger.info(
                "KVStream whole-chunk placement: pattern_len=%d, "
                "tiers=%s",
                len(self._wc_pattern),
                [t.config.index for t in self._tiers],
            )

        # -- Replicated-chunks state -------------------------------------
        self._work_steal_state: Optional[_WorkStealState] = None
        self._pending_steal_pool: list[tuple] = []
        if self._replicated_chunks:
            if len(self._tiers) < 2:
                raise ValueError(
                    "replicated_chunks requires at least 2 tiers."
                )
            self._steal_batch: int = int(
                extra.get("kvstream_steal_batch_size", 2)
            )
            self._pfs_max_pending_steals: int = int(
                extra.get("kvstream_pfs_max_pending_steals", 4)
            )
            # Persistent cross-step set of PFS steal sub-hashes
            # that were submitted but haven't completed yet.
            self._pfs_steal_pending: set[str] = set()
            # NVMe = tier 0, PFS = tier 1.
            # NVMe-exclusive layer range: layers that ONLY NVMe has.
            # Replicated layer range: layers on BOTH NVMe and PFS.
            self._nvme_tier_idx: int = 0
            self._pfs_tier_idx: int = 1
            self._excl_layer_start: int = (
                self._tiers[0].config.layer_start
            )
            self._excl_layer_end: int = (
                self._tiers[1].config.layer_start
            )
            self._repl_layer_start: int = (
                self._tiers[1].config.layer_start
            )
            self._repl_layer_end: int = (
                self._tiers[1].config.layer_end
            )
            logger.info(
                "KVStream replicated_chunks: steal_batch=%d, "
                "pfs_max_pending=%d, "
                "exclusive_layers=[%d,%d), "
                "replicated_layers=[%d,%d)",
                self._steal_batch,
                self._pfs_max_pending_steals,
                self._excl_layer_start,
                self._excl_layer_end,
                self._repl_layer_start,
                self._repl_layer_end,
            )

        # -- Consolidated kvstream config log ----------------------------
        logger.info(
            "KVStream config: kvstream_placement=%s, "
            "kvstream_slab_size_mb=%s, "
            "kvstream_deferred_writes=%s, "
            "kvstream_drain_poll_interval_s=%.3f, "
            "kvstream_read_chunk_size_kb=%d, "
            "kvstream_read_queue_depth=%d, "
            "kvstream_write_queue_depth=%d, "
            "kvstream_write_chunk_size_kb=%d, "
            "kvstream_max_fds=%d, "
            "kvstream_max_retries=%d, "
            "kvstream_try_odirect=%s, "
            "kvstream_split_ratios=%s, "
            "kvstream_tier_locking=%s, "
            "num_tiers=%d, num_layers=%d, "
            "per_layer_bytes=%d, kv_block_bytes=%d",
            placement_str,
            slab_str,
            self._deferred_writes_enabled,
            drain_interval,
            global_read_chunk_kb,
            global_read_qd,
            global_write_qd,
            global_write_chunk_kb,
            global_max_fds,
            global_max_retries,
            global_try_odirect,
            ratios_str,
            locking_str,
            actual_num_tiers,
            self._num_layers,
            self._per_layer_bytes,
            self._kv_block_bytes,
        )

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

    def _allocate_slab_space(
        self, tier_index: int, size: int
    ) -> tuple[str, int]:
        """Allocate space in the current slab file for a tier.

        Bump-allocator: appends ``size`` bytes to the current slab.
        When the slab would exceed the tier's slab size limit, a new
        slab file is started.

        Must be called under ``self.disk_lock`` or from a single-
        threaded context (``batched_submit_put_task`` is already
        single-writer).

        Args:
            tier_index: Which tier to allocate in.
            size: Number of bytes to allocate.

        Returns:
            ``(slab_path, file_offset)`` — the path to the slab file
            and the byte offset within it for this allocation.
        """
        cur_offset = self._slab_offsets[tier_index]
        slab_limit = (
            self._tiers[tier_index].config.slab_size_mb * 1024 * 1024
        )
        # Roll over to a new slab if this allocation would exceed limit
        if cur_offset > 0 and cur_offset + size > slab_limit:
            self._slab_counters[tier_index] += 1
            cur_offset = 0

        tier_path = self._tiers[tier_index].config.path
        slab_num = self._slab_counters[tier_index]
        slab_path = os.path.join(
            tier_path, f"slab_{slab_num:04d}.bin"
        )
        self._slab_offsets[tier_index] = cur_offset + size
        return slab_path, cur_offset

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

        # Delete the file on tiers using file-per-chunk mode.
        # In slab mode, do NOT delete the slab file — other chunks
        # share it.
        for tier_slice in meta.slices:
            tc = self._tiers[tier_slice.tier_index].config
            if tc.slab_size_mb > 0:
                continue
            try:
                os.remove(tier_slice.path)
            except FileNotFoundError:
                logger.warning(
                    "KVStream: file already removed: %s",
                    tier_slice.path,
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

    @_lmcache_nvtx_annotate
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
            t0_drain = time.perf_counter()
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
            groups_resolved = 0
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
                    groups_resolved += 1
                    if group.failed:
                        self._resolve_failed_group(group)
                    else:
                        self._resolve_completed_group(group)

            n_total = len(all_completed) + len(all_failed)
            if n_total > 0:
                elapsed_ms = (time.perf_counter() - t0_drain) * 1e3
                logger.debug(
                    "KVStream drain: %d completed, %d failed, "
                    "%d groups resolved in %.2f ms",
                    len(all_completed),
                    len(all_failed),
                    groups_resolved,
                    elapsed_ms,
                )

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

        # Use pre-built tier slices (slab mode) or reconstruct from key
        slices = (
            group.tier_slices
            if group.tier_slices is not None
            else self._build_tier_slices(group.key)
        )

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

        # Clean up any partially written files (best-effort).
        # Only delete files for file-per-chunk tiers; slab tiers
        # share files so deletion would corrupt other chunks.
        for tier in self._tiers:
            if tier.config.slab_size_mb > 0:
                continue
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

            # Build the inflight group for this chunk.
            # replicated_chunks produces 2 write segments:
            #   NVMe [0, num_layers) + PFS [repl_start, repl_end)
            if self._replicated_chunks:
                n_write_segments = 2
            elif self._whole_chunk_placement:
                n_write_segments = 1
            else:
                n_write_segments = num_tiers
            group = _InflightGroup(
                key=key,
                memory_obj=memory_obj,
                remaining=self._kv_size * n_write_segments,
                total_size=total_size,
                shape=shape,
                dtype=dtype,
                fmt=fmt,
                cached_positions=cached_positions,
                on_complete_callback=on_complete_callback,
            )

            # Build per-tier I/O descriptors
            save_ops: list[tuple[Any, str, torch.Tensor, str, int]] = []
            # Pre-built tier slices are required when slabs are used
            # (offsets can't be reconstructed) or in whole_chunk mode
            # (_build_tier_slices would wrongly produce slices for
            # ALL tiers).
            need_prebuilt_slices = (
                self._any_slab_enabled
                or self._whole_chunk_placement
                or self._replicated_chunks
            )
            slab_tier_slices: Optional[list[_TierSlice]] = (
                [] if need_prebuilt_slices else None
            )

            # In whole_chunk mode, pick ONE tier via weighted
            # round-robin; in layer_stripe mode, use all tiers.
            # In replicated_chunks mode, build 3 write segments
            # explicitly: NVMe [0, num_layers), PFS [repl, end).
            if self._whole_chunk_placement:
                tier_idx = self._wc_pattern[
                    self._wc_counter % len(self._wc_pattern)
                ]
                self._wc_counter += 1
                tiers_for_chunk = [self._tiers[tier_idx]]
            else:
                tiers_for_chunk = self._tiers

            if self._replicated_chunks:
                # Build 3 write segments explicitly:
                #  seg 0: NVMe, layers [0, num_layers) — full copy
                #  seg 1: PFS, layers [repl_start, repl_end) — tail
                #  seg 2: (conceptual — seg 0 already covers it)
                # Actually, NVMe writes ALL layers as a single file,
                # and PFS writes only its portion.
                write_segments: list[tuple[int, int, int]] = [
                    # (tier_index, layer_start, layer_end)
                    (self._nvme_tier_idx, 0, self._num_layers),
                    (self._pfs_tier_idx,
                     self._repl_layer_start,
                     self._repl_layer_end),
                ]
            else:
                write_segments = None  # use default tier loop

            if write_segments is not None:
                # replicated_chunks explicit segment loop
                for seg_tier_idx, seg_lstart, seg_lend in write_segments:
                    tier = self._tiers[seg_tier_idx]
                    tc = tier.config
                    seg_num_layers = seg_lend - seg_lstart
                    seg_layer_bytes = (
                        seg_num_layers * self._per_layer_bytes
                    )
                    seg_size = self._kv_size * seg_layer_bytes

                    path = self._key_to_tier_path(key, tier)
                    # For replicated_chunks with NVMe full copy,
                    # the NVMe file is larger than normal.  Use a
                    # distinct path suffix for the full copy.
                    if (
                        seg_tier_idx == self._nvme_tier_idx
                        and seg_lend == self._num_layers
                        and seg_lstart == 0
                    ):
                        # NVMe full-copy file
                        path = self._key_to_tier_path(key, tier)

                    slab_base = 0  # file-per-chunk only

                    if slab_tier_slices is not None:
                        slab_tier_slices.append(
                            _TierSlice(
                                tier_index=seg_tier_idx,
                                path=path,
                                layer_start=seg_lstart,
                                layer_end=seg_lend,
                                size=seg_size,
                                k_file_offset=0,
                                v_file_offset=seg_layer_bytes,
                            )
                        )

                    for kv_idx in range(self._kv_size):
                        kv_label = "k" if kv_idx == 0 else "v"
                        io_hash = self._next_io_hash(
                            key,
                            f"save_{kv_label}_t{seg_tier_idx}"
                            f"_l{seg_lstart}",
                        )
                        group.io_hashes.append(io_hash)

                        buf_offset = (
                            kv_idx * self._kv_block_bytes
                            + seg_lstart * self._per_layer_bytes
                        )
                        raw_slice = raw_tensor[
                            buf_offset : buf_offset + seg_layer_bytes
                        ]
                        file_offset = kv_idx * seg_layer_bytes

                        save_ops.append((
                            tier.engine, io_hash, raw_slice,
                            path, file_offset,
                        ))
            else:
                # layer_stripe / whole_chunk: original tier loop
                for tier in tiers_for_chunk:
                    tc = tier.config
                    tier_layer_bytes = (
                        tc.num_layers * self._per_layer_bytes
                    )
                    tier_size = self._kv_size * tier_layer_bytes

                    if tc.slab_size_mb > 0:
                        path, slab_base = self._allocate_slab_space(
                            tc.index, tier_size
                        )
                    else:
                        path = self._key_to_tier_path(key, tier)
                        slab_base = 0

                    if slab_tier_slices is not None:
                        slab_tier_slices.append(
                            _TierSlice(
                                tier_index=tc.index,
                                path=path,
                                layer_start=tc.layer_start,
                                layer_end=tc.layer_end,
                                size=tier_size,
                                k_file_offset=slab_base,
                                v_file_offset=(
                                    slab_base + tier_layer_bytes
                                ),
                            )
                        )

                    for kv_idx in range(self._kv_size):
                        kv_label = "k" if kv_idx == 0 else "v"
                        io_hash = self._next_io_hash(
                            key, f"save_{kv_label}{tc.index}"
                        )
                        group.io_hashes.append(io_hash)

                        buf_offset = (
                            kv_idx * self._kv_block_bytes
                            + tc.layer_start * self._per_layer_bytes
                        )
                        raw_slice = raw_tensor[
                            buf_offset : buf_offset + tier_layer_bytes
                        ]
                        file_offset = (
                            slab_base + kv_idx * tier_layer_bytes
                        )

                        save_ops.append((
                            tier.engine, io_hash, raw_slice,
                            path, file_offset,
                        ))

            group.tier_slices = slab_tier_slices

            if self._deferred_writes_enabled:
                self._deferred_queue.append((group, save_ops))
                # Register hashes in the group map so drain can find
                # them if the drain thread runs before flush.
                for h in group.io_hashes:
                    self._hash_to_group[h] = group
            else:
                # Submit immediately
                for engine, io_hash, raw_slice, path, foff in save_ops:
                    torch.cuda.nvtx.range_push("kvs_save_imm")
                    engine.save(io_hash, raw_slice, path, foff)
                    torch.cuda.nvtx.range_pop()
                for h in group.io_hashes:
                    self._hash_to_group[h] = group

    @_lmcache_nvtx_annotate
    def flush_deferred_writes(self) -> int:
        """Submit all deferred writes to the io_uring engines.

        Called at the start of the next step, before reads begin.
        For tiers with ``locking="rw"``, blocks until all in-flight
        writes on that tier's engine complete, preventing read/write
        contention on the storage device.

        Returns:
            Number of logical chunks flushed.
        """
        if not self._deferred_queue:
            # Even with no new deferred writes, we must still
            # wait for previously-submitted writes on rw-locked
            # tiers to finish before the caller starts reads.
            self._wait_rw_locked_tiers()
            return 0

        self._drain_completed()

        t0 = time.perf_counter()
        n = 0
        total_bytes = 0
        for group, save_ops in self._deferred_queue:
            for engine, io_hash, raw_slice, path, foff in save_ops:
                torch.cuda.nvtx.range_push("kvs_save_flush")
                engine.save(io_hash, raw_slice, path, foff)
                torch.cuda.nvtx.range_pop()
                total_bytes += raw_slice.nbytes
            # Hashes are already registered in _hash_to_group during
            # batched_submit_put_task.
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

        # Block until writes complete on rw-locked tiers so
        # subsequent reads don't contend with in-flight writes.
        self._wait_rw_locked_tiers()

        return n

    def _wait_rw_locked_tiers(self) -> None:
        """Block until all in-flight writes complete on rw-locked tiers.

        Called by ``flush_deferred_writes`` to ensure write I/O is
        fully drained before reads start on tiers configured with
        ``locking="rw"``.  Tiers with ``locking="none"`` are skipped.
        """
        io_queue_write = self.kvstream_core.IOQueue.WRITE
        for tier in self._tiers:
            if tier.config.locking != "rw":
                continue
            t0 = time.perf_counter()
            torch.cuda.nvtx.range_push(
                f"kvs_wait_wr_t{tier.config.index}"
            )
            tier.engine.wait_all(io_queue_write)
            torch.cuda.nvtx.range_pop()
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            if elapsed_ms > 1.0:
                logger.info(
                    "KVStream: wait_all(WRITE) tier %d: %.2f ms",
                    tier.config.index,
                    elapsed_ms,
                )

    # ------------------------------------------------------------------ #
    #  Get (read) path — P-tier layer-striped                             #
    # ------------------------------------------------------------------ #

    @_lmcache_nvtx_annotate
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

                torch.cuda.nvtx.range_push(
                    f"kvs_load_t{tier_slice.tier_index}_{kv_label}"
                )
                tier.engine.load(
                    io_hash, raw_slice, tier_slice.path, file_offset
                )
                torch.cuda.nvtx.range_pop()
                load_refs.append((tier, io_hash))

        return load_refs

    # ------------------------------------------------------------------ #
    #  replicated_chunks: submit exclusive-only loads                     #
    # ------------------------------------------------------------------ #

    def _submit_exclusive_loads(
        self,
        key: CacheEngineKey,
        meta: _TieredChunkMeta,
        raw_tensor: torch.Tensor,
    ) -> list[tuple[_TierState, str]]:
        """Submit reads for the NVMe-exclusive layer range only.

        In ``replicated_chunks`` mode, layers [0, repl_layer_start)
        exist only on NVMe.  This submits K+V reads for that range.
        The replicated range [repl_layer_start, repl_layer_end) is
        handled by the work-stealing scheduler.

        Args:
            key: Cache engine key.
            meta: Tiered metadata (with 2 slices: NVMe full + PFS tail).
            raw_tensor: Destination buffer (flat uint8).

        Returns:
            List of ``(tier, io_hash)`` for the submitted reads.
        """
        load_refs: list[tuple[_TierState, str]] = []
        nvme_tier = self._tiers[self._nvme_tier_idx]

        # Find the NVMe slice (layer_start=0, layer_end=num_layers)
        nvme_slice: Optional[_TierSlice] = None
        for s in meta.slices:
            if s.tier_index == self._nvme_tier_idx and s.layer_start == 0:
                nvme_slice = s
                break
        assert nvme_slice is not None, (
            "replicated_chunks: NVMe slice not found in metadata"
        )

        # Read only the exclusive portion [0, repl_layer_start)
        excl_layers = self._repl_layer_start
        excl_layer_bytes = excl_layers * self._per_layer_bytes

        for kv_idx in range(self._kv_size):
            kv_label = "k" if kv_idx == 0 else "v"
            io_hash = self._next_io_hash(
                key, f"load_{kv_label}_excl"
            )
            buf_offset = (
                kv_idx * self._kv_block_bytes
                # layer_start = 0, so no offset from layer_start
            )
            raw_slice = raw_tensor[
                buf_offset : buf_offset + excl_layer_bytes
            ]
            # File offset: NVMe file has ALL layers.
            # K-block at k_file_offset, V-block at v_file_offset.
            # Within each block, layers are contiguous from 0.
            # Exclusive layers are [0, repl_layer_start), which is
            # the first excl_layer_bytes of the K/V block.
            if kv_idx == 0:
                file_offset = nvme_slice.k_file_offset
            else:
                file_offset = nvme_slice.v_file_offset

            nvme_tier.engine.load(
                io_hash, raw_slice, nvme_slice.path, file_offset
            )
            load_refs.append((nvme_tier, io_hash))

        return load_refs

    # ------------------------------------------------------------------ #
    #  replicated_chunks: work-stealing scheduler                        #
    # ------------------------------------------------------------------ #

    def _init_work_stealing(
        self,
        steal_pool: list[tuple],
    ) -> None:
        """Initialize per-step work-stealing and submit first batches.

        Args:
            steal_pool: List of
                ``(group_hash, key, meta, raw_tensor,
                  repl_layer_start, repl_layer_end)``
                for chunks whose replicated range is stealable.
        """
        ws = _WorkStealState(
            items=steal_pool,
            pfs_cursor=len(steal_pool) - 1,
            max_inflight_per_tier=self._steal_batch,
        )
        self._work_steal_state = ws
        self._submit_next_steal_batch(self._nvme_tier_idx)
        self._submit_next_steal_batch(self._pfs_tier_idx)

    def _submit_next_steal_batch(self, tier_idx: int) -> None:
        """Advance cursor and submit steal reads for one tier.

        For NVMe, after the normal cursor range is exhausted, any
        remaining undone items (including those PFS is working on)
        are also submitted on NVMe.  This provides full tail-latency
        immunity — NVMe can complete all items even if PFS stalls.
        """
        ws = self._work_steal_state
        if ws is None:
            return
        is_nvme = tier_idx == self._nvme_tier_idx

        if is_nvme:
            # Phase 1: normal cursor-based stealing from the front.
            while (
                ws.nvme_inflight < ws.max_inflight_per_tier
                and ws.nvme_cursor <= ws.pfs_cursor
            ):
                idx = ws.nvme_cursor
                ws.nvme_cursor += 1
                if idx in ws.done_items:
                    continue
                if idx in ws.nvme_submitted:
                    continue
                if self._submit_steal_read(idx, tier_idx):
                    ws.nvme_submitted.add(idx)
                    ws.nvme_inflight += 1

            # Phase 2: re-steal PFS's items that haven't completed.
            # Scan all items beyond the cursor range (PFS territory)
            # and submit NVMe reads for any that are undone and not
            # yet submitted to NVMe.
            if ws.nvme_inflight < ws.max_inflight_per_tier:
                for idx in range(len(ws.items)):
                    if ws.nvme_inflight >= ws.max_inflight_per_tier:
                        break
                    if idx in ws.done_items:
                        continue
                    if idx in ws.nvme_submitted:
                        continue
                    if self._submit_steal_read(idx, tier_idx):
                        ws.nvme_submitted.add(idx)
                        ws.nvme_inflight += 1
            return

        # PFS: from the back, normal cursor-based only.
        # Skip if PFS has too many unresolved steal reads from
        # this or previous steps (tail spike backlog).
        if len(self._pfs_steal_pending) >= self._pfs_max_pending_steals:
            return

        while (
            ws.pfs_inflight < ws.max_inflight_per_tier
            and ws.pfs_cursor >= ws.nvme_cursor
        ):
            idx = ws.pfs_cursor
            ws.pfs_cursor -= 1
            if idx in ws.done_items:
                continue
            if idx in ws.pfs_submitted:
                continue
            if self._submit_steal_read(idx, tier_idx):
                ws.pfs_submitted.add(idx)
                ws.pfs_inflight += 1

    def _submit_steal_read(
        self, item_idx: int, tier_idx: int
    ) -> bool:
        """Submit K+V reads for one item's replicated range on one tier.

        Returns:
            True if reads were submitted, False on failure.
        """
        ws = self._work_steal_state
        assert ws is not None
        group_hash, key, meta, raw_tensor, l_start, l_end = (
            ws.items[item_idx]
        )

        # Find the slice for this tier covering [l_start, l_end)
        target: Optional[_TierSlice] = None
        for s in meta.slices:
            if (
                s.tier_index == tier_idx
                and s.layer_start <= l_start
                and s.layer_end >= l_end
            ):
                target = s
                break
        if target is None:
            return False

        tier = self._tiers[tier_idx]
        repl_layer_bytes = (l_end - l_start) * self._per_layer_bytes

        for kv_idx in range(self._kv_size):
            kv_label = "k" if kv_idx == 0 else "v"
            io_hash = self._next_io_hash(
                key, f"steal_{kv_label}_t{tier_idx}"
            )
            # Buffer offset: replicated layers start at l_start
            buf_offset = (
                kv_idx * self._kv_block_bytes
                + l_start * self._per_layer_bytes
            )
            raw_slice = raw_tensor[
                buf_offset : buf_offset + repl_layer_bytes
            ]
            # File offset: within this tier's file, the replicated
            # layers are at an offset relative to the slice's start.
            layer_offset_in_slice = (
                (l_start - target.layer_start)
                * self._per_layer_bytes
            )
            if kv_idx == 0:
                file_offset = (
                    target.k_file_offset + layer_offset_in_slice
                )
            else:
                file_offset = (
                    target.v_file_offset + layer_offset_in_slice
                )

            tier.engine.load(
                io_hash, raw_slice, target.path, file_offset
            )
            self._sub_to_group[io_hash] = group_hash
            self._sub_to_tier_idx[io_hash] = tier_idx
            self._tier_pending_subs[tier_idx] += 1
            ws.sub_to_item[io_hash] = item_idx

            # Track PFS steal sub-hashes across steps so we can
            # detect PFS backlog and bench it.
            if tier_idx == self._pfs_tier_idx:
                self._pfs_steal_pending.add(io_hash)

        return True

    def _pump_work_stealing(
        self, completed_sub_hashes: list[str]
    ) -> list[str]:
        """Process steal completions, replenish batches.

        Returns list of steal sub-hashes that were processed (so
        ``_process_sub_completions`` can skip them).
        """
        ws = self._work_steal_state
        if ws is None:
            return []

        processed: list[str] = []
        nvme_items_done = 0
        pfs_items_done = 0

        # Track late-arrival tier decrements separately.
        nvme_late = 0
        pfs_late = 0

        for sub_hash in completed_sub_hashes:
            item_idx = ws.sub_to_item.get(sub_hash)
            if item_idx is None:
                continue  # not a steal sub-hash
            processed.append(sub_hash)

            if item_idx in ws.done_items:
                # Late arrival from the other tier — harmless
                # overwrite of identical data.  But we need to
                # account for inflight on the late tier so it
                # can submit more work.
                late_tier = self._sub_to_tier_idx.get(sub_hash)
                # Only count once per item per tier (kv_size
                # sub-hashes per item, count on the last one).
                late_count = ws.item_completions.get(
                    item_idx, 0
                ) + 1
                ws.item_completions[item_idx] = late_count
                # After kv_size *additional* late arrivals, the
                # other tier's pair is fully done.
                if late_count % self._kv_size == 0:
                    if late_tier == self._nvme_tier_idx:
                        nvme_late += 1
                    else:
                        pfs_late += 1
                continue

            count = ws.item_completions.get(item_idx, 0) + 1
            ws.item_completions[item_idx] = count
            if count < self._kv_size:
                continue  # K arrived but V hasn't (or vice versa)

            # Item done (K+V complete from some tier).
            ws.done_items.add(item_idx)

            # Figure out which tier completed it for inflight
            # accounting.
            tier_idx = self._sub_to_tier_idx.get(sub_hash)
            if tier_idx == self._nvme_tier_idx:
                nvme_items_done += 1
            else:
                pfs_items_done += 1

            # Decrement group pending count.
            group_hash = self._sub_to_group.get(sub_hash)
            if group_hash and group_hash in self._group_pending_count:
                self._group_pending_count[group_hash] -= (
                    self._kv_size
                )

        ws.nvme_inflight = max(
            0, ws.nvme_inflight - nvme_items_done - nvme_late
        )
        ws.pfs_inflight = max(
            0, ws.pfs_inflight - pfs_items_done - pfs_late
        )
        self._submit_next_steal_batch(self._nvme_tier_idx)
        self._submit_next_steal_batch(self._pfs_tier_idx)

        return processed

    @_lmcache_nvtx_annotate
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

    @_lmcache_nvtx_annotate
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

    @_lmcache_nvtx_annotate
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
        steal_pool: list[tuple] = []

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

            group_hash = self._next_io_hash(key, "load_group")

            if self._replicated_chunks:
                # Submit only the NVMe-exclusive layers.  The
                # replicated range goes through work-stealing.
                load_refs = self._submit_exclusive_loads(
                    key, meta, raw_tensor
                )
                self._overlapped_load_refs[group_hash] = load_refs
                # pending = exclusive K+V + replicated K+V
                self._group_pending_count[group_hash] = (
                    len(load_refs) + self._kv_size
                )
                self._group_meta[group_hash] = (key, memory_obj)
                for _tier, sub_hash in load_refs:
                    self._sub_to_group[sub_hash] = group_hash
                    tier_idx = self._tiers.index(_tier)
                    self._sub_to_tier_idx[sub_hash] = tier_idx
                    self._tier_pending_subs[tier_idx] += 1

                steal_pool.append((
                    group_hash, key, meta, raw_tensor,
                    self._repl_layer_start,
                    self._repl_layer_end,
                ))
            else:
                load_refs = self._submit_tiered_loads(
                    key, meta, raw_tensor
                )
                self._overlapped_load_refs[group_hash] = load_refs
                self._group_pending_count[group_hash] = len(
                    load_refs
                )
                self._group_meta[group_hash] = (key, memory_obj)
                for _tier, sub_hash in load_refs:
                    self._sub_to_group[sub_hash] = group_hash
                    tier_idx = self._tiers.index(_tier)
                    self._sub_to_tier_idx[sub_hash] = tier_idx
                    self._tier_pending_subs[tier_idx] += 1

            self.local_cpu_backend.submit_put_task(key, memory_obj)
            results.append((group_hash, key, memory_obj))

        if steal_pool:
            # Accumulate — do NOT init yet.  submit_batch_load may
            # be called multiple times per step (once per request).
            # Work-stealing is initialized lazily on the first
            # wait_any_load call.
            self._pending_steal_pool.extend(steal_pool)

        return results

    @_lmcache_nvtx_annotate
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
        for tier_idx, (tier, sub_hash) in enumerate(load_refs):
            torch.cuda.nvtx.range_push(
                f"kvs_wait_t{tier_idx}"
            )
            tier.engine.wait_one(
                sub_hash, self.kvstream_core.IOQueue.READ
            )
            torch.cuda.nvtx.range_pop()

        # Recover cached_positions metadata
        disk_meta = self.dict.get(key, None)
        if disk_meta is not None:
            memory_obj.metadata.cached_positions = (
                disk_meta.cached_positions
            )

    def _process_sub_completions(
        self, sub_hashes: list[str]
    ) -> list[tuple[str, CacheEngineKey, MemoryObj]]:
        """Accumulate sub-hash completions and return fully-ready chunks.

        For each completed sub-hash, decrements the pending count of
        its parent group.  When a group reaches zero, its metadata is
        finalised and the group is yielded as ready.

        Steal sub-hashes (from work-stealing) are handled by
        ``_pump_work_stealing`` first; this method then cleans up
        their tracking entries.

        Args:
            sub_hashes: Sub-hashes that just completed.

        Returns:
            List of ``(group_hash, key, memory_obj)`` for chunks whose
            I/O across all tiers is now complete.
        """
        # Drain PFS steal completions (current step or stale from
        # previous steps).  This must happen before work-stealing
        # processing so the PFS backlog counter is accurate.
        if self._replicated_chunks and self._pfs_steal_pending:
            for sub_hash in sub_hashes:
                self._pfs_steal_pending.discard(sub_hash)

        # Let work-stealing process its sub-hashes first (adjusts
        # group pending counts and replenishes batches).
        steal_set: set[str] = set()
        if self._work_steal_state is not None:
            steal_set = set(
                self._pump_work_stealing(sub_hashes)
            )

        ready: list[tuple[str, CacheEngineKey, MemoryObj]] = []
        for sub_hash in sub_hashes:
            is_steal = sub_hash in steal_set

            # Always clean up tier tracking.
            tier_idx = self._sub_to_tier_idx.pop(sub_hash, None)
            if tier_idx is not None:
                self._tier_pending_subs[tier_idx] -= 1

            group_hash = self._sub_to_group.pop(sub_hash, None)
            if group_hash is None:
                continue

            if is_steal:
                # Steal sub-hash: _pump_work_stealing already
                # adjusted group pending count.  Check if group
                # is ready (may have been resolved by a prior
                # steal sub-hash in this same batch).
                if group_hash not in self._group_pending_count:
                    continue
                remaining = self._group_pending_count[group_hash]
                if remaining > 0:
                    continue
            else:
                # Normal sub-hash: decrement group pending count.
                remaining = (
                    self._group_pending_count[group_hash] - 1
                )
                if remaining > 0:
                    self._group_pending_count[group_hash] = remaining
                    continue

            # All sub-hashes for this group are done.
            del self._group_pending_count[group_hash]
            key, memory_obj = self._group_meta.pop(group_hash)
            self._overlapped_load_refs.pop(group_hash, None)
            # Recover cached_positions metadata
            disk_meta = self.dict.get(key, None)
            if disk_meta is not None:
                memory_obj.metadata.cached_positions = (
                    disk_meta.cached_positions
                )
            ready.append((group_hash, key, memory_obj))
        return ready

    @_lmcache_nvtx_annotate
    def wait_any_load(
        self,
    ) -> list[tuple[str, CacheEngineKey, MemoryObj]]:
        """Block until at least one chunk has all tier I/O complete.

        Uses completion-driven reaping: drains all tier engines
        non-blockingly, then blocks on one engine if nothing is
        ready yet.  Returns **all** chunks that are fully complete
        at the time of return.

        Returns:
            List of ``(group_hash, key, memory_obj)`` for every
            chunk whose reads across all tiers finished.
        """
        # Lazy init: collect all steal items from possibly multiple
        # submit_batch_load calls before starting the scheduler.
        if (
            self._work_steal_state is None
            and self._pending_steal_pool
        ):
            self._init_work_stealing(self._pending_steal_pool)
            self._pending_steal_pool = []

        ready: list[
            tuple[str, CacheEngineKey, MemoryObj]
        ] = []
        io_queue_read = self.kvstream_core.IOQueue.READ

        while not ready:
            # Phase 1: non-blocking drain from every tier engine
            for tier in self._tiers:
                newly_done = tier.engine.drain_completed_queue(
                    io_queue_read
                )
                if newly_done:
                    ready.extend(
                        self._process_sub_completions(newly_done)
                    )

            if ready:
                break

            # Phase 2: nothing fully ready — block on a tier
            # engine that still has pending sub-hashes.  Blocking
            # on a tier with zero pending would deadlock.
            blocked = False
            for ti, tier in enumerate(self._tiers):
                if self._tier_pending_subs[ti] > 0:
                    torch.cuda.nvtx.range_push(
                        f"kvs_wait_any_t{ti}"
                    )
                    completed = tier.engine.wait_any_completed(
                        io_queue_read
                    )
                    torch.cuda.nvtx.range_pop()
                    ready.extend(
                        self._process_sub_completions(completed)
                    )
                    blocked = True
                    break

            if not blocked:
                # Should not happen: no tier has pending subs but
                # we still have pending groups.  Defensive break.
                logger.error(
                    "wait_any_load: no tier has pending subs but "
                    "%d groups remain — breaking to avoid hang",
                    len(self._group_pending_count),
                )
                break
            # If still not ready (sub-hashes from one tier done but
            # the other tier's sub-hashes still pending), loop back
            # to drain all tiers again.

        # Clean up work-stealing state when all groups are resolved.
        if (
            self._work_steal_state is not None
            and not self._group_pending_count
        ):
            self._work_steal_state = None
            self._pending_steal_pool = []

        return ready

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

    def get_tier_read_stats(self) -> list[dict[str, Any]]:
        """Snapshot per-tier read statistics (no reset).

        Returns a list of dicts (one per tier) with read stats from the
        C++ io_uring engines.  Does NOT reset counters.

        Returns:
            List of per-tier dicts with keys ``tier``,
            ``read_bytes``, ``read_bw_mb_s``, ``read_elapsed_ms``.
        """
        IOQueue = self.kvstream_core.IOQueue
        results: list[dict[str, Any]] = []
        for i, tier in enumerate(self._tiers):
            rs = tier.engine.get_stats(IOQueue.READ)
            results.append({
                "tier": i,
                "read_bytes": rs.total_bytes_completed,
                "read_bw_mb_s": round(rs.bandwidth_mb_s(), 1),
                "read_elapsed_ms": round(
                    rs.elapsed_us() / 1e3, 2
                ),
            })
        return results

    def get_tier_stats_and_reset(
        self,
    ) -> list[dict[str, Any]]:
        """Snapshot and reset per-tier I/O statistics from C++ engines.

        Returns a list of dicts (one per tier) with read and write
        stats.  After returning, all engine counters are zeroed so the
        next call reports a clean interval.

        Returns:
            List of per-tier stat dicts with keys:
            ``tier``, ``read_bytes``, ``read_ops``, ``read_bw_mb_s``,
            ``read_elapsed_ms``, ``write_bytes``, ``write_ops``,
            ``write_bw_mb_s``, ``write_elapsed_ms``.
        """
        IOQueue = self.kvstream_core.IOQueue
        results: list[dict[str, Any]] = []
        for i, tier in enumerate(self._tiers):
            rs = tier.engine.get_stats(IOQueue.READ)
            ws = tier.engine.get_stats(IOQueue.WRITE)
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
            tier.engine.reset_stats(IOQueue.READ)
            tier.engine.reset_stats(IOQueue.WRITE)
        return results

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
