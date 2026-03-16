# SPDX-License-Identifier: Apache-2.0
"""KVStream disk backend for LMCache.

Uses the KVStream library (io_uring-based async I/O engine) to perform
high-performance disk reads and writes for KV cache data, replacing the
Python open()/write()/read() calls used by LocalDiskBackend.

Architecture
~~~~~~~~~~~~
KVStream internally maintains **two** io_uring engines (read and write) that
share a single FD cache (files opened ``O_RDWR | O_CREAT``).  This backend
submits writes as non-blocking SQEs on the main thread.  A lightweight
background drain thread polls completed write notifications every 50 ms
(configurable via ``kvstream_drain_poll_interval_s``), freeing CPU memory
buffers as soon as the kernel finishes writing them to disk.

Reads use ``wait_one`` / ``wait_all`` with ``IOQueue.READ`` so they never
block on outstanding writes.

Activated when ``config.kvstream_enable`` is ``True`` and
``config.local_disk`` points to a directory.
"""

# Standard
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
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
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

# Type alias for inflight write bookkeeping entries.
# (key, memory_obj, size, shape, dtype, fmt, cached_positions, callback)
_InflightEntry = tuple[
    CacheEngineKey,
    MemoryObj,
    int,
    "torch.Size",
    "torch.dtype",
    MemoryFormat,
    Optional["torch.Tensor"],
    Optional[Callable[[CacheEngineKey], None]],
]



class KVStreamDiskBackend(StorageBackendInterface):
    """Disk backend using KVStream (io_uring) for async I/O.

    KVStream submits I/O operations via io_uring and processes completions
    on dedicated C++ background threads (one per engine).  The Python write
    path is **drain-based**: ``batched_submit_put_task`` submits SQEs on the
    main thread and returns immediately.  A background drain thread polls
    ``_drain_completed()`` every 50 ms to free CPU memory buffers as soon
    as writes finish, preventing buffer starvation when no new writes are
    being submitted.

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
            lmcache_worker: Optional cache controller worker for ADMIT/EVICT
                messages.
            metadata: Optional LMCache metadata (provides worker_id etc.).
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

        # -- Directory ---------------------------------------------------
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

        # -- Inflight writes (drain-based bookkeeping) -------------------
        # Maps io_hash -> bookkeeping tuple for writes that have been
        # submitted to C++ but not yet drained as completed/failed.
        self._inflight: dict[str, _InflightEntry] = {}
        # Serializes _drain_completed() calls across the background
        # drain thread and the main thread (batched_submit_put_task).
        self._drain_lock = threading.Lock()

        # -- KVStream engine ---------------------------------------------
        extra = config.extra_config or {}
        read_chunk_size_kb: int = int(
            extra.get("kvstream_read_chunk_size_kb", 4096)
        )
        read_queue_depth: int = int(
            extra.get("kvstream_read_queue_depth", 64)
        )
        write_queue_depth: int = int(
            extra.get("kvstream_write_queue_depth", 4)
        )
        write_chunk_size_kb: int = int(
            extra.get("kvstream_write_chunk_size_kb", 256)
        )
        max_fds: int = int(extra.get("kvstream_max_fds", 4096))
        max_retries: int = int(extra.get("kvstream_max_retries", 10))
        try_odirect: bool = bool(
            extra.get("kvstream_try_odirect", True)
        )

        try:
            from kvstream import kvstream_core

            self.kvstream_core = kvstream_core
            self.engine = kvstream_core.KVStream(
                read_chunk_size_kb=read_chunk_size_kb,
                read_queue_depth=read_queue_depth,
                write_queue_depth=write_queue_depth,
                write_chunk_size_kb=write_chunk_size_kb,
                max_fds_open=max_fds,
                try_using_odirect=try_odirect,
                max_retries=max_retries,
            )
            logger.info(
                "KVStream engine initialized: "
                "read_chunk_kb=%d, read_qd=%d, write_qd=%d, "
                "write_chunk_kb=%d, max_fds=%d, "
                "try_odirect=%s, max_retries=%d",
                read_chunk_size_kb,
                read_queue_depth,
                write_queue_depth,
                write_chunk_size_kb,
                max_fds,
                try_odirect,
                max_retries,
            )
        except ImportError:
            raise ImportError(
                "kvstream_core is not installed. "
                "Build and install the kvstream package before using "
                "KVStreamDiskBackend."
            )

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
                "KVStreamDiskBackend: controller message sender not initialized"
            )

        # -- Background drain thread --------------------------------------
        # Polls the C++ write engine for completed/failed writes every
        # ``_DRAIN_POLL_INTERVAL_S`` seconds, ensuring buffers are freed
        # promptly even when no new writes are being submitted.
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

    # ------------------------------------------------------------------ #
    #  String / helpers                                                    #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        """Return human-readable backend name."""
        return "KVStreamDiskBackend"

    def _key_to_path(self, key: CacheEngineKey) -> str:
        """Convert a cache key to a filesystem path.

        Args:
            key: The cache engine key.

        Returns:
            Absolute path for the cached file.
        """
        return os.path.join(
            self.path, key.to_string().replace("/", "-") + ".bin"
        )

    def _next_io_hash(self, key: CacheEngineKey, op: str) -> str:
        """Generate a globally unique hash for a KVStream I/O operation.

        The hash must be unique across the entire engine lifetime because
        KVStream tracks per-hash state.  A monotonic counter suffix
        guarantees uniqueness even if the same cache key is written multiple
        times (e.g. after eviction and re-write).

        Args:
            key: The cache engine key.
            op: Operation prefix (``"save"`` or ``"load"``).

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

        Stops at the first miss (prefix-match semantics required by
        LMCache's lookup protocol).

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
        """Update cache recency for keys accumulated during lookup.

        Keys are processed in reverse order (suffix-to-prefix to
        prefix-to-suffix) so that the most recently accessed key ends up
        at the MRU position.
        """
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
            ``True`` if the key was found and unpinned, ``False`` otherwise.
        """
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove *key* from the disk cache.

        Args:
            key: The cache engine key.
            force: If ``True`` (default, external removal), acquires the disk
                lock and notifies the cache policy.  If ``False`` (internal
                eviction), assumes the caller already holds the lock.

        Returns:
            ``True`` if the key was removed, ``False`` if not found.
        """
        if force:
            self.disk_lock.acquire()

        meta = self.dict.pop(key, None)
        if not meta:
            if force:
                self.disk_lock.release()
            return False

        path = meta.path
        size = meta.size
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        try:
            os.remove(path)
        except FileNotFoundError:
            logger.warning("KVStream: file already removed: %s", path)

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

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
    ) -> None:
        """Register a successfully written key in the metadata dictionary.

        If the key already exists (duplicate write), the cache policy is
        updated for recency but no new metadata entry is created.

        Args:
            key: The cache engine key.
            size: Physical size of the cached data in bytes.
            shape: Logical tensor shape.
            dtype: Tensor data type.
            fmt: Memory format enum.
            cached_positions: Optional position tensor.
        """
        path = self._key_to_path(key)

        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path, size, shape, dtype, cached_positions, fmt, 0
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
                    "KVStream: put task for %s not found during removal", key
                )

    # ------------------------------------------------------------------ #
    #  Background drain thread                                             #
    # ------------------------------------------------------------------ #

    def _drain_loop(self, interval: float) -> None:
        """Background thread target: periodically drain completed writes.

        Runs until ``_drain_stop`` is set.  Each iteration calls
        ``_drain_completed()`` to free buffers held by finished
        async writes, then sleeps for *interval* seconds.

        This eliminates the deadlock where no new writes arrive
        to trigger the drain, leaving completed-write buffers
        pinned and starving the allocator.

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
    #  Drain-based write completion bookkeeping                            #
    # ------------------------------------------------------------------ #

    def _drain_completed(self) -> None:
        """Poll the C++ write engine for completed and failed writes.

        For each completed write hash:
          1. ``ref_count_down`` the ``MemoryObj`` (release the buffer).
          2. ``insert_key`` to register metadata.
          3. ``_remove_put_task`` to allow future puts for the same key.
          4. Invoke the ``on_complete_callback`` if provided.

        For each failed write hash:
          1. ``ref_count_down`` the ``MemoryObj``.
          2. ``_remove_put_task``.
          3. Roll back ``current_cache_size`` (the space was pre-reserved).

        Called from three places:
          1. The background drain thread (every ``kvstream_drain_poll_interval_s``).
          2. The *start* of each ``batched_submit_put_task`` (belt-and-suspenders).
          3. Once more during ``close()`` after ``engine.shutdown()``.

        Thread-safe: serialized via ``_drain_lock``.
        """
        with self._drain_lock:
            # -- Drain completions --
            completed_hashes = self.engine.drain_completed()
            for io_hash in completed_hashes:
                entry = self._inflight.pop(io_hash, None)
                if entry is None:
                    logger.warning(
                        "KVStream: drained completed hash %s with no "
                        "inflight entry",
                        io_hash,
                    )
                    continue

                (
                    key,
                    memory_obj,
                    size,
                    shape,
                    dtype,
                    fmt,
                    cached_positions,
                    on_complete_callback,
                ) = entry

                self.usage += size
                self.stats_monitor.update_local_storage_usage(self.usage)

                # ref_count_down before insert_key (matches LocalDiskBackend
                # ordering for mem-leak test compatibility)
                memory_obj.ref_count_down()

                self.insert_key(
                    key,
                    size,
                    shape,
                    dtype,
                    fmt,
                    cached_positions=cached_positions,
                )

                self._remove_put_task(key)

                if on_complete_callback is not None:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            "on_complete_callback failed for key %s: %s",
                            key,
                            e,
                        )

            # -- Drain failures --
            failed_hashes = self.engine.drain_failed()
            for io_hash in failed_hashes:
                entry = self._inflight.pop(io_hash, None)
                if entry is None:
                    logger.warning(
                        "KVStream: drained failed hash %s with no "
                        "inflight entry",
                        io_hash,
                    )
                    continue

                (
                    key,
                    memory_obj,
                    size,
                    _shape,
                    _dtype,
                    _fmt,
                    _cached_positions,
                    _on_complete_callback,
                ) = entry

                logger.error(
                    "KVStream: write failed for key %s (io_hash=%s)",
                    key,
                    io_hash,
                )

                memory_obj.ref_count_down()
                self._remove_put_task(key)

                # Roll back the pre-reserved capacity
                with self.disk_lock:
                    self.current_cache_size -= size

    # ------------------------------------------------------------------ #
    #  Put (write) path — drain-based, no background threads               #
    # ------------------------------------------------------------------ #

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Submit a batch of KV chunks for async disk write via KVStream.

        **Write path flow (all on the calling thread, no blocking):**

        1. ``_drain_completed()`` — reap any previously finished writes.
        2. Per-key dedup check, eviction loop, capacity reservation.
        3. ``ref_count_up`` + ``engine.save()`` per key — the C++ engine
           queues entries internally and a background flushing thread
           submits them to io_uring, absorbing any SQ ring backpressure.
        4. Record in ``_inflight`` for later drain bookkeeping.
        5. Return immediately.

        Args:
            keys: Cache keys for the KV chunks.
            objs: Memory objects containing the KV data.
            transfer_spec: Unused (present for interface compatibility).
            on_complete_callback: Optional callback invoked per key after
                that key's disk write completes.  Exceptions are caught and
                logged.
        """
        # Step 1: drain previously completed/failed writes
        self._drain_completed()

        # Step 2-3: process each key
        for key, memory_obj in zip(keys, objs, strict=False):
            assert memory_obj.tensor is not None

            # Skip repeated save
            if self.exists_in_put_tasks(key):
                logger.debug("Put task for %s is already in progress.", key)
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
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.dict, num_candidates=1
                    )
                    if not evict_keys:
                        logger.warning(
                            "KVStream: no eviction candidates. "
                            "Disk space under pressure."
                        )
                        evict_success = False
                        break

                    for evict_key in evict_keys:
                        self.current_cache_size -= self.dict[evict_key].size

                    self.batched_remove(evict_keys, force=False)

                if evict_success:
                    self.current_cache_size += required_size

            if not evict_success:
                self._remove_put_task(key)
                continue

            self.cache_policy.update_on_put(key)

            # ref_count_up BEFORE returning — storage_manager does
            # ref_count_down after batched_submit_put_task returns, so
            # the buffer must stay alive while the async write is in-flight.
            memory_obj.ref_count_up()

            # Collect metadata for inflight bookkeeping
            raw_tensor = memory_obj.raw_tensor
            path = self._key_to_path(key)
            size = memory_obj.get_physical_size()
            shape = memory_obj.metadata.shape
            dtype = memory_obj.metadata.dtype
            fmt = memory_obj.metadata.fmt
            cached_positions = memory_obj.metadata.cached_positions

            io_hash = self._next_io_hash(key, "save")

            # Submit non-blocking SQE to the C++ write engine.
            # The C++ engine queues the entry internally and a background
            # flushing thread submits it to io_uring, absorbing any
            # backpressure from a full SQ ring.
            self.engine.save(io_hash, raw_tensor, path, 0)

            # Record in inflight dict for later drain
            self._inflight[io_hash] = (
                key,
                memory_obj,
                size,
                shape,
                dtype,
                fmt,
                cached_positions,
                on_complete_callback,
            )

    # ------------------------------------------------------------------ #
    #  Get (read) path                                                     #
    # ------------------------------------------------------------------ #

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking read of a KV chunk from disk via KVStream.

        Allocates a ``MemoryObj`` from the CPU backend, submits a
        KVStream load, and blocks until the read completes.

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

        disk_meta = self.dict[key]
        path = disk_meta.path
        dtype = disk_meta.dtype
        shape = disk_meta.shape
        fmt = disk_meta.fmt
        assert dtype is not None
        assert shape is not None

        self.disk_lock.release()

        # Allocate destination memory
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, (
            "Memory allocation failed during KVStream disk load."
        )

        # Submit load and wait (READ engine only)
        raw_tensor = memory_obj.raw_tensor
        io_hash = self._next_io_hash(key, "load")

        start_time = time.time()
        self.engine.load(io_hash, raw_tensor, path, 0)
        self.engine.wait_one(io_hash, self.kvstream_core.IOQueue.READ)
        elapsed = time.time() - start_time

        size = memory_obj.get_physical_size()
        if elapsed > 0:
            logger.debug(
                "KVStream blocking load: %d bytes, %.2f MB/s",
                size,
                size / elapsed / 1e6,
            )

        # Recover cached_positions metadata
        cached_positions = self.dict.get(key, None)
        if cached_positions is not None:
            memory_obj.metadata.cached_positions = (
                cached_positions.cached_positions
            )

        return memory_obj

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Batched blocking read using io_uring batch submission.

        Overrides the default sequential ``get_blocking`` loop from
        ``StorageBackendInterface``.  All reads are submitted to io_uring
        in a single ``load_batch`` call, allowing the kernel to process
        them concurrently.  Then ``wait_all`` blocks until every read
        completes.

        Args:
            keys: Ordered list of cache keys to load.

        Returns:
            List of ``MemoryObj`` (or ``None`` for missing keys), in the
            same order as *keys*.
        """
        mem_objs: List[Optional[MemoryObj]] = []
        load_entries: list[tuple[str, "torch.Tensor", str, int]] = []
        # Track which indices in mem_objs have valid io_uring loads
        valid_indices: list[int] = []

        for i, key in enumerate(keys):
            with self.disk_lock:
                if key not in self.dict:
                    mem_objs.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.dict)
                disk_meta = self.dict[key]
                path = disk_meta.path
                dtype = disk_meta.dtype
                shape = disk_meta.shape
                fmt = disk_meta.fmt

            assert dtype is not None
            assert shape is not None

            memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
            assert memory_obj is not None, (
                "Memory allocation failed during KVStream batched disk load."
            )

            io_hash = self._next_io_hash(key, "load")
            load_entries.append((io_hash, memory_obj.raw_tensor, path, 0))
            mem_objs.append(memory_obj)
            valid_indices.append(i)

        if load_entries:
            start_time = time.time()
            self.engine.load_batch(load_entries)
            self.engine.wait_all(self.kvstream_core.IOQueue.READ)
            elapsed = time.time() - start_time

            total_bytes = sum(
                mem_objs[i].get_physical_size()  # type: ignore[union-attr]
                for i in valid_indices
            )
            if elapsed > 0:
                logger.debug(
                    "KVStream batched_get_blocking: %d entries, %d bytes, "
                    "%.2f MB/s",
                    len(load_entries),
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
        """Submit all reads via ``load_batch`` without waiting.

        Pre-allocates ``MemoryObj`` instances for every valid key and
        submits all io_uring read SQEs in one batch.  Returns immediately
        so the caller can interleave ``wait_one_load`` completions with
        GPU transfers.

        Args:
            keys: Ordered list of cache keys to load.

        Returns:
            A list with one entry per key.  Each entry is either
            ``(io_hash, key, memory_obj)`` for a valid key, or ``None``
            if the key was not found in the disk cache.
        """
        results: List[Optional[tuple[str, CacheEngineKey, MemoryObj]]] = []
        load_entries: list[tuple[str, "torch.Tensor", str, int]] = []

        for key in keys:
            with self.disk_lock:
                if key not in self.dict:
                    results.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.dict)
                disk_meta = self.dict[key]
                path = disk_meta.path
                dtype = disk_meta.dtype
                shape = disk_meta.shape
                fmt = disk_meta.fmt

            assert dtype is not None
            assert shape is not None

            memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
            assert memory_obj is not None, (
                "Memory allocation failed during KVStream overlapped load."
            )

            io_hash = self._next_io_hash(key, "load")
            load_entries.append((io_hash, memory_obj.raw_tensor, path, 0))
            results.append((io_hash, key, memory_obj))

        if load_entries:
            self.engine.load_batch(load_entries)

        return results

    def wait_one_load(
        self,
        io_hash: str,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """Block until a single read operation completes.

        Called in a loop by the overlapped retrieval path in
        ``CacheEngine._retrieve_overlapped_kvstream`` to wait for
        one chunk at a time, allowing GPU transfers to be interleaved.

        Args:
            io_hash: The KVStream operation hash from ``submit_batch_load``.
            key: The cache engine key (used to recover metadata).
            memory_obj: The pre-allocated ``MemoryObj`` receiving the data.
        """
        self.engine.wait_one(io_hash, self.kvstream_core.IOQueue.READ)

        # Recover cached_positions metadata
        disk_meta = self.dict.get(key, None)
        if disk_meta is not None:
            memory_obj.metadata.cached_positions = disk_meta.cached_positions

    def _sync_batch_load(
        self,
        io_hashes: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
    ) -> list[MemoryObj]:
        """Block until all KVStream reads complete, then do bookkeeping.

        Runs in a worker thread via ``asyncio.to_thread()``.

        Args:
            io_hashes: KVStream operation hashes (one per entry).
            keys: Cache keys (one per entry).
            memory_objs: Pre-allocated MemoryObjs to receive data.

        Returns:
            The same ``memory_objs`` list, now populated with data.
        """
        start_time = time.time()
        self.engine.wait_all(self.kvstream_core.IOQueue.READ)
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
                mem_obj.metadata.cached_positions = disk_meta.cached_positions
                with self.disk_lock:
                    disk_meta.unpin()

        return memory_objs

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Async batched read of KV chunks from disk via KVStream.

        Pre-allocates all ``MemoryObj`` instances, submits a batch of
        KVStream loads (non-blocking SQE submission), then awaits
        completion on a background thread.

        Args:
            lookup_id: Opaque lookup identifier for logging/tracing.
            keys: Ordered list of cache keys to load.
            transfer_spec: Unused (present for interface compatibility).

        Returns:
            List of ``MemoryObj`` instances populated with the loaded data.
        """
        mem_objs: list[MemoryObj] = []
        io_hashes: list[str] = []
        load_entries: list[tuple[str, torch.Tensor, str, int]] = []

        logger.debug(
            "lookup_id: %s; KVStream prefetching %d keys from disk.",
            lookup_id,
            len(keys),
        )

        for key in keys:
            self.disk_lock.acquire()
            assert key in self.dict, (
                f"Key {key} not found in KVStream disk cache after pinning"
            )

            path = self.dict[key].path
            dtype = self.dict[key].dtype
            shape = self.dict[key].shape
            fmt = self.dict[key].fmt

            assert dtype is not None
            assert shape is not None

            _alloc_t0 = time.time()
            memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
            _alloc_elapsed = time.time() - _alloc_t0
            if _alloc_elapsed > 0.01:
                logger.warning(
                    "D2H allocate for KVStream disk read took %.3fs (key=%s)",
                    _alloc_elapsed,
                    key,
                )
            assert memory_obj is not None, (
                "Memory allocation failed during async KVStream disk load."
            )

            self.dict[key].pin()
            self.cache_policy.update_on_hit(key, self.dict)

            self.disk_lock.release()

            memory_obj.pin()
            mem_objs.append(memory_obj)

            io_hash = self._next_io_hash(key, "load")
            io_hashes.append(io_hash)
            load_entries.append((io_hash, memory_obj.raw_tensor, path, 0))

        # Submit all loads (non-blocking SQE submission to read engine)
        self.engine.load_batch(load_entries)

        # Wait for completion on a background thread (READ engine only)
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
        """Shut down the KVStream engine and flush pending messages.

        1. Stops the background drain thread.
        2. Calls ``engine.shutdown()`` which internally drains the
           submit queue, waits for all pending ops on both read and
           write engines, then tears down the io_uring rings and
           closes the FD cache.
        3. Calls ``_drain_completed()`` one final time to do remaining
           ``ref_count_down`` / ``insert_key`` bookkeeping for any writes
           that completed during shutdown.
        4. Closes the batched message sender.
        """
        # Stop background drain thread
        self._drain_stop.set()
        self._drain_thread.join(timeout=2.0)
        if self._drain_thread.is_alive():
            logger.warning(
                "KVStream drain thread did not stop within 2s"
            )

        self.engine.shutdown()
        self._drain_completed()

        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()

        logger.info("KVStreamDiskBackend closed.")
