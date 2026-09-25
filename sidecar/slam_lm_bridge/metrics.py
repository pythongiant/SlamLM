"""Real telemetry: hardware facts, memory and token rates.

Everything here is measured, never estimated:

* machine identity from ``sysctl`` / ``system_profiler``;
* memory from ``mlx.core`` (MLX-owned arrays, its cache pool, its peak) plus
  the bridge process' own resident set;
* token rates from wall-clock timestamps the runner records per token.

A field with no data reports ``0`` (or ``0.0``); the UI renders an empty state.
"""

from __future__ import annotations

import ctypes
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

import mlx.core as mx

from .protocol import Hardware, LiveMetrics, StatePayload, sanitise_float

#: Live `decodeTps` is measured over this trailing window so the number stays
#: stable while streaming (see PROTOCOL.md's metric definitions).
DECODE_WINDOW_SECONDS = 2.0
SAMPLE_INTERVAL_SECONDS = 0.2  # 5 Hz


# MARK: - Machine facts


def _sysctl(key: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _gpu_cores() -> int:
    """GPU core count from system_profiler; 0 when it does not report one."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    for line in out.stdout.splitlines():
        text = line.strip()
        if not text.startswith("Total Number of Cores:"):
            continue
        _, _, value = text.partition(":")
        try:
            return int(value.strip())
        except ValueError:
            continue
    return 0


class _MachTaskBasicInfo(ctypes.Structure):
    """``mach_task_basic_info`` (task_basic_info_64 shape, flavour 20)."""

    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", ctypes.c_int32 * 2),
        ("system_time", ctypes.c_int32 * 2),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    ]


_MACH_TASK_BASIC_INFO = 20
_mach: Optional[tuple] = None


def _mach_resident_bytes() -> int:
    """Live resident set size of this process, asked of the kernel."""
    global _mach
    if _mach is None:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        task = ctypes.c_uint.in_dll(lib, "mach_task_self_")
        lib.task_info.restype = ctypes.c_int
        lib.task_info.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        _mach = (lib, task)
    lib, task = _mach
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int))
    if lib.task_info(task.value, _MACH_TASK_BASIC_INFO, ctypes.byref(info), ctypes.byref(count)):
        raise OSError("task_info failed")
    return int(info.resident_size)


def _rusage_peak_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; every other platform reports kibibytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _process_rss_bytes() -> int:
    """Resident set size of the bridge process, right now.

    ``getrusage``'s ``ru_maxrss`` is a high-water mark that never falls, so it
    would keep reporting a model's memory long after an unload. Ask Mach for the
    live figure and use the high-water mark only if Mach cannot be reached.
    """
    if sys.platform == "darwin":
        try:
            return _mach_resident_bytes()
        except Exception:
            pass
    return _rusage_peak_bytes()


class _HostVMStatistics64(ctypes.Structure):
    """``vm_statistics64``: the machine-wide page counts behind Activity Monitor."""

    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class _SwapUsage(ctypes.Structure):
    """``vm.swapusage`` as the kernel returns it: the ``xsu_*`` fields are bytes.

    The trailing ``xsu_pagesize`` is the page size, not a multiplier, so there
    is no scale to apply — verified against ``sysctl -n vm.swapusage``'s
    human-readable form, which renders the same number in mebibytes.
    """

    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_int32),
    ]


#: ``host_statistics`` flavour for ``vm_statistics64``.
_HOST_VM_INFO64 = 4
_host_vm: Optional[ctypes.CDLL] = None
_swap_lib: Optional[ctypes.CDLL] = None
_page_size_bytes: Optional[int] = None
_physical_bytes: Optional[int] = None


def _page_size() -> int:
    """The kernel's page size; 0 when it cannot be read (never assumed)."""
    global _page_size_bytes
    if _page_size_bytes is None:
        size = int(_sysctl("vm.pagesize") or 0)
        if size <= 0:
            try:
                size = int(os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError):
                size = 0
        _page_size_bytes = size
    return _page_size_bytes


def _physical_bytes_now() -> int:
    """Installed RAM from ``hw.memsize``; hardware does not change at runtime."""
    global _physical_bytes
    if _physical_bytes is None:
        _physical_bytes = int(_sysctl("hw.memsize") or 0)
    return _physical_bytes


def _mach_vm_statistics() -> _HostVMStatistics64:
    """Machine-wide page counts, one ``host_statistics64`` call."""
    global _host_vm
    if _host_vm is None:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.mach_host_self.restype = ctypes.c_uint
        lib.host_statistics64.restype = ctypes.c_int
        lib.host_statistics64.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        _host_vm = lib
    lib = _host_vm
    info = _HostVMStatistics64()
    count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
    if lib.host_statistics64(
        lib.mach_host_self(), _HOST_VM_INFO64, ctypes.byref(info), ctypes.byref(count)
    ):
        raise OSError("host_statistics64 failed")
    return info


def _swap_used_bytes() -> int:
    """Swap in use, straight from ``vm.swapusage``."""
    global _swap_lib
    if _swap_lib is None:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.sysctlbyname.restype = ctypes.c_int
        lib.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        _swap_lib = lib
    usage = _SwapUsage()
    size = ctypes.c_size_t(ctypes.sizeof(usage))
    if _swap_lib.sysctlbyname(
        b"vm.swapusage", ctypes.byref(usage), ctypes.byref(size), None, 0
    ):
        raise OSError("sysctl vm.swapusage failed")
    return int(usage.xsu_used)


#: Every machine-wide field, zeroed: what the sampler emits when the kernel
#: will not answer, so the wire always carries the protocol's keys.
ZERO_SYSTEM_MEMORY: Dict[str, int] = {
    "systemUsedBytes": 0,
    "systemAppBytes": 0,
    "systemWiredBytes": 0,
    "systemCompressedBytes": 0,
    "systemCachedBytes": 0,
    "systemSwapBytes": 0,
}


def system_memory() -> Dict[str, int]:
    """What the machine itself is using, in Activity Monitor's terms.

    One ``host_statistics64`` call (times the real page size) plus one
    ``vm.swapusage`` read — both syscalls, so the 5 Hz sampler can afford them.
    Definitions are PROTOCOL.md's: ``systemUsedBytes`` is physical RAM minus
    free minus cached; the page counts are the kernel's own.

    Raises ``OSError`` when the kernel reports no page counts. A ``vm.swapusage``
    that cannot be read costs only ``systemSwapBytes``, which reports 0.
    """
    physical = int(_physical_bytes_now())
    page = int(_page_size())
    if physical <= 0 or page <= 0:
        raise OSError("physical memory or page size unavailable")
    info = _mach_vm_statistics()
    free = int(info.free_count) * page
    cached = (int(info.external_page_count) + int(info.purgeable_count)) * page
    try:
        swap = _swap_used_bytes()
    except Exception:
        swap = 0  # the affected field reports 0; the page counts still stand
    return {
        "systemUsedBytes": max(0, physical - free - cached),
        "systemAppBytes": max(
            0, (int(info.internal_page_count) - int(info.purgeable_count)) * page
        ),
        "systemWiredBytes": int(info.wire_count) * page,
        "systemCompressedBytes": int(info.compressor_page_count) * page,
        "systemCachedBytes": cached,
        "systemSwapBytes": max(0, swap),
    }


def mlx_memory() -> tuple[int, int, int]:
    """(active, peak, cache) bytes as MLX itself counts them."""
    return (
        int(mx.get_active_memory()),
        int(mx.get_peak_memory()),
        int(mx.get_cache_memory()),
    )


def device_memory() -> tuple[int, int]:
    """(total, recommended working set) bytes for the MLX device."""
    info = mx.device_info()
    total = int(info.get("memory_size") or 0)
    recommended = int(info.get("max_recommended_working_set_size") or 0)
    return total, recommended


class HardwareProbe:
    """Machine identity, computed once and cached.

    ``system_profiler`` takes a noticeable moment, so the bridge warms the
    probe in the background at startup and ``identity()`` waits for that first
    computation instead of racing it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._identity: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._on_error: Optional[Callable[[str], None]] = None
        self._system_memory_reported = False

    def set_error_sink(self, on_error: Optional[Callable[[str], None]]) -> None:
        """Where a failing machine-memory read is reported (the sampler wires
        this to the writer's `log` events)."""
        self._on_error = on_error

    def system_memory(self) -> Dict[str, int]:
        """The six machine-wide fields, zeros when the kernel will not answer.

        A failure is reported through the error sink exactly once, however many
        times the 5 Hz sampler calls this.
        """
        try:
            return system_memory()
        except Exception as exc:
            if not self._system_memory_reported:
                self._system_memory_reported = True
                if self._on_error is not None:
                    self._on_error(f"system memory unavailable: {exc}")
            return dict(ZERO_SYSTEM_MEMORY)

    def warm(self) -> None:
        with self._lock:
            if self._identity is not None or self._thread is not None:
                return
            thread = threading.Thread(
                target=self._compute, name="slam-lm-hardware", daemon=True
            )
            self._thread = thread
        thread.start()

    def _compute(self) -> None:
        identity = {
            "model": _sysctl("hw.model"),
            "chip": _sysctl("machdep.cpu.brand_string") or platform.processor(),
            "gpuCores": _gpu_cores(),
        }
        total, recommended = device_memory()
        identity["totalBytes"] = total
        identity["recommendedBytes"] = recommended
        with self._lock:
            self._identity = identity

    def identity(self) -> Dict[str, Any]:
        self.warm()
        thread = self._thread
        with self._lock:
            cached = self._identity
        if cached is not None:
            return dict(cached)
        # The warm thread is still probing: wait for its result instead of
        # running a second `system_profiler` and blocking twice as long.
        if thread is not None and thread.is_alive():
            thread.join(timeout=30)
        with self._lock:
            cached = self._identity
        if cached is not None:
            return dict(cached)
        self._compute()
        with self._lock:
            return dict(self._identity or {})

    def sample(self) -> Hardware:
        """Identity plus a live memory sample (used for `hello`)."""
        facts = self.identity()
        active, peak, cache = mlx_memory()
        return Hardware(
            model=str(facts.get("model", "")),
            chip=str(facts.get("chip", "")),
            gpuCores=int(facts.get("gpuCores", 0)),
            totalBytes=int(facts.get("totalBytes", 0)),
            recommendedBytes=int(facts.get("recommendedBytes", 0)),
            memoryActiveBytes=active,
            memoryPeakBytes=peak,
            cacheBytes=cache,
            processRssBytes=_process_rss_bytes(),
            **self.system_memory(),
        )


# MARK: - Runner-owned counters


class RunnerStats:
    """The runner's shared telemetry, read at 5 Hz by the sampler.

    The runner mutates it on its worker thread; the sampler, the command
    dispatcher and HTTP handler threads read it. One lock covers all of it.
    `tokensGenerated` rises as tokens are produced; `requests` counts finished
    requests, including failures.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = "idle"
        self._phase = "idle"
        self._model: Optional[str] = None
        self._load_ms = 0.0
        self._tokens_generated = 0
        self._requests = 0
        self._ttft_ms = 0.0
        self._prefill_tps = 0.0
        self._token_times: Deque[float] = deque()

    # -- transitions

    def set_state(
        self,
        status: str,
        phase: str,
        model: Optional[str] = None,
        message: Optional[str] = None,
    ) -> StatePayload:
        with self._lock:
            self._status = status
            self._phase = phase
            self._model = model
        return StatePayload(status=status, phase=phase, model=model, message=message)

    def set_load_ms(self, load_ms: float) -> None:
        with self._lock:
            self._load_ms = load_ms

    # -- per-request accumulation

    def note_ttft(self, ttft_ms: float, prefill_tps: float) -> None:
        with self._lock:
            self._ttft_ms = ttft_ms
            self._prefill_tps = prefill_tps

    def note_token(self, when: Optional[float] = None) -> None:
        """Record one produced token: feeds the window and the session total."""
        stamp = time.monotonic() if when is None else when
        with self._lock:
            self._token_times.append(stamp)
            self._tokens_generated += 1

    def finish_request(self) -> None:
        """Count one finished request, successful or not."""
        with self._lock:
            self._requests += 1

    def clear_token_window(self) -> None:
        with self._lock:
            self._token_times.clear()

    # -- reads

    def decode_tps(self, now: Optional[float] = None) -> float:
        """Tokens/s over the trailing window; 0.0 when it cannot be measured."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            times = self._token_times
            cutoff = moment - DECODE_WINDOW_SECONDS
            while times and times[0] < cutoff:
                times.popleft()
            count = len(times)
            if count < 2:
                return 0.0
            span = times[-1] - times[0]
            if span <= 0:
                return 0.0
            return sanitise_float((count - 1) / span)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "phase": self._phase,
                "model": self._model,
                "loadMs": sanitise_float(self._load_ms),
                "tokensGenerated": int(self._tokens_generated),
                "requests": int(self._requests),
                "ttftMs": sanitise_float(self._ttft_ms),
                "prefillTps": sanitise_float(self._prefill_tps),
            }


def live_metrics(stats: RunnerStats, probe: HardwareProbe) -> LiveMetrics:
    """One `metrics` sample: all real numbers, zeroed when there is no data."""
    facts = probe.identity()
    active, peak, cache = mlx_memory()
    snap = stats.snapshot()
    return LiveMetrics(
        ts=time.time(),
        memoryActiveBytes=active,
        memoryPeakBytes=peak,
        memoryCacheBytes=cache,
        memoryTotalBytes=int(facts.get("totalBytes", 0)),
        memoryRecommendedBytes=int(facts.get("recommendedBytes", 0)),
        processRssBytes=_process_rss_bytes(),
        **probe.system_memory(),
        decodeTps=stats.decode_tps(),
        prefillTps=snap["prefillTps"],
        ttftMs=snap["ttftMs"],
        tokensGenerated=snap["tokensGenerated"],
        requests=snap["requests"],
        status=snap["status"],
        phase=snap["phase"],
        model=snap["model"],
        loadMs=snap["loadMs"],
    )


class MetricsSampler:
    """Emits a `metrics` event every 200 ms for the whole bridge lifetime.

    Memory fields stay valid with no model loaded; rates stay 0.0 until real
    tokens have been produced.
    """

    def __init__(
        self,
        stats: RunnerStats,
        emit: Callable[[LiveMetrics], None],
        probe: Optional[HardwareProbe] = None,
        interval: float = SAMPLE_INTERVAL_SECONDS,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self._stats = stats
        self._emit = emit
        self._probe = probe or HardwareProbe()
        if on_error is not None:
            self._probe.set_error_sink(on_error)
        self._interval = interval
        self._on_error = on_error
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._probe.warm()
        self._thread = threading.Thread(
            target=self._run, name="slam-lm-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while True:
            try:
                self._emit(live_metrics(self._stats, self._probe))
            except Exception as exc:  # a sampler hiccup must never kill telemetry
                if self._on_error is not None:
                    self._on_error(f"metrics sample failed: {exc}")
            if self._stop.wait(self._interval):
                return
