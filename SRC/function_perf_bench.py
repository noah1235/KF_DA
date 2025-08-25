import time
import tracemalloc
from statistics import median
from typing import Any, Callable, Tuple, Dict
import pandas as pd
try:
    import psutil, os
    _PS = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PS = None

def _maybe_sync(x):
    # JAX arrays (and some other device arrays) have block_until_ready
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()

def bench(
    func: Callable, 
    *args, 
    runs: int = 10, 
    sync: Callable[[Any], None] | None = _maybe_sync, 
    warmup: bool = True, 
    **kwargs
) -> Dict[str, float]:
    """
    Returns dict with median/mean runtime (ms), best/worst, peak tracemalloc (MB),
    and optional RSS delta (MB) if psutil is available.
    """
    # --- warmup (important for JIT / caching) ---
    if warmup:
        out = func(*args, **kwargs)
        if sync is not None:
            # ensure device work finished before benchmarking
            if isinstance(out, tuple):
                for o in out: sync(o)
            else:
                sync(out)

    times = []
    peaks_mb = []
    rss_deltas_mb = []

    for _ in range(runs):
        # tracemalloc tracks Python allocations; reset peak for per-run peak
        tracemalloc.start()
        try:
            if hasattr(tracemalloc, "reset_peak"):
                tracemalloc.reset_peak()

            rss_before = _PS.memory_info().rss if _PS else None

            t0 = time.perf_counter()
            out = func(*args, **kwargs)
            # For JAX: force completion before stopping timer
            if sync is not None:
                if isinstance(out, tuple):
                    for o in out: sync(o)
                else:
                    sync(out)
            t1 = time.perf_counter()

            cur, peak = tracemalloc.get_traced_memory()  # bytes
            times.append((t1 - t0) * 1e3)  # ms
            peaks_mb.append(peak / (1024**2))

            if _PS:
                rss_after = _PS.memory_info().rss
                rss_deltas_mb.append((rss_after - rss_before) / (1024**2))
        finally:
            tracemalloc.stop()

    def _mean(xs): return sum(xs) / len(xs)

    stats = {
        "time_ms_median": median(times),
        "time_ms_mean": _mean(times),
        "time_ms_best": min(times),
        "time_ms_worst": max(times),
        "peak_tracemalloc_mb_median": median(peaks_mb),
        "peak_tracemalloc_mb_mean": _mean(peaks_mb),
    }
    if rss_deltas_mb:
        stats.update({
            "rss_delta_mb_median": median(rss_deltas_mb),
            "rss_delta_mb_mean": _mean(rss_deltas_mb),
        })
    return pd.DataFrame([stats])
