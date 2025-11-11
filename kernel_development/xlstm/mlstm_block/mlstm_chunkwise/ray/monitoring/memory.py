import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil optional
    psutil = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class MemStats:
    ts: float
    rss_mb: float
    avail_mb: float | None
    total_mb: float | None
    mps_alloc_mb: float | None
    mps_reserved_mb: float | None


class MemoryPressureAbort(RuntimeError):
    pass


def _get_process_mem_mb() -> float:
    if psutil is not None:
        try:
            p = psutil.Process()
            return p.memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS; normalize by assuming KB if < 1e9
        val = usage.ru_maxrss
        if val < 1e9:
            return float(val) / 1024.0
        else:
            return float(val) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _get_system_mem_mb() -> tuple[Optional[float], Optional[float]]:
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return vm.available / (1024 * 1024), vm.total / (1024 * 1024)
        except Exception:
            pass
    return None, None


def _get_mps_mem_mb() -> tuple[Optional[float], Optional[float]]:
    if torch is None:
        return None, None
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            alloc = getattr(torch.mps, "current_allocated_memory", None)
            reserv = getattr(torch.mps, "current_reserved_memory", None)
            a = float(alloc()) / (1024 * 1024) if callable(alloc) else None
            r = float(reserv()) / (1024 * 1024) if callable(reserv) else None
            return a, r
    except Exception:
        pass
    return None, None


def snapshot() -> MemStats:
    """

    :return:
    """
    ts = time.time()
    rss = _get_process_mem_mb()
    avail, total = _get_system_mem_mb()
    mps_alloc, mps_reserved = _get_mps_mem_mb()
    return MemStats(ts, rss, avail, total, mps_alloc, mps_reserved)


class MemoryMonitor:
    """
    Lightweight memory monitor with optional watchdog actions.

    Env configuration (overridable by ctor args):
    - XLSTM_MEM_WATCHDOG: enable (default "1")
    - XLSTM_MEM_POLL_MS: poll interval in ms (default 200)
    - XLSTM_MEM_SOFT_PCT: soft threshold as fraction of total (default 0.85)
    - XLSTM_MEM_HARD_PCT: hard threshold as fraction of total (default 0.92)
    - XLSTM_MEM_SOFT_MB / XLSTM_MEM_HARD_MB: absolute MB thresholds (override pct)
    - XLSTM_MEM_ACTION: comma list of actions on soft threshold: warn,empty_cache
    """

    def __init__(
            self,
            poll_ms: Optional[int] = None,
            soft_mb: Optional[float] = None,
            hard_mb: Optional[float] = None,
            on_soft: Optional[Callable[[MemStats], None]] = None,
            on_hard: Optional[Callable[[MemStats], None]] = None,
            log_csv_path: Optional[str] = None,
    ):
        self.enabled = os.environ.get("XLSTM_MEM_WATCHDOG", "1") == "1"
        self.poll_ms = poll_ms or int(os.environ.get("XLSTM_MEM_POLL_MS", "200"))
        self.soft_mb = soft_mb or float(os.environ.get("XLSTM_MEM_SOFT_MB", "0") or 0)
        self.hard_mb = hard_mb or float(os.environ.get("XLSTM_MEM_HARD_MB", "0") or 0)
        self.soft_pct = float(os.environ.get("XLSTM_MEM_SOFT_PCT", "0.85"))
        self.hard_pct = float(os.environ.get("XLSTM_MEM_HARD_PCT", "0.92"))
        self.actions = [s.strip() for s in os.environ.get("XLSTM_MEM_ACTION", "warn,empty_cache").split(",") if
                        s.strip()]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.on_soft = on_soft
        self.on_hard = on_hard
        self.log_csv_path = log_csv_path
        self._csv_fp = None

    def start(self):
        """

        :return:
        """
        if not self.enabled:
            return self
        if self.log_csv_path:
            try:
                import os as _os
                _os.makedirs(_os.path.dirname(self.log_csv_path) or ".", exist_ok=True)
                self._csv_fp = open(self.log_csv_path, "w")
                self._csv_fp.write("ts,rss_mb,avail_mb,total_mb,mps_alloc_mb,mps_reserved_mb\n")
            except Exception:
                self._csv_fp = None
        self._thread = threading.Thread(target=self._run, name="xLSTM-MemWatch", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """

        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._csv_fp is not None:
            try:
                self._csv_fp.close()
            except Exception:
                pass

    @staticmethod
    def _maybe_empty_cache():
        if torch is None:
            return
        try:
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _thresholds_mb(self, total_mb: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        soft_mb = self.soft_mb
        hard_mb = self.hard_mb
        if (soft_mb == 0 or hard_mb == 0) and total_mb:
            if soft_mb == 0:
                soft_mb = total_mb * self.soft_pct
            if hard_mb == 0:
                hard_mb = total_mb * self.hard_pct
        return soft_mb or None, hard_mb or None

    def _run(self):
        while not self._stop.is_set():
            st = snapshot()
            if self._csv_fp is not None:
                try:
                    self._csv_fp.write(
                        f"{st.ts:.6f},{st.rss_mb:.1f},{(st.avail_mb or 0):.1f},{(st.total_mb or 0):.1f},{(st.mps_alloc_mb or 0):.1f},{(st.mps_reserved_mb or 0):.1f}\n"
                    )
                except Exception:
                    pass

            total_mb = st.total_mb
            soft_mb, hard_mb = self._thresholds_mb(total_mb)

            # Evaluate thresholds against process RSS as a proxy for unified memory
            try:
                if hard_mb and st.rss_mb >= hard_mb:
                    if self.on_hard:
                        self.on_hard(st)
                    # If on_hard did not raise, escalate
                    raise MemoryPressureAbort(
                        f"Hard memory limit reached: rss={st.rss_mb:.1f} MB >= {hard_mb:.1f} MB"
                    )
                if soft_mb and st.rss_mb >= soft_mb:
                    # Soft actions
                    if "warn" in self.actions:
                        print(
                            f"[xLSTM][mem] Soft limit: rss={st.rss_mb:.1f} MB (soft={soft_mb:.1f} MB).",
                            flush=True,
                        )
                    if "empty_cache" in self.actions:
                        self._maybe_empty_cache()
                    if self.on_soft:
                        self.on_soft(st)
            except MemoryPressureAbort:
                # Surface to main thread by storing exception and stopping
                self._stop.set()
                # Best-effort cache clear to free memory quickly
                self._maybe_empty_cache()
                # Re-raise on main thread when polled (drivers call .check())
                self._exc = MemoryPressureAbort  # type: ignore[attr-defined]
                return
            time.sleep(max(self.poll_ms, 50) / 1000.0)

    def check(self):
        """

        """
        # If background thread recorded abort, raise here
        exc = getattr(self, "_exc", None)
        if exc is not None:
            raise MemoryPressureAbort("Aborted due to memory pressure (watchdog)")
