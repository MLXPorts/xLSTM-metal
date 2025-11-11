try:
    import ray  # type: ignore
    from ray.util.metrics import Gauge  # type: ignore
except Exception:  # pragma: no cover
    ray = None  # type: ignore
    Gauge = None  # type: ignore


class _Noop:
    @staticmethod
    def set(*a, **k):
        """

        :param a:
        :param k:
        :return:
        """
        return None


def make_gauges(prefix: str = "xlstm"):
    """Create Ray dashboard gauges if Ray is initialized; otherwise return no-ops.

    Returns a dict of callables: {name: gauge.set(value)}
    """
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return {
            "mem_rss_mb": _Noop(),
            "mps_alloc_mb": _Noop(),
            "tok_s_decode": _Noop(),
            "tok_s_prefill": _Noop(),
        }
    try:
        g_mem = Gauge(f"{prefix}_mem_rss_mb", "Process RSS in MB")
        g_mps = Gauge(f"{prefix}_mps_alloc_mb", "MPS allocated MB")
        g_dec = Gauge(f"{prefix}_tok_s_decode", "Decode tokens/second")
        g_pre = Gauge(f"{prefix}_tok_s_prefill", "Prefill tokens/second")
        return {
            "mem_rss_mb": g_mem,
            "mps_alloc_mb": g_mps,
            "tok_s_decode": g_dec,
            "tok_s_prefill": g_pre,
        }
    except Exception:
        return {
            "mem_rss_mb": _Noop(),
            "mps_alloc_mb": _Noop(),
            "tok_s_decode": _Noop(),
            "tok_s_prefill": _Noop(),
        }

