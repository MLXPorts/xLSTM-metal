"""
"xltop": lightweight terminal monitor for xLSTM on Apple/MPS.

Features
- Live memory snapshot (process RSS, system avail/total, MPS allocated/reserved, recommended cap).
- Optional Ray cluster summary (via `ray status` if available).
- Top processes by RSS (psutil if present, else ps fallback).
- Basic controls:
  - q: quit
  - p: pause/resume refresh
  - s: set sampling interval
  - l: toggle CSV logging (uses MemoryMonitor) → runs/xltop/mem_<ts>.csv
  - C: clear MPS cache (this process)
  - r: refresh Ray status now
  - k: ray stop --force (cleanup lingering daemons)
  - K: kill a PID (TERM, then KILL)
  - h: help

Fallback
- Use --no-curses to print a single snapshot (or loop with --poll).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

try:
    import curses  # type: ignore
except Exception:
    curses = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from pathlib import Path

# Ensure repo root on sys.path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import signal as _signal


def fmt_mb(x: Optional[float]) -> str:
    """

    :param x:
    :return:
    """
    if x is None:
        return "-"
    return f"{x:8.1f} MB"


def fmt_gb(x: Optional[float]) -> str:
    """

    :param x:
    :return:
    """
    if x is None:
        return "-"
    return f"{x / 1024.0:6.2f} GB"


def get_recommended_mps_gb() -> Optional[float]:
    """

    :return:
    """
    try:
        if torch is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
            rm = getattr(torch.mps, "recommended_max_memory", None)
            if callable(rm):
                return float(rm()) / (1024.0 * 1024.0 * 1024.0)
    except Exception:
        return None
    return None


def get_top_processes(limit: int = 6) -> List[Tuple[int, str, float]]:
    """

    :param limit:
    :return:
    """
    # returns list of (pid, name, rss_mb)
    out: List[Tuple[int, str, float]] = []
    if psutil is not None:
        try:
            for p in psutil.process_iter(["pid", "name", "memory_info"]):
                try:
                    mi = p.info.get("memory_info")
                    rss = float(mi.rss) / (1024 * 1024) if mi else 0.0
                    out.append((int(p.info.get("pid")), str(p.info.get("name")), rss))
                except Exception:
                    continue
            out.sort(key=lambda t: t[2], reverse=True)
            return out[:limit]
        except Exception:
            pass
    # Fallback: ps
    try:
        ps = subprocess.run(
            ["ps", "-axo", "pid,rss,comm"], capture_output=True, text=True
        )
        lines = ps.stdout.strip().splitlines()[1:]
        for ln in lines:
            try:
                pid_s, rss_kb_s, comm = ln.strip().split(None, 2)
                rss_mb = float(rss_kb_s) / 1024.0
                out.append((int(pid_s), comm, rss_mb))
            except Exception:
                continue
        out.sort(key=lambda t: t[2], reverse=True)
        return out[:limit]
    except Exception:
        return out


def ray_status_text(timeout: float = 2.5) -> str:
    """

    :param timeout:
    :return:
    """
    ray_bin = shutil.which("ray")
    if not ray_bin:
        return "ray: not installed"
    try:
        proc = subprocess.run(
            [ray_bin, "status"], capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode == 0:
            # Collapse to a short section: Node status + Resources
            text = proc.stdout
            keep = []
            capture = False
            for ln in text.splitlines():
                if "======== Autoscaler status" in ln or ln.strip() == "Node status":
                    keep.append(ln)
                    capture = True
                    continue
                if capture and ln.strip().startswith("Resources"):
                    keep.append(ln)
                    continue
                if capture and ln.strip().startswith("Demands"):
                    keep.append(ln)
                    capture = False
                if capture:
                    keep.append(ln)
            return "\n".join(keep) or text
        else:
            err = (proc.stderr or proc.stdout or "").strip()
            if "Could not find any running Ray instance" in err:
                return "ray: not running"
            return err or "ray status error"
    except subprocess.TimeoutExpired:
        return "ray status: timeout"
    except Exception as e:
        return f"ray status error: {e}"


def ray_stop_force() -> str:
    """

    :return:
    """
    ray_bin = shutil.which("ray")
    if not ray_bin:
        return "ray not found"
    try:
        proc = subprocess.run([ray_bin, "stop", "--force"], capture_output=True, text=True)
        if proc.returncode == 0:
            return "ray stop --force: done"
        return f"ray stop failed: {proc.stderr.strip() or proc.stdout.strip()}"
    except Exception as e:
        return f"ray stop error: {e}"


def kill_pid(pid: int) -> str:
    """

    :param pid:
    :return:
    """
    try:
        os.kill(pid, 15)
        time.sleep(0.5)
        # check if still alive
        try:
            os.kill(pid, 0)
        except OSError:
            return f"TERM {pid}: ok"
        os.kill(pid, 9)
        return f"KILL {pid}: ok"
    except Exception as e:
        return f"kill {pid}: {e}"


def clear_mps_cache() -> str:
    """

    :return:
    """
    try:
        if torch is not None and hasattr(torch, "mps"):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
            return "torch.mps.empty_cache(): ok"
        return "MPS not available"
    except Exception as e:
        return f"empty_cache error: {e}"


@dataclass
class State:
    interval: float = 1.0
    paused: bool = False
    last_msg: str = ""
    log_path: Optional[str] = None
    mem_logger: Optional[MemoryMonitor] = None
    show_ray: bool = True
    stats_path: Optional[str] = None
    stats_inst: Optional[float] = None
    stats_avg: Optional[float] = None


def read_stats_csv_tail(path: str) -> tuple[Optional[float], Optional[float]]:
    """

    :param path:
    :return:
    """
    try:
        p = Path(path)
        if not p.exists():
            return None, None
        # Read last ~4KB
        with p.open('rb') as f:
            try:
                f.seek(-4096, 2)
            except Exception:
                f.seek(0)
            tail = f.read().decode(errors='ignore').splitlines()
        # Find last non-header data line
        for ln in reversed(tail):
            if not ln or ln.startswith('step'):
                continue
            parts = ln.split(',')
            if len(parts) >= 5:
                inst = float(parts[3]) if parts[3] else None
                avg = float(parts[4]) if parts[4] else None
                return inst, avg
        return None, None
    except Exception:
        return None, None


def read_stats_series(path: str, max_points: int = 24) -> list[float]:
    """

    :param path:
    :param max_points:
    :return:
    """
    try:
        p = Path(path)
        if not p.exists():
            return []
        with p.open('rb') as f:
            try:
                f.seek(-8192, 2)
            except Exception:
                f.seek(0)
            tail = f.read().decode(errors='ignore').splitlines()
        vals: list[float] = []
        for ln in tail:
            if not ln or ln.startswith('step'):
                continue
            parts = ln.split(',')
            if len(parts) >= 5 and parts[3]:
                try:
                    vals.append(float(parts[3]))
                except Exception:
                    continue
        return vals[-max_points:]
    except Exception:
        return []


def render_sparkline(vals: list[float], width: int = 40) -> str:
    """

    :param vals:
    :param width:
    :return:
    """
    if not vals:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    mmin, mmax = min(vals), max(vals)
    if mmax <= mmin:
        return blocks[0] * min(len(vals), width)
    # resample to width
    n = len(vals)
    if n > width:
        step = n / width
        xs = [vals[int(i * step)] for i in range(width)]
    else:
        xs = vals
    out = []
    for v in xs:
        idx = int((v - mmin) / (mmax - mmin) * (len(blocks) - 1))
        idx = max(0, min(idx, len(blocks) - 1))
        out.append(blocks[idx])
    return ''.join(out)


def draw_screen(stdscr, st: State):
    """

    :param stdscr:
    :param st:
    """
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    stitle = f"xltop — {now}  interval={st.interval:.1f}s  paused={'yes' if st.paused else 'no'}"
    stdscr.addnstr(0, 0, stitle.ljust(w), w)

    s = snapshot()
    rec_gb = get_recommended_mps_gb()
    used_mb = s.total_mb - s.avail_mb if (s.total_mb and s.avail_mb) else None
    used_pct = (used_mb / s.total_mb * 100.0) if (used_mb and s.total_mb) else None
    line1 = (
        f"Process RSS: {fmt_mb(s.rss_mb)}   System: used={fmt_mb(used_mb)} ({used_pct:.1f}% )  total={fmt_gb(s.total_mb)}"
        if used_pct is not None
        else f"Process RSS: {fmt_mb(s.rss_mb)}   System total={fmt_gb(s.total_mb)}"
    )
    stdscr.addnstr(2, 0, line1.ljust(w), w)
    line2 = (
        f"MPS: alloc={fmt_mb(s.mps_alloc_mb)}  reserved={fmt_mb(s.mps_reserved_mb)}  recommended={rec_gb:.2f} GB"
        if rec_gb is not None
        else f"MPS: alloc={fmt_mb(s.mps_alloc_mb)}  reserved={fmt_mb(s.mps_reserved_mb)}"
    )
    stdscr.addnstr(3, 0, line2.ljust(w), w)

    # Top processes
    stdscr.addnstr(5, 0, "Top processes by RSS (MB):".ljust(w), w)
    procs = get_top_processes()
    for i, (pid, name, rss) in enumerate(procs, start=6):
        stdscr.addnstr(i, 0, f"  {pid:>7}  {rss:8.1f}  {name}".ljust(w), w)

    row = 13
    # Stats (tokens/sec) if provided
    if st.stats_path:
        st.stats_inst, st.stats_avg = read_stats_csv_tail(st.stats_path)
        if st.stats_inst is not None or st.stats_avg is not None:
            row_line = f"Decode tok/s: inst={st.stats_inst or 0:.2f} avg={st.stats_avg or 0:.2f}"
            stdscr.addnstr(row, 0, row_line.ljust(w), w)
            row += 1
        series = read_stats_series(st.stats_path)
        if series:
            spark = render_sparkline(series, width=min(48, max(16, w - 20)))
            stdscr.addnstr(row, 0, f"tok/s sparkline: {spark}"[: w - 1], w)
            row += 1
    if st.show_ray:
        stdscr.addnstr(row, 0, "Ray status:".ljust(w), w)
        row += 1
        for ln in ray_status_text().splitlines():
            if row >= h - 5:
                break
            stdscr.addnstr(row, 0, ln[: w - 1], w)
            row += 1

    # Footer / Controls
    help1 = "q quit  p pause  s set-interval  l log CSV  D set-stats  C clear-mps-cache  E empty-cache-PID  G graceful-stop-PID  r ray-status  k ray-stop  K kill-pid  h help"
    stdscr.addnstr(h - 3, 0, (help1[: w - 1]).ljust(w), w)
    if st.last_msg:
        stdscr.addnstr(h - 2, 0, st.last_msg[: w - 1].ljust(w), w)
    if st.log_path:
        stdscr.addnstr(h - 1, 0, f"logging → {st.log_path}"[: w - 1].ljust(w), w)
    stdscr.refresh()


def curses_main(stdscr, st: State):
    """

    :param stdscr:
    :param st:
    """
    curses.curs_set(0)
    stdscr.nodelay(True)
    next_tick = time.monotonic()
    while True:
        now = time.monotonic()
        if not st.paused and now >= next_tick:
            draw_screen(stdscr, st)
            next_tick = now + st.interval
        try:
            ch = stdscr.getch()
        except Exception:
            ch = -1
        if ch == -1:
            time.sleep(0.05)
            continue
        key = chr(ch) if 0 <= ch < 256 else ""
        if key == "q":
            break
        elif key == "p":
            st.paused = not st.paused
            st.last_msg = f"paused={st.paused}"
        elif key == "s":
            st.last_msg = "set interval (seconds): type and Enter"
            curses.echo()
            stdscr.addstr(1, 0, "interval: ")
            try:
                val = stdscr.getstr(1, 10).decode().strip()
                st.interval = max(0.2, float(val))
                st.last_msg = f"interval set to {st.interval:.2f}s"
            except Exception:
                st.last_msg = "invalid interval"
            finally:
                curses.noecho()
        elif key == "l":
            if st.mem_logger is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join("runs", "xltop", f"mem_{ts}.csv")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                st.mem_logger = MemoryMonitor(log_csv_path=path).start()
                st.log_path = path
                st.last_msg = f"logging started → {path}"
            else:
                st.mem_logger.stop()
                st.mem_logger = None
                st.last_msg = "logging stopped"
                st.log_path = None
        elif key == "C":
            st.last_msg = clear_mps_cache()
        elif key == "D":
            st.last_msg = "set stats CSV path"
            curses.echo()
            stdscr.addstr(1, 0, "stats csv: ")
            try:
                val = stdscr.getstr(1, 11).decode().strip()
                st.stats_path = val or None
                st.last_msg = f"stats set to {st.stats_path}"
            except Exception:
                st.last_msg = "invalid stats path"
            finally:
                curses.noecho()
        elif key == "E":
            st.last_msg = "Send SIGUSR1 (empty_cache) to PID:"
            curses.echo()
            stdscr.addstr(1, 0, "PID:    ")
            try:
                val = stdscr.getstr(1, 5).decode().strip()
                pid = int(val)
                os.kill(pid, _signal.SIGUSR1)
                st.last_msg = f"SIGUSR1 sent to {pid}"
            except Exception as e:
                st.last_msg = f"SIGUSR1 error: {e}"
            finally:
                curses.noecho()
        elif key == "G":
            st.last_msg = "Send SIGTERM (graceful stop) to PID:"
            curses.echo()
            stdscr.addstr(1, 0, "PID:    ")
            try:
                val = stdscr.getstr(1, 5).decode().strip()
                pid = int(val)
                os.kill(pid, _signal.SIGTERM)
                st.last_msg = f"SIGTERM sent to {pid}"
            except Exception as e:
                st.last_msg = f"SIGTERM error: {e}"
            finally:
                curses.noecho()
        elif key == "r":
            st.last_msg = "refreshed ray status"
            draw_screen(stdscr, st)
        elif key == "k":
            st.last_msg = ray_stop_force()
        elif key == "K":
            st.last_msg = "Enter PID to kill:"
            curses.echo()
            stdscr.addstr(1, 0, "PID:    ")
            try:
                val = stdscr.getstr(1, 5).decode().strip()
                pid = int(val)
                st.last_msg = kill_pid(pid)
            except Exception as e:
                st.last_msg = f"kill: {e}"
            finally:
                curses.noecho()
        elif key == "h":
            st.last_msg = "q quit | p pause | s set interval | l log CSV | C clear MPS cache | r ray status | k ray stop | K kill pid"


_CLI_STATS_PATH: Optional[str] = None


def print_once():
    """

    """
    s = snapshot()
    rec_gb = get_recommended_mps_gb()
    print("xltop snapshot")
    print(f"  rss_mb={s.rss_mb:.1f} avail_mb={s.avail_mb} total_mb={s.total_mb}")
    print(f"  mps_alloc_mb={s.mps_alloc_mb} mps_reserved_mb={s.mps_reserved_mb} recommended_gb={rec_gb}")
    if _CLI_STATS_PATH:
        inst, avg = read_stats_csv_tail(_CLI_STATS_PATH)
        if inst is not None or avg is not None:
            print(f"  decode_tok_s: inst={inst or 0:.2f} avg={avg or 0:.2f}")
    print("Top processes:")
    for pid, name, rss in get_top_processes():
        print(f"  {pid:>7}  {rss:8.1f}  {name}")
    print("\nRay:")
    print(ray_status_text())


def loop_print(poll: float):
    """

    :param poll:
    :return:
    """
    try:
        while True:
            print_once()
            sys.stdout.flush()
            time.sleep(max(0.5, poll))
            print("\n---\n")
    except KeyboardInterrupt:
        return


def snapshot_json() -> Dict[str, Any]:
    """

    :return:
    """
    s = snapshot()
    rec_gb = get_recommended_mps_gb()
    top = get_top_processes()
    return {
        "ts": time.time(),
        "rss_mb": s.rss_mb,
        "avail_mb": s.avail_mb,
        "total_mb": s.total_mb,
        "mps_alloc_mb": s.mps_alloc_mb,
        "mps_reserved_mb": s.mps_reserved_mb,
        "mps_recommended_gb": rec_gb,
        "top_processes": [{"pid": pid, "name": name, "rss_mb": rss} for pid, name, rss in top],
        "ray": ray_status_text(),
    }


def json_stream(poll: float, count: int = 0):
    """

    :param poll:
    :param count:
    :return:
    """
    import json as _json
    i = 0
    try:
        while True:
            print(_json.dumps(snapshot_json()), flush=True)
            i += 1
            if count and i >= count:
                break
            time.sleep(max(0.2, poll))
    except KeyboardInterrupt:
        return


def main():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-curses", action="store_true",
                    help="Disable curses UI and print a snapshot (or loop with --poll)")
    ap.add_argument("--poll", type=float, default=0.0, help="Polling interval for --no-curses mode (0 = one-shot)")
    ap.add_argument("--once", action="store_true", help="Alias for --no-curses --poll 0 (single snapshot)")
    ap.add_argument("--count", type=int, default=0, help="For --poll modes, stop after N iterations (0 = infinite)")
    ap.add_argument("--json", action="store_true", help="Emit a single JSON snapshot and exit")
    ap.add_argument("--json-stream", action="store_true",
                    help="Emit NDJSON snapshots at --poll interval; use --count to limit")
    ap.add_argument("--stdin-commands", action="store_true",
                    help="Read simple commands from stdin (kill <pid>, ray stop, empty_cache, interval <sec>)")
    ap.add_argument("--stats-path", type=str, default=None,
                    help="Optional decode stats CSV (from runner --stats-log) to display tok/s")
    args = ap.parse_args()

    global _CLI_STATS_PATH
    _CLI_STATS_PATH = args.stats_path

    if args.once:
        print_once()
        return

    if args.json:
        import json as _json
        print(_json.dumps(snapshot_json()))
        return

    if args.json_stream:
        json_stream(max(0.2, args.poll or 1.0), count=args.count)
        return

    if args.no_curses or curses is None:
        if args.poll and args.poll > 0:
            # bounded loop if --count > 0
            if args.count > 0:
                for _ in range(args.count):
                    print_once()
                    sys.stdout.flush()
                    time.sleep(max(0.5, args.poll))
                    print("\n---\n")
            else:
                loop_print(args.poll)
        else:
            print_once()
        return

    st = State()
    st.stats_path = args.stats_path

    # Optional stdin control loop for non-interactive agents
    if args.stdin_commands:
        import threading

        def _reader():
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().split()
                if not cmd:
                    continue
                try:
                    if cmd[0] == "kill" and len(cmd) > 1:
                        st.last_msg = kill_pid(int(cmd[1]))
                    elif cmd[0] == "ray" and len(cmd) > 1 and cmd[1] == "stop":
                        st.last_msg = ray_stop_force()
                    elif cmd[0] == "empty_cache":
                        st.last_msg = clear_mps_cache()
                    elif cmd[0] == "interval" and len(cmd) > 1:
                        st.interval = max(0.2, float(cmd[1]))
                        st.last_msg = f"interval set to {st.interval:.2f}s"
                    elif cmd[0] == "quit":
                        os._exit(0)
                except Exception as e:
                    st.last_msg = f"cmd error: {e}"

        threading.Thread(target=_reader, daemon=True).start()

    curses.wrapper(curses_main, st)


if __name__ == "__main__":
    main()
