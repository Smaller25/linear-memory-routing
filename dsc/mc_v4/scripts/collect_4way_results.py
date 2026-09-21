#!/usr/bin/env python3
"""Auto-fill the 4-way 1B comparison doc with pipeline outputs.

Reads:
  - dsc/runs/eval/mc_v{2,3,4}_1bt_full_*/mc_v*/ppl.json
  - dsc/runs/eval/mc_v{2,3,4}_1bt_full_*/mc_v*/bench.json
  - dsc/runs/eval/ruler_standard/mc_v*_1bt_vs_*/*/S-NIAH-*.json, MK-NIAH-*.json
  - dsc/runs/outputs/tsz128x4k_1B_mc_370m_fineweb_edu_1bt_v{3,4}/v4_profile/components_*.json
  - dsc/runs/logs/mc_370m_fineweb_edu_1bt_v{3,4}_*.log (iter time = avg of last 100)

Prints: a markdown summary suitable for pasting into MC_V234_VANILLA_1B_COMPARISON_KO.md.
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path("/home/work/.projects/LLM-OS-Models/long-gdn")
DSC = REPO / "dsc"


def latest_dir(pattern: str) -> Path | None:
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [warn] failed to read {path}: {e}", file=sys.stderr)
        return None


def avg_iter_time_from_log(log_path: Path, last_n: int = 100) -> tuple[float, float] | None:
    """Extract iter times from log, return (mean_ms, median_ms) over last_n entries."""
    if not log_path.exists():
        return None
    times = []
    pat = re.compile(r"iter time:\s+([\d.]+)ms")
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if m:
            times.append(float(m.group(1)))
    if not times:
        return None
    tail = times[-last_n:]
    mean = sum(tail) / len(tail)
    sorted_tail = sorted(tail)
    median = sorted_tail[len(sorted_tail) // 2]
    return mean, median


def collect_iter_times():
    """Get avg iter time per model."""
    out = {}
    # vanilla 1B (use vanilla 5B log if 1B not available separately)
    # Actually, vanilla 1B uses same run as vanilla 5B (just diff ckpt), so we don't have a separate log
    # Use vanilla 30B log if exists
    va_log = latest_dir(str(DSC / "runs/logs/vanilla*30b*.log"))
    if va_log:
        t = avg_iter_time_from_log(va_log)
        if t:
            out["vanilla"] = t
    # v2 1B
    v2_log = latest_dir(str(DSC / "runs/logs/mc_370m_fineweb_edu_5bt_v2*.log"))
    if v2_log:
        t = avg_iter_time_from_log(v2_log)
        if t:
            out["v2"] = t
    # v3 1B
    v3_log = latest_dir(str(DSC / "runs/logs/mc_370m_fineweb_edu_1bt_v3*.log"))
    if v3_log:
        t = avg_iter_time_from_log(v3_log)
        if t:
            out["v3"] = t
    # v4 1B
    v4_log = latest_dir(str(DSC / "runs/logs/mc_370m_fineweb_edu_1bt_v4*.log"))
    if v4_log:
        t = avg_iter_time_from_log(v4_log)
        if t:
            out["v4"] = t
    return out


def collect_ppl():
    out = {}
    for label, pat in [
        ("vanilla", "mc_v*_1bt_full_*/vanilla_1B_paper_matched/ppl.json"),
        ("v2", "mc_v2_1bt_full_*/mc/ppl.json"),
        ("v3", "mc_v3_1bt_full_*/mc_v3/ppl.json"),
        ("v4", "mc_v4_1bt_full_*/mc_v4/ppl.json"),
    ]:
        path = latest_dir(str(DSC / "runs/eval" / pat))
        if path:
            d = load_json(path)
            if d:
                out[label] = d
    return out


def collect_bench():
    out = {}
    for label, pat in [
        ("vanilla", "mc_v*_1bt_full_*/vanilla_1B_paper_matched/bench.json"),
        ("v2", "mc_v2_1bt_full_*/mc/bench.json"),
        ("v3", "mc_v3_1bt_full_*/mc_v3/bench.json"),
        ("v4", "mc_v4_1bt_full_*/mc_v4/bench.json"),
    ]:
        path = latest_dir(str(DSC / "runs/eval" / pat))
        if path:
            d = load_json(path)
            if d:
                out[label] = d
    return out


def collect_ruler():
    out = {}
    # 4 task × 6 len × 4 model
    for label, pat in [
        ("vanilla", "mc_v*_1bt_vs_vanilla_*/vanilla_1B_paper_matched/*.json"),
        ("v2", "mc_v2_1bt_vs_vanilla_*/mc_1B/*.json"),
        ("v3", "mc_v3_1bt_vs_vanilla_*/mc_v3_1B/*.json"),
        ("v4", "mc_v4_1bt_vs_vanilla_*/mc_v4_1B/*.json"),
    ]:
        files = sorted(glob.glob(str(DSC / "runs/eval/ruler_standard" / pat)))
        cells = {}
        for f in files:
            stem = Path(f).stem  # e.g. "S-NIAH-1_4096"
            if "_" not in stem:
                continue
            task, length = stem.rsplit("_", 1)
            d = load_json(Path(f))
            if d is None:
                continue
            # Score is in results[0].score or top-level
            r = d.get("results", d)
            score = r[0]["score"] if isinstance(r, list) else r
            cells[(task, int(length))] = score
        if cells:
            out[label] = cells
    return out


def collect_v4_profile():
    """v4 in-training profiler output (per-component breakdown)."""
    out = []
    files = sorted(glob.glob(str(DSC / "runs/outputs/tsz128x4k_1B_mc_370m_fineweb_edu_1bt_v4/v4_profile/components_*.json")))
    for f in files:
        d = load_json(Path(f))
        if d:
            iter_num = int(re.search(r"iter(\d+)_", f).group(1))
            out.append((iter_num, d))
    return out


def main():
    print("=" * 70)
    print("4-WAY 1B COMPARISON DATA — auto-extracted from pipeline outputs")
    print("=" * 70)

    iter_times = collect_iter_times()
    ppls = collect_ppl()
    benches = collect_bench()
    rulers = collect_ruler()
    v4_profs = collect_v4_profile()

    print("\n## §2. Iter time (training)")
    print(f"{'Model':<10} {'mean (ms)':<12} {'median (ms)':<12} {'vs vanilla':<12}")
    va_t = iter_times.get("vanilla", (None, None))[0]
    for k in ("vanilla", "v2", "v3", "v4"):
        if k in iter_times:
            mean, median = iter_times[k]
            ratio = f"{mean / va_t:.2f}x" if (va_t and k != "vanilla") else "1.00x"
            print(f"{k:<10} {mean:<12.2f} {median:<12.2f} {ratio:<12}")
        else:
            print(f"{k:<10} {'MISSING':<12}")

    print("\n## §3. PPL")
    print(f"{'Model':<10} {'NLL':<10} {'PPL':<10}")
    for k in ("vanilla", "v2", "v3", "v4"):
        if k in ppls:
            d = ppls[k]
            print(f"{k:<10} {d.get('nll', '?'):<10.4f} {d.get('ppl', '?'):<10.3f}")
        else:
            print(f"{k:<10} MISSING")

    print("\n## §4. RULER (4 tasks × 6 lengths)")
    tasks = ["S-NIAH-1", "S-NIAH-2", "S-NIAH-3", "MK-NIAH-1"]
    lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    print(f"{'Task':<12} {'Length':<8} {'vanilla':<10} {'v2':<10} {'v3':<10} {'v4':<10}")
    for task in tasks:
        for length in lengths:
            row = [task, str(length)]
            for k in ("vanilla", "v2", "v3", "v4"):
                v = rulers.get(k, {}).get((task, length))
                row.append(str(v) if v is not None else "-")
            print(f"{row[0]:<12} {row[1]:<8} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")

    print("\n## §5. Bench (fwd & fwd+bwd, single GPU)")
    print(f"{'Model':<10} {'seq':<6} {'mb':<4} {'fwd ms':<8} {'fwd GB':<8} {'f+b ms':<8} {'f+b GB':<8}")
    for k in ("vanilla", "v2", "v3", "v4"):
        if k not in benches:
            continue
        for c in benches[k].get("configurations", []):
            seq = c["seq_len"]; mb = c["micro_batch"]
            f = c["forward"]; b = c["forward_backward"]
            print(f"{k:<10} {seq:<6} {mb:<4} {f['latency_ms']:<8.0f} {f['peak_vram_gb']:<8.1f} {b['latency_ms']:<8.0f} {b['peak_vram_gb']:<8.1f}")

    if v4_profs:
        print("\n## v4 per-iter component breakdown (in-training profiler)")
        print(f"{'iter':<8} {'fwd ms':<10} {'bwd ms':<10} {'optim ms':<10}")
        for iter_num, d in v4_profs[-5:]:  # last 5 samples
            fwd = d.get("fwd", {}).get("mean_ms", "?")
            bwd = d.get("bwd", {}).get("mean_ms", "?")
            opt = d.get("optim", {}).get("mean_ms", "?")
            print(f"{iter_num:<8} {fwd:<10.2f} {bwd:<10.2f} {opt:<10.2f}")


if __name__ == "__main__":
    main()
