"""arm별 shard -> 병합 JSON + 그림 + main-table-ready 마크다운 표 (GPU 불필요).

입력은 `routing_stats.py` 가 잡마다 따로 쓴 shard들이다:

    results/{tag}/{arm_key}.json  및  $MC_OUT/results/{tag}/{arm_key}.json

여기(오프라인, GPU 없음)서 glob으로 모아 병합한다. 병합 산출물:

    results/{tag}.json  ·  $MC_OUT/results/{tag}.json   (arm 전체 합본)
    results/{tag}.png   ·  $MC_OUT/results/{tag}.png    (layer × hit@2 프로파일)
    results/{tag}_table.md

잡 쪽에서 합본을 읽고-고쳐-쓰지 않는 이유는 GPU 2장에서 잡이 동시에 돌면
나중에 끝난 잡이 먼저 끝난 잡의 arm을 통째로 덮어쓰기 때문이다. shard가 깨져
읽히지 않으면 여기서 **조용히 넘어가지 않고 즉시 실패**한다(원자료 소실을
경고 없이 넘기는 것이 가장 위험).

shard 디렉터리가 없으면 옛 단일 파일 스키마(`{tag}.json`, 0024 e1_routing)를
그대로 읽는 legacy 경로로 떨어진다.

프로토콜의 main-table-ready 요건을 그대로 강제한다:
  - ssketch 시드를 mean±sd 로 묶는다. 시드가 3개 미만이면 표에 `(seeds=k<3)` 로
    표시하고 판정에 쓰지 못하게 한다.
  - 셀마다 (유효/전체) 표본 크기를 같이 적는다.
  - meank 대비 상승폭을 **SE 몇 배**인지로 환산한다. 이항 SE = sqrt(p(1-p)/n)
    을 두 arm에 대해 합성(sqrt(se_a^2 + se_b^2))한다. paired는 조건당 16쌍이라
    SE가 크고, 그걸 숨기지 않는 게 이 표의 목적이다.
  - single_1은 포화 셀이라 회귀 감시용으로만 표시한다(headline 아님).

    python ssketch_table.py [--tag ssketch_routing] [--metric hit_at_2|macro_hit2]
                            [--no-merge] [--no-figure]
"""
import argparse, glob, json, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")

HEADLINE = ("paired_D_multi", "niah_multivalue")
WATCHDOG = ("niah_single_1",)
# 0024 E1 / 0025 X2 의 VESSL A100 수치 — **방향 확인용**이지 비교선이 아니다.
PRIOR = {
    ("mc-5B", "niah_single_1"): 0.933, ("mc-30B", "niah_single_1"): 1.000,
    ("mc-5B", "niah_multikey_1"): 0.660, ("mc-30B", "niah_multikey_1"): 0.809,
    ("mc-5B", "paired_S_multi"): 0.812, ("mc-30B", "paired_S_multi"): 0.750,
    ("mc-5B", "paired_D_multi"): 0.562, ("mc-30B", "paired_D_multi"): 0.500,
    ("mc-5B", "niah_multivalue"): 0.442, ("mc-30B", "niah_multivalue"): 0.435,
}


def arm_slug(key):
    """routing_stats.arm_slug 와 같은 규칙(`|` -> `__`)."""
    return key.replace("|", "__")


def _atomic_json_dump(obj, path, **kw):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, **kw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def load_shards(tag):
    """results/{tag}/ 와 $MC_OUT/results/{tag}/ 의 arm shard를 모아 병합 객체를
    만든다. shard가 하나도 없으면 None(호출부가 legacy 단일 파일로 폴백).

    한 arm이 두 루트에 다 있으면 파싱되는 쪽을 쓴다(한쪽만 깨진 경우 복구).
    양쪽 다 깨졌으면 조용히 넘기지 않고 SystemExit 로 죽는다."""
    dirs = [os.path.join(RES, tag), os.path.join(MC_OUT, "results", tag)]
    shards, broken, srcs = {}, {}, []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            base = os.path.basename(p)
            if base.endswith("_per_sample.json"):
                continue
            name = base[: -len(".json")]
            try:
                obj = json.load(open(p))
            except Exception as e:                      # 절대 pass 로 삼키지 않는다
                broken.setdefault(name, []).append(f"{p}: {e}")
                continue
            key = obj.get("arm_key") or obj.get("meta", {}).get("arm_key") or name
            if arm_slug(key) != name:
                print(f"[table][warn] {p}: arm_key={key!r} 가 파일명 {name!r} 과 다르다",
                      file=sys.stderr)
            if key in shards:
                continue                                 # 첫 루트(repo) 우선
            shards[key] = obj
            srcs.append(p)
    fatal = {k: v for k, v in broken.items() if k not in {arm_slug(x) for x in shards}}
    if fatal:
        msg = "\n".join(f"  {k}: " + "; ".join(v) for k, v in sorted(fatal.items()))
        raise SystemExit(
            "[table] 읽을 수 없는 arm shard가 있다 — 조용히 빼고 표를 만들지 않는다.\n"
            + msg + "\n해당 arm을 다시 돌리거나 깨진 파일을 치운 뒤 다시 실행할 것.")
    if not shards:
        return None
    if broken:
        for k, v in sorted(broken.items()):
            print(f"[table][warn] shard {k}: 한쪽 사본이 깨졌지만 다른 사본으로 복구됨 "
                  + "; ".join(v), file=sys.stderr)

    datasets, seen = [], set()
    for obj in shards.values():
        for ds in obj.get("meta", {}).get("datasets", []):
            if ds not in seen:
                seen.add(ds)
                datasets.append(ds)
    base_meta = next(iter(shards.values())).get("meta", {})
    meta = {k: v for k, v in base_meta.items()
            if k not in ("arm_key", "model", "n_layers", "slurm_job_id")}
    meta["datasets"] = datasets
    meta["arms"] = sorted(shards.keys())
    meta["models"] = sorted({obj.get("arm", {}).get("model", k.split("|")[0])
                             for k, obj in shards.items()})
    meta["shard_sources"] = srcs
    merged = {"meta": meta,
              "results": {k: obj.get("results", {}) for k, obj in shards.items()},
              "arms": {k: obj.get("arm", {}) for k, obj in shards.items()},
              "e2_join": {k: obj["e2_join"] for k, obj in shards.items()
                          if obj.get("e2_join")}}
    return merged


def load_legacy(tag):
    for p in (os.path.join(RES, f"{tag}.json"), os.path.join(MC_OUT, "results", f"{tag}.json")):
        if os.path.exists(p):
            return json.load(open(p)), p
    raise SystemExit(
        f"no arm shards under {RES}/{tag}/ (or {MC_OUT}/results/{tag}/) "
        f"and no legacy {tag}.json — 잡이 아직 안 끝났거나 --tag 가 틀렸다.")


def load(tag, merge=True):
    """(obj, source_label) — shard 우선, 없으면 legacy 단일 파일."""
    merged = load_shards(tag)
    if merged is None:
        return load_legacy(tag)
    label = f"{RES}/{tag}/*.json ({len(merged['results'])} arms)"
    if merge:
        for p in (os.path.join(RES, f"{tag}.json"),
                  os.path.join(MC_OUT, "results", f"{tag}.json")):
            _atomic_json_dump(merged, p, indent=2)
            print(f"[table] merged -> {p}", file=sys.stderr)
    return merged, label


def make_figure(results, out_paths):
    """arm × dataset 의 layer별 hit@2 프로파일. 잡이 아니라 이 오프라인 단계에서
    그린다(같은 PNG를 여러 잡이 동시에 덮어쓰는 것을 막기 위해)."""
    keys = sorted(results.keys())
    if not keys:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[table][warn] matplotlib 없음 — 그림 건너뜀 ({e})", file=sys.stderr)
        return
    ncol = min(len(keys), 4)
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.2 * nrow), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax, key in zip(flat, keys):
        for dsname, agg in results[key].items():
            layers = [pl["layer"] for pl in agg["per_layer"]]
            hits = [pl["hit_at_2"] if pl["hit_at_2"] is not None else float("nan")
                    for pl in agg["per_layer"]]
            ax.plot(layers, hits, marker="o", label=dsname)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel("hit@2")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in flat[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    for p in out_paths:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(p, dpi=150)
        print(f"[table] wrote {p}", file=sys.stderr)
    plt.close(fig)


def cell(agg, metric):
    """best-layer(=hit@2 기준) 에서의 metric 값과 유효 표본 수."""
    bl = agg.get("best_layer")
    if bl is None:
        return None, 0, agg.get("n_total", 0)
    pl = agg["per_layer"][bl]
    n = pl["n_eligible"] if metric == "hit_at_2" else pl.get("macro_hit2_n", 0)
    return pl.get(metric), n, pl.get("n_total", 0)


def binom_se(p, n):
    if p is None or not n:
        return None
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def fmt(v, sd=None, n=None, tot=None):
    if v is None:
        return "—"
    s = f"{v:.3f}"
    if sd is not None:
        s += f"±{sd:.3f}"
    if n is not None:
        s += f" ({n}/{tot})"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ssketch_routing")
    ap.add_argument("--metric", default="hit_at_2", choices=["hit_at_2", "macro_hit2"])
    ap.add_argument("--no-merge", action="store_true",
                    help="shard를 읽어 표만 만들고 병합 {tag}.json 은 쓰지 않는다")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    obj, path = load(a.tag, merge=not a.no_merge)
    results, arms = obj["results"], obj.get("arms", {})
    datasets = obj["meta"]["datasets"]
    if not a.no_figure:
        make_figure(results, [os.path.join(RES, f"{a.tag}.png"),
                              os.path.join(MC_OUT, "results", f"{a.tag}.png")])

    # (model, arm_label) -> list over seeds of (value, n, total)
    groups = {}
    for key, by_ds in results.items():
        meta = arms.get(key, {"model": key.split("|")[0], "descriptor": "meank"})
        model = meta.get("model", key.split("|")[0])
        if meta.get("descriptor") == "meank":
            label = "meank"
        elif meta.get("full_state") or meta.get("rank") == meta.get("head_v_dim"):
            label = "full-state"
        else:
            label = f"ssketch r={meta.get('rank')}"
            if meta.get("router_query", "q") != "q":
                label += f" rq={meta['router_query']}"
            if meta.get("online_score", "read_norm") != "read_norm":
                label += f" on={meta['online_score']}"
        for ds in datasets:
            if ds not in by_ds:
                continue
            v, n, tot = cell(by_ds[ds], a.metric)
            groups.setdefault((model, label, ds), []).append((v, n, tot, meta.get("seed")))

    def agg_group(g):
        vals = [x[0] for x in g if x[0] is not None]
        if not vals:
            return None, None, 0, 0, 0
        n = max(x[1] for x in g)
        tot = max(x[2] for x in g)
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        return statistics.mean(vals), sd, n, tot, len(vals)

    models = sorted({k[0] for k in groups})
    labels_order = ["meank", "ssketch r=4", "ssketch r=8", "ssketch r=16", "full-state"]
    labels = [l for l in labels_order if any(k[1] == l for k in groups)]
    labels += sorted({k[1] for k in groups} - set(labels))

    out = []
    out.append(f"# ssketch routing — metric = `{a.metric}` (best layer, @2048, chunk=256, topk=2)")
    out.append("")
    out.append(f"source: `{path}`  ·  arms: {len(results)}")
    out.append("")
    out.append("헤드라인 셀은 **paired_D_multi / niah_multivalue**. `niah_single_1`은 포화 셀이라")
    out.append("회귀 감시용이다. 괄호는 (유효/전체) 표본 수, ± 는 시드 간 sd.")
    out.append("")
    for model in models:
        out.append(f"## {model}")
        out.append("")
        out.append("| dataset | " + " | ".join(labels) + " | Δ(best ssketch − meank) | SE배수 | 0024/0025 (A100, 참고) |")
        out.append("|---" * (len(labels) + 4) + "|")
        for ds in datasets:
            row = [ds + (" *(포화)*" if ds in WATCHDOG else (" **(headline)**" if ds in HEADLINE else ""))]
            base = None
            best_s, best_lab = None, None
            for lab in labels:
                g = groups.get((model, lab, ds))
                if not g:
                    row.append("—")
                    continue
                m, sd, n, tot, k = agg_group(g)
                txt = fmt(m, sd, n, tot)
                if lab.startswith("ssketch") and k < 3:
                    txt += f" ⚠seeds={k}"
                row.append(txt)
                if lab == "meank":
                    base = (m, n)
                elif m is not None and (best_s is None or m > best_s[0]):
                    best_s, best_lab = (m, n), lab
            if base and base[0] is not None and best_s:
                delta = best_s[0] - base[0]
                se_a, se_b = binom_se(base[0], base[1]), binom_se(best_s[0], best_s[1])
                se = math.sqrt((se_a or 0) ** 2 + (se_b or 0) ** 2)
                row.append(f"{delta:+.3f} ({best_lab})")
                row.append(f"{delta / se:+.2f}×" if se > 0 else "—")
            else:
                row += ["—", "—"]
            prior = PRIOR.get((model, ds))
            row.append(f"{prior:.3f}" if prior is not None else "—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    out.append("## 판정 규칙 (계획서)")
    out.append("")
    out.append("- hit@2 상승 + downstream 상승 → 성공")
    out.append("- hit@2 상승 + downstream 정체 → read-side 병목으로 pivot (이것도 결과)")
    out.append("- **hit@2 정체 → 가설 기각**")
    out.append("")
    out.append("상승은 헤드라인 셀에서 SE 2배 이상일 때만 상승으로 읽는다. 시드가 3개 미만인")
    out.append("셀(⚠)은 판정에 쓰지 않는다. 참고 열의 A100 수치는 방향 확인용이며 하드웨어가")
    out.append("달라 비교선으로 쓸 수 없다 — meank 열이 이번 실행의 기준선이다.")

    text = "\n".join(out) + "\n"
    dest = os.path.join(RES, f"{a.tag}_table.md")
    open(dest, "w").write(text)
    print(text)
    print(f"[table] wrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
