"""Dataset A(RULER 표준) 준비 + needle annotation. GPU 불필요."""
import argparse, json, os, re, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
TOKENIZER = "TinyLlama/TinyLlama_v1.1"
CHUNK = 256

NEEDLE_RE = re.compile(r"One of the special magic numbers? for ([\w-]+) is:? (\d+)")
QUERY_RE = re.compile(r"What (?:is|are all) the special magic numbers? for ([\w-]+)")


def annotate(input_text, tokenizer, chunk=CHUNK):
    """needle들의 토큰 위치·segment와 질의 key를 식별."""
    q_matches = QUERY_RE.findall(input_text)
    if not q_matches:
        raise ValueError("no query found")
    query_key = q_matches[-1]
    enc = tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.offset_mapping

    def tok_at(char_pos):
        return next(i for i, (s, e) in enumerate(offsets) if s <= char_pos < e)

    needles = []
    for m in NEEDLE_RE.finditer(input_text):
        key, value = m.group(1), m.group(2)
        ts, te = tok_at(m.start(2)), tok_at(m.end(2) - 1)
        needles.append({"key": key, "value": value,
                        "tok_start": ts, "tok_end": te, "seg": te // chunk})
    gold = [n for n in needles if n["key"] == query_key]
    if not gold:
        raise ValueError(f"queried key {query_key} not among needles")
    return {"query_key": query_key, "needles": needles, "gold_seg": gold[0]["seg"],
            "n_seg": (len(offsets) + chunk - 1) // chunk, "n_tok": len(offsets)}


def prepare_a(num_samples=50, length=2048, tasks=("niah_single_1", "niah_multikey_1")):
    """vendored RULER prepare.py를 TinyLlama 토크나이저로 호출."""
    gen_dir = os.path.join(REPO, "src", "ruler", "gen")
    essays = os.path.join(gen_dir, "synthetic", "json", "PaulGrahamEssays.json")
    if not os.path.exists(essays):
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "ruler.py"),
                        "download-essays"], check=True)
    save_dir = os.path.join(MC_OUT, "data", str(length))
    # prepare.py internally re-invokes the per-task generator via a hardcoded
    # `python <script>` shell command (not sys.executable), so we must put
    # this interpreter's bin dir first on PATH or it silently falls back to
    # whatever `python` resolves to on the caller's PATH (e.g. base conda,
    # missing wonderwords/nltk). See task-3-report.md.
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + env.get("PATH", "")
    for task in tasks:
        r = subprocess.run(
            [sys.executable, "prepare.py", "--save_dir", save_dir,
             "--benchmark", "synthetic", "--task", task,
             "--tokenizer_path", TOKENIZER, "--tokenizer_type", "hf",
             "--max_seq_length", str(length), "--model_template_type", "base",
             "--num_samples", str(num_samples)],
            cwd=gen_dir, capture_output=True, text=True, env=env)
        out = os.path.join(save_dir, task, "validation.jsonl")
        assert os.path.exists(out), f"{task} prepare failed:\n{r.stderr[-2000:]}"
        n = sum(1 for _ in open(out))
        print(f"[prepare-a] {task}: {n} samples -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare-a", "prepare-b"])
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--n-pairs", type=int, default=16, help="쌍 수 per condition (prepare-b)")
    a = ap.parse_args()
    if a.cmd == "prepare-a":
        prepare_a(num_samples=a.num_samples)
    else:
        from paired_gen import prepare_b  # Task 4에서 추가
        prepare_b(n_pairs_per_cond=a.n_pairs)
