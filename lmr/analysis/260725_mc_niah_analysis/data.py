"""Dataset A(RULER 표준) 준비 + needle annotation. GPU 불필요."""
import argparse, json, os, re, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MC_OUT = os.environ.get("MC_OUT", "/data2/sohyung/mc_niah")
TOKENIZER = "TinyLlama/TinyLlama_v1.1"
CHUNK = 256

# Non-greedy `.+?` key group, NOT `[\w-]+` — wonderwords' adjectivelist.txt
# contains a genuine two-word entry ("ad hoc"), so RULER's word-type keys
# (f"{adj}-{noun}") can be e.g. "ad hoc-picture", which has an internal
# space. `[\w-]+` can't match past that space, so the whole NEEDLE_RE match
# silently failed at that position and the needle was dropped entirely from
# ann["needles"] (not mis-keyed — just invisible), surfaced when
# niah_single_2's 50-sample draw happened to include it (data-generation
# review, task-3 corrections round). `.+?` up to the fixed " is:? (\d+)"
# (colon is always present in the actual template — "is: {value}" — kept
# optional here only because that's how this regex already handled
# hand-built test fixtures without the colon) resolves to the nearest
# following occurrence, which is the same needle's own value in practice
# since "One of the special magic numbers for ... is:" doesn't otherwise
# appear in haystack text.
NEEDLE_RE = re.compile(r"One of the special magic numbers? for (.+?) is:? (\d+)")
# Captures the *full* query-key list, not just the first word — RULER's niah
# multiquery template writes "for K1, K2, and K3 mentioned in the provided
# text" (see src/ruler/gen/synthetic/niah.py:189 `query` construction). The
# old QUERY_RE (`for ([\w-]+)`) stopped at the first comma and silently
# dropped keys 2..N. Non-greedy `.+?` up to the fixed " mentioned in the
# provided text" phrase (present in both niah_single/multikey singular and
# multiquery/multivalue plural template forms) captures the whole list.
QUERY_LIST_RE = re.compile(
    r"What (?:is|are all) the special magic numbers? for (.+?) mentioned in the provided text"
)


def _split_query_keys(query_text):
    """"K1, K2, and K3" -> ["K1","K2","K3"]; single key text -> [text] unchanged.

    Mirrors the join niah.py uses:
    `', '.join(queries[:-1]) + ', and ' + queries[-1] if len(queries) > 1 else queries[0]`.
    Splitting on the literal ", and " first (which appears exactly once,
    right before the last key, regardless of list length) and then on ", "
    for the remainder inverts that join exactly.
    """
    parts = query_text.split(", and ")
    if len(parts) == 2:
        head = parts[0].split(", ") if parts[0] else []
        return head + [parts[1]]
    return [query_text]


def annotate(input_text, tokenizer, chunk=CHUNK):
    """needle들의 토큰 위치·segment와 질의 key(들)를 식별.

    single/multikey(질의 1개)에서는 기존 필드(query_key, gold_seg)가 그대로
    채워진다(회귀 없음). multiquery(질의 key 여러 개)와 multivalue(질의 key
    1개, needle 여러 개 — 값마다 별도 문장)는 queried_keys/gold_needles/
    gold_segs로 다건을 표현한다. gold_needles는 "질문된 key에 해당하는 모든
    needle 인스턴스" (multiquery: 4 needles, 서로 다른 key 1개씩;
    multivalue: 4 needles, 같은 key 값 4개)."""
    q_matches = QUERY_LIST_RE.findall(input_text)
    if not q_matches:
        raise ValueError("no query found")
    queried_keys = _split_query_keys(q_matches[-1])
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
    gold_needles = [n for n in needles if n["key"] in queried_keys]
    if not gold_needles:
        raise ValueError(f"queried keys {queried_keys} not among needles")
    query_key = queried_keys[0] if len(queried_keys) == 1 else None
    return {"query_key": query_key, "queried_keys": queried_keys,
            "needles": needles, "gold_needles": gold_needles,
            "gold_seg": gold_needles[0]["seg"],
            "gold_segs": sorted({n["seg"] for n in gold_needles}),
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


# niah_multiquery의 synthetic.yaml 항목은 num_needle_q=4로 고정돼 있다(prepare.py는
# yaml 태스크 이름으로만 config를 찾으므로 num_needle_q를 CLI로 오버라이드하는 경로가
# 없다). needle-수 스윕(X2: q=2)을 위해 vendored niah.py를 직접 호출한다 — prepare.py가
# 하는 일(TASKS['niah'] 템플릿 조립 + answer_prefix 부착 + subprocess 호출)을 그대로
# 재현하되 num_needle_q만 인자로 받는다. save_name으로 별도 디렉터리에 저장되므로
# niah_multiquery(q=4)와 충돌하지 않는다.
def prepare_a_niah_q(num_samples, length, num_needle_q, save_name):
    """niah.py 직접 호출로 num_needle_q(질의 needle 수)를 오버라이드한 multiquery 변종 생성.

    다른 인자는 synthetic.yaml의 niah_multiquery 항목과 동일
    (type_haystack=essay, type_needle_k=words, type_needle_v=numbers, num_needle_v=1).
    num_needle_k는 niah.py가 내부에서 max(num_needle_k, num_needle_q)로 클램프하므로
    1로 둬도 num_needle_q에 맞춰 자동으로 늘어난다(niah.py:77)."""
    gen_dir = os.path.join(REPO, "src", "ruler", "gen")
    synth_dir = os.path.join(gen_dir, "synthetic")
    essays = os.path.join(synth_dir, "json", "PaulGrahamEssays.json")
    if not os.path.exists(essays):
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "ruler.py"),
                        "download-essays"], check=True)
    sys.path.insert(0, synth_dir)
    from constants import TASKS  # noqa: E402 (vendored RULER module, path above)
    niah_cfg = TASKS["niah"]
    template = niah_cfg["template"] + niah_cfg["answer_prefix"]  # model_template_type=base -> identity

    save_dir = os.path.join(MC_OUT, "data", str(length))
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + env.get("PATH", "")
    r = subprocess.run(
        [sys.executable, "niah.py", "--save_dir", save_dir, "--save_name", save_name,
         "--tokenizer_path", TOKENIZER, "--tokenizer_type", "hf",
         "--max_seq_length", str(length), "--tokens_to_generate", str(niah_cfg["tokens_to_generate"]),
         "--num_samples", str(num_samples),
         "--type_haystack", "essay", "--type_needle_k", "words", "--type_needle_v", "numbers",
         "--num_needle_k", "1", "--num_needle_v", "1", "--num_needle_q", str(num_needle_q),
         "--template", template],
        cwd=synth_dir, capture_output=True, text=True, env=env)
    out = os.path.join(save_dir, save_name, "validation.jsonl")
    assert os.path.exists(out), f"{save_name} (q={num_needle_q}) prepare failed:\n{r.stderr[-2000:]}"
    n = sum(1 for _ in open(out))
    print(f"[prepare-a-niah-q] {save_name} (num_needle_q={num_needle_q}): {n} samples -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare-a", "prepare-b", "prepare-a-niah-q"])
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--tasks", type=str, default="niah_single_1,niah_multikey_1",
                    help="comma-separated RULER task names (prepare-a)")
    ap.add_argument("--num-needle-q", type=int, default=2, help="prepare-a-niah-q only")
    ap.add_argument("--save-name", type=str, default="niah_multiquery_q2",
                    help="prepare-a-niah-q only: output subdir name under data/<length>/")
    ap.add_argument("--n-pairs", type=int, default=16, help="쌍 수 per condition (prepare-b)")
    a = ap.parse_args()
    if a.cmd == "prepare-a":
        prepare_a(num_samples=a.num_samples, length=a.length,
                  tasks=tuple(t.strip() for t in a.tasks.split(",") if t.strip()))
    elif a.cmd == "prepare-a-niah-q":
        prepare_a_niah_q(num_samples=a.num_samples, length=a.length,
                         num_needle_q=a.num_needle_q, save_name=a.save_name)
    else:
        from paired_gen import prepare_b  # Task 4에서 추가
        prepare_b(n_pairs_per_cond=a.n_pairs)
