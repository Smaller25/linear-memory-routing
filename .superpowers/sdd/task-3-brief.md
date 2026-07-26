### Task 3: Dataset A 준비 + gold-chunk annotation (CPU 테스트)

**Files:**
- Create: `lmr/analysis/260725_mc_niah_analysis/data.py`
- Test: `tests/lmr/test_mc_niah_data.py`

**Interfaces:**
- Produces: `annotate(input_text, tokenizer, chunk=256) -> dict(query_key, needles=[{key,value,tok_start,tok_end,seg}], gold_seg:int, n_seg:int)`; CLI `data.py prepare-a` → `$MC_OUT/data/2048/{niah_single_1,niah_multikey_1}/validation.jsonl` (50샘플, TinyLlama 토크나이저)

- [ ] **Step 1: 실패하는 annotation 테스트 작성**

```python
# tests/lmr/test_mc_niah_data.py
import importlib.util, os, sys
import pytest

ANA = os.path.join(os.path.dirname(__file__), "..", "..",
                   "lmr", "analysis", "260725_mc_niah_analysis")
sys.path.insert(0, os.path.abspath(ANA))
import data as mcdata  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")


def test_annotate_finds_gold_needle(tok):
    filler = "The grass is green. The sky is blue. " * 120
    needle = "One of the special magic numbers for apple-pie is: 7301562."
    text = (filler[:2000] + " " + needle + " " + filler[2000:4000]
            + "\nWhat is the special magic number for apple-pie mentioned in the provided text?")
    ann = mcdata.annotate(text, tok)
    assert ann["query_key"] == "apple-pie"
    assert len(ann["needles"]) == 1
    n = ann["needles"][0]
    assert n["value"] == "7301562"
    # 토큰 위치가 실제 needle 문자 위치와 일치하는 segment를 가리킴
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    char_pos = text.find("7301562")
    tok_idx = next(i for i, (s, e) in enumerate(enc.offset_mapping) if s <= char_pos < e)
    assert n["seg"] == tok_idx // 256 == ann["gold_seg"]


def test_annotate_multikey_picks_queried(tok):
    needles = [f"One of the special magic numbers for key-{i} is: 100000{i}." for i in range(4)]
    filler = "The grass is green. " * 60
    text = (" ".join([filler, needles[0], filler, needles[1], filler, needles[2],
                      filler, needles[3], filler])
            + "\nWhat is the special magic number for key-2 mentioned in the provided text?")
    ann = mcdata.annotate(text, tok)
    assert ann["query_key"] == "key-2"
    assert len(ann["needles"]) == 4
    assert ann["gold_seg"] == next(n["seg"] for n in ann["needles"] if n["key"] == "key-2")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `PY=/data2/sohyung/conda-envs/sh_infocap/bin/python; HF_HOME=/data2/sohyung/hf_home $PY -m pytest tests/lmr/test_mc_niah_data.py -x -q`
Expected: FAIL (`data` 모듈/`annotate` 없음)

- [ ] **Step 3: data.py 작성 (annotate + prepare-a)**

```python
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
    essays = os.path.join(REPO, "data", "PaulGrahamEssays.json")
    if not os.path.exists(essays):
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "ruler.py"),
                        "download-essays"], check=True)
    save_dir = os.path.join(MC_OUT, "data", str(length))
    for task in tasks:
        r = subprocess.run(
            [sys.executable, "prepare.py", "--save_dir", save_dir,
             "--benchmark", "synthetic", "--task", task,
             "--tokenizer_path", TOKENIZER, "--tokenizer_type", "hf",
             "--max_seq_length", str(length), "--model_template_type", "base",
             "--num_samples", str(num_samples)],
            cwd=gen_dir, capture_output=True, text=True)
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `HF_HOME=/data2/sohyung/hf_home $PY -m pytest tests/lmr/test_mc_niah_data.py -x -q`
Expected: 2 passed

- [ ] **Step 5: prepare-a 실행 (CPU, 로컬 실행 가능)**

Run: `source lmr/analysis/260725_mc_niah_analysis/env_common.sh && $PY lmr/analysis/260725_mc_niah_analysis/data.py prepare-a`
Expected: 두 태스크 각 `50 samples` 출력. `head -c 500 $MC_OUT/data/2048/niah_multikey_1/validation.jsonl`로 needle 4개 형식 눈검사. 추가 검증: annotate가 50샘플 전부에서 예외 없이 gold_seg를 찾는지 —

```bash
$PY - <<'EOF'
import json, os, sys
sys.path.insert(0, "lmr/analysis/260725_mc_niah_analysis")
import data as d
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(d.TOKENIZER)
for task in ("niah_single_1", "niah_multikey_1"):
    p = os.path.join(d.MC_OUT, "data", "2048", task, "validation.jsonl")
    anns = [d.annotate(json.loads(l)["input"], tok) for l in open(p)]
    segs = [a["gold_seg"] for a in anns]
    print(task, "n=", len(anns), "gold_seg range", min(segs), max(segs),
          "n_tok max", max(a["n_tok"] for a in anns))
EOF
```
Expected: 예외 없음, n=50, n_tok ≤ 1920 부근, gold_seg 0~7 분포

- [ ] **Step 6: 커밋**

```bash
git add lmr/analysis/260725_mc_niah_analysis/data.py tests/lmr/test_mc_niah_data.py
git commit -m "mc-niah: Dataset A prepare (RULER, TinyLlama tok) + needle annotation"
```

---

