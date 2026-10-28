"""Dataset B: single/multi paired NIAH, distractor 배치 통제(S=같은 segment, D=다른 segment).

토큰 정밀 배치: 문장 단위로 토큰 수를 누적하며 목표 offset에 needle 삽입 후
annotate()로 실측 segment를 검증, 어긋나면 삽입점을 한 문장씩 밀며 재시도.

single/multi가 같은 haystack을 공유하도록: multi를 먼저 조합한 뒤, single은 multi
텍스트에서 distractor needle 문장만 토큰 수가 정확히 같은 중립 필러로 치환해서 만든다
(_neutralize). 이렇게 하면 gold needle의 절대 토큰 위치·segment가 single/multi 간에
완전히 동일해지고, distractor span을 제외한 나머지 토큰 시퀀스도 완전히 동일해진다.
"""
import json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as mcdata

WORDS = ("apple bridge candle dragon engine forest guitar harbor island jungle kettle "
         "ladder magnet needle orchid pillow quartz rocket saddle temple umbrella violin "
         "walnut yonder zephyr anchor bamboo canyon dolphin ember falcon glacier").split()
NEEDLE_FMT = "One of the special magic numbers for {key} is: {value}."
TEMPLATE = ("Some special magic numbers are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the numbers afterwards.\n"
            "{context}\n"
            "What is the special magic number for {query} mentioned in the provided text?")
CHUNK = mcdata.CHUNK

# 중립 필러 단어 풀 — TinyLlama 토크나이저에서 각 단어가 (선행 공백 포함) 정확히
# 1토큰씩 늘어나는 것을 실측 확인함 (일반 소문자 단어, 특수문자/숫자 없음).
NEUTRAL_ATOMS = ("grass green sky blue sun yellow here go there back again soon later "
                  "quiet slow small large open close good calm steady gentle plain simple "
                  "clear round smooth deep wide thin bright still soft warm cool dry damp "
                  "firm loose bold faint keen mild rough").split()


def _essay_sentences(tokenizer, budget_toks):
    essays = json.load(open(os.path.join(
        mcdata.REPO, "src", "ruler", "gen", "synthetic", "json", "PaulGrahamEssays.json")))["text"]
    sents, total = [], 0
    for s in re.split(r"(?<=[.!?]) +", essays):
        s = s.strip()
        if not s:
            continue
        n = len(tokenizer(s, add_special_tokens=False).input_ids)
        sents.append((s, n)); total += n
        if total > budget_toks * 3:
            break
    return sents


def _compose(sents, inserts, tokenizer, body_budget):
    """inserts: [(target_tok_offset, text)] 오름차순. 문장 누적으로 배치."""
    inserts = sorted(inserts)
    out, cum, ii = [], 0, 0
    for s, n in sents:
        while ii < len(inserts) and cum >= inserts[ii][0]:
            out.append(inserts[ii][1]); ii += 1
            cum += len(tokenizer(inserts[ii - 1][1], add_special_tokens=False).input_ids)
        if cum + n > body_budget:
            break
        out.append(s); cum += n
    while ii < len(inserts):          # budget 끝에 못 넣었으면 마지막에
        out.append(inserts[ii][1]); ii += 1
    return " ".join(out)


def _fit_filler(tokenizer, target_n, max_tries=50):
    """NEUTRAL_ATOMS로 만든, 단독 토큰 수(add_special_tokens=False)가 정확히
    target_n인 중립 문장을 구성. 각 단어가 +1토큰씩 늘어남을 실측 확인했으므로
    target_n-1개 단어 + 마침표로 거의 항상 한 번에 맞아떨어지지만, 혹시 모를 경계
    효과에 대비해 순서를 바꿔가며 최대 max_tries번 재시도한다."""
    n_atoms = len(NEUTRAL_ATOMS)
    k = max(1, target_n - 1)
    for attempt in range(max_tries):
        rng = random.Random(attempt * 104729 + target_n)
        pool = NEUTRAL_ATOMS[:]
        rng.shuffle(pool)
        words = [pool[i % n_atoms] for i in range(k)]
        cand = " ".join(words) + "."
        if len(tokenizer(cand, add_special_tokens=False).input_ids) == target_n:
            return cand
    raise RuntimeError(f"filler search failed for target_n={target_n} after {max_tries} tries")


def _neutralize(tokenizer, ctx_multi, distractor_sentences):
    """multi context에서 distractor needle 문장들을, 실측 in-context 토큰 수가
    정확히 같은 중립 필러로 치환해 single context를 만든다. 문자 길이는 달라져도
    되지만 토큰 수는 정확히 같아야 gold needle 이후 텍스트의 토큰 정렬이 보존된다.
    오른쪽(뒤쪽) 문장부터 치환해 앞쪽 char offset이 안 틀어지게 한다."""
    enc = tokenizer(ctx_multi, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.offset_mapping

    def tok_at(p):
        return next(i for i, (s, e) in enumerate(offsets) if s <= p < e)

    spans = []
    for d in distractor_sentences:
        start = ctx_multi.index(d)
        end = start + len(d)
        lo, hi = tok_at(start), tok_at(end - 1)
        spans.append((start, end, hi - lo + 1))
    spans.sort(key=lambda t: -t[0])   # 뒤쪽부터 치환 (앞쪽 char offset 보존)

    ctx_single = ctx_multi
    for start, end, target_n in spans:
        filler = _fit_filler(tokenizer, target_n)
        ctx_single = ctx_single[:start] + filler + ctx_single[end:]
    return ctx_single


def _make_one(tokenizer, sents, rng, condition, seq_len=2048, n_gen=128, num_keys=4):
    overhead = 96                      # 템플릿+질문 여유
    body = seq_len - n_gen - overhead  # ≈1824 tokens
    n_seg_body = body // CHUNK         # 7 — gold는 1..n_seg_body-2에서 선택
    keys = rng.sample(WORDS, num_keys)
    vals = [str(rng.randint(1000000, 9999999)) for _ in range(num_keys)]
    gold_key, gold_val = keys[0], vals[0]

    for attempt in range(8):
        gold_seg = rng.randint(1, n_seg_body - 2)
        others = [s for s in range(1, n_seg_body - 1) if s != gold_seg]
        if condition == "S":
            d_segs = [gold_seg] + rng.sample(others, num_keys - 2)
        else:
            d_segs = rng.sample(others, num_keys - 1)
        gold_off = gold_seg * CHUNK + rng.randint(16, CHUNK - 96)
        ins = [(gold_off, NEEDLE_FMT.format(key=gold_key, value=gold_val))]
        distractor_sentences = []
        codist_dist = None
        for k, v, s in zip(keys[1:], vals[1:], d_segs):
            if s == gold_seg:
                delta = rng.choice([-64, 64])
                off = min(max(s * CHUNK + 8, gold_off + delta), (s + 1) * CHUNK - 40)
                codist_dist = abs(off - gold_off)
            else:
                off = s * CHUNK + rng.randint(16, CHUNK - 96)
            sent = NEEDLE_FMT.format(key=k, value=v)
            ins.append((off, sent))
            distractor_sentences.append(sent)

        ctx_multi = _compose(sents, ins, tokenizer, body)
        try:
            # single은 multi 텍스트에서 distractor 문장만 길이 맞춘 필러로 치환해
            # 만든다 — 같은 haystack을 공유하도록 (재조합하지 않음).
            ctx_single = _neutralize(tokenizer, ctx_multi, distractor_sentences)
        except RuntimeError:
            continue  # 필러 탐색 실패 — 이번 attempt 통째로 재추첨

        row_m = {"input": TEMPLATE.format(context=ctx_multi, query=gold_key),
                 "outputs": [gold_val]}
        row_s = {"input": TEMPLATE.format(context=ctx_single, query=gold_key),
                 "outputs": [gold_val]}
        try:
            am = mcdata.annotate(row_m["input"], tokenizer)
            asg = mcdata.annotate(row_s["input"], tokenizer)
        except ValueError:
            continue
        d_actual = sorted(n["seg"] for n in am["needles"] if n["key"] != gold_key)
        # 두 분기 모두 실측(annotate) gold_seg를 기준으로 판정한다.
        ok_cond = ((am["gold_seg"] in d_actual) if condition == "S"
                   else (am["gold_seg"] not in d_actual))
        gold_m = next(n for n in am["needles"] if n["key"] == gold_key)
        gold_s = next(n for n in asg["needles"] if n["key"] == gold_key)
        # single/multi에서 gold가 완전히 같은 토큰 위치·segment에 실측 배치됐고,
        # 전체 토큰 수도 정확히 같은지까지 확인 (haystack 공유 불변식).
        if (len(am["needles"]) == num_keys and len(asg["needles"]) == 1
                and am["gold_seg"] == asg["gold_seg"]
                and gold_m["tok_start"] == gold_s["tok_start"]
                and am["n_tok"] == asg["n_tok"]
                and ok_cond and am["n_tok"] <= seq_len - n_gen
                and asg["n_tok"] <= seq_len - n_gen):
            meta = {"gold_seg": am["gold_seg"], "distractor_segs": d_actual,
                    "needle_key": gold_key, "condition": condition,
                    "codist_tok_dist": codist_dist}
            return {**row_s, **meta, "variant": "single"}, {**row_m, **meta, "variant": "multi"}
    raise RuntimeError(f"placement failed after 8 attempts (condition={condition})")


def build_pairs(tokenizer, n_pairs, condition, seed=42, seq_len=2048, n_gen=128):
    rng = random.Random(seed + (0 if condition == "S" else 1000))
    sents = _essay_sentences(tokenizer, seq_len)
    rows = []
    for pid in range(n_pairs):
        s, m = _make_one(tokenizer, sents, rng, condition, seq_len, n_gen)
        s["pair_id"] = m["pair_id"] = pid
        rows += [s, m]
    return rows


def prepare_b(n_pairs_per_cond=16, seq_len=2048, n_gen=128, seed=42):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mcdata.TOKENIZER)
    out_dir = os.path.join(mcdata.MC_OUT, "data", "paired")
    os.makedirs(out_dir, exist_ok=True)
    for cond in ("S", "D"):
        rows = build_pairs(tok, n_pairs_per_cond, cond, seed, seq_len, n_gen)
        p = os.path.join(out_dir, f"{cond}.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[prepare-b] {cond}: {len(rows)} rows ({n_pairs_per_cond} pairs) -> {p}")
