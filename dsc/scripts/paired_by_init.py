import json, math
V = ["mean1", "maxsim", "mixq"]
R = {}
for v in V:
    d = json.load(open(f"/root/a19_query_conditional/{v}.json"))["variants"]
    R[v] = list(d.values())[0]

def paired(a, b, sub):
    ra, rb = R[a][f"runs_{sub}"], R[b][f"runs_{sub}"]
    d = [x - y for x, y in zip(ra, rb)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    t = m / se if se else float("inf")
    return m, se, t, f"{sum(1 for x in d if x > 0)}/{n}"

print("paired by init seed -- every variant uses inits 0-3 with the same data order\n")
for sub in ("all", "few_needles", "many_needles"):
    print(f"--- {sub} ---")
    for a, b in [("mixq", "maxsim"), ("mixq", "mean1"),
                 ("maxsim", "mean1")]:
        m, se, t, pos = paired(a, b, sub)
        print(f"  {a:>7} - {b:<11} diff={m:+.4f}  se={se:.4f}  "
              f"t={t:+6.2f} (df=3)  {pos} positive")
    print()

for v in V:
    r = R[v]
    few = "/".join(f"{x:.3f}" for x in r["runs_few_needles"])
    many = "/".join(f"{x:.3f}" for x in r["runs_many_needles"])
    print(f"{v:<11} few {few}   many {many}")
