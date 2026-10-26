"""F3 — additive merge violates the delta algebra (the original argument, no model, CPU).
Store key k -> value A in segment 1; UPDATE same k -> value B in segment 2.
(a) single continuous delta state: the erase term removes A, read = B (correct).
(b) two segments cached separately + ADDITIVE read (any scalar weights): read = w1*A + w2*B,
    so the *overwritten* value A cannot be removed -> 'resurrection of a deleted value'.
y = |component of read along A| and |along B|. This is our analysis (not from a paper)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0); d = 16
k = rng.standard_normal(d); k /= np.linalg.norm(k)          # the (shared) key
# value basis: A, B orthonormal so we can read off components cleanly
A = rng.standard_normal(d); A -= (A@k)*k; A /= np.linalg.norm(A)
B = rng.standard_normal(d); B -= (B@k)*k; B -= (B@A)*A;      B /= np.linalg.norm(B)
def write(S, kk, vv, beta=1.0):                              # delta rule (unit key): erase then write
    return S - beta*np.outer(S@kk, kk) + beta*np.outer(vv, kk)
# (a) single continuous state over both segments
S = np.zeros((d, d)); S = write(S, k, A); S = write(S, k, B)
read_single = S @ k
# (b) two separately-cached segment states, additive read (optimal-ish equal weights shown; any w>0 keeps A)
S1 = write(np.zeros((d, d)), k, A); S2 = write(np.zeros((d, d)), k, B)
read_add = (S1 @ k) + (S2 @ k)
def comp(v): return abs(v@A), abs(v@B)
sa, sb = comp(read_single); aa, ab = comp(read_add)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
x = np.arange(2); w = 0.36
ax.bar(x-w/2, [sa, aa], w, label="A component (overwritten)", color="#c0392b")
ax.bar(x+w/2, [sb, ab], w, label="B component (current)", color="#2471a3")
ax.set_xticks(x); ax.set_xticklabels(["single delta state\n(continuous)", "2 segments\n+ additive read"])
ax.set_ylabel("|read · value|"); ax.set_title("Additive merge resurrects the deleted value A\n(delta erase is lost)")
ax.legend(fontsize=8, frameon=False); ax.set_ylim(0, 1.15)
for xi, (va, vb) in zip(x, [(sa, sb), (aa, ab)]):
    ax.text(xi-w/2, va+0.02, f"{va:.2f}", ha="center", fontsize=8)
    ax.text(xi+w/2, vb+0.02, f"{vb:.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("/home/sohyung/labmeeting_figures/F3_delta_toy.png", dpi=200)
print(f"single: A={sa:.3f} B={sb:.3f}  (want A~0, B~1)")
print(f"additive: A={aa:.3f} B={ab:.3f}  (A resurrected)")
print("saved F3_delta_toy.png")
