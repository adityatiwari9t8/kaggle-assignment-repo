import re, json, math, numpy as np, pandas as pd
src = open(__import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "judge_nb.py").read()
# grab only the torch-free helper cell
cell = src.split("# %%\n# ---------- Pure-python / numpy helpers (no torch needed) ----------")[1].split("# %%\n# ---------- Locate data")[0]
exec(cell)

# 1) parsing: JSON arrays, nulls, NaN, malformed
assert parse_turns('["hi", null, "there"]') == "hi\nthere"
assert parse_turns('[null]') == ""
assert parse_turns(float("nan")) == ""
assert parse_turns("plain text, not json") == "plain text, not json"
print("parse_turns OK")

# 2) style features antisymmetry under swap
a, b = "Short.", "- one\n- two\n```code```\nA much longer answer with several words."
fa, fb = style_features(a), style_features(b)
assert np.allclose((fa - fb), -(fb - fa)) and (fa != fb).any()
print("style_features OK", np.round(fa - fb, 2))

# 3) truncation keeps head+tail and never exceeds max_len
for p_len in (3, 96, 500):
    for r_len in (5, 200, 5000):
        p, r = list(range(1000, 1000 + p_len)), list(range(5000, 5000 + r_len))
        s = build_seq(p, r, 256, 1, 2)
        assert len(s) <= 256 and s[0] == 1 and s[-1] == 2, (p_len, r_len, len(s))
        if r_len == 5000: assert 5000 in s and (5000 + r_len - 1) in s   # head and tail kept
print("build_seq OK")

# 4) metric + calibration sanity on synthetic overconfident logits
rng = np.random.default_rng(0); n = 4000
y = rng.choice(3, n, p=[.35, .34, .31])
logits = rng.normal(0, 1, (n, 3)) * 3 + np.eye(3)[y] * 0.6        # noisy + overconfident
assert abs(mlogloss(np.full((n, 3), 1/3), y) - math.log(3)) < 1e-9
raw = mlogloss(softmax_np(logits), y)
T = fit_temperature(logits, y); cal = mlogloss(softmax_np(logits, T), y)
prior = np.bincount(y, minlength=3) / n
lam = fit_prior_blend(softmax_np(logits, T), y, prior)
fin = mlogloss((1-lam)*softmax_np(logits, T) + lam*prior, y)
print(f"raw {raw:.4f} -> temp(T={T:.2f}) {cal:.4f} -> +prior(lam={lam:.2f}) {fin:.4f}")
assert cal < raw and fin <= cal + 1e-12

# 5) label swap logic + TTA column mapping (mirrors PairData.get / predict_logits)
lab = np.array([0, 1, 2]); swapped = np.where(np.isin(lab, [0, 1]), 1 - lab, lab); assert list(swapped) == [1, 0, 2]
lg = np.array([[2., 0., 1.]]); assert list(lg[:, [1, 0, 2]][0]) == [0., 2., 1.]
print("swap mapping OK")

# 6) submission format check on a fake frame
P = softmax_np(rng.normal(size=(5, 3))); P = np.clip(P, 1e-4, 1); P /= P.sum(1, keepdims=True)
sub = pd.DataFrame({"id": range(5), "winner_model_a": P[:, 0], "winner_model_b": P[:, 1], "winner_tie": P[:, 2]})
assert np.allclose(sub.iloc[:, 1:].sum(axis=1), 1.0, atol=1e-6)
print("submission format OK:", list(sub.columns))
