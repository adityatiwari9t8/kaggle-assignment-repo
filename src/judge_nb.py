# %% [markdown]
# # LLM Classification Finetuning — Siamese Pairwise Judge
#
# **Task.** Each row has a user `prompt` and two chatbot answers (`response_a`, `response_b`).
# Predict three probabilities: A preferred, B preferred, or tie. Metric: multiclass **log loss**.
#
# **My approach (original design, no code reused from other repos).**
# 1. Encode `(prompt, response_a)` and `(prompt, response_b)` *separately* with one **shared DeBERTa-v3 encoder**
#    (mean pooling), instead of cramming everything into one text.
# 2. Compare the two vectors: `[hA, hB, hA-hB, hA*hB]` plus four cheap *style-difference* features
#    (length, words, lines, bullets/code) -> small MLP -> 3 logits.
# 3. **Swap symmetry.** During training, A/B are swapped at random (labels and style features swapped too).
#    At inference, I average the logits of the normal and the swapped orientation (test-time augmentation).
# 4. **Length-aware truncation** (keep the head *and* the tail of long answers) so conclusions are not cut off.
# 5. **Calibration** on a held-out split: temperature scaling + a small blend toward the class prior.
#
# **How to run on Kaggle (internet OFF is fine).**
# - Add the competition data, add a DeBERTa-v3 model (base or small) as an *Input*, pick a GPU.
# - Run with `MODE = "smoke"` first (a few minutes), then set `MODE = "full"`.
# - The notebook writes `submission.csv` (`id, winner_model_a, winner_model_b, winner_tie`).

# %%
import os, re, json, math, time, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

MODE = "smoke"          # "smoke" = quick pipeline check, "full" = real training run
SEED = 42
VAL_FRAC = 0.08         # held-out share used for honest scoring and calibration
LR_ENCODER, LR_HEAD = 2e-5, 2e-4

CFG = {
    "smoke": dict(n_train=800, n_val=240, max_len=256, bs=4, accum=1, epochs=1, time_budget_min=10),
    "full":  dict(n_train=None, n_val=None, max_len=384, bs=8, accum=2, epochs=2, time_budget_min=240),
}[MODE]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("mode:", MODE, "| device:", DEVICE, "| cfg:", CFG)

# %%
# ---------- Pure-python / numpy helpers (no torch needed) ----------

def clip_chars(s, head=3000, tail=1500):
    """Cheap pre-clip on characters so tokenising 57k long texts stays fast; keeps start and end."""
    if len(s) <= head + tail:
        return s
    return s[:head] + "\n...\n" + s[-tail:]


def parse_turns(cell):
    """A cell is a JSON array of conversation turns (entries can be null). Return one clean string."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return ""
    try:
        turns = json.loads(cell)
    except (TypeError, ValueError):
        return str(cell)
    if isinstance(turns, str):
        return turns
    if not isinstance(turns, list):
        return str(turns)
    return "\n".join(t for t in turns if isinstance(t, str) and t.strip())


def style_features(text):
    """Four log-scaled surface statistics. Differences between A and B feed the classifier head."""
    n_chars = len(text)
    n_words = len(text.split())
    n_lines = text.count("\n")
    n_struct = len(re.findall(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s", text)) + text.count("```")
    return np.array([math.log1p(n_chars), math.log1p(n_words),
                     math.log1p(n_lines), math.log1p(n_struct)], dtype=np.float32)


def build_seq(p_ids, r_ids, max_len, cls_id, sep_id, p_cap=96):
    """[CLS] prompt [SEP] response [SEP]; long prompts/responses keep their head AND tail."""
    if len(p_ids) > p_cap:
        h = p_cap * 2 // 3
        p_ids = p_ids[:h] + p_ids[len(p_ids) - (p_cap - h):]
    room = max_len - 3 - len(p_ids)
    if len(r_ids) > room:
        h = int(room * 0.65)
        r_ids = r_ids[:h] + r_ids[len(r_ids) - (room - h):]
    return [cls_id] + list(p_ids) + [sep_id] + list(r_ids) + [sep_id]


def softmax_np(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def mlogloss(P, y):
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def fit_temperature(logits, y):
    grid = np.linspace(0.4, 8.0, 153)
    best = min(grid, key=lambda T: mlogloss(softmax_np(logits, T), y))
    fine = np.linspace(max(0.2, best - 0.05), best + 0.05, 21)
    return float(min(fine, key=lambda T: mlogloss(softmax_np(logits, T), y)))


def fit_prior_blend(P, y, prior):
    grid = np.linspace(0.0, 0.30, 31)
    return float(min(grid, key=lambda lam: mlogloss((1 - lam) * P + lam * prior, y)))


# %%
# ---------- Locate data + pretrained model (works offline on Kaggle) ----------

def find_data_dir():
    if os.environ.get("DATA_DIR"):
        return os.environ["DATA_DIR"]
    for pat in ("/kaggle/input/*/train.csv", "/kaggle/input/*/*/train.csv"):
        hits = glob.glob(pat)
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError("train.csv not found - add the competition data as an Input.")


def find_model_dir():
    if os.environ.get("MODEL_DIR"):
        return os.environ["MODEL_DIR"]
    cands = []
    for cfg_path in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg_path)
        try:
            mt = json.load(open(cfg_path)).get("model_type", "")
        except Exception:
            continue
        has_tok = any(os.path.exists(os.path.join(d, f)) for f in ("spm.model", "tokenizer.json"))
        if mt == "deberta-v2" and has_tok:
            cands.append(d)
    if not cands:
        raise FileNotFoundError("No DeBERTa-v3 model found - add one (e.g. deberta-v3-base) as an Input.")
    cands.sort(key=lambda d: ("base" not in d.lower(), d))
    return cands[0]


DATA_DIR, MODEL_DIR = find_data_dir(), find_model_dir()
print("data :", DATA_DIR, "\nmodel:", MODEL_DIR)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print("train rows:", len(train_df), "| test rows:", len(test_df))

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(train_df))
n_val = CFG["n_val"] or int(VAL_FRAC * len(train_df))
val_idx = perm[:n_val]
train_idx = perm[n_val:]
if CFG["n_train"]:
    train_idx = train_idx[:CFG["n_train"]]
print("fit rows:", len(train_idx), "| held-out rows:", len(val_idx))

# labels: 0 = A wins, 1 = B wins, 2 = tie
y_all = train_df[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy().argmax(axis=1)
prior = np.bincount(y_all[train_idx], minlength=3) / len(train_idx)
print("class prior (A, B, tie):", np.round(prior, 4))

# %%
# ---------- Tokenise once ----------
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
CLS_ID, SEP_ID, PAD_ID = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id


def encode_texts(texts, head, tail, chunk=2000):
    out = []
    for i in range(0, len(texts), chunk):
        batch = [clip_chars(t, head, tail) for t in texts[i:i + chunk]]
        out.extend(tok(batch, add_special_tokens=False, truncation=False)["input_ids"])
    return out


class PairData:
    """Holds pre-tokenised prompt / answers and style differences for one dataframe."""

    def __init__(self, df, labels=None):
        P = [parse_turns(c) for c in df["prompt"]]
        A = [parse_turns(c) for c in df["response_a"]]
        B = [parse_turns(c) for c in df["response_b"]]
        t0 = time.time()
        self.P = encode_texts(P, 600, 300)
        self.A = encode_texts(A, 3000, 1500)
        self.B = encode_texts(B, 3000, 1500)
        self.extra = np.stack([style_features(a) - style_features(b) for a, b in zip(A, B)])
        self.y = labels if labels is not None else np.full(len(df), -1)
        print(f"tokenised {len(df)} rows in {time.time() - t0:.0f}s")

    def approx_len(self, i):
        m = CFG["max_len"]
        return min(len(self.A[i]), m) + min(len(self.B[i]), m)

    def get(self, i, swap=False):
        a = build_seq(self.P[i], self.A[i], CFG["max_len"], CLS_ID, SEP_ID)
        b = build_seq(self.P[i], self.B[i], CFG["max_len"], CLS_ID, SEP_ID)
        ex, y = self.extra[i], int(self.y[i])
        if swap:                                  # exchange A/B everywhere -> same example, mirrored
            a, b, ex = b, a, -ex
            if y in (0, 1):
                y = 1 - y
        return a, b, ex, y


def collate(items):
    seqs = [it[0] for it in items] + [it[1] for it in items]     # first half = A side, second half = B side
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    extra = torch.tensor(np.stack([it[2] for it in items]), dtype=torch.float32)
    y = torch.tensor([it[3] for it in items], dtype=torch.long)
    return ids, mask, extra, y


fit_data = PairData(train_df.iloc[train_idx].reset_index(drop=True), y_all[train_idx])
val_data = PairData(train_df.iloc[val_idx].reset_index(drop=True), y_all[val_idx])

# %%
# ---------- Model ----------

class PairJudge(nn.Module):
    def __init__(self, path, n_extra=4, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(path).float()
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(4 * h + n_extra, h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(h, 3),
        )

    def pool(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, ids, mask, extra):
        pooled = self.pool(ids, mask)
        n = extra.shape[0]
        ha, hb = pooled[:n], pooled[n:]
        z = torch.cat([ha, hb, ha - hb, ha * hb, extra.to(ha.dtype)], dim=-1)
        return self.head(z)


model = PairJudge(MODEL_DIR).to(DEVICE)

# %%
# ---------- Training ----------

def train_batches(n, data, bs, gen):
    """Shuffle, then sort inside mega-chunks by length so padding is small; shuffle the batches."""
    order = gen.permutation(n)
    batches = []
    for s in range(0, n, bs * 40):
        part = sorted(order[s:s + bs * 40], key=data.approx_len)
        batches += [part[k:k + bs] for k in range(0, len(part), bs)]
    gen.shuffle(batches)
    return batches


bs, accum, epochs = CFG["bs"], CFG["accum"], CFG["epochs"]
est_batches = math.ceil(len(fit_data.y) / bs) + len(fit_data.y) // (bs * 40) + 1
total_steps = max(1, epochs * math.ceil(est_batches / accum))

enc_params = [p for n_, p in model.named_parameters() if n_.startswith("enc.")]
head_params = [p for n_, p in model.named_parameters() if n_.startswith("head.")]
opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENCODER},
                         {"params": head_params, "lr": LR_HEAD}], weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

gen = np.random.default_rng(SEED)
t_start, run_loss, run_n, stop = time.time(), 0.0, 0, False
model.train()
for ep in range(epochs):
    batches = train_batches(len(fit_data.y), fit_data, bs, gen)
    for bi, batch in enumerate(batches):
        items = [fit_data.get(int(i), swap=bool(gen.random() < 0.5)) for i in batch]
        ids, mask, extra, y = collate(items)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            logits = model(ids.to(DEVICE), mask.to(DEVICE), extra.to(DEVICE))
        loss = F.cross_entropy(logits.float(), y.to(DEVICE)) / accum
        scaler.scale(loss).backward()
        run_loss += loss.item() * accum; run_n += 1
        if (bi + 1) % accum == 0 or bi == len(batches) - 1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
        if run_n % 100 == 0:
            print(f"ep {ep} batch {bi}/{len(batches)} train-loss {run_loss / run_n:.4f} "
                  f"elapsed {(time.time() - t_start) / 60:.1f} min")
            run_loss, run_n = 0.0, 0
        if (time.time() - t_start) / 60 > CFG["time_budget_min"]:
            print("time budget reached - stopping training early")
            stop = True
            break
    if stop:
        break
print(f"training done in {(time.time() - t_start) / 60:.1f} min")

# %%
# ---------- Inference helpers ----------

@torch.no_grad()
def predict_logits(data, idx, swap, ibs=32):
    """Logits for rows `idx` (returned in that order). If swap, columns are mapped back to A/B/tie."""
    model.eval()
    idx = [int(i) for i in idx]
    order = sorted(idx, key=data.approx_len)
    pos = {i: k for k, i in enumerate(idx)}
    out = np.zeros((len(idx), 3), dtype=np.float32)
    for s in range(0, len(order), ibs):
        chunk = order[s:s + ibs]
        ids, mask, extra, _ = collate([data.get(i, swap=swap) for i in chunk])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            lg = model(ids.to(DEVICE), mask.to(DEVICE), extra.to(DEVICE))
        lg = lg.float().cpu().numpy()
        if swap:
            lg = lg[:, [1, 0, 2]]
        for r, i in enumerate(chunk):
            out[pos[i]] = lg[r]
    return out


# %%
# ---------- Honest validation + calibration ----------
yv = val_data.y
lg_plain = predict_logits(val_data, range(len(yv)), swap=False)
lg_swap = predict_logits(val_data, range(len(yv)), swap=True)
lg_tta = 0.5 * (lg_plain + lg_swap)

uniform = np.full((len(yv), 3), 1 / 3)
prior_P = np.tile(prior, (len(yv), 1))
T = fit_temperature(lg_tta, yv)
P_T = softmax_np(lg_tta, T)
lam = fit_prior_blend(P_T, yv, prior)
P_final = (1 - lam) * P_T + lam * prior

report = {
    "mode": MODE, "n_fit": int(len(fit_data.y)), "n_heldout": int(len(yv)),
    "logloss_uniform": mlogloss(uniform, yv),
    "logloss_train_prior": mlogloss(prior_P, yv),
    "logloss_single_orientation": mlogloss(softmax_np(lg_plain), yv),
    "logloss_swap_tta": mlogloss(softmax_np(lg_tta), yv),
    "temperature": T, "logloss_tta_temperature": mlogloss(P_T, yv),
    "prior_blend": lam, "logloss_final_calibrated": mlogloss(P_final, yv),
    "accuracy": float((lg_tta.argmax(1) == yv).mean()),
}
for k, v in report.items():
    print(f"{k:>30}: {v:.4f}" if isinstance(v, float) else f"{k:>30}: {v}")
json.dump(report, open(os.path.join(OUT_DIR, "run_report.json"), "w"), indent=2)
print("\nCalibration is fitted on this held-out split, so the last score is slightly optimistic.")

# %%
# ---------- Predict the test set and write submission.csv ----------
test_data = PairData(test_df.reset_index(drop=True))
n_test = len(test_df)
lt = 0.5 * (predict_logits(test_data, range(n_test), swap=False) +
            predict_logits(test_data, range(n_test), swap=True))
P_test = (1 - lam) * softmax_np(lt, T) + lam * prior
P_test = np.clip(P_test, 1e-4, 1.0)
P_test = P_test / P_test.sum(axis=1, keepdims=True)

sub = pd.DataFrame({
    "id": test_df["id"].values,
    "winner_model_a": P_test[:, 0],
    "winner_model_b": P_test[:, 1],
    "winner_tie": P_test[:, 2],
})
assert len(sub) == n_test and sub["id"].is_unique
assert np.isfinite(sub.iloc[:, 1:].to_numpy()).all()
assert np.allclose(sub.iloc[:, 1:].sum(axis=1), 1.0, atol=1e-6)
sub.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print(sub.head())
print("wrote", os.path.join(OUT_DIR, "submission.csv"), sub.shape)
