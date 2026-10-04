"""AI651 PA1 - Task 2 (Leaderboard): Autoformer forecaster for a 168-step horizon.

Layout
  autoformer/AutoCorrelation.py, autoformer/Autoformer_EncDec.py
      Auto-Correlation and series-decomposition blocks, copied from the official
      implementation (github.com/thuml/Autoformer, MIT licence; Wu et al., 2021).
  task2.py (this file)
      data handling, phase features, covariate handling, the model wrapper
      (embedding written here so encoder and decoder can take different covariates),
      training, chronological validation, baselines and the final forecast.

Run
  python task2.py baselines
  python task2.py run --variant none --seed 0
  python task2.py final --variant <v> --seed 0
"""
import argparse, json, math, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from autoformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from autoformer.Autoformer_EncDec import (Encoder, Decoder, EncoderLayer, DecoderLayer,
                                          my_Layernorm, series_decomp)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "Data")
RES = os.path.join(HERE, "results")

# ----------------------------------------------------------------------------- config
SEQ_LEN, LABEL_LEN, PRED_LEN = 168, 48, 168
N_HIST = 43656                      # observed history length
N_EVAL_BLOCKS, N_ES_BLOCKS = 24, 12  # non-overlapping 168-step blocks
EVAL_START = N_HIST - N_EVAL_BLOCKS * PRED_LEN     # 39624
ES_START = EVAL_START - N_ES_BLOCKS * PRED_LEN     # 37608  (training targets end here)
PERIODS = (24.0, 8766.0)             # recovered from the periodogram (daily, yearly)

MODEL_CFG = dict(d_model=32, n_heads=4, e_layers=1, d_layers=1, d_ff=64,
                 moving_avg=25, factor=1, dropout=0.1, exo_head=True)
TRAIN_CFG = dict(batch=64, lr=3e-4, exo_lr=1e-2, max_epochs=10, patience=3, train_stride=2)

CONT = ["feature_A", "feature_B", "feature_C", "feature_D", "feature_E", "feature_F"]
BIN = ["feature_G", "feature_H", "feature_I", "feature_J"]
LOG_COLS = ["feature_D", "feature_E", "feature_F"]   # heavy-tailed cumulative counters


# ----------------------------------------------------------------------------- data
def load():
    y = pd.read_csv(os.path.join(DATA, "student_train.csv"))["value"].to_numpy(float)
    X = pd.read_csv(os.path.join(DATA, "optional_external_data.csv"))
    assert len(y) == N_HIST and len(X) == N_HIST + PRED_LEN
    t = X["time_idx"].to_numpy(float)
    phase = np.concatenate([np.stack([np.sin(2 * np.pi * t / p), np.cos(2 * np.pi * t / p)], 1)
                            for p in PERIODS], 1)
    cov = X[CONT + BIN].copy()
    for c in LOG_COLS:
        cov[c] = np.log1p(cov[c])
    cov = cov.to_numpy(float)
    mu, sd = cov[:ES_START].mean(0), cov[:ES_START].std(0) + 1e-8
    mu[len(CONT):], sd[len(CONT):] = 0.0, 1.0          # leave one-hot columns as 0/1
    cov = (cov - mu) / sd
    return y, phase.astype(np.float32), cov.astype(np.float32)


def marks_for(variant, phase, cov):
    """Return (encoder marks, decoder marks) arrays over the full 43,824-step clock."""
    if variant == "none":
        return phase, phase
    if variant == "past":            # covariates only up to the forecast origin
        return np.concatenate([phase, cov], 1), phase
    if variant == "pastfuture":      # covariates also across the forecast horizon
        both = np.concatenate([phase, cov], 1)
        return both, both
    raise ValueError(variant)


class Scaler:
    def __init__(self, y):
        self.m, self.s = float(y[:ES_START].mean()), float(y[:ES_START].std())
    def f(self, v): return (v - self.m) / self.s
    def inv(self, v): return v * self.s + self.m


def make_batch(origins, ys, menc, mdec):
    """origin o = index of the first forecast step (0-based)."""
    xe = np.stack([ys[o - SEQ_LEN:o] for o in origins])[..., None]
    me = np.stack([menc[o - SEQ_LEN:o] for o in origins])
    md = np.stack([mdec[o - LABEL_LEN:o + PRED_LEN] for o in origins])
    return torch.tensor(xe), torch.tensor(me), torch.tensor(md)


def target(origins, ys):
    return torch.tensor(np.stack([ys[o:o + PRED_LEN] for o in origins])[..., None])


# ----------------------------------------------------------------------------- model
class Embedding(nn.Module):
    """Value embedding (circular conv, as in Autoformer's TokenEmbedding) plus a linear
    map of the per-step marks. No positional embedding, as in Autoformer."""
    def __init__(self, c_in, mark_dim, d_model, dropout):
        super().__init__()
        self.value = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular", bias=False)
        self.mark = nn.Linear(mark_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, m):
        return self.drop(self.value(x.permute(0, 2, 1)).transpose(1, 2) + self.mark(m))


class Autoformer(nn.Module):
    """Wiring follows thuml/Autoformer models/Autoformer.py: input decomposition,
    trend initialised with the window mean, seasonal part with zeros, Auto-Correlation
    in the encoder and in both decoder attentions, progressive decomposition inside
    every block, trend accumulated through the decoder."""
    def __init__(self, enc_mark, dec_mark, d_model, n_heads, e_layers, d_layers, d_ff,
                 moving_avg, factor, dropout, exo_head=True):
        super().__init__()
        # Exogenous linear head: a per-step linear map from the decoder marks over the
        # horizon straight to the output, added to Autoformer's seasonal + trend output.
        # Auto-Correlation aggregates time-shifted copies of the sequence, which smears a
        # covariate effect that acts at a specific step; this path keeps it pointwise.
        # Cost: dec_mark + 1 parameters.
        self.exo = nn.Linear(dec_mark, 1) if exo_head else None
        self.decomp = series_decomp(moving_avg)
        self.enc_emb = Embedding(1, enc_mark, d_model, dropout)
        self.dec_emb = Embedding(1, dec_mark, d_model, dropout)
        AC = lambda mask: AutoCorrelationLayer(
            AutoCorrelation(mask, factor, attention_dropout=dropout), d_model, n_heads)
        self.encoder = Encoder([EncoderLayer(AC(False), d_model, d_ff, moving_avg=moving_avg,
                                             dropout=dropout, activation="gelu")
                                for _ in range(e_layers)], norm_layer=my_Layernorm(d_model))
        self.decoder = Decoder([DecoderLayer(AC(True), AC(False), d_model, 1, d_ff,
                                             moving_avg=moving_avg, dropout=dropout,
                                             activation="gelu")
                                for _ in range(d_layers)], norm_layer=my_Layernorm(d_model),
                               projection=nn.Linear(d_model, 1, bias=True))

    def forward(self, x, m_enc, m_dec):
        mean = x.mean(1, keepdim=True).repeat(1, PRED_LEN, 1)
        zeros = torch.zeros(x.shape[0], PRED_LEN, 1)
        seasonal, trend = self.decomp(x)
        trend_init = torch.cat([trend[:, -LABEL_LEN:], mean], 1)
        seas_init = torch.cat([seasonal[:, -LABEL_LEN:], zeros], 1)
        enc, _ = self.encoder(self.enc_emb(x, m_enc))
        s, t = self.decoder(self.dec_emb(seas_init, m_dec), enc, trend=trend_init)
        out = (s + t)[:, -PRED_LEN:]
        if self.exo is not None:
            out = out + self.exo(m_dec[:, -PRED_LEN:])
        return out


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- metrics
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y - p)
    smape = np.mean(np.where((np.abs(y) + np.abs(p)) > 0, 200 * d / (np.abs(y) + np.abs(p) + 1e-12), 0.0))
    return dict(MAE=float(d.mean()), RMSE=float(np.sqrt(np.mean(d ** 2))), sMAPE=float(smape))


def block_metrics(preds, y, origins):
    """Mean over 168-step blocks of the per-block metric (each block = one leaderboard-like score)."""
    rows = [metrics(y[o:o + PRED_LEN], p) for o, p in zip(origins, preds)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, rows


EVAL_ORIGINS = [EVAL_START + k * PRED_LEN for k in range(N_EVAL_BLOCKS)]


# ----------------------------------------------------------------------------- baselines
def baselines():
    y, _, _ = load()
    train_mean = y[:ES_START].mean()
    fc = {
        "Persistence (last value)": lambda o: np.full(PRED_LEN, y[o - 1]),
        "Mean of last 168": lambda o: np.full(PRED_LEN, y[o - 168:o].mean()),
        "Seasonal naive (24)": lambda o: np.tile(y[o - 24:o], 7),
        "Training mean": lambda o: np.full(PRED_LEN, train_mean),
    }
    out = {}
    for name, f in fc.items():
        agg, rows = block_metrics([f(o) for o in EVAL_ORIGINS], y, EVAL_ORIGINS)
        out[name] = dict(agg, blocks=rows)
        print(f"{name:28s} " + "  ".join(f"{k}={v:8.2f}" for k, v in agg.items()))
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "baselines.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- training
def predict(model, origins, ys, menc, mdec, sc, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(origins), bs):
            o = origins[i:i + bs]
            out.append(model(*make_batch(o, ys, menc, mdec)).squeeze(-1).numpy())
    return np.clip(sc.inv(np.concatenate(out)), 0, None)


def train(variant, seed, train_end=ES_START, es=True, epochs=None, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    y, phase, cov = load()
    sc = Scaler(y)
    ys = sc.f(y).astype(np.float32)
    menc, mdec = marks_for(variant, phase, cov)
    model = Autoformer(menc.shape[1], mdec.shape[1], **MODEL_CFG)
    # the exogenous head is a tiny linear model: give it a larger step size so it can
    # converge within the few epochs before the Autoformer body starts to overfit
    exo_ids = {id(p) for p in model.exo.parameters()} if model.exo is not None else set()
    groups = [{"params": [p for p in model.parameters() if id(p) not in exo_ids], "lr": TRAIN_CFG["lr"]}]
    if exo_ids:
        groups.append({"params": list(model.exo.parameters()), "lr": TRAIN_CFG["exo_lr"]})
    opt = torch.optim.Adam(groups)
    lossf = nn.MSELoss()
    tr = np.arange(SEQ_LEN, train_end - PRED_LEN + 1, TRAIN_CFG["train_stride"])
    es_orig = np.arange(ES_START, EVAL_START - PRED_LEN + 1, 24)
    best, best_ep, bad, state, hist = np.inf, 0, 0, None, []
    max_ep = epochs or TRAIN_CFG["max_epochs"]
    rng = np.random.default_rng(seed)
    for ep in range(1, max_ep + 1):
        model.train(); t0 = time.time(); perm = rng.permutation(tr); tl = []
        for i in range(0, len(perm), TRAIN_CFG["batch"]):
            o = perm[i:i + TRAIN_CFG["batch"]]
            loss = lossf(model(*make_batch(o, ys, menc, mdec)), target(o, ys))
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        for g in opt.param_groups:          # Autoformer's "type1" schedule: halve each epoch
            g["lr"] *= 0.5
        rec = dict(epoch=ep, train_loss=float(np.mean(tl)), sec=round(time.time() - t0, 1))
        if es:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([lossf(model(*make_batch(es_orig[i:i + 256], ys, menc, mdec)),
                                          target(es_orig[i:i + 256], ys)).item()
                                    for i in range(0, len(es_orig), 256)]))
            rec["es_loss"] = vl
            if vl < best - 1e-4:
                best, best_ep, bad = vl, ep, 0
                state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        hist.append(rec); log(json.dumps(rec))
        if es and bad >= TRAIN_CFG["patience"]:
            break
    if es:
        model.load_state_dict(state)
    return model, dict(history=hist, best_epoch=best_ep if es else max_ep,
                       epochs_run=len(hist)), (ys, menc, mdec, sc, y)


def run(variant, seed):
    model, info, (ys, menc, mdec, sc, y) = train(variant, seed)
    preds = predict(model, EVAL_ORIGINS, ys, menc, mdec, sc)
    agg, rows = block_metrics(preds, y, EVAL_ORIGINS)
    out = dict(variant=variant, seed=seed, params=n_params(model), **info, **agg, blocks=rows,
               preds=preds.round(2).tolist(), model_cfg=MODEL_CFG, train_cfg=TRAIN_CFG)
    os.makedirs(os.path.join(RES, "runs"), exist_ok=True)
    json.dump(out, open(os.path.join(RES, "runs", f"{variant}_s{seed}.json"), "w"))
    torch.save(model.state_dict(), os.path.join(RES, "runs", f"{variant}_s{seed}.pt"))
    print(variant, seed, {k: round(v, 2) for k, v in agg.items()}, "P =", n_params(model),
          "epochs_run =", info["epochs_run"], "best =", info["best_epoch"])


def final(variant, seed):
    """Forecast t = 43657..43824 with the early-stopped model of (variant, seed).
    No refit: E = epochs actually run in that training (incl. patience epochs)."""
    rpath = os.path.join(RES, "runs", f"{variant}_s{seed}")
    info = json.load(open(rpath + ".json"))
    y, phase, cov = load()
    sc = Scaler(y)
    ys = np.concatenate([sc.f(y), np.zeros(PRED_LEN)]).astype(np.float32)
    menc, mdec = marks_for(variant, phase, cov)
    model = Autoformer(menc.shape[1], mdec.shape[1], **MODEL_CFG)
    model.load_state_dict(torch.load(rpath + ".pt"))   # errors if the run used a different architecture
    pred = predict(model, [N_HIST], ys, menc, mdec, sc)[0]
    assert len(pred) == 168 and np.all(np.isfinite(pred))
    s = ", ".join(f"{v:.2f}" for v in pred)
    open(os.path.join(RES, "submission.txt"), "w").write(s + "\n")
    pd.DataFrame({"time_idx": np.arange(N_HIST + 1, N_HIST + PRED_LEN + 1), "value": pred}) \
        .to_csv(os.path.join(RES, "submission_forecast.csv"), index=False)
    meta = dict(variant=variant, seed=seed, P=n_params(model), E=info["epochs_run"])
    json.dump(meta, open(os.path.join(RES, "submission_meta.json"), "w"), indent=1)
    print(meta); print(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baselines", "run", "final"])
    ap.add_argument("--variant", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    {"baselines": baselines, "run": lambda: run(a.variant, a.seed),
     "final": lambda: final(a.variant, a.seed)}[a.cmd]()
