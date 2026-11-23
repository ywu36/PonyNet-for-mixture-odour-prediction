# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   M = molecules with SMILES (235)        L = tokens per SMILES (padded)     E = MIST embedding size (512)
#   B = pairs in a batch                   K = molecules per mixture (padded) U = distinct molecules in a batch
#   P = pairs in the set being processed
# ---------------------------------------------------------------------------------------------
"""B2-ft: MIST fine-tuned on mixture distances, the way MIST's own mixture model does it.

MIST's olfactory mixture model (external/mist_repo, electrolyte_fm/models/mixture_similarity.py,
MixtureSimilarityModel; config submit/chemixhub.jsonnet + chemixhub_tasks.libsonnet,
'olfactory_similarity'): a MIST encoder, not frozen; per molecule the first-token (CLS) embedding;
per mixture a weighted sum with weights from a small network and a softmax over its molecules;
distance = sigmoid(w * (1 - cos) + b); loss = 1 - R^2; AdamW lr 1.6e-4, weight decay 0.01; learning
rate warmed up, then cosine decay to 0.1 x. Three small classes of that file (R2Loss, ScaledCosineDistance,
WeightedSumPooling) are read from your MIST clone at run time (mist_parts below), not copied here; the file
imports lightning, which the `mist` env lacks, so the class definitions are executed on their own.

Encoders ("variant"): ``base`` = dream2_data/mist_28M_base (pretrained on SMILES only; MIST's
config starts from this stage), ``odour`` = the encoder of dream2_data/mist_odour (the base
fine-tuned on single-molecule odour labels; B1 uses this model frozen). Input: RDKit canonical
Kekulé SMILES (mixdist.mist.kekulize, PLAN.md rule 9).

Version 1 (SETTINGS; train, fit_predict). MIST's recipe; where it differs from MIST's config, fixed
before any score was seen:
  * batch 16 pairs, not 2: the loss is 1 - R^2 over a batch, which is unstable for 2 pairs;
  * a fixed budget, not 100,000 steps: the number of epochs (at most SETTINGS['max_epochs']) is the
    one with the lowest loss on an inner validation split of the training pairs (whole mixtures held
    out; early stopping), then the model is trained again from the start on all training pairs for
    that many epochs;
  * warmup over the same number of pairs as MIST's (2,000 steps x 2 pairs = 4,000 pairs).
No calibration line: the model outputs a distance in 0-1, trained on the 0-1 labels.
It did not reach the label level: the head starts at sigmoid(6 * (1 - cos) - 6), about 0.003, and its
two numbers move about lr per step, so the encoder had to push mixtures apart to raise the outputs.

Version 2 (SETTINGS_V2; train_v2, fit_predict_v2). Two stages, as POMMix (CheMix on frozen POM
embeddings, then a joint fine-tune with its own learning rate for the graph network):
  A "head only": encoder frozen (its CLS embeddings of the 235 molecules computed once); the weighted
    sum and the head are trained on the whole fit split per step;
  B "joint": from stage A's weighted sum and head, the encoder is unfrozen with a learning rate 10 x
    below MIST's; the weighted sum and head keep theirs.
The normalisation B1 lacked and B1a adds (finding 1: quantile per word over the 235 molecules, each
mixture scaled to length 1, Euclidean distance) is built in: before the weighted sum each embedding
dimension is z-scored with a fixed mean and sd computed once from the starting encoder's CLS
embeddings of the 235 molecules (no labels; the quantile transform is not differentiable, so the
z-score stands in for it, in both stages); after it each mixture is scaled to length 1 and the head
takes their Euclidean distance: sigmoid(w * |a/|a| - b/|b|| + b0).
Head start: weight 0.5, bias so that the mean fit pair starts at the mean label (sigmoid(0.5 * mean_d
+ b0) = mean label, mean_d = mean distance of the fit pairs before training). POMMix's head starts at
weight 0.5, bias 0.5. Loss: mean squared error on the label scale (POMMix's Huber loss with delta 1
is half the squared error for errors below 1, which all are on 0-1 labels). Constant learning rates.
Each stage stops early on the inner validation pairs (whole mixtures held out) and keeps its best
epoch; there is no retraining on all pairs, so that each stage's model can also be reported with a
line fitted on its predictions for the inner validation pairs, which it was not trained on. The line
is never fitted on test pairs.

B1a readouts (readouts): after each stage, and before training ("frozen"), the encoder's molecule
outputs (CLS embedding; for the odour model also its 135 odour-word probabilities from its own word
layer on the current encoder) go through B1a's recipe: quantile per dimension over the 235 molecules,
mean per mixture (or the model's learned weights), length 1, Euclidean, one line on the fold's
training pairs. This asks whether fine-tuning improves the representation once B1a's normalisation
is applied.

Used by evaluation/B2ft/eval_mist_mixture.py. Needs the `mist` conda environment and a GPU.
"""
from __future__ import annotations

import ast
import copy
import functools
import math
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from . import cv, data, evaluate, mist, mixture

SETTINGS = {"lr": 1.6e-4, "weight_decay": 0.01, "batch_pairs": 16, "max_epochs": 40, "patience": 8,
            "warmup_pairs": 4000, "rel_decay": 0.1, "pool_hidden": 128, "dropout": 0.2, "inner_k": 5}
SETTINGS_V2 = {"normalise": "z-score per CLS dimension over the 235 molecules (starting encoder), fixed",
               "distance": "mixtures scaled to length 1, Euclidean, sigmoid(w * d + b)", "loss": "mean squared error", "weight_decay": 0.01, "pool_hidden": 128, "dropout": 0.2, "inner_k": 5,
               "head_weight": 0.5, "head_bias": "mean fit pair starts at the mean label", "schedule": "constant",
               "A_lr": 1e-3, "A_batch": "all fit pairs", "A_max_epochs": 5000, "A_patience": 1000,
               "B_encoder_lr": 1.6e-5, "B_lr": 1e-3, "B_batch_pairs": 64, "B_max_epochs": 40, "B_patience": 8,
               "line": "fitted on the inner validation pairs", "refit_on_all_pairs": False}
VARIANTS = {"base": "mist_28M_base", "odour": "mist_odour"}


@functools.lru_cache(maxsize=None)
def mist_parts() -> SimpleNamespace:
    """R2Loss, ScaledCosineDistance and WeightedSumPooling from MIST's own code (Apache-2.0, not redistributed here).

    Reads electrolyte_fm/models/mixture_similarity.py from your MIST clone (data.locate("mist_repo")) and executes
    only those class definitions and their base class, so lightning is not needed.
    """
    path = data.locate("mist_repo", needs="electrolyte_fm/models/mixture_similarity.py") / \
        "electrolyte_fm/models/mixture_similarity.py"
    source = path.read_text()
    wanted = {"EmbeddingDistance", "R2Loss", "ScaledCosineDistance", "WeightedSumPooling"}
    body = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in wanted]
    missing = wanted - {n.name for n in body}
    if missing:
        raise ImportError(f"{path} lacks {sorted(missing)}")
    space = {"math": math, "torch": torch, "nn": nn, "F": nn.functional, "__name__": "mist_mixture_similarity"}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), space)
    return SimpleNamespace(**{k: space[k] for k in ("R2Loss", "ScaledCosineDistance", "WeightedSumPooling")})


class ScaledUnitEuclidean(nn.Module):
    """Version 2 head: as ScaledCosineDistance, on the Euclidean distance of the length-1 mixtures."""

    def __init__(self):
        super().__init__()
        self.scaler = nn.Linear(1, 1, bias=True)

    @staticmethod
    def raw(x1, x2):
        return (nn.functional.normalize(x1, dim=1) - nn.functional.normalize(x2, dim=1)).norm(dim=1)

    def forward(self, x1, x2):
        return torch.sigmoid(self.scaler(self.raw(x1, x2).unsqueeze(-1)))


def relative_cosine_warmup(step: int, total: int, warmup: int, rel_decay: float) -> float:
    """Learning-rate factor of mist_repo's RelativeCosineWarmup (electrolyte_fm/utils/lr_schedule.py)."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (min(step, total) - warmup) / max(1, total - warmup)
    amp = 0.5 * (1 - rel_decay)
    return max(rel_decay, amp * math.cos(math.pi * progress) + amp + rel_decay)
# ------------------------------------------------------------------------------------------------


def load_encoder(variant: str, full: bool = False):
    """-> (encoder, tokenizer). ``base``: the SMILES-pretrained encoder; ``odour``: the encoder inside
    the odour model (its CLS embedding is the one mist_odour's .embed returns); with ``full``, the
    whole odour model (its .encoder is the encoder)."""
    from smirk import SmirkTokenizerFast
    from transformers import AutoModel, AutoTokenizer
    try:                                                     # as dream2_data/mist_odour/modeling_mist_finetuned.py
        AutoTokenizer.register("SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast)
    except ValueError:                                       # already registered in this process
        pass
    path = data.locate(VARIANTS[variant])
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    if variant == "base":
        return AutoModel.from_pretrained(str(path), add_pooling_layer=False), tokenizer
    odour = AutoModel.from_pretrained(str(path), trust_remote_code=True)
    return (odour if full else odour.encoder), tokenizer


class MixtureDistance(nn.Module):
    """Encoder -> CLS per molecule -> weighted sum per mixture -> scaled cosine distance."""

    def __init__(self, encoder, pool_hidden: int = SETTINGS["pool_hidden"], dropout: float = SETTINGS["dropout"],
                 normalised: bool = False):
        """``normalised`` (version 2): fixed z-score per CLS dimension (set_normalisation) and the
        length-1 Euclidean head; otherwise (version 1) MIST's model as it is."""
        super().__init__()
        self.encoder = encoder
        self.pool = mist_parts().WeightedSumPooling(encoder.config.hidden_size, pool_hidden, dropout)
        self.normalised = normalised
        self.distance = ScaledUnitEuclidean() if normalised else mist_parts().ScaledCosineDistance()
        if normalised:
            self.register_buffer("center_E", torch.zeros(encoder.config.hidden_size))
            self.register_buffer("scale_E", torch.ones(encoder.config.hidden_size))

    def set_normalisation(self, cls_MxE: torch.Tensor):
        self.center_E.copy_(cls_MxE.mean(0))
        self.scale_E.copy_(cls_MxE.std(0) + 1e-6)

    def cls(self, ids_UxL, mask_UxL):
        return self.encoder(ids_UxL, attention_mask=mask_UxL).last_hidden_state[:, 0]

    def pooling_logits(self, cls_UxE):
        """The weighted sum's score per molecule (its weights are a softmax of these within a mixture)."""
        if self.normalised:
            cls_UxE = (cls_UxE - self.center_E) / self.scale_E
        return self.pool.weight_predictor(cls_UxE).squeeze(-1)

    def from_cls(self, cls_UxE, a_BxK, b_BxK, raw_distance: bool = False):
        """a_BxK, b_BxK: rows of cls_UxE, -1 = padding. -> distance_B (``raw_distance``: before the head)"""
        if self.normalised:
            cls_UxE = (cls_UxE - self.center_E) / self.scale_E

        def mixture(idx_BxK):
            pad_BxK = idx_BxK < 0
            emb_BxKxE = cls_UxE[idx_BxK.clamp(min=0)] * (~pad_BxK).unsqueeze(-1)
            return self.pool(emb_BxKxE, pad_BxK)

        a_BxE, b_BxE = mixture(a_BxK), mixture(b_BxK)
        if raw_distance:
            return self.distance.raw(a_BxE, b_BxE) if self.normalised else 1 - self.distance.cosine(a_BxE, b_BxE)
        return self.distance(a_BxE, b_BxE).squeeze(-1)

    def forward(self, ids_UxL, mask_UxL, a_BxK, b_BxK):
        return self.from_cls(self.cls(ids_UxL, mask_UxL), a_BxK, b_BxK)

    def head(self) -> tuple[float, float]:
        return float(self.distance.scaler.weight.detach()), float(self.distance.scaler.bias.detach())


@dataclass
class Tokens:
    cids_M: list
    ids_MxL: torch.Tensor
    mask_MxL: torch.Tensor

    @classmethod
    def build(cls, tokenizer) -> "Tokens":
        smiles = data.load_smiles()
        cids_M = sorted(smiles)
        enc = tokenizer([mist.kekulize(smiles[c]) for c in cids_M], padding=True, return_tensors="pt")
        return cls(cids_M, enc["input_ids"], enc["attention_mask"])

    def batch(self, pairs, device):
        """Encode each distinct molecule of the batch once: -> (ids_UxL, mask_UxL, a_BxK, b_BxK)."""
        row = {c: i for i, c in enumerate(self.cids_M)}
        used = sorted({row[c] for p in pairs for c in p.a | p.b})
        pos = {m: u for u, m in enumerate(used)}
        k = max(max(len(p.a), len(p.b)) for p in pairs)

        def index(mix):
            idx = [pos[row[c]] for c in sorted(mix)]
            return idx + [-1] * (k - len(idx))

        a_BxK = torch.tensor([index(p.a) for p in pairs], device=device)
        b_BxK = torch.tensor([index(p.b) for p in pairs], device=device)
        used_t = torch.tensor(used)
        ids, mask = self.ids_MxL[used_t], self.mask_MxL[used_t]
        width = int(mask.sum(1).max())                       # drop padding no molecule of the batch needs
        return ids[:, :width].to(device), mask[:, :width].to(device), a_BxK, b_BxK

    def rows(self, pairs, device):
        """Rows of all M molecules (for cached embeddings): -> (a_PxK, b_PxK), -1 = padding."""
        row = {c: i for i, c in enumerate(self.cids_M)}
        k = max(max(len(p.a), len(p.b)) for p in pairs)

        def index(mix):
            idx = [row[c] for c in sorted(mix)]
            return idx + [-1] * (k - len(idx))

        return (torch.tensor([index(p.a) for p in pairs], device=device),
                torch.tensor([index(p.b) for p in pairs], device=device))


def predict(model, tokens: Tokens, pairs, device, batch_pairs: int = 64) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_pairs):
            out.append(model(*tokens.batch(pairs[i:i + batch_pairs], device)).float().cpu().numpy())
    return np.concatenate(out)


def inner_split(pairs, seed: int) -> tuple[list, list]:
    """Inner training and validation pairs: one fold of mixdist.cv, whole mixtures held out. Pairs
    with one mixture on each side are left out of both."""
    fold = cv.mixture_holdout_folds(pairs, k=SETTINGS["inner_k"], repeats=1, seed=1000 + seed)[0]
    return [pairs[i] for i in fold.train], [pairs[i] for i in fold.test]


# ---- version 1 -----------------------------------------------------------------------------------
def _fit(variant: str, fit, device: str, seed: int, epochs: int, val=None, log=None) -> tuple:
    """Train a fresh model on ``fit`` for up to ``epochs`` epochs. With ``val``: keep the epoch with the
    lowest validation loss and stop after SETTINGS['patience'] epochs without a better one.
    -> (model, tokens, validation loss per epoch)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    encoder, tokenizer = load_encoder(variant)
    tokens = Tokens.build(tokenizer)
    model = MixtureDistance(encoder).to(device)
    s = SETTINGS
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    steps_per_epoch = math.ceil(len(fit) / s["batch_pairs"])
    total, warmup = s["max_epochs"] * steps_per_epoch, math.ceil(s["warmup_pairs"] / s["batch_pairs"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: relative_cosine_warmup(t, total, warmup, s["rel_decay"]))
    loss_fn, rng = mist_parts().R2Loss(), random.Random(seed)
    y_val = torch.tensor(evaluate.labels(val), dtype=torch.float32) if val else None
    best, best_loss, best_epoch, history = None, float("inf"), -1, []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(fit)))
        rng.shuffle(order)
        for i in range(0, len(order), s["batch_pairs"]):
            batch = [fit[j] for j in order[i:i + s["batch_pairs"]]]
            if len(batch) < 2:
                continue
            y = torch.tensor([p.y for p in batch], dtype=torch.float32, device=device)
            loss = loss_fn(model(*tokens.batch(batch, device)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if val:
            val_loss = float(loss_fn(torch.tensor(predict(model, tokens, val, device)), y_val))
            history.append(val_loss)
            if val_loss < best_loss:
                best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), val_loss, epoch
            if log:
                log(f"{variant} seed {seed} epoch {epoch + 1}: inner validation 1 - R^2 {val_loss:.3f}")
            if epoch - best_epoch >= s["patience"]:
                break
    if val:
        model.load_state_dict(best)
    return model, tokens, history


def train(variant: str, pairs, device: str, seed: int = 0, log=None) -> tuple[MixtureDistance, Tokens, dict]:
    """Version 1. Fine-tune on ``pairs``: choose the number of epochs by early stopping on an inner
    validation split (whole mixtures held out), then train again from the start on all ``pairs`` for
    that many epochs. -> (model, tokens, info)."""
    fit, val = inner_split(pairs, seed)
    model, _, history = _fit(variant, fit, device, seed, SETTINGS["max_epochs"], val, log)
    epochs = int(np.argmin(history)) + 1
    del model
    torch.cuda.empty_cache()
    model, tokens, _ = _fit(variant, list(pairs), device, seed, epochs)
    return model, tokens, {"epochs": epochs, "epochs_run": len(history), "inner_val_loss": history,
                           "n_fit": len(fit), "n_val": len(val), "n_train": len(pairs), "head": model.head()}


def fit_predict(variant: str, pairs, device: str, seed: int = 0, infos: list | None = None):
    """Version 1: fit_predict(train_idx, test_idx) for mixdist.evaluate.cross_validate."""
    def run(train_idx, test_idx):
        model, tokens, info = train(variant, [pairs[i] for i in train_idx], device, seed)
        if infos is not None:
            infos.append(info)
        pred = predict(model, tokens, [pairs[i] for i in test_idx], device)
        del model
        torch.cuda.empty_cache()
        return pred
    return run


# ---- version 2 -----------------------------------------------------------------------------------
def molecule_cls(model: MixtureDistance, tokens: Tokens, device, batch: int = 64) -> torch.Tensor:
    """CLS embedding of every molecule under the current (frozen) encoder -> cls_MxE on ``device``."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tokens.cids_M), batch):
            ids, mask = tokens.ids_MxL[i:i + batch], tokens.mask_MxL[i:i + batch]
            width = int(mask.sum(1).max())
            out.append(model.cls(ids[:, :width].to(device), mask[:, :width].to(device)))
    return torch.cat(out)


def odour_word_probs(odour_model, tokens: Tokens, device, batch: int = 64) -> np.ndarray:
    """The odour model's 135 odour-word probabilities per molecule, on its current encoder -> probs_MxW."""
    odour_model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tokens.cids_M), batch):
            ids, mask = tokens.ids_MxL[i:i + batch], tokens.mask_MxL[i:i + batch]
            width = int(mask.sum(1).max())
            out.append(torch.sigmoid(odour_model(ids[:, :width].to(device), mask[:, :width].to(device))).float().cpu())
    return torch.cat(out).numpy()


def readouts(model, tokens: Tokens, device, pairs, odour_model=None) -> dict:
    """B1a's recipe on the current encoder's molecule outputs: quantile per dimension over the M molecules,
    mean per mixture or the model's learned weights, length 1, Euclidean -> {name: dist_P} for ``pairs``."""
    cls_MxE = molecule_cls(model, tokens, device)
    with torch.no_grad():
        logit_M = model.pooling_logits(cls_MxE).float().cpu().numpy()
    feats = {"CLS": cls_MxE.float().cpu().numpy()}
    if odour_model is not None:
        feats["words"] = odour_word_probs(odour_model, tokens, device)
    row = {c: i for i, c in enumerate(tokens.cids_M)}

    def weighted(q_MxD, mix):
        idx = [row[c] for c in sorted(mix)]
        w = np.exp(logit_M[idx] - logit_M[idx].max())
        return (w / w.sum()) @ q_MxD[idx]

    out = {}
    for name, feat in feats.items():
        q = mixture.quantile_per_word(feat)
        out[f"{name}, mean"] = mixture.unit_euclidean_distance(*mixture.mean_profiles(pairs, tokens.cids_M, q))
        out[f"{name}, learned weights"] = mixture.unit_euclidean_distance(
            np.stack([weighted(q, p.a) for p in pairs]), np.stack([weighted(q, p.b) for p in pairs]))
    return out


LOG_COLUMNS = ["time", "variant", "repeat", "fold", "stage", "epoch", "train_loss", "val_loss", "val_rmse", "val_r",
               "head_weight", "head_bias", "lr", "best"]
SUMMARY_COLUMNS = ["time", "variant", "repeat", "fold", "stage", "epochs_run", "best_epoch", "val_first", "val_best",
                   "val_last", "train_at_best", "val_pred_mean", "val_pred_sd", "val_label_mean", "val_label_sd",
                   "test_pred_mean", "test_pred_sd", "test_label_mean", "test_label_sd", "head_weight", "head_bias",
                   "minutes"]


class TsvLog:
    """Tab-separated lines appended to ``path`` and flushed one by one, so a run can be watched (tail -f)."""

    def __init__(self, path, columns, **fixed):
        self.columns, self.fixed = columns, fixed
        new = not path.exists()
        self.f = open(path, "a", buffering=1)
        if new:
            self.f.write("\t".join(columns) + "\n")

    def write(self, **values):
        values = {"time": time.strftime("%H:%M:%S"), **self.fixed, **values}
        fmt = lambda v: f"{v:.5g}" if isinstance(v, float) else str(v)     # noqa: E731
        self.f.write("\t".join(fmt(values.get(c, "")) for c in self.columns) + "\n")
        self.f.flush()


def _val_metrics(pred, y) -> dict:
    """Inner-validation loss (mean squared error), RMSE and r of ``pred`` against ``y``."""
    mse = float(np.mean((pred - y) ** 2))
    r = float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 else float("nan")
    return {"val_loss": mse, "val_rmse": math.sqrt(mse), "val_r": r}


def _record(history, log, stage, epoch, train_loss, val, model, lr, best):
    for k, v in {"train": train_loss, **{k.replace("val_loss", "val"): v for k, v in val.items()}}.items():
        history.setdefault(k, []).append(v)
    if log is not None:
        w, b = model.head()
        log.write(stage=stage, epoch=epoch + 1, train_loss=train_loss, **val, head_weight=w, head_bias=b, lr=lr,
                  best=int(best))


def _stage_a(model, tokens, cls_MxE, fit, val, device, log=None) -> dict:
    """Head only: train the weighted sum and the head on frozen embeddings, whole fit split per step;
    keep the epoch with the lowest validation loss. -> per-epoch history (train, val, val_rmse, val_r)."""
    s = SETTINGS_V2
    a_fit, b_fit = tokens.rows(fit, device)
    a_val, b_val = tokens.rows(val, device)
    y_fit = torch.tensor(evaluate.labels(fit), dtype=torch.float32, device=device)
    y_val = evaluate.labels(val)
    trained = [*model.pool.parameters(), *model.distance.parameters()]
    opt = torch.optim.AdamW(trained, lr=s["A_lr"], weight_decay=s["weight_decay"])
    best, best_loss, best_epoch, history = None, float("inf"), -1, {}
    for epoch in range(s["A_max_epochs"]):
        model.train()
        loss = nn.functional.mse_loss(model.from_cls(cls_MxE, a_fit, b_fit), y_fit)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            m = _val_metrics(model.from_cls(cls_MxE, a_val, b_val).float().cpu().numpy(), y_val)
        is_best = m["val_loss"] < best_loss
        if is_best:
            best_loss, best_epoch = m["val_loss"], epoch
            best = (copy.deepcopy(model.pool.state_dict()), copy.deepcopy(model.distance.state_dict()))
        _record(history, log, "A", epoch, float(loss), m, model, f"{s['A_lr']:g}", is_best)
        if epoch - best_epoch >= s["A_patience"]:
            break
    model.pool.load_state_dict(best[0])
    model.distance.load_state_dict(best[1])
    return history


def _stage_b(model, tokens, fit, val, device, seed, log=None) -> dict:
    """Joint: unfreeze the encoder at its own lower learning rate; batches of pairs; keep the epoch with
    the lowest validation loss. -> per-epoch history (train = mean over the epoch's batches, val, ...)."""
    s = SETTINGS_V2
    rest = [*model.pool.parameters(), *model.distance.parameters()]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": s["B_encoder_lr"]},
                             {"params": rest, "lr": s["B_lr"]}], weight_decay=s["weight_decay"])
    lr = f"encoder {s['B_encoder_lr']:g}, rest {s['B_lr']:g}"
    rng = random.Random(seed)
    y_val = evaluate.labels(val)
    best, best_loss, best_epoch, history = None, float("inf"), -1, {}
    for epoch in range(s["B_max_epochs"]):
        model.train()
        order = list(range(len(fit)))
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), s["B_batch_pairs"]):
            batch = [fit[j] for j in order[i:i + s["B_batch_pairs"]]]
            y = torch.tensor([p.y for p in batch], dtype=torch.float32, device=device)
            loss = nn.functional.mse_loss(model(*tokens.batch(batch, device)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        m = _val_metrics(predict(model, tokens, val, device), y_val)
        is_best = m["val_loss"] < best_loss
        if is_best:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), m["val_loss"], epoch
        _record(history, log, "B", epoch, float(np.mean(losses)), m, model, lr, is_best)
        if epoch - best_epoch >= s["B_patience"]:
            break
    model.load_state_dict(best)
    return history


def train_v2(variant: str, pairs, device: str, seed: int = 0, stages: str = "AB", predict_pairs=(),
             readout_pairs=None, log: TsvLog | None = None) -> dict:
    """Version 2 on ``pairs``: stage A, then (if in ``stages``) stage B from stage A's model. After each
    stage: its raw predictions for ``predict_pairs`` (their labels are not read) and the line fitted on
    its inner validation predictions; with ``readout_pairs``, also the B1a readouts' distances for them
    (also before training, as "frozen"). ``log``: one line per epoch. -> {stage: info}."""
    s = SETTINGS_V2
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    fit, val = inner_split(pairs, seed)
    loaded, tokenizer = load_encoder(variant, full=True)
    odour_model = loaded.to(device) if variant == "odour" else None
    tokens = Tokens.build(tokenizer)
    model = MixtureDistance(loaded.encoder if odour_model is not None else loaded, s["pool_hidden"], s["dropout"],
                            normalised=True).to(device)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    cls_MxE = molecule_cls(model, tokens, device)
    model.set_normalisation(cls_MxE)
    out = {}
    if readout_pairs is not None:
        out["frozen"] = {"readout": {k: d.tolist() for k, d in
                                     readouts(model, tokens, device, readout_pairs, odour_model).items()}}
    a_fit, b_fit = tokens.rows(fit, device)
    with torch.no_grad():
        mean_d = float(model.from_cls(cls_MxE, a_fit, b_fit, raw_distance=True).mean())
        mean_y = float(evaluate.labels(fit).mean())
        model.distance.scaler.weight.fill_(s["head_weight"])
        model.distance.scaler.bias.fill_(math.log(mean_y / (1 - mean_y)) - s["head_weight"] * mean_d)
    y_val = evaluate.labels(val)
    for stage in stages:
        t = time.time()
        if stage == "A":
            history = _stage_a(model, tokens, cls_MxE, fit, val, device, log)
        else:
            for p in model.encoder.parameters():
                p.requires_grad_(True)
            history = _stage_b(model, tokens, fit, val, device, seed, log)
        val_pred = predict(model, tokens, val, device)
        best = int(np.argmin(history["val"]))
        info = {"epochs": best + 1, "epochs_run": len(history["val"]), "history": history,
                "inner_val_loss": history["val"], "train_loss_at_best": history["train"][best],
                "head": model.head(), "line": evaluate.fit_line(val_pred, y_val),
                "val_pred_mean_sd": [float(val_pred.mean()), float(val_pred.std())],
                "val_label_mean_sd": [float(y_val.mean()), float(y_val.std())],
                "minutes": (time.time() - t) / 60, "n_fit": len(fit), "n_val": len(val), "n_train": len(pairs),
                "pred": predict(model, tokens, list(predict_pairs), device).tolist() if predict_pairs else []}
        if stage == "A":
            info.update(start_mean_d=mean_d, start_output=mean_y)
        if readout_pairs is not None:
            info["readout"] = {k: d.tolist() for k, d in readouts(model, tokens, device, readout_pairs, odour_model).items()}
        out[stage] = info
    del model, cls_MxE, odour_model
    torch.cuda.empty_cache()
    return out


def fit_predict_v2(variant: str, pairs, device: str, seed: int = 0, infos: list | None = None, stages: str = "AB",
                   k: int = 5, log_dir=None):
    """Version 2: fit_predict(train_idx, test_idx) for mixdist.evaluate.cross_validate (``k`` folds per
    repeat). Returns the last stage's raw predictions; every stage's predictions, line, head and history,
    and the B1a readouts' predictions for the test pairs (one line on the fold's training pairs) are
    appended to ``infos``. ``log_dir``: train_log.tsv (one line per epoch) and fit_summary.tsv (one line
    per fit and stage), written as training goes."""
    y_P = evaluate.labels(pairs)
    calls = []

    def run(train_idx, test_idx):
        repeat, fold = len(calls) // k + 1, len(calls) % k + 1
        calls.append(1)
        tag = {"variant": variant, "repeat": repeat, "fold": fold}
        log = TsvLog(log_dir / "train_log.tsv", LOG_COLUMNS, **tag) if log_dir else None
        info = train_v2(variant, [pairs[i] for i in train_idx], device, seed, stages,
                        predict_pairs=[pairs[i] for i in test_idx], readout_pairs=pairs, log=log)
        y_test = y_P[test_idx]
        for part, d in info.items():                          # distances for all pairs -> lined test predictions
            d["readout"] = {name: evaluate.linear_calibration(np.array(d_P), y_P, np.zeros(len(y_P)))(
                train_idx, test_idx).tolist() for name, d_P in d["readout"].items()}
            if part == "frozen":
                continue
            h, pred = d["history"], np.array(d["pred"])
            row = dict(stage=part, epochs_run=d["epochs_run"], best_epoch=d["epochs"], val_first=h["val"][0],
                       val_best=min(h["val"]), val_last=h["val"][-1], train_at_best=d["train_loss_at_best"],
                       val_pred_mean=d["val_pred_mean_sd"][0], val_pred_sd=d["val_pred_mean_sd"][1],
                       val_label_mean=d["val_label_mean_sd"][0], val_label_sd=d["val_label_mean_sd"][1],
                       test_pred_mean=float(pred.mean()), test_pred_sd=float(pred.std()),
                       test_label_mean=float(y_test.mean()), test_label_sd=float(y_test.std()),
                       head_weight=d["head"][0], head_bias=d["head"][1], minutes=d["minutes"])
            d["summary"] = row
            if log_dir:
                TsvLog(log_dir / "fit_summary.tsv", SUMMARY_COLUMNS, **tag).write(**row)
            print(" ".join(f"{c}={row[c]:.4g}" if isinstance(row[c], float) else f"{c}={row[c]}"
                           for c in row), f"variant={variant} repeat={repeat} fold={fold}", flush=True)
        if infos is not None:
            infos.append(info)
        return np.array(info[stages[-1]]["pred"])
    return run
