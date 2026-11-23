# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes
#   T = all molecules in the embedding table (DREAM + extra)     E = embedding size
#   R = receptor units      N = mixtures in a batch      K = molecules per mixture (padded)
#   W = GS-LF odour words   G = blend groups of the pair data    B = items in a batch
# ---------------------------------------------------------------------------------------------
"""O1: receptor-competition model (design/o1_design.html; PLAN.md "Bio-inspired hidden layer").

Per molecule m (frozen encoder vector h(m)) and receptor unit j:
    x(m, j) = softplus(W_x h(m) + b_x)_j        binding (>= 0)
    e(m, j) = sigmoid(W_e h(m) + b_e)_j         efficacy (0..1)
Per mixture A (competitive binding, no weights):
    r_j(A) = sum_(m in A) e(m,j) x(m,j) / (1 + sum_(m in A) x(m,j))
Distance:  d(A, B) = sigmoid(alpha * sqrt(sum_j w_j^2 (r_j(A) - r_j(B))^2) + beta),  w, alpha > 0
Readouts (training only): odour words of single molecules, sigmoid(U r(m)); blend groups of
molecule pairs, sigmoid(V r({m1, m2})). A pair is a 2-molecule mixture through the same layer, so
a molecule that does not bind receptor j leaves r_j unchanged: no averaging over pairs, no dilution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class O1(nn.Module):
    def __init__(self, n_emb: int, n_receptors: int = 64, n_words: int = 0, n_groups: int = 0):
        super().__init__()
        self.head_x = nn.Linear(n_emb, n_receptors)
        self.head_e = nn.Linear(n_emb, n_receptors)
        nn.init.constant_(self.head_x.bias, -1.0)            # start at x ~ 0.3: about a quarter of the sites
        self.w_raw = nn.Parameter(torch.zeros(n_receptors))  # w = softplus(w_raw) >= 0
        self.alpha_raw = nn.Parameter(torch.tensor(1.0))     # alpha = softplus(alpha_raw) > 0
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.words = nn.Linear(n_receptors, n_words) if n_words else None
        self.groups = nn.Linear(n_receptors, n_groups) if n_groups else None

    def binding(self, emb_BxE):
        """-> x_BxR, e_BxR."""
        return F.softplus(self.head_x(emb_BxE)), torch.sigmoid(self.head_e(emb_BxE))

    def receptors(self, emb_TxE, idx_NxK, mask_NxK):
        """Competitive binding: r_NxR for N mixtures given as padded rows of the table, plus x of
        the molecules involved (for the sparsity penalty)."""
        x_NxKxR, e_NxKxR = self.binding(emb_TxE[idx_NxK])
        m = mask_NxK.unsqueeze(-1).to(x_NxKxR.dtype)
        r_NxR = (m * x_NxKxR * e_NxKxR).sum(1) / (1.0 + (m * x_NxKxR).sum(1))
        return r_NxR, x_NxKxR[mask_NxK]

    def raw_distance(self, rA_NxR, rB_NxR, weighted: bool = True):
        w = F.softplus(self.w_raw) if weighted else torch.ones_like(self.w_raw)
        return torch.sqrt(((w * (rA_NxR - rB_NxR)) ** 2).sum(-1) + 1e-8)

    def distance(self, rA_NxR, rB_NxR):
        return torch.sigmoid(F.softplus(self.alpha_raw) * self.raw_distance(rA_NxR, rB_NxR) + self.beta)


def asymmetric_loss(logits_BxG, y_BxG, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05):
    """Asymmetric loss (Ridnik et al. 2021) for incomplete multi-label targets: listed labels get full
    weight; unlisted ones are down-weighted (gamma_neg) and easy negatives below `clip` are ignored,
    so labels the catalogue simply did not list are not pushed hard to 0."""
    p = torch.sigmoid(logits_BxG)
    p_neg = (p - clip).clamp(min=0)
    loss_pos = y_BxG * (1 - p) ** gamma_pos * torch.log(p.clamp(min=1e-8))
    loss_neg = (1 - y_BxG) * p_neg ** gamma_neg * torch.log((1 - p_neg).clamp(min=1e-8))
    return -(loss_pos + loss_neg).sum(-1).mean()


def pad(groups: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Lists of table rows -> (idx_NxK, mask_NxK), padded with row 0 and mask False."""
    k = max(len(g) for g in groups)
    idx = torch.zeros(len(groups), k, dtype=torch.long)
    mask = torch.zeros(len(groups), k, dtype=torch.bool)
    for i, g in enumerate(groups):
        idx[i, :len(g)] = torch.tensor(g)
        mask[i, :len(g)] = True
    return idx.to(device), mask.to(device)


# ---------------------------------------------------------------------------------------------
# Data: one embedding table for every molecule, and the three training signals
# ---------------------------------------------------------------------------------------------
import csv  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import data as _data, features as _features  # noqa: E402

GSLF = "Training_Dataset/curated_GS_LF_merged_4983.csv"
PAIRS_CSV = Path(__file__).resolve().parents[1] / "external/odor_pair/pairs.csv"
GROUP_MIN_COUNT = 1000          # odor-pair code's cutoff (pairing/data.py CUTOFF): 33 groups


@dataclass
class Tables:
    emb_TxE: np.ndarray                     # standardized open-POM embeddings, DREAM rows first
    row_of_cid: dict                        # DREAM CID -> table row
    words_rows: np.ndarray                  # GS-LF molecules: table rows
    words_y: np.ndarray                     # GS-LF molecules: 0/1 odour words (n x W)
    word_names: list
    pair_rows: np.ndarray                   # blend pairs: (n x 2) table rows
    pair_y: np.ndarray                      # blend pairs: 0/1 groups (n x G)
    group_names: list
    pair_train: np.ndarray = field(default=None)   # pair indices for training (molecule-disjoint split)
    pair_monitor: np.ndarray = field(default=None)  # pair indices kept out, for the AUROC check only

    def describe(self) -> str:
        return (f"table {self.emb_TxE.shape[0]} molecules x {self.emb_TxE.shape[1]}; GS-LF {len(self.words_rows)} molecules x "
                f"{len(self.word_names)} words; blend pairs {len(self.pair_rows)} x {len(self.group_names)} groups "
                f"(train {len(self.pair_train)}, monitor {len(self.pair_monitor)}, molecule-disjoint)")


ENCODERS = {"openpom": ("openpom.npz", "openpom_extra.npz"),   # embed_openpom.py, embed_openpom_extra.py
            "mist": ("mist.npz", "mist_extra.npz")}            # embed_mist.py, embed_mist_extra.py


def load_tables(seed: int = 0, monitor_frac: float = 0.1, encoder: str = "openpom") -> Tables:
    """DREAM molecules (features/<encoder>.npz) + extra molecules (features/<encoder>_extra.npz,
    keyed by the SMILES string of the GS-LF and pair files); GS-LF labels; blend pairs
    (external/odor_pair/pairs.csv). Embeddings are standardized per column over the whole table
    (inputs only, no labels)."""
    dream_file, extra_file = ENCODERS[encoder]
    fs = np.load(_features.features_dir() / dream_file, allow_pickle=False)
    z = np.load(_features.features_dir() / extra_file, allow_pickle=False)
    emb = np.concatenate([fs["emb"], z["emb"]]).astype(np.float32)
    emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
    row_of_cid = {int(c): i for i, c in enumerate(fs["cids"])}
    row_of_smiles = {s: len(fs["cids"]) + i for i, s in enumerate(z["smiles"])}

    gslf = list(csv.DictReader(open(_data.locate("pnas_repo", needs=GSLF) / GSLF)))
    word_names = [c for c in gslf[0] if c not in ("nonStereoSMILES", "descriptors")]
    keep = [r for r in gslf if r["nonStereoSMILES"] in row_of_smiles]
    words_rows = np.array([row_of_smiles[r["nonStereoSMILES"]] for r in keep])
    words_y = np.array([[float(r[w]) for w in word_names] for r in keep], dtype=np.float32)

    rows = list(csv.DictReader(open(PAIRS_CSV)))
    total = {}
    for r in rows:
        for item in r["blender_groups"].split(";"):
            g, n = item.rsplit(":", 1)
            total[g] = total.get(g, 0) + int(n)
    group_names = sorted(g for g, n in total.items() if n > GROUP_MIN_COUNT)
    gi = {g: i for i, g in enumerate(group_names)}
    pr, py = [], []
    for r in rows:
        if r["smiles_a"] not in row_of_smiles or r["smiles_b"] not in row_of_smiles:
            continue
        y = np.zeros(len(group_names), np.float32)
        for item in r["blender_groups"].split(";"):
            g = item.rsplit(":", 1)[0]
            if g in gi:
                y[gi[g]] = 1
        if y.any():
            pr.append((row_of_smiles[r["smiles_a"]], row_of_smiles[r["smiles_b"]]))
            py.append(y)
    pair_rows, pair_y = np.array(pr), np.array(py)
    mols = np.unique(pair_rows)
    held = set(np.random.default_rng(seed).choice(mols, int(monitor_frac * len(mols)), replace=False).tolist())
    in_held = np.array([[a in held, b in held] for a, b in pair_rows])
    t = Tables(emb, row_of_cid, words_rows, words_y, word_names, pair_rows, pair_y, group_names)
    t.pair_train = np.where(~in_held.any(1))[0]
    t.pair_monitor = np.where(in_held.all(1))[0]
    return t


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
@dataclass
class Settings:
    receptors: int = 64
    steps: int = 2000               # steps on DREAM distances (strict, joint)
    pretrain_steps: int = 3000      # steps on blend pairs before DREAM (joint, pairs)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_words: float = 0.05      # GS-LF odour-word readout (BCE)
    lambda_pairs: float = 0.05      # blend-group readout (asymmetric loss)
    mu: float = 1e-3                # L1 on binding x
    batch_words: int = 256
    batch_pairs: int = 512
    log_every: int = 100
    seed: int = 0
    inner_calibration: bool = False  # early stopping + line fitted on an inner held-out split of the training fold
    xcal: bool = False               # cross-fitted calibration: line fitted on out-of-fold predictions (5 inner splits)


class Trainer:
    """Holds the tables on the device and trains O1 models for one data setting:
    strict = DREAM distances (+ GS-LF words); joint = pretrain on blend pairs, then DREAM + pairs;
    pairs = pretrain on blend pairs only, then a line on the unweighted receptor distance."""

    def __init__(self, tables: Tables, pairs_dream, setting: str, s: Settings, device: str, progress=None):
        assert setting in ("strict", "joint", "pairs")
        self.t, self.s, self.setting, self.dev = tables, s, setting, device
        self.pairs = pairs_dream
        self.progress, self.tag = progress, ""              # progress(line): live training log, e.g. print
        self.emb = torch.tensor(tables.emb_TxE, device=device)
        self.y = torch.tensor([p.y for p in pairs_dream], dtype=torch.float32, device=device)
        self.idxA, self.maskA = pad([[tables.row_of_cid[c] for c in sorted(p.a)] for p in pairs_dream], device)
        self.idxB, self.maskB = pad([[tables.row_of_cid[c] for c in sorted(p.b)] for p in pairs_dream], device)
        self.words_rows = torch.tensor(tables.words_rows, device=device)
        self.words_y = torch.tensor(tables.words_y, device=device)
        self.pair_rows = torch.tensor(tables.pair_rows, device=device)
        self.pair_y = torch.tensor(tables.pair_y, device=device)
        self.gen = torch.Generator(device="cpu").manual_seed(s.seed)
        self.pretrained, self.pretrain_log = None, []
        if setting in ("joint", "pairs"):
            self.pretrained, self.pretrain_log = self._pretrain()

    def new_model(self):
        torch.manual_seed(self.s.seed)
        return O1(self.emb.shape[1], self.s.receptors, len(self.t.word_names), len(self.t.group_names)).to(self.dev)

    def _sample(self, n_total, n, pool=None):
        i = torch.randint(0, n_total, (n,), generator=self.gen)
        return torch.as_tensor(pool)[i] if pool is not None else i

    def _aux_losses(self, model, use_pairs: bool) -> dict:
        s, out = self.s, {}
        pos = self._sample(len(self.words_rows), s.batch_words).to(self.dev)
        x, e = model.binding(self.emb[self.words_rows[pos]])
        r = x * e / (1 + x)                                  # a single molecule is a 1-molecule mixture
        out["words"] = F.binary_cross_entropy_with_logits(model.words(r), self.words_y[pos])
        out["l1"] = x.mean()
        if use_pairs:
            pi = self._sample(len(self.t.pair_train), s.batch_pairs, self.t.pair_train).to(self.dev)
            pr = self.pair_rows[pi]
            mask = torch.ones_like(pr, dtype=torch.bool)
            rp, xp = model.receptors(self.emb, pr, mask)
            out["pairs"] = asymmetric_loss(model.groups(rp), self.pair_y[pi])
            out["l1"] = (out["l1"] + xp.mean()) / 2
        return out

    def _pretrain(self):
        model, s, log = self.new_model(), self.s, []
        opt = torch.optim.AdamW(model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
        for step in range(1, s.pretrain_steps + 1):
            parts = self._aux_losses(model, use_pairs=True)
            loss = s.lambda_pairs * parts["pairs"] + s.lambda_words * parts["words"] + s.mu * parts["l1"]
            opt.zero_grad(); loss.backward(); opt.step()
            if step % s.log_every == 0 or step == s.pretrain_steps:
                log.append({"step": step, **{k: float(v.detach()) for k, v in parts.items()}})
                self._say("pretrain", log[-1])
        return {k: v.detach().clone() for k, v in model.state_dict().items()}, log

    def _say(self, tag, entry):
        if self.progress:
            self.progress(f"[{tag}] " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()))

    def all_receptors(self, model):
        rA, _ = model.receptors(self.emb, self.idxA, self.maskA)
        rB, _ = model.receptors(self.emb, self.idxB, self.maskB)
        return rA, rB

    def fit(self, train_idx):
        """Train one model on DREAM pairs train_idx -> (model, log). For 'pairs' only a line is fitted.
        With inner_calibration: hold out whole mixtures of the training fold (one inner zero-overlap
        fold), train on the rest, keep the step with the lowest inner-validation MSE, and fit a line
        prediction -> label on the inner-validation predictions. Only training-fold labels are used."""
        if self.s.xcal and self.setting != "pairs":
            return self._fit_xcal(np.asarray(train_idx))
        if self.s.inner_calibration and self.setting != "pairs":
            return self._fit_calibrated(np.asarray(train_idx))
        return self._train(train_idx)

    def _fit_xcal(self, train_idx):
        """Cross-fitted calibration (stacking). Split the training fold's mixtures 5 ways (zero
        overlap); for each split train on the rest and predict it, collecting out-of-fold
        predictions; fit y = a * pred + c on them. Then train on the whole training fold and apply
        the line. The model is frozen when the line is fitted; only training-fold labels are used."""
        from .cv import mixture_holdout_folds
        folds = mixture_holdout_folds([self.pairs[i] for i in train_idx], k=5, repeats=1, seed=self.s.seed)
        oof_idx, oof_pred, tag = [], [], self.tag
        for j, f in enumerate(folds):
            self.tag = f"{tag} inner {j + 1}/5"
            m, _ = self._train(train_idx[np.array(f.train)])
            te = train_idx[np.array(f.test)]
            oof_idx.append(te)
            oof_pred.append(self.predict(m, te, raw=True))
        self.tag = tag
        oof_idx, oof_pred = np.concatenate(oof_idx), np.concatenate(oof_pred)
        y_oof = self.y.cpu().numpy()[oof_idx]
        a, c = np.polyfit(oof_pred, y_oof, 1)
        model, log = self._train(train_idx)
        model.line = (float(a), float(c))
        log.append({"step": -1, "line_a": float(a), "line_c": float(c), "oof_n": int(len(oof_idx)),
                    "oof_bias_before": float(oof_pred.mean() - y_oof.mean())})
        self._say(tag, log[-1])
        return model, log


    def _fit_calibrated(self, train_idx):
        from .cv import mixture_holdout_folds
        fold = mixture_holdout_folds([self.pairs[i] for i in train_idx], k=5, repeats=1, seed=self.s.seed)[0]
        inner_tr, inner_va = train_idx[np.array(fold.train)], train_idx[np.array(fold.test)]
        model, log = self._train(inner_tr, val_idx=inner_va)
        with torch.no_grad():
            p = self.predict(model, inner_va)
        a, c = np.polyfit(p, self.y.cpu().numpy()[inner_va], 1)
        model.line = (float(a), float(c))
        log.append({"step": -1, "best_step": model.best_step, "line_a": float(a), "line_c": float(c),
                    "inner_val_n": int(len(inner_va))})
        return model, log

    def _train(self, train_idx, val_idx=None):
        """Train on DREAM pairs train_idx; with val_idx, keep the weights with the lowest val MSE."""
        s = self.s
        model = self.new_model()
        if self.pretrained is not None:
            model.load_state_dict(self.pretrained)
        log = []
        if self.setting == "pairs":
            with torch.no_grad():
                rA, rB = self.all_receptors(model)
                d = model.raw_distance(rA, rB, weighted=False).cpu().numpy()
            a, c = np.polyfit(d[train_idx], self.y.cpu().numpy()[train_idx], 1)
            model.line = (float(a), float(c))
            return model, log
        tr = torch.tensor(train_idx, device=self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
        best = (float("inf"), None, 0)
        for step in range(1, s.steps + 1):
            rA, xA = model.receptors(self.emb, self.idxA[tr], self.maskA[tr])
            rB, xB = model.receptors(self.emb, self.idxB[tr], self.maskB[tr])
            pred = model.distance(rA, rB)
            parts = {"mse": F.mse_loss(pred, self.y[tr])}
            parts.update(self._aux_losses(model, use_pairs=self.setting == "joint"))
            parts["l1"] = (parts["l1"] + torch.cat([xA, xB]).mean()) / 2
            loss = parts["mse"] + s.lambda_words * parts["words"] + s.mu * parts["l1"]
            if "pairs" in parts:
                loss = loss + s.lambda_pairs * parts["pairs"]
            opt.zero_grad(); loss.backward(); opt.step()
            if step % s.log_every == 0 or step == s.steps:
                p, yv = pred.detach().cpu().numpy(), self.y[tr].cpu().numpy()
                entry = {"step": step, **{k: float(v) for k, v in parts.items()},
                         "train_rmse": float(np.sqrt(((p - yv) ** 2).mean())), "train_r": float(np.corrcoef(p, yv)[0, 1])}
                if val_idx is not None:
                    with torch.no_grad():
                        pv = self.predict(model, val_idx, raw=True)
                    mse_v = float(((pv - self.y.cpu().numpy()[val_idx]) ** 2).mean())
                    entry["inner_val_rmse"] = mse_v ** 0.5
                    if mse_v < best[0]:
                        best = (mse_v, {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
                log.append(entry)
                self._say(self.tag, entry)
        if val_idx is not None:
            model.load_state_dict(best[1])
            model.best_step = best[2]
        return model, log

    @torch.no_grad()
    def predict(self, model, idx, raw: bool = False):
        """Predicted distances for DREAM training pairs idx; a fitted line (pairs setting, or inner
        calibration) is applied unless raw."""
        return self._predict(model, *self.all_receptors(model), raw)[idx]

    @torch.no_grad()
    def predict_pairs(self, model, pairs) -> np.ndarray:
        """Predicted distances for any pairs whose molecules are in the table (e.g. V50, T46)."""
        rA, _ = model.receptors(self.emb, *pad([[self.t.row_of_cid[c] for c in sorted(p.a)] for p in pairs], self.dev))
        rB, _ = model.receptors(self.emb, *pad([[self.t.row_of_cid[c] for c in sorted(p.b)] for p in pairs], self.dev))
        return self._predict(model, rA, rB)

    def _predict(self, model, rA, rB, raw: bool = False) -> np.ndarray:
        if self.setting == "pairs":
            a, c = model.line
            return (a * model.raw_distance(rA, rB, weighted=False) + c).cpu().numpy()
        d = model.distance(rA, rB).cpu().numpy()
        if not raw and getattr(model, "line", None) is not None:
            a, c = model.line
            d = a * d + c
        return d

    @torch.no_grad()
    def pair_monitor_auroc(self, model) -> float:
        """Macro AUROC of the blend-group readout on the molecule-disjoint monitor pairs."""
        from .metrics import auroc
        pi = torch.as_tensor(self.t.pair_monitor, device=self.dev)
        pr = self.pair_rows[pi]
        rp, _ = model.receptors(self.emb, pr, torch.ones_like(pr, dtype=torch.bool))
        p, y = torch.sigmoid(model.groups(rp)).cpu().numpy(), self.pair_y[pi].cpu().numpy()
        vals = [auroc(y[:, g], p[:, g]) for g in range(y.shape[1]) if 0 < y[:, g].sum() < len(y)]
        return float(np.mean(vals))
