# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   N = GoodScents + Ma molecules   S = GS-LF molecules
#   F = group indicators (input)   W = words (138)   G = DREAM's 24 merged groups   P = development pairs (514)
# ---------------------------------------------------------------------------------------------
"""Multi-hot probe (the user, 2026-11-19: "worst case throw away the deep network, do group-level multi-hot encoding
and train like a DNN not GNN and see; give some clue on these data"). The encoder is a plain MLP on RDKit functional-group
indicators (F -> 256 -> 138 words). It cannot memorize single molecules: molecules with the same groups get the same
profile. Question: do GoodScents blends and Ma dominance carry information that this encoder can use, and does that
information help single-molecule words and DREAM?

Training streams (as O3-E): singles (GS-LF train, open-POM word loss); GoodScents train blends (mix the two molecules'
decoded profiles with w = sigmoid(theta . (x1 - x2)), rank loss over the 79 notes with temperature); Ma (a theta . dx + c).
Mixture side: theta 24 + tau 1 (+ Ma line 2). Variants listed in VARIANTS; each 3 seeds, profiles averaged.

Checks: single held-out AUROC (GS-LF audit test); GoodScents held-out pairs (both molecules unseen): rank loss, word AUROC;
Ma: r of theta . dx vs L (in-sample for variants trained on Ma); DREAM development pairs, O1.2b form on the variant's
profiles (27 numbers fitted per fold), change from O1.2b, paired, 4 subsets.

Env:     mist (GPU)
Usage:   python evaluation/O3/probe_multihot.py
Writes:  results/probe_multihot/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o2j import word_auroc  # noqa: E402

F32 = torch.float32
SEEDS = (0, 1, 2)
# name: (input set, singles weight, GoodScents weight, Ma weight, mix space)
VARIANTS = {
    "A singles only, 85 groups": ("fr85", 1, 0, 0, "prob"),
    "B singles + GoodScents, 85 groups": ("fr85", 1, 1, 0, "prob"),
    "C singles + GoodScents + Ma, 85 groups": ("fr85", 1, 1, 1, "prob"),
    "D GoodScents only, 85 groups": ("fr85", 0, 1, 0, "prob"),
    "E singles + GoodScents, mix logits not probabilities": ("fr85", 1, 1, 0, "logit"),
    "F singles + GoodScents + Ma, 24 DREAM groups only": ("g24", 1, 1, 1, "prob"),
}


class MLP(nn.Module):
    def __init__(self, n_in, hidden=256, n_out=138, drop=0.2):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.f(x)


def mix(net, Fin, X, a, b, theta, tau, space):
    la, lb = net(Fin[a]), net(Fin[b]); w = torch.sigmoid((X[a] - X[b]) @ theta).unsqueeze(1)
    if space == "logit":
        return (w * la + (1 - w) * lb) / tau
    p = (w * torch.sigmoid(la) + (1 - w) * torch.sigmoid(lb)).clamp(1e-6, 1 - 1e-6)
    return torch.logit(p) / tau


def train(D, var, seed, dev, epochs=62, bs=128, pair_bs=128, ma_bs=32):
    inp, w_s, w_g, w_m, space = var; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Fs, Fn = D[inp]["single"], D[inp]["scent"]; X = D["X_N"]
    net = MLP(Fs.shape[1]).to(dev).to(F32)
    theta = torch.zeros(X.shape[1], device=dev, dtype=F32, requires_grad=True)
    log_tau = torch.zeros((), device=dev, dtype=F32, requires_grad=True)
    ma_line = torch.tensor([1.0, 0.0], device=dev, dtype=F32, requires_grad=True)
    opt = torch.optim.Adam(list(net.parameters()) + [theta, log_tau, ma_line], lr=1e-3, weight_decay=1e-5)
    tr = D["single_tr"]; nb = int(np.ceil(len(tr) / bs))
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr)
        for k in range(nb):
            loss = 1e-3 * (theta ** 2).sum()
            if w_s:
                b = perm[k * bs:(k + 1) * bs]; loss = loss + w_s * o2.pom_loss(net(Fs[b]), D["Y"][b], D["class_w"])
            if w_g:
                q = rng.choice(D["pairs_tr"], pair_bs, replace=False)
                lg = mix(net, Fn, X, D["pa"][q], D["pb"][q], theta, torch.exp(log_tau), space)
                loss = loss + w_g * o3.rank_loss(lg, D["PY"][q], D["T"]).mean()
            if w_m:
                t = rng.choice(len(D["L"]), ma_bs, replace=False); dx = (X[D["ma_a"][t]] - X[D["ma_b"][t]]) @ theta
                loss = loss + w_m * ((ma_line[0] * dx + ma_line[1] - D["L"][t]) ** 2).mean() / D["varL"]
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, theta.detach(), torch.exp(log_tau).detach()


@torch.no_grad()
def checks(D, var, net, theta, tau):
    inp, space = var[0], var[4]
    p_single = torch.sigmoid(net(D[inp]["single"][D["single_te"]])).cpu().numpy()
    te = D["pairs_te"]; lg = mix(net, D[inp]["scent"], D["X_N"], D["pa"][te], D["pb"][te], theta, tau, space)
    gl = o3.rank_loss(lg, D["PY"][te], D["T"]).mean().item(); ga, _ = word_auroc(lg.cpu().numpy(), D["PY"][te].cpu().numpy(), D["T"].cpu().numpy())
    dx = ((D["X_N"][D["ma_a"]] - D["X_N"][D["ma_b"]]) @ theta).cpu().numpy(); L = D["L"].cpu().numpy()
    return {"single": p_single, "scent_loss": gl, "scent_auc": ga, "ma_r": float(np.corrcoef(dx, L)[0, 1]) if dx.std() > 0 else float("nan"),
            "dream": torch.sigmoid(net(D[inp]["dream"])).cpu().numpy(), "theta": theta.cpu().numpy()}


def main():
    dev = pick_device(); rep = Report("probe_multihot", {"seeds": SEEDS, "variants": {k: list(v) for k, v in VARIANTS.items()}, "device": dev, "lam": LAM})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Xd, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    gslf = np.arange(G.n_gslf); smi_s = G.smiles_N[:G.n_gslf]
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
    D = {"fr85": {"single": t(o3.fr_units(smi_s)[0]), "scent": t(o3.fr_units(z["smiles"])[0]), "dream": t(o3.fr_units(op.smiles_M)[0])},
         "g24": {"single": t(o3.group_matrix(smi_s, gn)), "scent": t(o3.group_matrix(z["smiles"], gn)), "dream": t(Xd)}}
    D.update(X_N=D["g24"]["scent"], Y=t(G.Y[:G.n_gslf]), single_tr=gslf[~G.audit_test], single_te=gslf[G.audit_test],
             pairs_tr=np.where(S.pair_split == 0)[0], pairs_te=np.where(S.pair_split == 1)[0], pa=S.pair_a, pb=S.pair_b,
             PY=t(S.pair_Y), T=torch.as_tensor(S.targets, device=dev), ma_a=S.ma_a, ma_b=S.ma_b, L=t(S.ma_L), varL=float(S.ma_L.var()))
    cnt = G.Y[D["single_tr"]].sum(0); D["class_w"] = t(np.log1p(cnt / cnt.max()))
    rep.line(f"singles train {len(D['single_tr'])}, held-out {len(D['single_te'])}; GoodScents train pairs {len(D['pairs_tr'])}, held-out "
             f"{len(D['pairs_te'])} (both molecules unseen); Ma {len(S.ma_L)}; inputs: 85 RDKit fr_ indicators or DREAM's 24 merged groups; "
             f"encoder MLP in -> 256 -> 138 (85 groups: {85 * 256 + 256 + 256 * 138 + 138} weights); mixture side theta 24 + tau 1 (+ Ma 2)")
    # how many distinct inputs: the probe cannot tell apart molecules with the same groups
    for k in ("fr85", "g24"):
        u = len(np.unique(D[k]["single"].cpu().numpy(), axis=0)); rep.line(f"{k}: GS-LF molecules {G.n_gslf}, distinct group patterns {u}")

    out = {}
    for name, var in VARIANTS.items():
        cs = [checks(D, var, *train(D, var, s, dev)) for s in SEEDS]
        ps = np.mean([c["single"] for c in cs], 0)
        out[name] = {"single_auc": o2.mean_auroc(ps, G.Y[D["single_te"]]), "scent_loss": np.mean([c["scent_loss"] for c in cs]),
                     "scent_loss_sd": np.std([c["scent_loss"] for c in cs]), "scent_auc": np.mean([c["scent_auc"] for c in cs]),
                     "ma_r": np.mean([c["ma_r"] for c in cs]), "dream": np.mean([c["dream"] for c in cs], 0), "theta": np.mean([c["theta"] for c in cs], 0)}
        rep.line(f"{name}: trained")

    rep.section("Molecule-level checks (data never used in training); 3 seeds")
    rep.table(["variant", "single held-out AUROC (O2-C GNN 0.875-0.883)", "GoodScents held-out rank loss (frozen O2-C + theta 2.141)",
               "GoodScents held-out word AUROC", "Ma r (theta . dx vs L)"],
              [[k, f"{v['single_auc']:.4f}", f"{v['scent_loss']:.4f} (sd {v['scent_loss_sd']:.4f})", f"{v['scent_auc']:.4f}",
                ("in-sample " if VARIANTS[k][3] else "not trained on Ma ") + f"{v['ma_r']:+.3f}"] for k, v in out.items()])

    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M); q_op = mixture.quantile_per_word(op.probs_MxW)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    res = {"O1.2b (reference)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, Xd, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te))}
    for k, v in out.items():
        q = mixture.quantile_per_word(v["dream"])
        res[k] = evaluate.cross_validate(P, lambda tr, te, q=q: o3.O3(q, Xd, A, B, True, LAM).fit(tr, y[tr]).predict(te))
    ref = res["O1.2b (reference)"]

    def score(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    rep.section(f"DREAM development pairs, O1.2b form on each variant's profiles (27 numbers per fold), {evaluate.CV_REPEATS} x 5: change from O1.2b (paired)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = score(r_, sub, None if r_ is ref else ref)
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["profiles", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)

    rep.section("Paired steps (same folds): what each stream adds, first minus second")
    rows = []
    for hi, lo in (("B singles + GoodScents, 85 groups", "A singles only, 85 groups"),
                   ("C singles + GoodScents + Ma, 85 groups", "B singles + GoodScents, 85 groups"),
                   ("E singles + GoodScents, mix logits not probabilities", "B singles + GoodScents, 85 groups"),
                   ("D GoodScents only, 85 groups", "A singles only, 85 groups")):
        for sub in SUB:
            s_ = score(res[hi], sub, res[lo])
            rows.append([f"{hi}  minus  {lo}", sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["comparison", "subset", "RMSE change", "r change [95% CI]"], rows)

    rep.section("Group loudness theta learned from GoodScents (+ Ma), mean of 3 seeds")
    keys = [k for k in out if VARIANTS[k][2]]; order = np.argsort(-np.abs(out[keys[0]]["theta"]))
    rep.table(["group"] + keys, [[gn[g]] + [f"{out[k]['theta'][g]:+.3f}" for k in keys] for g in order])
    rep.save()


if __name__ == "__main__":
    main()
