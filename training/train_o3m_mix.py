# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   Q = GoodScents pairs   T = Ma mixtures
#   H = hidden layer (64)   W = words (138)
# ---------------------------------------------------------------------------------------------
"""O3-M mixture fine-tuning: start from the singles-trained width-64 network (checkpoints/o3m_w64), add mixtures up at
the hidden vector v (mixture z = plain mean of v; a molecule's pull = its length |v|; NO group weights), and fine-tune
on real mixture data with the losses that held up so far (findings/clues.md):

  singles     open-POM word loss through the linear head (keeps the backbone's words; held-out AUROC must stay > 0.85)
  GoodScents  destination: head(z) / tau vs the blend's real notes (rank loss over the 79 notes; tau = 1 number)
              lean-to:     softplus(-s (cos(z, v_A) - cos(z, v_B)) / 0.1), s = +1 if the blend's real notes are closer to
                           A's own notes than to B's (train_o3t.scent_rel_labels: real labels only, gap >= 0.05)
  Ma          who wins:    softplus(-s (cos(z, v_A) - cos(z, v_B)) / 0.1), s = sign(L) only, never the value
Because the head is linear, head(z) = mean of head(v_m): adding up at v and reading words are the same thing.

Against memorizing (RoFt-Mol arXiv 2509.00614, L2-SP arXiv 1802.01483, POMMix arXiv 2501.16271):
  encoder learning rate 0.1 x the head's; L2 pull of the encoder toward its starting weights (--l2sp);
  early stopping on held-out singles (10%) + held-out GoodScents molecules (10%, both losses);
  checks on molecules never trained on (official GoodScents held-out pairs, Ma fold 0, official single held-out);
  --mode shuffle: same as ft with GoodScents notes, lean labels and Ma signs permuted across training items (control:
  anything it gains is not real information).
--freeze-head: the word head is locked (only the encoder learns; singles words change only through the encoder).
Modes: head (encoder frozen: only the head and tau learn; lean-to and Ma have nothing to move), ft, shuffle.

Env:     mist (GPU)
Usage:   python training/train_o3m_mix.py --mode ft --members 0 1 2
Writes:  checkpoints/o3m_mix_<mode>/member_<k>.pt, results/train_o3m_mix/<date_time>_<mode>.{txt,json}
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import cv, evaluate, features, o2, o2j, o3  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o2j import word_auroc  # noqa: E402
from train_o3s_scent import MA_HOLDOUT_FOLD, split_molecules, split_singles  # noqa: E402
from train_o3t import anchor_pairs, scent_rel_labels  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import train_o3m  # noqa: E402

cos = nn.functional.cosine_similarity
F32 = torch.float32


def hidden(net, store, ids, dev):
    """v for molecules `ids` of a graph store, centring with the single-molecule statistics (never a mixture batch)."""
    x, e, s, d, g = store.batch(ids, dev)
    with o2j.single_stats(net):
        return net.latent(net.encode(x, e, s, d, g, int(g.max()) + 1))[0]


def pair_terms(net, S, a, b, dev):
    mols, inv = np.unique(np.r_[a, b], return_inverse=True); v = hidden(net, S, mols, dev)
    va, vb = v[inv[:len(a)]], v[inv[len(a):]]; z = 0.5 * (va + vb)
    return z, cos(z, va, dim=1) - cos(z, vb, dim=1)


def dream_dist(net, D, idx, log_beta, dev):
    """Mixture = plain mean of v over its molecules; d = sqrt(|u_A - u_B|^2 + beta (h_A - h_B)^2), u = z/|z|, h = log|z|."""
    R = D["dream"]; v = hidden(net, D["G"], R["rows"], dev)
    zA, zB = (R["A"][idx] @ v) / R["A"][idx].sum(1, keepdim=True), (R["B"][idx] @ v) / R["B"][idx].sum(1, keepdim=True)
    du = ((nn.functional.normalize(zA, dim=1) - nn.functional.normalize(zB, dim=1)) ** 2).sum(1)
    dh = (torch.log(zA.norm(dim=1)) - torch.log(zB.norm(dim=1))) ** 2
    return torch.sqrt(du + torch.exp(log_beta) * dh + 1e-12)


def dream_comparison_loss(net, D, y_t, groups, rng, log_beta, dev, n=64):
    """Two training pairs sharing a mixture, (A, B) and (A, C): the one people rated less similar must be farther."""
    g = [groups[j] for j in rng.integers(0, len(groups), n)]; ij = np.array([rng.choice(x, 2, replace=False) for x in g])
    i, j = ij[:, 0], ij[:, 1]; dy = y_t[i] - y_t[j]; keep = dy.abs() >= 0.02
    if not keep.any():
        return None
    d = dream_dist(net, D, np.r_[i, j], log_beta, dev); di, dj = d[:len(i)], d[len(i):]
    return nn.functional.softplus(-torch.sign(dy[keep]) * (di - dj)[keep] / 0.05).mean()


def lean_loss(d, y01):
    return nn.functional.softplus(-(2 * y01 - 1) * d / 0.1)


@torch.no_grad()
def checks(net, log_tau, D, dev, tr_q_sample):
    net.eval(); G, S = D["G"], D["S"]; out = {}
    out["single_auroc"] = o2.mean_auroc(o2.predict_probs(net, G, D["single_te"], dev), G.Y[D["single_te"]])
    out["single_train_auroc"] = o2.mean_auroc(o2.predict_probs(net, G, D["single_tr_sample"], dev), G.Y[D["single_tr_sample"]])
    for name, qs in (("heldout", D["te_q"]), ("train", tr_q_sample)):   # heldout first: it fits tau
        lg, dd = [], []
        for i in range(0, len(qs), 1000):
            q = qs[i:i + 1000]; z, d = pair_terms(net, S, S.pair_a[q], S.pair_b[q], dev); lg.append(net.head(z)); dd.append(d)
        lg, dd = torch.cat(lg), torch.cat(dd); Y = torch.as_tensor(S.pair_Y[qs], dtype=F32, device=dev)
        if name == "heldout":   # one temperature for the notes loss, chosen on training pairs (fair before vs after)
            lt_, yt_ = [], torch.as_tensor(S.pair_Y[tr_q_sample], dtype=F32, device=dev)
            for i in range(0, len(tr_q_sample), 1000):
                q = tr_q_sample[i:i + 1000]; lt_.append(net.head(pair_terms(net, S, S.pair_a[q], S.pair_b[q], dev)[0]))
            lt_ = torch.cat(lt_); grid = torch.logspace(-2, 1.5, 71, device=dev)
            tau = grid[int(torch.stack([o3.rank_loss(lt_ / g, yt_, D["T"]).mean() for g in grid]).argmin())]; out["tau_fitted"] = float(tau)
        out[f"scent_dest_loss_{name}"] = o3.rank_loss(lg / tau, Y, D["T"]).mean().item()
        lab = D["rel"][qs]; k = ~np.isnan(lab)
        out[f"scent_lean_auroc_{name}"] = roc_auc_score(lab[k], dd.cpu().numpy()[k])
        if name == "heldout":
            out["scent_word_auroc_heldout"] = word_auroc(lg.cpu().numpy(), S.pair_Y[qs], S.targets)[0]
    t = D["ma_te"]; _, d = pair_terms(net, S, S.ma_a[t], S.ma_b[t], dev)
    out["ma_heldout_auroc"] = roc_auc_score(S.ma_L[t] > 0, d.cpu().numpy())
    return out


def finetune(D, member, mode, dev, l2sp, enc_mult, epochs, log, pair_batch=128, ma_batch=32, freeze_head=False, w_single=1.0,
             fold=None, w_dream=1.0):
    G, S = D["G"], D["S"]; seed = member; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{member}.pt", dev); start = copy.deepcopy(net.state_dict())
    log_tau = torch.zeros((), device=dev, requires_grad=True); log_beta = torch.tensor(-2.0, device=dev, requires_grad=True)
    head_p = list(net.head.parameters()); enc_p = [p for n_, p in net.named_parameters() if not n_.startswith("head.")]
    enc_ref = [p.detach().clone() for p in enc_p]
    if mode == "head":
        for p in enc_p:
            p.requires_grad_(False)
    if freeze_head:   # locked word head: single-molecule words can only change through the encoder
        for p in head_p:
            p.requires_grad_(False)
    groups = [{"params": ([] if freeze_head else head_p) + [log_tau] + ([log_beta] if fold is not None else []), "lr": 1e-3, "base": 1e-3}]
    if mode != "head":
        groups.append({"params": enc_p, "lr": 1e-3 * enc_mult, "base": 1e-3 * enc_mult})
    opt = torch.optim.Adam(groups)
    before = checks(net, log_tau, D, dev, D["tr_q_sample"])
    tr_q, val_q = split_molecules(S, seed); val_q = val_q[rng.permutation(len(val_q))[:2000]]
    single_tr, single_va = split_singles(D["single_tr"], seed)
    PY = torch.as_tensor(S.pair_Y, dtype=F32, device=dev); rel = D["rel"].copy(); ma_y = (S.ma_L > 0).astype(np.float32)
    if mode == "shuffle":   # control: labels moved to other training items (structure no longer matches its labels)
        perm = rng.permutation(tr_q); PY[tr_q] = PY[perm]; rel[tr_q] = rel[perm]
        ma_y[D["ma_tr"]] = ma_y[rng.permutation(D["ma_tr"])]
    rel_t = torch.as_tensor(rel, dtype=F32, device=dev); ma_t = torch.as_tensor(ma_y, device=dev)
    if fold is not None:   # DREAM: the fold's training pairs, an inner mixture-disjoint 10% for early stopping
        R = D["dream"]; ftr = np.array(R["folds"][fold].train); sub = [R["P"][i] for i in ftr]
        inner = cv.mixture_holdout_folds(sub, k=10, repeats=1, seed=seed)[0]
        d_tr, d_va = ftr[np.array(inner.train)], ftr[np.array(inner.test)]; yD = R["y"].copy()
        if mode == "shuffle":
            yD[d_tr] = yD[rng.permutation(d_tr)]
        y_t = torch.as_tensor(yD, dtype=F32, device=dev); dgroups = anchor_pairs(R["P"], d_tr)
        log(f"  fold {fold}: DREAM training pairs {len(d_tr)} (anchor mixtures {len(dgroups)}), inner validation {len(d_va)}, "
            f"test {len(R['folds'][fold].test)}")
    by_mol = {}
    for q in tr_q:
        by_mol.setdefault(int(S.pair_a[q]), []).append(q); by_mol.setdefault(int(S.pair_b[q]), []).append(q)
    mol_list = np.array(sorted(by_mol))
    Y = torch.as_tensor(G.Y, device=dev); counts = G.Y[single_tr].sum(0)
    class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=F32, device=dev)
    bs = 128; nb = math.ceil(len(single_tr) / bs); step = 0; best = (np.inf, -1, None)
    for epoch in range(epochs):
        net.train(); perm_s = rng.permutation(single_tr); tot = {}
        for k in range(nb):
            for gr in opt.param_groups:
                gr["lr"] = gr["base"] * 0.5 ** (step // 640)
            b = perm_s[k * bs:(k + 1) * bs]
            L = {"singles": w_single * o2.pom_loss(net(G.batch(b, dev))[1], Y[b], class_w)}
            q = np.array([rng.choice(by_mol[int(m)]) for m in rng.choice(mol_list, pair_batch, replace=False)])
            z, d = pair_terms(net, S, S.pair_a[q], S.pair_b[q], dev)
            L["scent_dest"] = o3.rank_loss(net.head(net.drop(z)) / torch.exp(log_tau), PY[q], D["T"]).mean()
            lab = rel_t[torch.as_tensor(q, device=dev)]; kk = ~torch.isnan(lab)
            if mode != "head":
                if kk.any():
                    L["scent_lean"] = lean_loss(d[kk], lab[kk]).mean()
                t = rng.choice(D["ma_tr"], ma_batch, replace=False); _, dm = pair_terms(net, S, S.ma_a[t], S.ma_b[t], dev)
                L["ma_who_wins"] = lean_loss(dm, ma_t[torch.as_tensor(t, device=dev)]).mean()
            if fold is not None:
                ld = dream_comparison_loss(net, D, y_t, dgroups, rng, log_beta, dev)
                if ld is not None:
                    L["dream"] = w_dream * ld
            loss = sum(L.values())
            if mode != "head" and l2sp > 0:
                loss = loss + l2sp * sum(((p - r) ** 2).sum() for p, r in zip(enc_p, enc_ref))
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            for kk_, vv in L.items():
                tot[kk_] = tot.get(kk_, 0.0) + vv.item() / nb
        net.eval()
        with torch.no_grad():
            vs = 0.0
            for i in range(0, len(single_va), 256):
                bb = single_va[i:i + 256]; vs += o2.pom_loss(net(G.batch(bb, dev))[1], Y[bb], class_w).item() * len(bb) / len(single_va)
            z, d = pair_terms(net, S, S.pair_a[val_q], S.pair_b[val_q], dev)
            vd = o3.rank_loss(net.head(z) / torch.exp(log_tau), PY[val_q], D["T"]).mean().item()
            lab = rel_t[torch.as_tensor(val_q, device=dev)]; kk = ~torch.isnan(lab)
            vl = lean_loss(d[kk], lab[kk]).mean().item() if mode != "head" else 0.0
            vD = (1.0 - float(np.corrcoef(dream_dist(net, D, d_va, log_beta, dev).cpu().numpy(), D["dream"]["y"][d_va])[0, 1])) if fold is not None else 0.0
        crit = w_single * vs + vd + vl + vD
        if crit < best[0]:
            best = (crit, epoch, (copy.deepcopy(net.state_dict()), log_tau.detach().clone(), log_beta.detach().clone()))
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            log(f"  {mode} member {member} epoch {epoch + 1}: " + ", ".join(f"{k_} {v_:.4f}" for k_, v_ in tot.items())
                + f"; validation singles {vs:.4f}, GoodScents notes {vd:.4f}, lean {vl:.4f}" + (f", DREAM 1 - r {vD:.4f}" if fold is not None else "")
                + f" (best epoch {best[1] + 1})")
    net.load_state_dict(best[2][0]); log_tau, log_beta = best[2][1], best[2][2]
    drift = math.sqrt(sum(((net.state_dict()[n_] - start[n_]).float() ** 2).sum().item() for n_ in start if not n_.startswith("head.") and start[n_].is_floating_point()))
    after = checks(net, log_tau, D, dev, D["tr_q_sample"])
    if fold is not None:
        net.eval(); R = D["dream"]; te = np.array(R["folds"][fold].test)
        with torch.no_grad():
            d_te, d_f = dream_dist(net, D, te, log_beta, dev).cpu().numpy(), dream_dist(net, D, ftr, log_beta, dev).cpu().numpy()
        a_, c_ = evaluate.fit_line(d_f, R["y"][ftr])
        after.update({"fold": fold, "test_idx": te, "test_pred": a_ * d_te + c_, "beta": float(torch.exp(log_beta)),
                      "dream_train_r": float(np.corrcoef(d_f, R["y"][ftr])[0, 1]), "dream_test_r": float(np.corrcoef(d_te, R["y"][te])[0, 1])})
        log(f"  fold {fold} member {member}: DREAM r training pairs {after['dream_train_r']:.3f}, test pairs {after['dream_test_r']:.3f}")
    return net, log_tau, best[1] + 1, before, after, drift


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["head", "ft", "shuffle"], required=True)
    ap.add_argument("--members", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--label", choices=["cosine", "excess"], default="cosine", help="GoodScents lean-to label (scent_rel_labels)")
    ap.add_argument("--fold", type=int, default=None, help="DREAM in training: this fold of 5 x 1 (whole mixtures held out)")
    ap.add_argument("--w-dream", type=float, default=1.0)
    ap.add_argument("--w-single", type=float, default=1.0, help="weight of the single-molecule word loss (mixture losses weight 1)")
    ap.add_argument("--freeze-head", action="store_true", help="lock the linear word head (only the encoder and tau learn)")
    ap.add_argument("--tag", default="", help="suffix for the checkpoint folder and report")
    ap.add_argument("--l2sp", type=float, default=1e-3); ap.add_argument("--enc-mult", type=float, default=0.1)
    a = ap.parse_args(); dev = pick_device()
    rep = Report("train_o3m_mix", {**vars(a), "device": dev}, tag=a.mode + a.tag + ("" if a.fold is None else f"_fold{a.fold}"))
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    gslf = np.arange(G.n_gslf); ma_tr, ma_te = S.ma_split(MA_HOLDOUT_FOLD); te_q = np.where(S.pair_split == 1)[0]
    tr_all = np.where(S.pair_split == 0)[0]
    D = {"G": G, "S": S, "single_tr": gslf[~G.audit_test], "single_te": gslf[G.audit_test], "te_q": te_q, "ma_tr": ma_tr, "ma_te": ma_te,
         "T": torch.as_tensor(S.targets, device=dev), "rel": scent_rel_labels(G, S, measure=a.label),
         "single_tr_sample": np.random.default_rng(0).choice(gslf[~G.audit_test], 1000, replace=False),
         "tr_q_sample": np.random.default_rng(0).choice(tr_all, 2000, replace=False)}
    if a.fold is not None:
        P = dev_pairs(); op = features.load("openpom"); A, B = membership(P, op.cids_M)
        assert np.array_equal(np.asarray(G.dream_cids), np.asarray(op.cids_M).astype(G.dream_cids.dtype)), "DREAM molecule order"
        D["dream"] = {"P": P, "y": evaluate.labels(P), "rows": G.dream_rows(), "folds": list(cv.mixture_holdout_folds(P, k=5, repeats=1, seed=0)),
                      "A": torch.as_tensor(A, dtype=F32, device=dev), "B": torch.as_tensor(B, dtype=F32, device=dev)}
    rep.line(f"start: checkpoints/o3m_w64 members {a.members}; mode {a.mode}; encoder lr x{a.enc_mult}, pull to start {a.l2sp}; "
             f"GoodScents held-out pairs {len(te_q)} (lean labels {int((~np.isnan(D['rel'][te_q])).sum())}), Ma held-out fold 0 {len(ma_te)}")
    ck = ROOT / f"checkpoints/o3m_mix_{a.mode}{a.tag}"; ck.mkdir(parents=True, exist_ok=True); rows = []
    keys = [("single_auroc", "single AUROC (held-out)"), ("single_train_auroc", "single AUROC (training)"),
            ("scent_word_auroc_heldout", "GoodScents notes AUROC (held-out)"), ("scent_dest_loss_heldout", "GoodScents notes loss (held-out)"),
            ("scent_dest_loss_train", "GoodScents notes loss (training)"), ("tau_fitted", "notes temperature (fitted on training pairs)"), ("scent_lean_auroc_heldout", "GoodScents lean-to AUROC (held-out)"),
            ("scent_lean_auroc_train", "GoodScents lean-to AUROC (training)"), ("ma_heldout_auroc", "Ma who-wins AUROC (held-out)")]
    for m in a.members:
        t0 = time.time(); net, lt, ep, before, after, drift = finetune(D, m, a.mode, dev, a.l2sp, a.enc_mult, a.epochs, rep.line, freeze_head=a.freeze_head,
                                                               w_single=a.w_single, fold=a.fold, w_dream=a.w_dream)
        arch = torch.load(ROOT / f"checkpoints/o3m_w64/member_{m}.pt", map_location="cpu", weights_only=False)["arch"]
        torch.save({"model_state_dict": net.state_dict(), "arch": arch,
                    "member": m, "mode": a.mode, "tau": float(torch.exp(lt)), "best_epoch": ep, "before": before, "after": after,
                    "encoder_drift": drift, "single_auroc": after["single_auroc"]}, ck / (f"member_{m}.pt" if a.fold is None else f"fold{a.fold}_member{m}.pt"))
        for k, lab in keys:
            if k not in before:
                continue
            rows.append([m, lab, f"{before[k]:.4f}", f"{after[k]:.4f}", f"{after[k] - before[k]:+.4f}"])
        rep.line(f"member {m}: best epoch {ep}, encoder moved {drift:.3f} (L2 norm), {time.time() - t0:.0f} s")
    rep.section(f"Before vs after fine-tuning ({a.mode}); held-out = molecules never trained on. AUROC: higher = better; loss: lower = better")
    rep.table(["member", "check", "before", "after", "change"], rows)
    rep.save()


if __name__ == "__main__":
    main()
