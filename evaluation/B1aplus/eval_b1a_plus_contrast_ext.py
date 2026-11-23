"""Two follow-ups to the shared-molecule contrast lead (eval_b1a_plus_shared_contrast.py: beta +0.17 [+0.03, +1.00],
survives an absolute-salience control). Both on B1a++ (lam 1e-3 on theta only); mixing stays B1a++'s weighted mean
of open-POM word profiles; only the weights change.

emb    HIDDEN-LAYER contrast. Same model, but contrast is measured in open-POM's internal 256-d embedding (the layer
       before the word outputs; each dimension z-scored over the 235 molecules) instead of the 138 word outputs.
       Both contrasts enter together:  [i shared] * (alpha + b_word * contrast_word_i + b_emb * contrast_emb_i).
       Reading: b_emb carries it and b_word drops -> the effect lives below the words (hidden layer).
       Primary subset setting 2 (only shared molecules change).
match  EXTENSION to molecules that are NOT shared: every molecule i gets a soft match m_i = exp(-d_i / d0), d_i = word-
       profile distance to the closest molecule in the OTHER mixture (0 for a shared molecule, so m = 1), d0 = median
       nearest-neighbour distance among the 235 molecules. Company contrast c_i = || q_i - mean q of the other
       molecules in its own mixture ||, z-scored over all (pair, side, molecule) instances.
           log w_i = theta . groups_i + b_m * m_i + b_mc * m_i * c_i
       b_mc > 0: a molecule that has a look-alike on the other side AND stands out from its company pulls the pair
       together (generalises "noticed in both" from identical to similar molecules). Primary subset setting 1.
unshared  Do UNSHARED molecules that stand out from their company also count more (and so push the pair apart,
       since only one mixture has them)? Every molecule gets its company contrast (shared: vs the mixture's unshared
       molecules, as before; unshared: vs all other molecules of its mixture), z-scored separately:
           log w_i = theta . groups_i + [i shared] * (alpha + b_sh * c_i) + [i unshared] * b_un * c_i
       Prediction: b_un > 0. In setting 1 every molecule is unshared, so b_un is the setting-1 version of the lead.
       Primary subset setting 2 (both kinds present); setting 1 reported. Also b_un fitted on setting-1 pairs only and on
       setting-2 pairs only (50 refits each): does it act the same in both?
Fixed before running: standard CV, change from B1a++ on the same folds; claim only if r-change CI > 0 and
within-study CI > 0 on the primary subset. Fitted numbers on all 820 pairs with 95% CI from 50 mixture-resample refits.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_contrast_ext.py {emb|match|unshared}
Writes:  results/eval_b1a_plus_contrast_ext_<mode>/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402
from eval_b1a_plus_shared_contrast import SharedTerms  # noqa: E402

LAM = 1e-3; N_BOOT = 2000; N_BOOT_X = 50


def company_contrast(F, Mx, Sh, P, shared_only):
    """|| F_i - mean F of company ||; company = unshared molecules (shared_only) or all others in the mixture."""
    C = np.zeros(Mx.shape)
    for p in range(P):
        mem = np.where(Mx[p] > 0)[0]
        if shared_only:
            sh = mem[Sh[p, mem] > 0]; un = mem[Sh[p, mem] == 0]
            if len(sh) and len(un):
                C[p, sh] = np.linalg.norm(F[sh] - F[un].mean(0), axis=1)
        elif len(mem) > 1:
            tot = F[mem].sum(0)
            C[p, mem] = np.linalg.norm(F[mem] - (tot[None, :] - F[mem]) / (len(mem) - 1), axis=1)
    return C


def zs(CA, CB, mA, mB):
    v = np.r_[CA[mA], CB[mB]]; mu, sd = v.mean(), v.std()
    return np.where(mA, (CA - mu) / sd, 0.0), np.where(mB, (CB - mu) / sd, 0.0)


def main():
    mode = sys.argv[1]; assert mode in ("emb", "match", "unshared")
    rep = Report(f"eval_b1a_plus_contrast_ext_{mode}", {"mode": mode, "lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_boot_x": N_BOOT_X})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    Sh = A * B

    if mode == "emb":
        E = (op.emb_MxE - op.emb_MxE.mean(0)) / (op.emb_MxE.std(0) + 1e-9)
        WA, WB = company_contrast(q, A, Sh, P, True), company_contrast(q, B, Sh, P, True)
        EA, EB = company_contrast(E, A, Sh, P, True), company_contrast(E, B, Sh, P, True)
        mA, mB = (Sh > 0) & (WA > 0), (Sh > 0) & (WB > 0)
        ZWA, ZWB = zs(WA, WB, mA, mB); ZEA, ZEB = zs(EA, EB, mA, mB)
        rep.line(f"shared instances {mA.sum() + mB.sum()}; corr(word contrast, embedding contrast) = {np.corrcoef(np.r_[ZWA[mA], ZWB[mB]], np.r_[ZEA[mA], ZEB[mB]])[0, 1]:+.2f}")
        make = lambda: SharedTerms(q, Xg, A, B, S, LAM, [ZWA, ZEA], [ZWB, ZEB], Sh)  # noqa: E731
        labels = ["alpha (shared)", "b_word (word-space contrast)", "b_emb (embedding contrast)"]
        primary = "setting 2"
    elif mode == "unshared":
        WA, WB = company_contrast(q, A, Sh, P, True), company_contrast(q, B, Sh, P, True)
        mA, mB = (Sh > 0) & (WA > 0), (Sh > 0) & (WB > 0)
        ZSA, ZSB = zs(WA, WB, mA, mB)
        UA_, UB_ = (A > 0) & (Sh == 0), (B > 0) & (Sh == 0)
        CA, CB = company_contrast(q, A, Sh, P, False), company_contrast(q, B, Sh, P, False)
        ZUA, ZUB = zs(CA, CB, UA_ & (CA > 0), UB_ & (CB > 0))
        ShI = (Sh > 0).astype(float); ones = np.ones(A.shape)
        rep.line(f"shared instances {mA.sum() + mB.sum()}; unshared instances {UA_.sum() + UB_.sum()}")
        make = lambda: SharedTerms(q, Xg, A, B, S, LAM, [ShI, ShI * ZSA, ZUA], [ShI, ShI * ZSB, ZUB], ones)  # noqa: E731
        labels = ["(cancels)", "alpha (shared)", "b_sh (shared stands out)", "b_un (unshared stands out)"]
        primary = "setting 2"
    else:
        D = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
        nn = np.sort(D + np.eye(len(D)) * 1e9, 1)[:, 0]; d0 = float(np.median(nn))
        def match(Mx, Other):
            Mm = np.zeros(Mx.shape)
            for p in range(P):
                mem = np.where(Mx[p] > 0)[0]; oth = np.where(Other[p] > 0)[0]
                Mm[p, mem] = np.exp(-D[np.ix_(mem, oth)].min(1) / d0)
            return Mm
        MA, MB = match(A, B), match(B, A)
        CA, CB = company_contrast(q, A, Sh, P, False), company_contrast(q, B, Sh, P, False)
        ZA, ZB = zs(CA, CB, A > 0, B > 0)
        s1 = (Sh.sum(1) == 0)
        rep.line(f"d0 = {d0:.2f}; match m in setting-1 pairs: mean {np.r_[MA[s1][A[s1] > 0], MB[s1][B[s1] > 0]].mean():.2f}, "
                 f"share of molecules with m > 0.5: {(np.r_[MA[s1][A[s1] > 0], MB[s1][B[s1] > 0]] > 0.5).mean():.0%}")
        ones = np.ones(A.shape)
        make = lambda: SharedTerms(q, Xg, A, B, S, LAM, [MA, MA * ZA], [MB, MB * ZB], ones)  # noqa: E731
        labels = ["alpha (cancels: same for every molecule)", "b_m (has a look-alike on the other side)", "b_mc (look-alike x stands out)"]
        primary = "setting 1"

    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: make().fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)
    s1 = base.no_overlap_N; kN = Sh.sum(1)[base.pair_N]
    rows, ok = [], False
    for subset, mask in [("setting 1", s1), ("setting 2", ~s1), ("  share 1-2", (kN >= 1) & (kN <= 2)), ("all pairs", np.ones(len(s1), bool))]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        d0_, (lo, hi), _ = within_change(train, base, res, mask)
        if subset == primary:
            ok = s["r_change_ci"][0] > 0 and lo > 0
        rows.append([subset + (" (primary)" if subset == primary else ""), s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                     f"{d0_:+.3f} [{lo:+.3f}, {hi:+.3f}]"])
    rep.section(f"1. {mode} vs B1a++ (same folds)")
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "new RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule ({primary}): {'MET' if ok else 'NOT met'}")

    allp = np.arange(P)
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_X)
    ab = lambda m: [float(v) for v in m.ab(m.theta)]  # noqa: E731
    f0 = ab(make().fit(allp, y))
    fb = np.array([ab(make().fit(allp, y, w_pairs=(cnt[i, ja] * cnt[i, jb]).astype(float) + 1e-12)) for i in range(N_BOOT_X)])
    rep.section("2. Fitted numbers (all 820 pairs)")
    for j, lab in enumerate(labels):
        rep.line(f"{lab:45s} {f0[j]:+.3f} [{np.percentile(fb[:, j], 2.5):+.3f}, {np.percentile(fb[:, j], 97.5):+.3f}]")
    if mode == "unshared":
        for nm, sub in [("setting-1 pairs only", np.where(Sh.sum(1) == 0)[0]), ("setting-2 pairs only", np.where(Sh.sum(1) > 0)[0])]:
            g0 = ab(make().fit(sub, y[sub]))[3]
            gb = np.array([ab(make().fit(sub, y[sub], w_pairs=(cnt[i, ja[sub]] * cnt[i, jb[sub]]).astype(float) + 1e-12))[3] for i in range(N_BOOT_X)])
            rep.line(f"b_un fitted on {nm:22s} {g0:+.3f} [{np.percentile(gb, 2.5):+.3f}, {np.percentile(gb, 97.5):+.3f}]")
    rep.value("result", {"mode": mode, "fit": dict(zip(labels, f0)), "ci": {lab: list(np.percentile(fb[:, j], [2.5, 97.5])) for j, lab in enumerate(labels)}, "claim_met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
