# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = molecules   W = words (138)   R = members
# ---------------------------------------------------------------------------------------------
"""Train O2-VAE (design/o2_vae_design.md): open-POM rebuilt in PyTorch with a variational bottleneck, single
molecules only, open-POM's recipe + beta_KL x KL. No mixture data.

Step A (--grid): one member (seed 0) per configuration on an inner split of the O2 training molecules (80 / 20 at
  random, seed 0): "det" = deterministic O2, "D:beta" = VAE with D latent dims and KL weight beta.
  --grid-report applies the rule fixed in the design (within 0.005 AUROC of det -> largest beta, then smaller D).
Step B (--latent D --beta b --seeds ...): members on all O2 training molecules; --report gives the gate
  (held-out 20% AUROC: mean single >= 0.873 and ensemble >= 0.889).

Env:     mist (GPU)
Usage:   python training/train_o2_vae.py --grid det 32:1e-3 32:1e-2      (one process per GPU is fine)
         python training/train_o2_vae.py --grid-report
         python training/train_o2_vae.py --latent 32 --beta 1e-2 --seeds 0 1 2 3 4
         python training/train_o2_vae.py --report
Writes:  checkpoints/o2_vae/{grid_<cfg>,member_<seed>}.pt, results/train_o2_vae/<date_time>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixdist import features, o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402

CKPT = ROOT / "checkpoints/o2_vae"
GATE_SINGLE, GATE_ENSEMBLE, WITHIN = 0.873, 0.889, 0.005


def parse_cfg(c):
    if c == "det":
        return None, 0.0
    d, b = c.split(":")
    return int(d), float(b)


def make(latent):
    return (lambda: o2.O2VAE(latent)) if latent else o2.O2Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="*", default=[]); ap.add_argument("--grid-report", action="store_true")
    ap.add_argument("--latent", type=int); ap.add_argument("--beta", type=float); ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    dev = pick_device(); rep = Report("train_o2_vae", {**vars(a), "device": dev, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    gslf = np.arange(G.n_gslf); train_ids, test_ids = gslf[~G.audit_test], gslf[G.audit_test]
    perm = np.random.default_rng(0).permutation(train_ids); n_val = len(perm) // 5
    inner_val, inner_tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    CKPT.mkdir(parents=True, exist_ok=True)

    for cfg in a.grid:
        latent, beta = parse_cfg(cfg); t0 = time.time()
        net, steps = o2.train_member(G, inner_tr, 0, dev, log=rep.line, make_net=make(latent), beta_kl=beta)
        auc = o2.mean_auroc(o2.predict_probs(net, G, inner_val, dev), G.Y[inner_val])
        torch.save({"model_state_dict": net.state_dict(), "latent": latent, "beta": beta, "inner_auroc": auc}, CKPT / f"grid_{cfg.replace(':', '_')}.pt")
        rep.line(f"grid {cfg}: inner train {len(inner_tr)}, inner val {len(inner_val)}, AUROC {auc:.4f}, {time.time() - t0:.0f} s")

    if a.grid_report:
        res = {}
        for f in sorted(CKPT.glob("grid_*.pt")):
            c = torch.load(f, map_location="cpu", weights_only=False); res[f.stem[5:].replace("_", ":")] = c
        det = res["det"]["inner_auroc"]
        rep.section("Step A: inner-validation AUROC (one member, seed 0)")
        rows = [[k, c["latent"] or "-", c["beta"], f"{c['inner_auroc']:.4f}", f"{c['inner_auroc'] - det:+.4f}"] for k, c in res.items()]
        rep.table(["config", "D", "beta_KL", "AUROC", "minus det"], rows)
        ok = [(c["beta"], -c["latent"], k) for k, c in res.items() if c["latent"] and c["inner_auroc"] >= det - WITHIN]
        if ok:
            chosen = max(ok)[2]; rep.line(f"rule (within {WITHIN} of det, largest beta, then smaller D): chosen {chosen}")
        else:
            chosen = None; rep.line(f"rule: no VAE configuration within {WITHIN} of det -> stop (VAE does not fit as well as open-POM)")
        rep.value("grid", {k: c["inner_auroc"] for k, c in res.items()}); rep.value("chosen", chosen)

    for seed in a.seeds:
        t0 = time.time()
        net, steps = o2.train_member(G, train_ids, seed, dev, log=rep.line, make_net=make(a.latent), beta_kl=a.beta)
        auc = o2.mean_auroc(o2.predict_probs(net, G, test_ids, dev), G.Y[test_ids])
        torch.save({"model_state_dict": net.state_dict(), "latent": a.latent, "beta": a.beta, "seed": seed, "steps": steps,
                    "held_out_auroc": auc}, CKPT / f"member_{seed}.pt")
        rep.line(f"seed {seed}: D {a.latent}, beta {a.beta}, steps {steps}, held-out AUROC {auc:.4f}, {time.time() - t0:.0f} s")

    if a.report:
        files = sorted(CKPT.glob("member_*.pt")); nets = o2.load_members(files, dev)
        probs_R = [o2.predict_probs(n, G, test_ids, dev) for n in nets]
        single = [o2.mean_auroc(p, G.Y[test_ids]) for p in probs_R]; ens = o2.mean_auroc(np.mean(probs_R, 0), G.Y[test_ids])
        rep.section("Step B: gate, held-out 20% AUROC (mean over words)")
        rep.table(["member", "held-out AUROC"], [[f.stem, f"{s:.4f}"] for f, s in zip(files, single)])
        passed = np.mean(single) >= GATE_SINGLE and ens >= GATE_ENSEMBLE
        rep.line(f"single mean {np.mean(single):.4f} (range {min(single):.4f}-{max(single):.4f}), ensemble {ens:.4f}; "
                 f"gate: mean single >= {GATE_SINGLE} and ensemble >= {GATE_ENSEMBLE} -> {'PASS' if passed else 'FAIL'}")
        rep.line("references: released members 0.873-0.881 single; O2 (deterministic) single 0.875-0.883, ensemble 0.894")
        rep.value("gate", {"single": single, "ensemble": ens, "pass": bool(passed)})
    rep.save()


if __name__ == "__main__":
    main()
