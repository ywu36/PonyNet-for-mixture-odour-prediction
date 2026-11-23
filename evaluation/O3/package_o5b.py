# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   T = atoms   P = pairs
# ---------------------------------------------------------------------------------------------
"""Package model (b) for use (user via compatible_test, 2026-11-20: "best model packaged and ready to use now, CV only
confirming later"). Finding 6 sec. 14, clue C40.
Model (b): O5 network (checkpoints/o3m_attpool, 10 members, frozen, GS-LF singles only) + unit merge + strongest-atom
soft max with the DREAM atom query = beta q0 + delta (delta shrunk toward O5's own atom attention) + odd-one-out +
angle + line: 69 numbers per member, fitted on all 820 DREAM training pairs (exactly the held-out fit). Lockbox not
scored. Check: re-predicts T46 and V50 from the saved files (must equal finding 6 sec. 14: r 0.551 / 0.599).

Env:     mist
Usage:   python evaluation/O3/package_o5b.py                   (fit + save + check)
         python evaluation/O3/package_o5b.py --constants-only  (only add each member's hidden scale and strength SD to
                                                                 the saved readout files; needed by the new-molecule path)
Writes:  checkpoints/final_o5b/readout_member_<s>.pt, checkpoints/final_o5b/dream_vectors.npz,
         results/package_o5b/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402

OUT = ROOT / "checkpoints/final_o5b"


def member_constants(dev, BB):
    """Per member: hidden scale (mean hidden-vector length over 2000 GS-LF training molecules, as eo.backbones divides by)
    and the SD of O5's own log molecule strengths over the 235 DREAM molecules (eo.backbones divides atom scores by it)."""
    import train_groupatoms
    import train_o3m
    from mixdist import o2
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); _, tr, _, _ = train_groupatoms.graphs_with_groups()
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); out = {}
    for s, m in zip(eo.SEEDS, BB["o5"]):
        n = train_o3m.load(ROOT / f"checkpoints/o3m_attpool/member_{s}.pt", dev)
        with torch.no_grad():
            out[s] = {"scale": n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item(), "strength_sd": float(m["strength"].std())}
    return out


def main():
    ecd.DIRECTION_ONLY = True; dev = pick_device(); OUT.mkdir(parents=True, exist_ok=True)
    if "--constants-only" in sys.argv:
        rep = Report("package_o5b", {"constants_only": True, "device": dev}, tag="constants")
        _, BB = eo.backbones(dev, rep)
        for s, c in member_constants(dev, BB).items():
            r = torch.load(OUT / f"readout_member_{s}.pt", weights_only=False); r.update(c); torch.save(r, OUT / f"readout_member_{s}.pt")
            rep.line(f"member {s}: hidden scale {c['scale']:.4f}, strength SD {c['strength_sd']:.4f}")
        rep.save(); return
    rep = Report("package_o5b", {"members": eo.SEEDS, "lam": eo.LAM, "device": dev})
    _, BB = eo.backbones(dev, rep); op = features.load("openpom"); cids_M = np.array([int(c) for c in op.cids_M])
    train = data.load_train(); A, B = membership(train, cids_M); y = evaluate.labels(train)
    rep.line(f"fit on {len(train)} DREAM training pairs (all); lockbox not scored")
    vec = {}; const = member_constants(dev, BB)
    for s, m in zip(eo.SEEDS, BB["o5"]):
        f = eo.PriorAtom(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], "shrink").fit(np.arange(len(train)), y)
        torch.save({"seed": s, "backbone": f"checkpoints/o3m_attpool/member_{s}.pt", "beta": f.beta, "delta": f.delta, "t": f.t,
                    "kappa": f.kappa, "lin": f.lin, "q0": torch.as_tensor(m["q0"]), "lam": eo.LAM, **const[s]}, OUT / f"readout_member_{s}.pt")
        vec[f"V_{s}"], vec[f"H_{s}"], vec[f"mol_{s}"], vec[f"s_{s}"] = m["V"], m["H"], m["mol"], m["s"]
        rep.line(f"member {s}: beta {f.beta.item():+.3f}, |delta| {f.delta.norm().item():.3f} (|beta q0| {(f.beta * f.q0).norm().item():.3f}), "
                 f"tau {np.exp(f.t.item()):.3g}, kappa {f.kappa.item():+.3f}, line {f.lin.numpy().round(4)}")
    np.savez_compressed(OUT / "dream_vectors.npz", cids=cids_M, **vec)
    import predict_o5b
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        p = predict_o5b.predict([pr.a for pr in hs.pairs], [pr.b for pr in hs.pairs])
        rep.line(f"check {hs.name}: r {metrics.pearson(p, hs.y_P):.3f}, RMSE {metrics.rmse(p, hs.y_P):.4f} (from saved files)")
    rep.save()


if __name__ == "__main__":
    main()
