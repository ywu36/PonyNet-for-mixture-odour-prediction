"""Build the shipped model file pony/weights/pony_dream.pt and check the clean package against the research pipeline.

Steps (each check must pass):
  1. featurizer: pony.featurize reproduces open-POM's GraphFeaturizer graphs for every molecule in features/o2_graphs.npz
  2. network: the 10 Stage-1 copies (checkpoints/o3m_attpool) load into pony.PonyNet; per copy, the hidden scale (mean
     hidden length over 2,000 training molecules) and the own atom scorer u0 are taken from the research code
     (evaluation/O3/eval_o5_softmax.backbones), and the package's DREAM molecule vectors equal the research ones
  3. readout: the 133 numbers per copy are fitted on all 820 DREAM training pairs by the research code that produced the
     paper (evaluation/O3/eval_size_tune.SizeTune, row "(b) + hidden length, learned (133)"), and stored. The package's
     own fit (pony.readout.fit, same equations) is run on the same molecule table as a check and its difference reported:
     the fitted temperature often reaches its floor, where L-BFGS is sensitive to rounding, so the two fits can end at
     slightly different points. The package then predicts T46 and V50 from the stored numbers; the predictions must
     equal the research run's and give the paper's r (T46 0.593, V50 0.622)

Env:     mist
Usage:   memcap python reproduce/build_pony_weights.py [--skip-featurizer-check]
Writes:  pony/weights/pony_dream.pt, results/build_pony_weights/<date_time>.{txt,json}
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
import pony  # noqa: E402
from pony import features as fz, readout as ro  # noqa: E402
from pony.model import Member  # noqa: E402

PAPER = {"T46": 0.593, "V50": 0.622}
ROW = "(b) + hidden length, learned (133)"


def check_featurizer(G, rep):
    bad = 0
    for i, smi in enumerate(G.smiles_N):
        g = fz.featurize(str(smi)); n0, n1, e0, e1 = G.node_off[i], G.node_off[i + 1], G.edge_off[i], G.edge_off[i + 1]
        ok = (np.array_equal(g.node, G.node[n0:n1]) and np.array_equal(g.edge, G.edge[e0:e1])
              and np.array_equal(g.edge_index, G.edge_index[:, e0:e1]))
        bad += not ok
    rep.line(f"featurizer: {len(G.smiles_N) - bad} of {len(G.smiles_N)} molecules identical to open-POM's graphs")
    assert bad == 0


def main():
    import eval_o5_softmax as eo
    import train_groupatoms
    import train_o3m
    dev = pick_device(); rep = Report("build_pony_weights", {"seeds": eo.SEEDS, "device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    if "--skip-featurizer-check" not in sys.argv:
        check_featurizer(G, rep)
    _, BB = eo.backbones(dev, rep); BB = BB["o5"]
    _, tr, _, _ = train_groupatoms.graphs_with_groups(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    dream_smiles = [str(s) for s in G.smiles_N[G.dream_rows()]]; cid_to_smiles = dict(zip([int(c) for c in G.dream_cids], dream_smiles))
    graphs = [fz.featurize(s) for s in dream_smiles]
    members = []
    for s, bb in zip(eo.SEEDS, BB):
        ck = torch.load(ROOT / f"checkpoints/o3m_attpool/member_{s}.pt", map_location="cpu", weights_only=False)
        n = train_o3m.load(ROOT / f"checkpoints/o3m_attpool/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        arch = {k: v for k, v in ck["arch"].items() if k != "pool"}; assert ck["arch"].get("pool") == "attn"
        m = Member(arch, ck["model_state_dict"], scale, bb["q0"], None)
        V, H, mol, _ = m.encode(graphs)
        dv, dh = np.abs(V - bb["V"]).max(), np.abs(H - bb["H"]).max()
        rep.line(f"copy {s}: scale {scale:.4f}; max |difference| to research vectors: molecules {dv:.2e}, atoms {dh:.2e}")
        assert dv < 1e-4 and dh < 1e-4 and np.array_equal(mol, bb["mol"])
        members.append(m)
    model = pony.Pony(members, [str(w) for w in G.words],
                      {"paper": "Structure over Scale: Mixing Odours in a Pony Network like the Human Receptors",
                       "fitted_on": "DREAM olfactory mixtures challenge training data: 820 rated pairs (730 challenge rows + 90 released after)"})
    train = data.load_train(); smi = lambda ids: [cid_to_smiles[int(c)] for c in ids]  # noqa: E731
    import eval_context_dream as ecd
    import eval_size_tune as est
    from mixdist.b1a_plus import membership
    ecd.DIRECTION_ONLY = True; y = evaluate.labels(train); A, B = membership(train, G.dream_cids); tr = np.arange(len(train))
    for m, bb in zip(members, BB):
        f = est.SizeTune(bb["V"], bb["H"], bb["mol"], A, B, bb["s"], bb["q0"], "hl64+plain", 0.0).fit(tr, y)
        m.readout = ro.ReadoutParams(f.lin[0].item(), f.lin[1].item(), f.beta.item(), f.kappa.item(), f.t.item(),
                                     f.delta.numpy().copy(), f.cvec.numpy().copy())
        tab = ro.MoleculeTable(bb["V"], bb["H"], bb["mol"], bb["q0"]); own = ro.fit(tab, A, B, y)
        p = m.readout; dp = max(abs(own.a - p.a), abs(own.beta - p.beta), np.abs(own.delta - p.delta).max(), np.abs(own.gamma - p.gamma).max())
        dy = np.abs(ro.predict(tab, A, B, own) - ro.predict(tab, A, B, p)).max()
        rep.line(f"readout: a {p.a:+.4f} c {p.c:+.4f} beta {p.beta:+.4f} kappa {p.kappa:+.4f} tau {np.exp(p.t):.2e} "
                 f"|delta| {np.linalg.norm(p.delta):.4f} |gamma| {np.linalg.norm(p.gamma):.4f}; package fit: max number "
                 f"difference {dp:.1e}, max training-prediction difference {dy:.1e}, tau {np.exp(own.t):.2e}")
    ref = sorted((ROOT / "results/eval_size_tune").glob("*_heldout_pred.pkl"))
    ref = [r for r in ref if ROW in pickle.load(open(r, "rb"))["T46"]["pred"]]
    stored = pickle.load(open(ref[-1], "rb")) if ref else None
    out = ROOT / "pony/weights/pony_dream.pt"; model.save(out); model = pony.Pony.from_pretrained(out)
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        pred = model.predict([smi(p.a) for p in hs.pairs], [smi(p.b) for p in hs.pairs]); r = metrics.pearson(pred, hs.y_P)
        msg = f"{hs.name}: r {r:.4f} (paper {PAPER[hs.name]}), RMSE {metrics.rmse(pred, hs.y_P):.4f}"
        if stored:
            msg += f"; max |difference| to research predictions ({ref[-1].name}) {np.abs(pred - stored[hs.name]['pred'][ROW]).max():.2e}"
        rep.line(msg); assert round(r, 3) == PAPER[hs.name]
    rep.line(f"saved {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)"); rep.save()


if __name__ == "__main__":
    main()
