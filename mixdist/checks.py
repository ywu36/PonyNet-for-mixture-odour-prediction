# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   M = our molecules (235)        G = our molecules that have GS-LF labels (229)
#   W = model odour words          L = GS-LF labels (138)        E = embedding size
#   S = odour words two models share
# ---------------------------------------------------------------------------------------------
"""Checks that a model's cached features are right for OUR molecules (used by check_<model>.py).

    run_standard_checks   the sections every check_<model>.py prints, in this order:
      1. inputs      the SMILES the model was given are our molecules (same structure as the data)
      2. outputs     odour-word probabilities vs real GS-LF labels: mean AUROC, shuffled control
      3. embedding   linear probe from the embedding vector to GS-LF labels, cross-validated over
                     molecules: mean AUROC, shuffled control
      4. agreement   per-molecule correlation with every other model whose features are cached

Both MIST and open-POM were trained on GS-LF, so 2 and 3 are in-sample: they show our pipeline
and the cached arrays are correct (wrong SMILES format, label order or row order would give ~0.5,
like the shuffled control), not how well the models generalise to new molecules.
Needs rdkit and scikit-learn (mist env).
"""
from __future__ import annotations

import numpy as np

from . import data, features, gslf, metrics

MIN_POSITIVES = 3        # labels with fewer positive molecules give unstable AUROCs and are skipped


def _rows(fs, cids) -> list[int]:
    row_of_cid = {int(c): i for i, c in enumerate(fs.cids_M)}
    return [row_of_cid[int(c)] for c in cids]


def _scored_labels(y_GxL) -> list[int]:
    return [k for k in range(y_GxL.shape[1]) if y_GxL[:, k].sum() >= MIN_POSITIVES]


def _shuffled(y_GxL, seed: int = 0):
    return y_GxL[np.random.default_rng(seed).permutation(len(y_GxL))]


def input_smiles(fs) -> dict:
    """Does each SMILES the model saw describe the same molecule as our data (stereo included)?"""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    def canon(s):
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol) if mol else None

    ours = data.load_smiles()
    bad = [int(c) for c, s in zip(fs.cids_M, fs.smiles_M) if canon(str(s)) != canon(ours[int(c)])]
    return {"n": len(fs.cids_M), "n_same_molecule": len(fs.cids_M) - len(bad), "mismatched_cids": bad}


def output_auroc(fs, cids_G, labels_L, y_GxL) -> dict:
    """Mean AUROC of the model's probability for each GS-LF label it predicts."""
    probs_GxW = fs.probs_MxW[_rows(fs, cids_G)]
    col_of_word = {w: j for j, w in enumerate(fs.words_W)}
    y_shuf_GxL = _shuffled(y_GxL)
    per_label, control = {}, []
    for k in _scored_labels(y_GxL):
        word = labels_L[k]
        if word in col_of_word:
            score_G = probs_GxW[:, col_of_word[word]]
            per_label[word] = metrics.auroc(y_GxL[:, k], score_G)
            control.append(metrics.auroc(y_shuf_GxL[:, k], score_G))
    return {"mean": float(np.nanmean(list(per_label.values()))), "shuffled": float(np.nanmean(control)),
            "per_label": per_label}


def embedding_probe_auroc(fs, cids_G, labels_L, y_GxL, n_splits: int = 5, seed: int = 0) -> dict:
    """Logistic regression from the embedding to each GS-LF label, out-of-fold AUROC over molecules."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    emb_GxE = fs.emb_MxE[_rows(fs, cids_G)]

    def probe(y_G):
        oof_G = np.zeros(len(y_G))
        folds = StratifiedKFold(n_splits=min(n_splits, int(y_G.sum())), shuffle=True, random_state=seed)
        for tr, te in folds.split(emb_GxE, y_G):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            oof_G[te] = clf.fit(emb_GxE[tr], y_G[tr]).predict_proba(emb_GxE[te])[:, 1]
        return metrics.auroc(y_G, oof_G)

    y_shuf_GxL = _shuffled(y_GxL)
    ks = _scored_labels(y_GxL)
    real = {labels_L[k]: probe(y_GxL[:, k]) for k in ks}
    control = [probe(y_shuf_GxL[:, k]) for k in ks if y_shuf_GxL[:, k].sum() >= MIN_POSITIVES]
    return {"mean": float(np.nanmean(list(real.values()))), "shuffled": float(np.nanmean(control)),
            "per_label": real, "embedding_size": int(fs.emb_MxE.shape[1])}


def agreement(fa, fb) -> dict:
    """Per-molecule Pearson between two models' probabilities on their shared odour words."""
    shared_S = sorted(set(fa.words_W) & set(fb.words_W))
    ca = {w: j for j, w in enumerate(fa.words_W)}
    cb = {w: j for j, w in enumerate(fb.words_W)}
    a_MxS = fa.probs_MxW[:, [ca[w] for w in shared_S]]
    b_MxS = fb.probs_MxW[_rows(fb, fa.cids_M)][:, [cb[w] for w in shared_S]]
    r_M = np.array([metrics.pearson(a, b) for a, b in zip(a_MxS, b_MxS)])
    low = np.argsort(r_M)[:5]
    return {"n_shared_words": len(shared_S), "median": float(np.median(r_M)),
            "p10": float(np.percentile(r_M, 10)), "p90": float(np.percentile(r_M, 90)),
            "n_below_0.5": int((r_M < 0.5).sum()),
            "least": [(int(fa.cids_M[i]), float(r_M[i])) for i in low]}


def _weakest(per_label: dict, n: int = 5) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in sorted(per_label.items(), key=lambda kv: kv[1])[:n])


def run_standard_checks(rep, model: str, recompute: bool = False):
    """Print sections 0-4 for ``model`` into the Report; returns the FeatureSet."""
    rep.section("0. Embeddings")
    fs = features.ensure(model, recompute=recompute, log=rep.line)
    rep.line(f"{fs.describe()}; computed {features.computed_at(model)}")

    rep.section("1. Inputs: are the SMILES given to the model our molecules?")
    inp = input_smiles(fs)
    rep.line(f"{inp['n_same_molecule']} of {inp['n']} same molecule as the data"
             + (f"; MISMATCH: {inp['mismatched_cids']}" if inp["mismatched_cids"] else ""))
    rep.value("inputs", inp)

    cids_G, labels_L, y_GxL = gslf.labels_for_our_molecules()
    rep.section("2-3. Outputs and embedding vs real GS-LF odour labels (in-sample: the model was trained on GS-LF)")
    rep.line(f"{len(cids_G)} of our molecules have GS-LF labels; labels with >= {MIN_POSITIVES} positives are scored")
    out = output_auroc(fs, cids_G, labels_L, y_GxL)
    emb = embedding_probe_auroc(fs, cids_G, labels_L, y_GxL)
    rep.table(["what is scored", "labels", "mean AUROC", "shuffled labels"], [
        [f"2. odour-word probabilities ({fs.probs_MxW.shape[1]} words)", len(out["per_label"]), out["mean"], out["shuffled"]],
        [f"3. {emb['embedding_size']}-d embedding, linear probe (5-fold CV)", len(emb["per_label"]), emb["mean"], emb["shuffled"]],
    ])
    rep.line(f"weakest labels, outputs:   {_weakest(out['per_label'])}")
    rep.line(f"weakest labels, embedding: {_weakest(emb['per_label'])}")
    rep.value("outputs_auroc", out)
    rep.value("embedding_probe_auroc", emb)

    others = [m for m in features.MODELS if m != model and features.path(m).exists()]
    rep.section("4. Agreement with the other models on our molecules")
    if not others:
        rep.line("no other model's features cached")
    rows = []
    for other in others:
        fo = features.load(other)
        a = agreement(fs, fo)
        rows.append([fo.label, a["n_shared_words"], a["p10"], a["median"], a["p90"], a["n_below_0.5"]])
        rep.value(f"agreement_{other}", a)
    if rows:
        rep.table(["other model", "shared words", "r 10th pct", "median r", "r 90th pct", "molecules r < 0.5"], rows)
    return fs
