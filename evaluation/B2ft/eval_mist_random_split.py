# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = training pairs (820)   T = test pairs of one split (82)
# ---------------------------------------------------------------------------------------------
"""B2-ft on MIST's split type: why the MIST paper reports a gain from fine-tuning and our
cross-validation does not. The MIST paper splits pairs at random (80/10/10), so a test pair's
mixtures can also occur in training pairs; our cross-validation holds whole mixtures out.

Per split: 10% of the 820 training pairs at random are test pairs, the other 90% train version 2 of
B2-ft (mixdist.mist_mixture.train_v2, odour encoder; its inner validation pairs come from the 90%).
On the same test pairs it scores the frozen readouts with a line fitted on the 90%: B1 MIST
(baseline) and B1a on the CLS embedding. Test pairs are also scored by kind: "both mixtures seen"
(each of its two mixtures is in some training pair) and "neither seen" (neither is, as in our
cross-validation and in T46 / V50); the few pairs with one mixture seen count only in "all".
Only training pairs are used; T46 and V50 are not read. findings/finding_2.py reuses change_table.

Env:     mist (GPU)
Usage:   python evaluation/B2ft/eval_mist_random_split.py [--splits 5]
Writes:  results/eval_mist_random_split/<date_time>.{txt,json} (per-pair predictions of every split),
         checkpoints/mist_mixture/random_split/train_log.tsv (one line per epoch)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixdist import data, device, evaluate, features, metrics, mixture  # noqa: E402
from mixdist import mist_mixture as mm  # noqa: E402
from mixdist.report import Report  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "checkpoints/mist_mixture/random_split"
BASE = "B1 MIST frozen (cosine, line)"
ROWS = (BASE, "B1a MIST frozen (CLS, quantile, length 1, Euclidean, line)",
        "B2-ft stage A (encoder frozen) + line", "B2-ft stage B (whole model) + line")
SUBSETS = {"all test pairs": (0, 1, 2), "both mixtures seen": (2,), "neither seen": (0,)}   # -> mixtures seen


def run_split(split: int, pairs, d_b1_P, d_cls_P, y_P, dev) -> dict:
    """Train on 90% at random, predict the other 10% -> {test indices, seen flags, predictions per row}."""
    perm_P = np.random.default_rng(split).permutation(len(pairs))
    test_T, train = perm_P[:len(pairs) // 10], perm_P[len(pairs) // 10:]
    mixes = {(pairs[i].dataset, m) for i in train for m in (pairs[i].id_a, pairs[i].id_b)}
    seen_T = np.array([(pairs[i].dataset, pairs[i].id_a) in mixes and (pairs[i].dataset, pairs[i].id_b) in mixes
                       for i in test_T])
    log = mm.TsvLog(LOG_DIR / "train_log.tsv", mm.LOG_COLUMNS, variant="odour", repeat=split, fold=0)
    info = mm.train_v2("odour", [pairs[i] for i in train], dev, seed=split, stages="AB",
                       predict_pairs=[pairs[i] for i in test_T], log=log)

    def lined(d_P):
        a, c = evaluate.fit_line(d_P[train], y_P[train])
        return a * d_P[test_T] + c
    pred = {ROWS[0]: lined(d_b1_P), ROWS[1]: lined(d_cls_P)}
    for stage, row in zip("AB", ROWS[2:]):
        a, c = info[stage]["line"]                                # fitted on the inner validation pairs
        pred[row] = a * np.array(info[stage]["pred"]) + c
    return {"split": split, "test": test_T.tolist(), "seen": seen_T.tolist(),
            "best_epoch": {s: info[s]["epochs"] for s in "AB"}, "pred": {k: v.tolist() for k, v in pred.items()}}


def mixtures_seen(pairs, test_T) -> np.ndarray:
    """Per test pair: how many of its two mixtures (0, 1, 2) occur in a training pair (all non-test pairs)."""
    train = set(range(len(pairs))) - set(int(i) for i in test_T)
    mixes = {(pairs[i].dataset, m) for i in train for m in (pairs[i].id_a, pairs[i].id_b)}
    return np.array([((pairs[i].dataset, pairs[i].id_a) in mixes) + ((pairs[i].dataset, pairs[i].id_b) in mixes)
                     for i in test_T])


def change_table(splits: list, pairs, y_P) -> tuple:
    """-> (header, rows, notes, {(subset, row): (mean RMSE, mean r, splits with r up)}); changes from BASE."""
    seen = [mixtures_seen(pairs, np.array(s["test"])) for s in splits]
    count = {k: [int((n_T == k).sum()) for n_T in seen] for k in (0, 1, 2)}
    notes = [f"test pairs per split: {len(splits[0]['test'])}; both mixtures seen {min(count[2])}-{max(count[2])}, "
             f"neither seen {min(count[0])}-{max(count[0])}, one seen {min(count[1])}-{max(count[1])} "
             f"({sum(count[1])} over all splits, counted only in 'all test pairs')"]
    table = {}
    for subset, kinds in SUBSETS.items():
        for row in ROWS:
            rmse, r = [], []
            for s, n_T in zip(splits, seen):
                mask_T = np.isin(n_T, kinds)
                y_T = y_P[np.array(s["test"])][mask_T]
                sc, base = (metrics.score(np.array(s["pred"][k])[mask_T], y_T) for k in (row, BASE))
                rmse.append(sc["rmse"] - base["rmse"] if row != BASE else sc["rmse"])
                r.append(sc["r"] - base["r"] if row != BASE else sc["r"])
            table[(subset, row)] = (np.mean(rmse), np.mean(r), int(np.sum(np.array(r) > 0)))
    header = [""] + [f"{s} {m}" for s in SUBSETS for m in ("RMSE", "r")]
    rows = [["baseline: " + BASE + ", absolute"] + [f"{table[(s, BASE)][i]:.3f}" for s in SUBSETS for i in (0, 1)]]
    for row in ROWS[1:]:
        rows.append([row] + [x for s in SUBSETS for x in (f"{table[(s, row)][0]:+.3f}",
                                                          f"{table[(s, row)][1]:+.3f} ({table[(s, row)][2]}/{len(splits)})")])
    return header, rows, notes, table


def summarise(rep: Report, splits: list, pairs, y_P) -> None:
    rep.section("2. Change from B1 MIST frozen on the same test pairs, mean over splits (RMSE / r)")
    rep.line("RMSE change: + is worse.  r change: - is worse.  In brackets: splits (of n) where the r change is > 0.")
    header, rows, notes, table = change_table(splits, pairs, y_P)
    for line in notes:
        rep.line(line)
    rep.table(header, rows)
    rep.value("summary", {f"{s} | {r}": list(v) for (s, r), v in table.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    rep = Report("eval_mist_random_split", vars(args))
    fs, pairs = features.load("mist"), data.load_train()
    y_P = evaluate.labels(pairs)
    d_b1_P = mixture.mean_cosine_distance(pairs, fs.cids_M, fs.probs_MxW)
    d_cls_P = mixture.paper_distance(pairs, fs.cids_M, fs.emb_MxE)
    dev = device.pick_device(min_free_gb=4)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "train_log.tsv").unlink(missing_ok=True)
    rep.section("1. Splits")
    rep.line(f"device {dev}; {len(pairs)} training pairs; per split 10% test at random, 90% train; "
             f"B2-ft version 2 settings: {mm.SETTINGS_V2}")
    rep.line(f"per-epoch log: {LOG_DIR.relative_to(REPO)}/train_log.tsv")
    splits = []
    for split in range(1, args.splits + 1):
        splits.append(run_split(split, pairs, d_b1_P, d_cls_P, y_P, dev))
        rep.line(f"split {split}: best epoch stage A {splits[-1]['best_epoch']['A']}, stage B {splits[-1]['best_epoch']['B']}")
    rep.value("splits", splits)
    summarise(rep, splits, pairs, y_P)
    rep.save()


if __name__ == "__main__":
    main()
