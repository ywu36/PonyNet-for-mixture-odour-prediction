# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes
#   R = training rows (20,578)      T = T46 rows (46)       L = molecules of the team's list (204)
#   K = rating slots (369)          Q = ratings per slot (21)
# ---------------------------------------------------------------------------------------------
"""Layout of the feature row that team D2Smell's XGBoost gets: column ranges and what fills them.

Builds the team's matrix with the team's own code (evaluation/D2Smell/ablation.py) and checks, on
every training row and every T46 row, what findings/finding_2.html says about the layout: the
column ranges, that a ratings column is one identity column times a fixed number, which molecules
the identity columns mark, and which odour words are kept. No label is used.

Env:     d2smell (the team's pinned environment, see evaluation/check_d2smell.py)
Usage:   python evaluation/D2Smell/check_d2smell_features.py
Writes:  results/check_d2smell_features/<date_time>.{txt,json}
"""
import ablation
import numpy as np
import pandas as pd
from mixdist.data import locate
from mixdist.report import Report

HALF = 184                                                   # the team's code halves its 369 columns: int(369 / 2)


def layout(rep: Report) -> None:
    rep.section("1. Columns of one row (one pair of mixtures), numbered from 1")
    (h0, h1), (r0, r1), (t0, t1), (o0, o1) = (ablation.BLOCKS[b] for b in ablation.TEAM_BLOCKS)
    n_list = (h1 - h0 + t1 - t0) // 2
    rep.table(["columns", "count", "group", "content"], [
        [f"{h0 + 1}-{n_list}", str(n_list), "identity", "mixture 1: 1 if molecule j of the list is in it"],
        [f"{n_list + 1}-{h1}", str(h1 - n_list), "identity", f"mixture 2, molecules 1-{h1 - n_list} of the list"],
        [f"{r0 + 1}-{r1}", str(r1 - r0), "ratings", f"{h1} slots of {(r1 - r0) // h1}, one slot per identity column above"],
        [f"{t0 + 1}-{t1}", str(t1 - t0), "identity", f"mixture 2, molecules {h1 - n_list + 1}-{n_list} (no rating slots)"],
        [f"{o0 + 1}-{o1}", str(o1 - o0), "odour", "per odour word, (profile of mixture 1 - profile of mixture 2) squared"]])
    rep.line("The team's code takes the last 64 columns of its 433 inputs as the odour block (it has 25), so 39 identity")
    rep.line("columns of mixture 2 stay next to the odour columns and get no rating slots.")


def rating_slots(rep: Report, source, td: ablation.TeamData) -> None:
    rep.section("2. Ratings columns")
    ratings = pd.read_csv(source / "output/percept_scaled.csv")
    table_LxQ = ratings.values[:, 1:22].astype(np.float32)
    row_K = np.array([k if k < HALF else k - HALF for k in range(2 * HALF)])       # ratings-table row behind slot k
    rep.line(f"ratings table: {len(ratings)} molecules x {table_LxQ.shape[1]} ratings of a single molecule, scaled to 0-1: "
             + ", ".join(c.lower() for c in ratings.columns[1:22]))
    list_cids = sorted(pd.read_csv(source / "data/processed/CID.csv")["CID"].astype(int).tolist())
    measured = set(pd.read_csv(source / "data/processed/percept.csv")["CID"].astype(int))
    rep.line(f"measured ratings (the team's percept.csv) for {len(measured & set(list_cids))} of the {len(list_cids)} list molecules; "
             "the others are predicted from chemical descriptors by the team's train/percept_regressor.py")
    for name, x in (("training rows", td.x_Rx8240), ("T46 rows", td.x_Tx8240)):
        (h0, h1), (r0, r1) = ablation.BLOCKS["identity_head"], ablation.BLOCKS["ratings"]
        slots = x[:, r0:r1].reshape(len(x), h1 - h0, -1)
        on = x[:, h0:h1] == 1
        for k in range(2 * HALF):
            assert not slots[~on[:, k], k].any(), "a slot is filled without its identity column"
            assert np.allclose(slots[on[:, k], k], table_LxQ[row_K[k]], atol=1e-6), "a slot does not hold one fixed row"
        assert not slots[:, 2 * HALF:].any(), "the last slot is expected to stay empty"
        rep.line(f"check  {name} ({len(x)}): slot k is zero when identity column k is 0, and holds row k of the ratings table "
                 f"(row k - {HALF} from slot {HALF + 1} on) when it is 1; slot {h1 - h0} is never filled")
    rep.line("so every ratings column equals one identity column times a fixed number")
    rated = ratings["CID"].astype(int).tolist()
    own = sum((list_cids + list_cids)[k] == rated[row_K[k]] for k in range(2 * HALF))
    rep.line(f"slots that hold the ratings of their own molecule: {own} of {2 * HALF} (the table is in another order)")
    rep.line(f"fixed numbers that are exactly 0 (columns that are always 0): {int((table_LxQ[row_K] == 0).sum())} of {2 * HALF * 21}")


def identity(rep: Report, source, td: ablation.TeamData) -> None:
    rep.section("3. Identity columns")

    def bits(x) -> tuple:                                    # -> (mixture 1, mixture 2), each rows x 204
        (h0, h1), (t0, t1) = ablation.BLOCKS["identity_head"], ablation.BLOCKS["identity_tail"]
        both = np.concatenate([x[:, h0:h1], x[:, t0:t1]], axis=1)
        return both[:, :both.shape[1] // 2], both[:, both.shape[1] // 2:]

    n_orig = len(td.pairs_O)
    a_RxL, b_RxL = bits(td.x_Rx8240)
    a_TxL, b_TxL = bits(td.x_Tx8240)
    in_train_L = (a_RxL[:n_orig].sum(0) + b_RxL[:n_orig].sum(0)) > 0
    in_t46_L = (a_TxL.sum(0) + b_TxL.sum(0)) > 0
    rep.line(f"molecules of the list of {len(in_train_L)} that occur in the {n_orig} training pairs: {int(in_train_L.sum())}")
    rep.line(f"molecules marked in T46 rows: {int(in_t46_L.sum())}, of which not in any training pair: {int((in_t46_L & ~in_train_L).sum())}")
    defs = pd.read_csv(source / "data/raw/mixtures/Mixure_Definitions_test_set.csv")
    in_file = defs.iloc[:, 1:].notna().sum(axis=1)
    marked = np.concatenate([a_TxL.sum(1), b_TxL.sum(1)])
    rep.line(f"molecules per T46 mixture: {in_file.min()}-{in_file.max()} in the definitions file, "
             f"{int(marked.min())}-{int(marked.max())} marked in the identity columns")
    rep.line("(the T46 definitions file has no Dataset column and the team's code skips two columns, so the first")
    rep.line(" molecule of every T46 mixture is not marked; the odour columns come from the saved profile file instead)")
    sizes = np.concatenate([a_RxL[:n_orig].sum(1), b_RxL[:n_orig].sum(1)])
    rep.line(f"molecules per training mixture: {int(sizes.min())}-{int(sizes.max())}, median {np.median(sizes):.0f}")


def odour_words(rep: Report, source) -> None:
    rep.section("4. Odour words of the odor-pair network (AUROC as listed in the team's labels.csv)")
    labels = pd.read_csv(source / "odor-pair-model/labels.csv", index_col=0, skipinitialspace=True)
    words, auroc = labels.loc["Note"].str.strip().tolist(), labels.loc["AUROC"].astype(float).tolist()
    dropped = [i for i in range(len(words)) if i + 1 in ablation.DROPPED_WORDS]
    kept = [i for i in range(len(words)) if i + 1 not in ablation.DROPPED_WORDS]
    rep.line(f"kept {len(kept)}: " + ", ".join(words[i] for i in kept))
    rep.line(f"dropped {len(dropped)}: " + ", ".join(f"{words[i]} ({auroc[i]:.2f})" for i in dropped))
    rep.line(f"lowest AUROC among the kept words: {min(auroc[i] for i in kept):.2f}; highest among the dropped: "
             f"{max(auroc[i] for i in dropped):.2f}")


def main():
    rep = Report("check_d2smell_features")
    source = locate("olfboost_repo", needs="output")
    rep.line(f"team code  {source}")
    td = ablation.load_team_data(source)
    layout(rep)
    rating_slots(rep, source, td)
    identity(rep, source, td)
    odour_words(rep, source)
    rep.save()


if __name__ == "__main__":
    main()
