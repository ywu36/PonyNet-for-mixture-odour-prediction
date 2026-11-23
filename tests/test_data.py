"""Step 1 checks: the loader reproduces the source repository's published numbers.

Run with pytest, or directly:  python3 tests/test_data.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import data  # noqa: E402


def test_train_counts():
    base = data.load_train(extra=False)
    assert len(base) == 730
    assert Counter(p.dataset for p in base) == {
        "Bushdid": 260, "Snitz 1": 147, "Snitz 2": 91, "Ravia 1": 91, "Ravia 2": 91, "Ravia 4": 50}
    full = data.load_train()
    assert Counter(p.dataset for p in full[730:]) == {"Ravia 5": 50, "Ravia 6": 37, "Metamer": 3}
    assert not {p.dataset for p in full} & {"random", "manual", "angle"}   # rule 2
    assert all(p.a and p.b and 0.0 <= p.y <= 1.0 for p in full)


def test_test_set():
    t = data.load_test()
    assert len(t) == 46 and all(p.a and p.b for p in t)


def test_validation_set():
    v = data.load_validation()
    assert len(v) == 50
    assert all(len(p.extra["ratings"]) == 16 for p in v)                    # 16 subjects
    assert all(set(r) == {1, 2} for p in v for r in p.extra["ratings"].values())
    assert all(p.jaccard == 0 for p in v)                                   # A and B share nothing


def test_every_molecule_has_smiles():
    smiles = data.load_smiles()
    for p in data.load_train() + data.load_test() + data.load_validation():
        assert (p.a | p.b) <= set(smiles)


def test_validation_is_new_mixtures_of_seen_molecules():
    train = data.load_train()
    seen_mols = set().union(*[p.a | p.b for p in train])
    seen_mixes = {m for p in train for m in (p.a, p.b)}
    v = data.load_validation()
    assert set().union(*[p.a | p.b for p in v]) <= seen_mols
    assert not {m for p in v for m in (p.a, p.b)} & seen_mixes


def _v50_scores(target):
    v = data.load_validation()
    preds = data.load_baseline_predictions("V50")
    y = [target(p) for p in v]
    return {m: data.pearson([col[(p.id_a, p.id_b)] for p in v], y) for m, col in preds.items()}


def test_pom_on_v50_against_combined_label():
    # The label the plan and the source repo (Figure 4, ExpMean_combined) both use.
    r = _v50_scores(lambda p: p.y)
    assert abs(r["POM"] - 0.4989) < 5e-4
    assert abs(r["belfaction"] - 0.5096) < 5e-4
    assert abs(r["Ensemble"] - 0.4736) < 5e-4


def test_pom_0537_is_against_repetition_1_only():
    # The plan's "POM 0.537" is reproduced only when scoring against repetition 1 alone.
    assert abs(_v50_scores(lambda p: p.extra["rep1"])["POM"] - 0.5376) < 5e-4
    assert abs(_v50_scores(lambda p: p.extra["rep2"])["POM"] - 0.3961) < 5e-4


def test_noise_ceiling_matches_source_repo():
    v = data.load_validation()
    r = data.pearson([p.extra["rep1"] for p in v], [p.extra["rep2"] for p in v])
    assert abs(r - 0.74154776) < 1e-6          # value hard-coded in pnas_repo Figure_4.ipynb


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
