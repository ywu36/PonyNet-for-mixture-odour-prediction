"""pony: featurizer, network, readout and the packaged paper model.

Run with pytest, or directly:  python tests/test_pony.py
Tests marked "needs DREAM data" are skipped when the DREAM challenge repository is not available (see README).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pony  # noqa: E402
from pony import features as fz, readout as ro  # noqa: E402

ETHYL_ACETATE, LIMONENE, VANILLIN = "CCOC(C)=O", "CC1=CCC(CC1)C(C)=C", "COc1cc(C=O)ccc1O"
HEXANAL, LINALOOL, INDOLE = "CCCCCC=O", "CC(C)=CCCC(C)(O)C=C", "c1ccc2[nH]ccc2c1"


@pytest.fixture(scope="module")
def model():
    return pony.Pony.from_pretrained()


# ------------------------------------------------------------------ featurizer
def test_featurize_shapes_and_one_hot_blocks():
    g = fz.featurize("CCO")
    assert g.node.shape == (3, fz.ATOM_DIM) and g.edge.shape == (4, fz.BOND_DIM) and g.edge_index.shape == (2, 4)
    assert (g.node.sum(1) == 6).all()                     # six one-hot blocks per atom
    assert (g.edge[:, 0] == 0).all() and (g.edge[:, 1:5].sum(1) == 1).all()


def test_featurize_same_molecule_same_graph_up_to_atom_order():
    a, b = fz.featurize("OC(C)CCc1ccccc1"), fz.featurize("c1ccc(cc1)CCC(O)C")
    rows = lambda X: sorted(map(tuple, X))  # noqa: E731
    assert rows(a.node) == rows(b.node) and rows(a.edge) == rows(b.edge) and a.edge_index.shape == b.edge_index.shape


def test_embedding_does_not_depend_on_smiles_spelling(model):
    P1, _ = model.odour_words(["OC(C)CCc1ccccc1"]); P2, _ = model.odour_words(["c1ccc(cc1)CCC(O)C"])
    assert np.allclose(P1, P2, atol=1e-5)


def test_featurize_aromatic_ring_bonds():
    g = fz.featurize("c1ccccc1")
    assert (g.edge[:, 4] == 1).all() and (g.edge[:, 5] == 1).all()


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        fz.featurize("not a molecule")


def test_featurizer_matches_open_pom_graphs():
    """Same graphs as open-POM's GraphFeaturizer, on molecules it featurized (needs features/o2_graphs.npz)."""
    path = Path(__file__).resolve().parents[1] / "features" / "o2_graphs.npz"
    if not path.exists():
        pytest.skip("features/o2_graphs.npz not available")
    z = np.load(path, allow_pickle=False)
    for i in np.random.default_rng(0).choice(len(z["smiles"]), 300, replace=False):
        g = fz.featurize(str(z["smiles"][i])); n0, n1 = z["node_off"][i], z["node_off"][i + 1]; e0, e1 = z["edge_off"][i], z["edge_off"][i + 1]
        assert np.array_equal(g.node, z["node"][n0:n1]) and np.array_equal(g.edge, z["edge"][e0:e1])
        assert np.array_equal(g.edge_index, z["edge_index"][:, e0:e1])


# ------------------------------------------------------------------ network
def test_network_parameter_count_matches_paper(model):
    assert len(model.members) == 10
    assert sum(p.numel() for p in model.members[0].net.parameters()) == 116_603   # Table A1


def test_molecule_vector_is_attention_weighted_mean_of_atom_vectors(model):
    m = model.members[0]; graphs = [fz.featurize(s) for s in (VANILLIN, LIMONENE)]
    x, e, s, d, gid = fz.batch(graphs)
    with torch.no_grad():
        H, sc, v, _ = m.net(torch.tensor(x), torch.tensor(e), torch.tensor(s), torch.tensor(d), torch.tensor(gid), 2)
    for k in range(2):
        w = torch.softmax(sc[torch.tensor(gid) == k], 0)
        assert torch.allclose(v[k], (w[:, None] * H[torch.tensor(gid) == k]).sum(0), atol=1e-4)


def test_odour_words(model):
    P, words = model.odour_words([VANILLIN, LIMONENE])
    assert P.shape == (2, 138) and len(words) == 138 and ((P > 0) & (P < 1)).all()
    assert P[0, words.index("vanilla")] > P[1, words.index("vanilla")]


# ------------------------------------------------------------------ readout and predictions
def test_predict_shape_and_rating_scale(model):
    y = model.predict([[ETHYL_ACETATE, LIMONENE], [VANILLIN]], [[HEXANAL], [VANILLIN, INDOLE, LINALOOL]])
    assert y.shape == (2,) and np.isfinite(y).all() and (-0.5 < y).all() and (y < 1.5).all()


def test_identical_mixtures_get_the_intercept(model):
    y = model.predict([[ETHYL_ACETATE, LIMONENE]], [[LIMONENE, ETHYL_ACETATE]])[0]
    c = np.mean([m.readout.c for m in model.members])
    assert abs(y - c) < 1e-3


def test_prediction_is_symmetric_and_order_free(model):
    A, B = [VANILLIN, HEXANAL, LINALOOL], [INDOLE, LIMONENE]
    y1 = model.predict([A], [B])[0]
    assert abs(y1 - model.predict([B], [A])[0]) < 1e-10
    assert abs(y1 - model.predict([A[::-1]], [B[::-1]])[0]) < 1e-10
    assert abs(y1 - model.predict([["O=CCCCCC", *A[::2]]], [B])[0]) < 1e-10   # another spelling of hexanal


def test_more_different_mixtures_rate_higher(model):
    base = [LIMONENE, LINALOOL, HEXANAL]
    near = model.predict([base], [[LIMONENE, LINALOOL]])[0]
    far = model.predict([base], [[VANILLIN, INDOLE]])[0]
    assert near < far


def test_explain_shares_sum_to_one(model):
    ex = model.explain([VANILLIN, LIMONENE, HEXANAL])
    assert len(ex["molecules"]) == 3 and abs(ex["shares"].sum() - 1) < 1e-9 and (ex["length_per_copy"] > 0).all()


def test_zero_readout_gives_equal_shares_and_unit_length(model):
    m = model.members[0]; tab = m.table([fz.featurize(s) for s in (VANILLIN, LIMONENE, HEXANAL)])
    _, beta, kappa, t, delta, gamma = ro.ReadoutParams().tensors(64); X = torch.ones(1, 3, dtype=ro.F64)
    _, pi, g = ro.mixture_points(tab, X, ro.context(tab, X), ro.strengths(tab, beta, delta, t), kappa, gamma)
    assert torch.allclose(pi, torch.full_like(pi, 1 / 3)) and abs(g.item() - 1) < 1e-12


def test_save_and_load_round_trip(model, tmp_path):
    model.save(tmp_path / "m.pt"); again = pony.Pony.from_pretrained(tmp_path / "m.pt")
    A, B = [[VANILLIN, HEXANAL]], [[LIMONENE]]
    assert np.allclose(model.predict(A, B), again.predict(A, B), atol=1e-12)


def test_empty_mixture_and_length_mismatch_raise(model):
    with pytest.raises(ValueError):
        model.predict([[]], [[VANILLIN]])
    with pytest.raises(ValueError):
        model.predict([[VANILLIN]], [])


def test_embed_and_line_give_predict(model):
    a, b = [ETHYL_ACETATE, VANILLIN], [LIMONENE, LINALOOL, INDOLE]
    P = model.embed([a, b]); L = model.line()
    assert P.shape == (len(model.members), 2, 64) and L.shape == (len(model.members), 2)
    manual = (L[:, 0] * np.linalg.norm(P[:, 0] - P[:, 1], axis=1) + L[:, 1]).mean()
    assert np.isclose(manual, model.predict([a], [b])[0], atol=1e-6)


def test_network_forward_refuses_training_mode(model):
    net = model.members[0].net
    try:
        net.train()
        x, e, s, d, g = fz.batch([fz.featurize("CCO")]); f, i = torch.float32, torch.long
        with pytest.raises(RuntimeError):
            net(torch.as_tensor(x, dtype=f), torch.as_tensor(e, dtype=f), torch.as_tensor(s, dtype=i), torch.as_tensor(d, dtype=i), torch.as_tensor(g, dtype=i), 1)
    finally:
        net.eval()


# ------------------------------------------------------------------ file readers
PAIR_CSV = ("mixture_a,mixture_b,rating\n"     # file format only; the rating is a placeholder, not a measurement
            "CCO;CC(=O)OCC;CC(C)=O;CCCCO,CC(C)CO;CCOC(C)=O;CC=O;CCCC(=O)O,0.5\n")


def test_read_pairs(tmp_path):
    from pony.data import read_pairs
    f = tmp_path / "pairs.csv"; f.write_text(PAIR_CSV)
    a, b, y = read_pairs(f)
    assert len(a[0]) == 4 and len(b[0]) == 4 and y.tolist() == [0.5]


def test_read_molecules_keeps_only_0_1_columns(tmp_path):
    from pony.data import read_molecules
    f = tmp_path / "mols.csv"; f.write_text("smiles,descriptors,fruity,sweet\nCCOC(C)=O,fruity;sweet,1,1\nc1ccc2[nH]ccc2c1,animal,0,0\n")
    smiles, labels, words = read_molecules(f)
    assert words == ["fruity", "sweet"] and labels.shape == (2, 2) and smiles[1] == INDOLE


def test_holdout_mixtures_keeps_test_mixtures_out_of_training():
    from pony.data import holdout_mixtures
    smiles, train, _ = _dream()
    a = [[smiles[c] for c in p.a] for p in train]; b = [[smiles[c] for c in p.b] for p in train]
    test = holdout_mixtures(a, b, 0.2, seed=0); key = lambda m: ";".join(sorted(m))  # noqa: E731
    in_training = {key(m) for x, z, t in zip(a, b, test) if not t for m in (x, z)}
    assert 0 < test.sum() < len(test)
    assert all(key(x) not in in_training or key(z) not in in_training for x, z, t in zip(a, b, test) if t)


# ------------------------------------------------------------------ training (needs open-POM's GS-LF molecules)
def _gslf():
    path = Path(__file__).resolve().parents[1] / "features" / "o2_graphs.npz"
    if not path.exists():
        pytest.skip("features/o2_graphs.npz not available")
    z = np.load(path, allow_pickle=False); n = int(z["n_gslf"])
    tr = np.arange(n)[~z["audit_test"]]
    return [str(s) for s in z["smiles"][tr]], z["Y"][tr], [str(s) for s in z["smiles"][n:]], [str(w) for w in z["words"]]


def test_train_network_short_run():
    from pony.train import train_network
    smiles, Y, ref, words = _gslf()
    net = train_network(smiles[:256], Y[:256], seed=0, recipe={"epochs": 2}, log=None)
    assert not net.training and not any(p.requires_grad for p in net.parameters())
    assert net.head.out_features == len(words) and sum(p.numel() for p in net.parameters()) == 116_603
    m = pony.Pony.from_networks([net], words, smiles[:256], ref[:50])
    assert m.members[0].scale > 0 and m.members[0].u0.shape == (64,) and np.isfinite(m.members[0].u0).all()


def test_calibrate_reproduces_shipped_constants(model):
    """scale and u0 computed by pony.train.calibrate equal the ones shipped with the paper's model (copy 0)."""
    from pony.train import calibrate
    smiles, _, ref, _ = _gslf()
    scale, u0 = calibrate(model.members[0].net, smiles, ref)
    assert np.isclose(scale, model.members[0].scale, rtol=1e-5)
    assert np.abs(u0 - model.members[0].u0).max() < 1e-3 * np.abs(model.members[0].u0).max()


# ------------------------------------------------------------------ the paper's numbers (needs DREAM data)
def _dream():
    try:
        from mixdist import data, evaluate
        return data.load_smiles(), data.load_train(), evaluate.held_out_sets()
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"needs DREAM data: {e}")


def test_pretrained_model_reproduces_paper_test_scores(model):
    smiles, _, held_out = _dream()
    paper = {"T46": 0.593, "V50": 0.622}
    for hs in held_out:
        pred = model.predict([[smiles[c] for c in p.a] for p in hs.pairs], [[smiles[c] for c in p.b] for p in hs.pairs])
        assert round(float(np.corrcoef(pred, hs.y_P)[0, 1]), 3) == paper[hs.name]


def test_refit_on_training_pairs_improves_on_zero_readout(model):
    smiles, train, _ = _dream()
    sub = train[:200]; A = [[smiles[c] for c in p.a] for p in sub]; B = [[smiles[c] for c in p.b] for p in sub]
    y = np.array([p.y for p in sub])
    two = pony.Pony(model.members[:2], model.words)
    saved = [m.readout for m in two.members]
    try:
        for m in two.members:
            m.readout = ro.ReadoutParams(a=1.0, delta=np.zeros(64), gamma=np.zeros(64))
        r0 = np.corrcoef(two.predict(A, B), y)[0, 1]
        two.fit(A, B, y)
        assert np.corrcoef(two.predict(A, B), y)[0, 1] > r0
    finally:
        for m, r in zip(two.members, saved):
            m.readout = r


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
