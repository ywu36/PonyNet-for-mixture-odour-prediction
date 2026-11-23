"""O1 model properties (mixdist/o1.py): the constraints in design/o1_design.html hold by construction."""
import pytest

torch = pytest.importorskip("torch")
from mixdist.o1 import O1, asymmetric_loss, pad  # noqa: E402


def _model():
    torch.manual_seed(0)
    return O1(n_emb=8, n_receptors=5, n_words=3, n_groups=2)


def test_symmetric_and_identical_mixtures():
    m, emb = _model(), torch.randn(6, 8)
    idxA, maskA = pad([[0, 1, 2], [3]], "cpu")
    idxB, maskB = pad([[3, 4], [3]], "cpu")
    rA, _ = m.receptors(emb, idxA, maskA)
    rB, _ = m.receptors(emb, idxB, maskB)
    assert torch.allclose(m.distance(rA, rB), m.distance(rB, rA))
    d_same = m.distance(rA, rA)                                  # identical mixtures -> sigmoid(beta)
    assert torch.allclose(d_same, torch.sigmoid(m.beta).expand_as(d_same), atol=1e-3)


def test_order_invariant_and_padding_ignored():
    m, emb = _model(), torch.randn(6, 8)
    r1, _ = m.receptors(emb, *pad([[0, 1, 2]], "cpu"))
    r2, _ = m.receptors(emb, *pad([[2, 0, 1]], "cpu"))
    r3, _ = m.receptors(emb, *pad([[2, 0, 1], [0, 1, 2, 3, 4]], "cpu"))
    assert torch.allclose(r1, r2, atol=1e-6) and torch.allclose(r1, r3[:1], atol=1e-6)


def test_competition_formula_and_saturation():
    m = _model()
    with torch.no_grad():                                        # binding x = softplus(z), efficacy e = sigmoid(z)
        m.head_x.weight.zero_(); m.head_e.weight.zero_()
        m.head_x.bias.fill_(torch.log(torch.expm1(torch.tensor(2.0))))  # x = 2 for every molecule
        m.head_e.bias.fill_(20.0)                                # e = 1
    emb = torch.zeros(3, 8)
    r1, _ = m.receptors(emb, *pad([[0]], "cpu"))
    r2, _ = m.receptors(emb, *pad([[0, 1]], "cpu"))
    assert torch.allclose(r1, torch.full_like(r1, 2 / 3), atol=1e-4)      # 2 / (1 + 2)
    assert torch.allclose(r2, torch.full_like(r2, 4 / 5), atol=1e-4)      # saturates: 0.8, not 1.33


def test_non_binding_molecule_changes_nothing():
    m = _model()
    with torch.no_grad():
        m.head_x.weight.zero_(); m.head_x.weight[:, 0] = 30.0    # binding depends on feature 0
        m.head_x.bias.fill_(-15.0)
    emb = torch.zeros(2, 8); emb[0, 0] = 1.0                      # molecule 0 binds, molecule 1 does not (x ~ 0)
    r1, _ = m.receptors(emb, *pad([[0]], "cpu"))
    r2, _ = m.receptors(emb, *pad([[0, 1]], "cpu"))
    assert torch.allclose(r1, r2, atol=1e-5)


def test_asymmetric_loss_spares_unlisted_labels():
    logits = torch.tensor([[2.0, 0.0]])                          # second label predicted p = 0.5, not listed
    y = torch.tensor([[1.0, 0.0]])
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
    assert asymmetric_loss(logits, y) < plain
