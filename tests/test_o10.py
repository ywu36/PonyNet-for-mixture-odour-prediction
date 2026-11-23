"""O1.0 / O1.1 properties (mixdist/o10.py)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mixdist.o10 import O10  # noqa: E402
from mixdist.o1 import pad  # noqa: E402


def _tables():
    g = torch.Generator().manual_seed(0)
    return torch.randn(6, 4, generator=g), torch.rand(6, 4, generator=g)   # e (signed), q in [0, 1]


def test_o11_at_zero_binding_is_b1a():
    e, q = _tables()
    m = O10(4, weights=False, occupancy=True)
    with torch.no_grad():
        m.rho.zero_()
    mix = pad([[0, 1, 2]], "cpu")
    r = m.receptors(e, *mix, q_TxW=q)
    mean = e[[0, 1, 2]].mean(0, keepdim=True)
    assert torch.allclose(torch.nn.functional.normalize(r, dim=1), torch.nn.functional.normalize(mean, dim=1), atol=1e-6)


def test_o11_gain_depends_on_which_molecules_not_only_how_many():
    e, q = _tables()
    m = O10(4, weights=False, occupancy=True)
    with torch.no_grad():
        m.rho.fill_(2.0)
    gain = lambda idx: (m.receptors(e, *pad([idx], "cpu"), q_TxW=q) / e[idx].sum(0)).squeeze(0)
    assert not torch.allclose(gain([0, 1]), gain([0, 2]))        # same size, different molecule


def test_o10_gain_depends_only_on_size():
    e, _ = _tables()
    m = O10(4, weights=False)
    with torch.no_grad():
        m.rho.copy_(torch.tensor([0.0, 1.0, -1.0, 2.0]))
    gain = lambda idx: (m.receptors(e, *pad([idx], "cpu")) / e[idx].sum(0)).squeeze(0)
    assert torch.allclose(gain([0, 1]), gain([0, 2]), atol=1e-6)


def test_o12_mean_at_zero_and_max_when_large():
    e, _ = _tables()
    m = O10(4, weights=False, dominance=True)
    mix = pad([[0, 1, 2], [3, 4]], "cpu")
    with torch.no_grad():
        m.rho.zero_()
    assert torch.allclose(m.receptors(e, *mix)[0], e[[0, 1, 2]].mean(0), atol=1e-6)
    with torch.no_grad():
        m.rho.fill_(200.0)
    assert torch.allclose(m.receptors(e, *mix)[1], e[[3, 4]].max(0).values, atol=1e-3)
