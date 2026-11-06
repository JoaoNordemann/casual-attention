import torch
import pytest
from src.attention import CausalSelfAttention


def test_equivalence():
    torch.manual_seed(0)
    m = CausalSelfAttention(d_model=64, n_heads=8, max_seq_len=128, use_flash=False).eval()
    x = torch.randn(2, 32, 64)

    out_manual = m(x)
    m.use_flash = True
    out_sdpa = m(x)
    assert torch.allclose(out_manual, out_sdpa, atol=1e-5), "manual != SDPA"


def test_causality():
    """Alterar tokens futuros não pode mudar saídas passadas."""
    torch.manual_seed(0)
    m = CausalSelfAttention(d_model=64, n_heads=8, max_seq_len=128).eval()
    x = torch.randn(1, 32, 64)
    y1 = m(x)

    x2 = x.clone()
    x2[:, 16:] = torch.randn_like(x2[:, 16:])
    y2 = m(x2)

    assert torch.allclose(y1[:, :16], y2[:, :16], atol=1e-6), "vazamento do futuro!"
    assert not torch.allclose(y1[:, 16:], y2[:, 16:])


def test_output_shape():
    m = CausalSelfAttention(d_model=64, n_heads=8, max_seq_len=128).eval()
    x = torch.randn(3, 20, 64)
    assert m(x).shape == x.shape


def test_eval_is_deterministic_with_dropout():
    """Com dropout > 0, .eval() desliga o dropout e a saída é idêntica entre chamadas."""
    m = CausalSelfAttention(d_model=64, n_heads=8, max_seq_len=128, dropout=0.5).eval()
    x = torch.randn(2, 32, 64)
    assert torch.equal(m(x), m(x))


@pytest.mark.parametrize("use_flash", [False, True])
def test_train_dropout_is_stochastic(use_flash):
    """Em .train() com dropout > 0, duas chamadas diferem; com a mesma seed, repetem."""
    m = CausalSelfAttention(
        d_model=64, n_heads=8, max_seq_len=128, dropout=0.5, use_flash=use_flash
    ).train()
    x = torch.randn(2, 32, 64)

    torch.manual_seed(1)
    a = m(x)
    b = m(x)
    assert not torch.allclose(a, b), "dropout não está ativo em train()"

    torch.manual_seed(1)
    assert torch.equal(a, m(x)), "mesma seed deveria reproduzir a máscara de dropout"


def test_train_without_dropout_is_deterministic():
    m = CausalSelfAttention(d_model=64, n_heads=8, max_seq_len=128, dropout=0.0).train()
    x = torch.randn(2, 32, 64)
    assert torch.equal(m(x), m(x))


@pytest.mark.parametrize("use_flash", [False, True])
def test_all_params_receive_gradients(use_flash):
    m = CausalSelfAttention(
        d_model=64, n_heads=8, max_seq_len=128, use_flash=use_flash
    ).train()
    x = torch.randn(2, 32, 64, requires_grad=True)

    m(x).pow(2).mean().backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name}: grad is None"
        assert torch.isfinite(p.grad).all(), f"{name}: grad com NaN/inf"
        assert p.grad.abs().sum() > 0, f"{name}: grad todo zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()    