"""Causal, normalized-density, gradient and GAN-update checks on small models."""
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from models import FVSBN, MADE, Generator, Discriminator


def run_checks():
    torch.manual_seed(5494)
    checks = []
    for model in [FVSBN(6), MADE(6, hidden=12, depth=1), MADE(6, hidden=12, depth=2)]:
        model = model.double()
        x = torch.rand(6, dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(lambda z: model(z[None])[0], x)
        assert torch.triu(jacobian).abs().max().item() == 0, "Current/future-pixel leakage"
        for i in range(6):
            changed = x.detach().clone()
            changed[i:] = 1 - changed[i:]
            torch.testing.assert_close(model(x[None])[:, :i+1], model(changed[None])[:, :i+1])
        configurations = torch.tensor(list(itertools.product([0., 1.], repeat=6)), dtype=torch.float64)
        logp = -F.binary_cross_entropy_with_logits(model(configurations), configurations, reduction="none").sum(1)
        torch.testing.assert_close(logp.exp().sum(), torch.tensor(1., dtype=torch.float64))
        # Finite difference through the network, independent of the mask proof.
        assert torch.autograd.gradcheck(model, (x[None],), eps=1e-6, atol=1e-4)
        model(configurations).sum().backward()
        for module in model.modules():
            if hasattr(module, "mask"):
                assert (module.weight.grad[module.mask == 0] == 0).all()
        random_values = torch.rand(12, 6, dtype=torch.float64)
        generated = model.sample(random_values)
        assert ((generated == 0) | (generated == 1)).all()
        if isinstance(model, FVSBN):
            reference = torch.zeros_like(random_values)
            for i in range(6):
                reference[:, i] = (random_values[:, i] < model(reference)[:, i].sigmoid()).double()
            torch.testing.assert_close(generated, reference)
        checks.append({"model": type(model).__name__, "layers": len(list(model.modules())),
                       "causality": True, "joint_probability_sum": logp.exp().sum().item(),
                       "finite_difference": True, "masked_gradients_zero": True})
    g, d = Generator(), Discriminator()
    noise = torch.randn(8, 100)
    fake = g(noise)
    assert fake.shape == (8, 784) and fake.min() >= -1 and fake.max() <= 1
    d_loss = F.binary_cross_entropy_with_logits(d(fake.detach()), torch.zeros(8))
    d_loss.backward()
    assert all(p.grad is None for p in g.parameters()), "D step must detach G output"
    d.zero_grad(set_to_none=True)
    for p in d.parameters():
        p.requires_grad_(False)
    g_loss = F.binary_cross_entropy_with_logits(d(g(noise)), torch.ones(8))
    g_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())
    assert all(p.grad is None for p in d.parameters())
    return {"passed": True, "autoregressive_checks": checks, "gan_gradient_routing": True}


if __name__ == "__main__":
    result = run_checks()
    Path("model_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
