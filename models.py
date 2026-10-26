"""Three required generative models; no torchvision dependency."""
import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class FVSBN(nn.Module):
    """p(x)=product_i Bernoulli(x_i; sigmoid(b_i + sum_{j<i} W_ij*x_j))."""
    def __init__(self, dimensions=784):
        super().__init__()
        self.dimensions = dimensions
        self.linear = MaskedLinear(dimensions, dimensions,
                                   torch.tril(torch.ones(dimensions, dimensions), diagonal=-1))

    def forward(self, x):
        return self.linear(x)

    @torch.no_grad()
    def sample(self, uniforms):
        # Incremental exact ancestral sampling avoids 784 full matrix multiplies.
        n, dimensions = uniforms.shape
        samples = torch.zeros_like(uniforms)
        logits = self.linear.bias.unsqueeze(0).expand(n, -1).clone()
        weights = self.linear.weight * self.linear.mask
        for i in range(dimensions):
            samples[:, i] = (uniforms[:, i] < logits[:, i].sigmoid()).to(samples.dtype)
            logits += samples[:, i:i+1] * weights[:, i].unsqueeze(0)
        return samples


class MADE(nn.Module):
    """Natural-order masked neural autoregressor, with fixed connectivity degrees.

    Hidden edges require m(previous)<=m(next); output edges require m(hidden)<i.
    Consequently output i depends only on pixels j<i. No input copying is possible.
    """
    def __init__(self, dimensions=784, hidden=512, depth=1, mask_seed=5494):
        super().__init__()
        self.dimensions = dimensions
        generator = torch.Generator().manual_seed(mask_seed)
        degrees = [torch.arange(1, dimensions + 1)]
        degrees += [torch.randint(1, dimensions, (hidden,), generator=generator) for _ in range(depth)]
        degrees += [torch.arange(1, dimensions + 1)]
        modules = []
        for i, (previous, following) in enumerate(zip(degrees, degrees[1:])):
            output = i == len(degrees) - 2
            mask = following[:, None] > previous[None, :] if output else following[:, None] >= previous[None, :]
            modules.append(MaskedLinear(len(previous), len(following), mask))
            if not output:
                modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def sample(self, uniforms):
        samples = torch.zeros_like(uniforms)
        for i in range(self.dimensions):
            probability = self(samples)[:, i].sigmoid()
            samples[:, i] = (uniforms[:, i] < probability).to(samples.dtype)
        return samples


class Generator(nn.Module):
    def __init__(self, latent=100):
        super().__init__()
        self.latent = latent
        self.net = nn.Sequential(nn.Linear(latent, 256), nn.LeakyReLU(.2),
                                 nn.Linear(256, 512), nn.LeakyReLU(.2),
                                 nn.Linear(512, 784), nn.Tanh())

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 512), nn.LeakyReLU(.2),
                                 nn.Linear(512, 256), nn.LeakyReLU(.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)  # Logits, not sigmoid probabilities.
