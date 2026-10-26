"""Train all required MNIST models and controlled ablations, with resumable runs.

Examples:
  python train.py --suite main --device cuda --output /content/drive/MyDrive/CS5494/runs
  python train.py --suite all --device cuda --output /content/drive/MyDrive/CS5494/runs
  python train.py --suite main --smoke --device cpu --output smoke_runs
Checkpoints are our own locally produced files; never load untrusted .pt files.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import load_data, preprocess
from models import Discriminator, FVSBN, Generator, MADE


@dataclass
class Experiment:
    name: str
    model: str
    epochs: int
    lr: float
    n_train: int = 50000
    batch_size: int = 256
    hidden: int = 512
    depth: int = 1
    seed: int = 5494


MAIN = [Experiment("fvsbn_main", "fvsbn", 30, .001),
        Experiment("made_main", "made", 40, .001),
        Experiment("gan_main", "gan", 50, .0002)]
ABLATIONS = [Experiment("made_depth2", "made", 40, .001, depth=2),
             Experiment("made_10k", "made", 40, .001, n_train=10000),
             Experiment("gan_lr1e4", "gan", 50, .0001)]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(data, temporary)
    temporary.replace(path)


def state_on_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_grid(samples, path, columns=8, title=None):
    samples = samples.detach().cpu().reshape(-1, 28, 28).numpy()
    rows = math.ceil(len(samples) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.05, rows * 1.05), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, sample in zip(axes.flat, samples):
        ax.imshow(sample, cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(pad=.15, rect=(0, 0, 1, .95 if title else 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def generate(model, kind, n, device, seed):
    model.eval()
    random = torch.Generator(device=device).manual_seed(seed)
    if kind == "gan":
        return (model(torch.randn(n, 100, device=device, generator=random)) + 1) / 2
    return model.sample(torch.rand(n, 784, device=device, generator=random))


@torch.no_grad()
def nll(model, data, device):
    model.eval()
    total = 0.
    for (batch,) in DataLoader(TensorDataset(data), batch_size=512):
        batch = batch.to(device)
        total += F.binary_cross_entropy_with_logits(model(batch), batch, reduction="sum").item()
    return total / len(data)


def plot_history(directory, history, kind):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    if kind == "gan":
        for key, label in [("g_loss", "Generator non-saturating loss"), ("d_loss", "Discriminator loss")]:
            axes[0].plot(epochs, [row[key] for row in history], label=label)
        for key, label in [("d_real", "D(real)"), ("d_fake", "D(fake)")]:
            axes[1].plot(epochs, [row[key] for row in history], label=label)
        axes[1].set(ylabel="Mean discriminator probability", ylim=(0, 1))
        axes[0].set(ylabel="Mean BCE objective (not likelihood)")
    else:
        axes[0].plot(epochs, [row["train_nll"] for row in history], label="Training (online batches)")
        axes[0].plot(epochs, [row["validation_nll"] for row in history], label="Held-out validation")
        axes[0].set(ylabel="NLL (nats/image)")
        axes[1].plot(epochs, [row["optimization_seconds"] for row in history], label="Optimization time")
        axes[1].set(ylabel="Seconds per epoch")
    for ax in axes:
        ax.set(xlabel="Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(directory / "learning_curves.png", dpi=170)
    plt.close(fig)


def run_experiment(config, data, output, device, smoke=False):
    directory = output / config.name
    directory.mkdir(parents=True, exist_ok=True)
    configuration = {**asdict(config), "smoke_test": smoke, "split_seed": 5494,
                     "binary_threshold": .5, "schema_version": 1}
    config_path = directory / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError(f"Config differs from saved run {directory}; choose a new output directory")
    else:
        atomic_json(config_path, configuration)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("complete"):
            print(config.name, "already complete; retained without retraining", flush=True)
            return summary
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    is_gan = config.model == "gan"
    train = preprocess(data["train"][:config.n_train], binary=not is_gan)
    validation = preprocess(data["validation"][:256] if smoke else data["validation"], binary=True)
    test = preprocess(data["test"][:256] if smoke else data["test"], binary=True)
    if is_gan:
        model, discriminator = Generator().to(device), Discriminator().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(.5, .999))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(.5, .999))
    else:
        model = (FVSBN() if config.model == "fvsbn" else MADE(hidden=config.hidden, depth=config.depth)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history, snapshots, best_state = [], [], None
    best_validation, best_epoch = float("inf"), 0
    first_epoch, previous_wall = 1, 0.
    checkpoint_path = directory / "last.pt"
    if checkpoint_path.exists():
        # Only read checkpoints generated in this run directory by this script.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if is_gan:
            discriminator.load_state_dict(checkpoint["discriminator"])
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        history, snapshots = checkpoint["history"], checkpoint["snapshots"]
        best_state, best_validation, best_epoch = checkpoint["best_state"], checkpoint["best_validation"], checkpoint["best_epoch"]
        first_epoch = checkpoint["epoch"] + 1
        previous_wall = checkpoint["wall_seconds"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(config.name, "resuming at epoch", first_epoch, flush=True)
    started = time.perf_counter()
    for epoch in range(first_epoch, config.epochs + 1):
        loader_generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(TensorDataset(train), batch_size=config.batch_size, shuffle=True,
                            generator=loader_generator, num_workers=0, pin_memory=device.type == "cuda")
        model.train()
        total, g_total, d_total, real_total, fake_total = 0., 0., 0., 0., 0.
        synchronize(device)
        start = time.perf_counter()
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            n = len(batch)
            if is_gan:
                discriminator.train()
                for p in discriminator.parameters():
                    p.requires_grad_(True)
                d_optimizer.zero_grad(set_to_none=True)
                noise = torch.randn(n, 100, device=device)
                fake = model(noise).detach()
                real_logits, fake_logits = discriminator(batch), discriminator(fake)
                d_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                d_loss += F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                d_loss.backward()
                d_optimizer.step()
                # Freeze D parameters while propagating its input gradient into G.
                for p in discriminator.parameters():
                    p.requires_grad_(False)
                optimizer.zero_grad(set_to_none=True)
                generated = model(torch.randn(n, 100, device=device))
                logits = discriminator(generated)
                g_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
                g_loss.backward()
                optimizer.step()
                g_total += g_loss.item() * n
                d_total += d_loss.item() * n
                real_total += real_logits.sigmoid().detach().sum().item()
                fake_total += fake_logits.sigmoid().detach().sum().item()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(batch), batch, reduction="none").sum(1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite autoregressive training loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * n
        synchronize(device)
        seconds = time.perf_counter() - start
        row = {"epoch": epoch, "optimization_seconds": seconds}
        if is_gan:
            row.update(g_loss=g_total/len(train), d_loss=d_total/len(train),
                       d_real=real_total/len(train), d_fake=fake_total/len(train))
        else:
            validation_score = nll(model, validation, device)
            row.update(train_nll=total/len(train), validation_nll=validation_score)
            if validation_score < best_validation:
                best_validation, best_epoch, best_state = validation_score, epoch, state_on_cpu(model)
        history.append(row)
        # Same random values at every stage: changes reflect the model, not the seed.
        snapshots.append(generate(model, config.model, 4, device, config.seed + 1000).cpu())
        checkpoint = {"epoch": epoch, "model": state_on_cpu(model), "optimizer": optimizer.state_dict(),
                      "history": history, "snapshots": snapshots, "best_state": best_state,
                      "best_validation": best_validation, "best_epoch": best_epoch,
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                      "wall_seconds": previous_wall + time.perf_counter() - started}
        if is_gan:
            checkpoint.update(discriminator=state_on_cpu(discriminator), d_optimizer=d_optimizer.state_dict())
        atomic_checkpoint(checkpoint_path, checkpoint)
        atomic_json(directory / "history.json", history)
        print(config.name, json.dumps(row), flush=True)
    if not is_gan:
        model.load_state_dict(best_state)
    model.eval()
    atomic_checkpoint(directory / "selected_model.pt", {"model": state_on_cpu(model), "config": configuration})
    # Final test likelihood is evaluated once after validation-based selection.
    test_nll = None if is_gan else nll(model, test, device)
    n_samples = 16 if smoke else 64
    _ = generate(model, config.model, 2, device, config.seed + 3000)  # inference warmup
    inference = []
    for repeat in range(3):
        synchronize(device)
        start = time.perf_counter()
        samples = generate(model, config.model, n_samples, device, config.seed + 2000)
        synchronize(device)
        inference.append(time.perf_counter() - start)
    np.save(directory / "samples.npy", samples.cpu().numpy())
    save_grid(samples, directory / "samples.png", title=config.name + (" (SMOKE TEST)" if smoke else ""))
    stages = np.unique(np.linspace(0, len(snapshots) - 1, min(5, len(snapshots))).round().astype(int))
    progress = torch.cat([snapshots[i] for i in stages])
    save_grid(progress, directory / "generation_progress.png", columns=4,
              title="Fixed randomness; epochs " + ", ".join(str(int(i)+1) for i in stages))
    plot_history(directory, history, config.model)
    summary = {"complete": True, "smoke_test": smoke, "config": configuration,
               "selection": "final fixed epoch" if is_gan else "minimum validation NLL",
               "selected_epoch": config.epochs if is_gan else best_epoch,
               "parameters_allocated": sum(p.numel() for p in model.parameters()),
               "parameters_active": sum(int(m.mask.sum()) + m.out_features for m in model.modules() if hasattr(m, "mask")) if not is_gan else sum(p.numel() for p in model.parameters()),
               "n_validation": len(validation) if not is_gan else None, "n_test": len(test) if not is_gan else None,
               "validation_nll_nats": None if is_gan else best_validation, "test_nll_nats": test_nll,
               "test_bits_per_dimension": None if is_gan else test_nll / (784 * math.log(2)),
               "training_optimization_seconds": sum(row["optimization_seconds"] for row in history),
               "run_wall_seconds": previous_wall + time.perf_counter() - started,
               "inference_samples_per_repeat": n_samples, "inference_seconds": inference,
               "inference_seconds_median": float(np.median(inference)),
               "sample_seed": config.seed + 2000, "progress_epochs": [int(i)+1 for i in stages],
               "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "python": platform.python_version(), "torch": torch.__version__,
               "sample_sha256": hashlib.sha256((directory / "samples.npy").read_bytes()).hexdigest()}
    atomic_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["main", "ablations", "all"], default="main")
    parser.add_argument("--only", help="One experiment name; restricts the selected suite")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--smoke", action="store_true", help="512 training images, 2 epochs: correctness only")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Select a GPU runtime in Colab, or pass --device cpu.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = load_data(args.data)
    configurations = copy.deepcopy(MAIN if args.suite == "main" else ABLATIONS if args.suite == "ablations" else MAIN + ABLATIONS)
    if args.only:
        configurations = [c for c in configurations if c.name == args.only]
        if not configurations:
            raise ValueError("--only is not in the selected suite")
    if args.smoke:
        for c in configurations:
            c.epochs, c.n_train, c.hidden = 2, 512, 64
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [run_experiment(c, data, args.output, device, args.smoke) for c in configurations]
    atomic_json(args.output / ("suite_" + args.suite + ("_" + args.only if args.only else "") + ".json"), summaries)
    print("Finished", len(summaries), "experiments. Results:", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
