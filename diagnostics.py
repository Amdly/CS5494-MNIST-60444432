"""Auxiliary diversity evidence, not a substitute for inspecting generated images.

Fit a small classifier on the training split only, assess it on validation, and
report predicted classes/confidence of the unfiltered saved samples. Report
nearest training neighbours and pairwise distances to help inspect memorization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import load_data
from train import atomic_json, save_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(5495)
    device = torch.device(args.device)
    dataset = load_data(args.data)
    n = 1000 if args.smoke else 50000
    train = dataset["train"][:n].float() / 255
    labels = dataset["train_labels"][:n].long()
    classifier = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=.001)
    for epoch in range(1 if args.smoke else 5):
        for x, y in DataLoader(TensorDataset(train, labels), batch_size=256, shuffle=True,
                               generator=torch.Generator().manual_seed(5495 + epoch)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(classifier(x), y)
            loss.backward()
            optimizer.step()
    classifier.eval()
    nv = 256 if args.smoke else 10000
    with torch.no_grad():
        validation = dataset["validation"][:nv].float().to(device) / 255
        y = dataset["validation_labels"][:nv].to(device)
        accuracy = (classifier(validation).argmax(1) == y).float().mean().item()
        records = {}
        for summary_path in sorted(args.output.glob("*/summary.json")):
            summary = json.loads(summary_path.read_text())
            if summary["smoke_test"] != args.smoke:
                raise ValueError("Cannot mix smoke and formal results")
            directory = summary_path.parent
            samples = torch.from_numpy(np.load(directory / "samples.npy")).to(device)
            probabilities = classifier(samples).softmax(1)
            confidence, prediction = probabilities.max(1)
            counts = torch.bincount(prediction, minlength=10).cpu().tolist()
            # Use matching binary/grayscale preprocessing for nearest-neighbour analysis.
            reference = train[:summary["config"]["n_train"]]
            reference = (reference >= .5).float() if summary["config"]["model"] != "gan" else reference
            best_distance = torch.full((len(samples),), float("inf"), device=device)
            best_index = torch.zeros(len(samples), dtype=torch.long, device=device)
            for start in range(0, len(reference), 2048):
                distances = torch.cdist(samples, reference[start:start+2048].to(device)) / np.sqrt(784)
                value, index = distances.min(1)
                improved = value < best_distance
                best_distance[improved] = value[improved]
                best_index[improved] = index[improved] + start
            pairs = torch.stack([samples[:8].cpu(), reference[best_index[:8].cpu()]], dim=1).flatten(0, 1)
            save_grid(pairs, directory / "nearest_training_pairs.png", columns=4,
                      title="Generated / nearest training image (alternating)")
            pairwise = torch.pdist(samples) / np.sqrt(784)
            record = {"n_unfiltered_samples": len(samples), "classifier_predicted_counts": counts,
                      "classifier_mean_confidence": confidence.mean().item(),
                      "predicted_classes_present": sum(c > 0 for c in counts),
                      "fraction_confidence_above_0_8": (confidence >= .8).float().mean().item(),
                      "mean_pairwise_pixel_rmse": pairwise.mean().item(),
                      "nearest_training_mean_pixel_rmse": best_distance.mean().item(),
                      "nearest_training_indices": best_index.cpu().tolist(),
                      "nearest_training_pool_size": len(reference),
                      "limitation": "Classifier predictions may be wrong on generated images. Pixel distances are not perceptual quality or a proof against memorization. Inspect sample grids."}
            atomic_json(directory / "diagnostics.json", record)
            records[directory.name] = record
    atomic_json(args.output / "diagnostics.json", {"classifier_validation_accuracy": accuracy,
                "classifier_training_count": n, "smoke_test": args.smoke, "experiments": records})
    print("Auxiliary classifier validation accuracy:", accuracy, flush=True)


if __name__ == "__main__":
    main()
