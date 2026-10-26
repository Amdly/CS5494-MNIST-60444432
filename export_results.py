"""Create a review packet with real result tables and figures, and export a ZIP."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    results = []
    for path in sorted(args.output.glob("*/summary.json")):
        result = json.loads(path.read_text())
        results.append(result)
    if not results:
        raise RuntimeError("No completed experiment results to export")
    rows = [{"experiment": r["config"]["name"], "smoke_test": r["smoke_test"],
             "n_train": r["config"]["n_train"], "epochs": r["config"]["epochs"],
             "lr": r["config"]["lr"], "selected_epoch": r["selected_epoch"],
             "test_nll_nats": r["test_nll_nats"], "test_bits_per_dimension": r["test_bits_per_dimension"],
             "training_seconds": r["training_optimization_seconds"],
             "inference_seconds": r["inference_seconds_median"],
             "n_generated_per_timing": r["inference_samples_per_repeat"], "device": r["device"]} for r in results]
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# MNIST experiment review packet", "", "Zian LI — student ID 60444432 (EID zli934). Final report filename: 60444432.pdf.", "",
             "This packet contains measured outputs. It is not the final assignment report.", ""]
    if any(r["smoke_test"] for r in results):
        lines += ["**SMOKE TEST ONLY: reduced data/epochs. Do not present these as trained-model results.**", ""]
    for r in results:
        name = r["config"]["name"]
        lines += ["## " + name, "", f"Configuration: `{json.dumps(r['config'])}`.", "",
                  f"Selected epoch: {r['selected_epoch']} ({r['selection']}). Training optimization time: {r['training_optimization_seconds']:.2f} s.", ""]
        if r["test_nll_nats"] is not None:
            lines += [f"Test NLL: {r['test_nll_nats']:.3f} nats/image; {r['test_bits_per_dimension']:.5f} bits/dimension.", ""]
        lines += [f"![Loss curves]({name}/learning_curves.png)", "", f"![Fixed-randomness progress]({name}/generation_progress.png)", "",
                  f"![Unfiltered samples]({name}/samples.png)", ""]
    lines += ["## Interpretation checklist", "",
              "- Compare FVSBN and MADE likelihood only on the same fixed binary test set. GAN has no reported tractable likelihood.",
              "- Compare depth and data ablations against MADE main; the data-size experiment fixes epochs, so fewer samples also mean fewer updates.",
              "- Compare GAN learning-rate curves, fixed-noise progress and unfiltered samples. Loss alone cannot establish visual quality or absence of mode collapse.",
              "- Report actual GPU model, training time, sample count and inference time. AR and GAN have different sampling computation.",
              "- Inspect any classifier diagnostics with their validation accuracy and generated-image distribution shift in mind.",
              "- Use author Zian LI, student ID 60444432, filename 60444432.pdf, and a frozen accessible code/data link in the final report."]
    (args.output / "REVIEW_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(p for p in args.output.rglob("*") if p.is_file() and p.suffix in
                   ({".json", ".csv", ".png", ".npy", ".md", ".pt"} if args.include_checkpoints else {".json", ".csv", ".png", ".npy", ".md"}))
    manifest = {p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    archive = args.output.parent / (args.output.name + "_results.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(args.output))
        z.writestr("export_manifest.json", json.dumps(manifest, indent=2))
    print("Exported", archive.resolve(), "bytes", archive.stat().st_size)


if __name__ == "__main__":
    main()
