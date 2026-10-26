"""Audit real smoke outputs and simulate interruption to test exact CPU resume."""
import json
from pathlib import Path
import tempfile

import nbformat
import numpy as np
import torch

from data import load_data
import train

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
results = {}
for path in sorted((ROOT / "smoke_runs").glob("*/summary.json")):
    item = json.loads(path.read_text())
    assert item["complete"] and item["smoke_test"] and item["config"]["n_train"] == 512
    samples = np.load(path.parent / "samples.npy")
    assert samples.shape == (16, 784) and np.isfinite(samples).all()
    assert samples.min() >= 0 and samples.max() <= 1
    if item["config"]["model"] != "gan":
        assert np.isin(samples, [0, 1]).all()
        assert np.isfinite(item["test_nll_nats"])
    for name in ["samples.png", "generation_progress.png", "learning_curves.png", "selected_model.pt", "last.pt"]:
        assert (path.parent / name).stat().st_size > 0
    results[path.parent.name] = {"passed": True, "sample_shape": list(samples.shape),
                               "n_train": 512, "epochs": 2, "device": item["device"]}
assert len(results) == 6
notebook = nbformat.read(ROOT / "MNIST_Colab.ipynb", as_version=4)
nbformat.validate(notebook)
embedded = {}
for cell in notebook.cells:
    if cell.cell_type == "code" and cell.source.startswith("%%writefile "):
        header, content = cell.source.split("\n", 1)
        filename = header.split(" ", 1)[1]
        assert content == (ROOT / filename).read_text(encoding="utf-8")
        compile(content, filename, "exec")
        embedded[filename] = True
    elif cell.cell_type == "code":
        compile(cell.source, "notebook cell", "exec")
assert len(embedded) == 6

# The GAN uses random training noise; exact equality after an interrupted epoch
# checks model, both optimizers, RNG state and deterministic batch ordering.
data = load_data(ROOT / "data")
temp = Path(tempfile.mkdtemp(prefix="resume_check_", dir=ROOT / "smoke_runs"))
assert temp.resolve().is_relative_to((ROOT / "smoke_runs").resolve())
configuration = train.Experiment("gan_resume", "gan", 2, .0002, n_train=64, batch_size=32)
continuous = train.run_experiment(configuration, data, temp / "continuous", torch.device("cpu"), smoke=True)
save = train.atomic_checkpoint
interrupted = False
def stop_after_first_checkpoint(path, state):
    global interrupted
    save(path, state)
    if path.name == "last.pt" and state["epoch"] == 1 and not interrupted:
        interrupted = True
        raise RuntimeError("SIMULATED_INTERRUPTION")
train.atomic_checkpoint = stop_after_first_checkpoint
try:
    train.run_experiment(configuration, data, temp / "resumed", torch.device("cpu"), smoke=True)
except RuntimeError as error:
    assert str(error) == "SIMULATED_INTERRUPTION"
finally:
    train.atomic_checkpoint = save
assert interrupted
resumed = train.run_experiment(configuration, data, temp / "resumed", torch.device("cpu"), smoke=True)
assert continuous["sample_sha256"] == resumed["sample_sha256"]
states = [torch.load(temp / kind / "gan_resume/selected_model.pt", weights_only=False)["model"]
          for kind in ["continuous", "resumed"]]
assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
record = {"smoke_experiments": results, "embedded_sources_match": embedded,
          "gan_resume_bitwise_equal_cpu": True,
          "note": "Correctness and workflow validation only. Formal GPU execution and final report pending."}
(ROOT / "pipeline_validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
