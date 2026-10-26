"""Build a self-contained Colab notebook embedding the reviewed local source."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [md("""# CS5494 — MNIST generative models
**Zian LI — student ID 60444432 (EID zli934)**

上传本 notebook 到 Colab，选择 GPU 运行时，再按顺序运行。无需远程连接助手。
默认将 checkpoint 保存到 Google Drive，重开运行时后重复运行相同配置即可续跑。
正式实验分为“三个主模型”和“三项消融”，都完成后导出结果 ZIP。

This notebook embeds all local source files. It trains FVSBN, MADE and GAN,
records actual measurements, and exports evidence for the final English report.
Reduced smoke runs are clearly labelled and must not be used as final results.
"""), md("## 1. Configuration and persistent output directory"), code('''from pathlib import Path
import os, sys, subprocess, importlib.util

USE_DRIVE = True  # Recommended on Colab: checkpoints survive runtime deletion.
SMOKE = False    # True runs 512 examples × 2 epochs for pipeline verification only.
IN_COLAB = importlib.util.find_spec("google.colab") is not None
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT = Path("/content/drive/MyDrive/CS5494_MNIST") / ("smoke_runs" if SMOKE else "runs")
else:
    OUTPUT = (Path("/content") if IN_COLAB else Path.cwd()) / "CS5494_MNIST" / ("smoke_runs" if SMOKE else "runs")
SOURCE = (Path("/content") if IN_COLAB else Path.cwd()) / "cs5494_mnist_source"
SOURCE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
DATA = SOURCE / "data"
os.chdir(SOURCE)
print("Source:", SOURCE, "\\nPersistent results:", OUTPUT)

# Keep Colab's CUDA-compatible torch; install only missing lightweight packages.
for package in ["numpy", "matplotlib"]:
    if importlib.util.find_spec(package) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch:", torch.__version__, "Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name())
elif not SMOKE:
    raise RuntimeError("Formal configuration: select a GPU runtime, reconnect, and rerun. Set SMOKE=True only for a CPU check.")
'''), md("## 2. Embedded source files\nThese cells write the implementation into the source directory. Reviewable copies accompany this notebook in the repository.")]
for filename in ["models.py", "data.py", "check_models.py", "train.py", "diagnostics.py", "export_results.py"]:
    cells.append(md("### " + filename))
    cells.append(code("%%writefile " + filename + "\n" + (ROOT / filename).read_text(encoding="utf-8")))
cells += [md("## 3. Mathematical verification"), code('''subprocess.run([sys.executable, "check_models.py"], check=True)
'''), md("""## 4. Train the three required models
All models use the fixed official training split. FVSBN and MADE use binary pixels;
GAN uses original grayscale. The first epoch log provides an actual throughput
measurement. Checkpoints are saved after every epoch. Rerunning resumes them.
"""), code('''common = ["--device", DEVICE, "--data", str(DATA), "--output", str(OUTPUT)]
if SMOKE:
    common.append("--smoke")
subprocess.run([sys.executable, "train.py", "--suite", "main", *common], check=True)
'''), md("""## 5. Controlled ablations
MADE: hidden depth 1 versus 2; 50k versus 10k training samples at a fixed epoch
budget. GAN: learning rate 2e-4 versus 1e-4. Keep this output directory/configuration
unchanged when resuming. These comparisons are needed for the experiment analysis.
"""), code('''subprocess.run([sys.executable, "train.py", "--suite", "ablations", *common], check=True)
'''), md("""## 6. Auxiliary diversity and nearest-neighbour diagnostics
The auxiliary classifier uses training labels only and reports validation accuracy.
Its generated-image predictions are imperfect evidence, not ground-truth labels.
Always inspect the unfiltered sample grids as well.
"""), code('''subprocess.run([sys.executable, "diagnostics.py", *common], check=True)
'''), md("## 7. Inspect real outputs"), code('''import json
from IPython.display import display, Image
for path in sorted(OUTPUT.glob("*/summary.json")):
    result = json.loads(path.read_text())
    print(path.parent.name, "selected epoch", result["selected_epoch"],
          "test NLL", result["test_nll_nats"], "train seconds", result["training_optimization_seconds"])
    display(Image(filename=str(path.parent / "samples.png")))
    display(Image(filename=str(path.parent / "generation_progress.png")))
    display(Image(filename=str(path.parent / "learning_curves.png")))
'''), md("""## 8. Export for report preparation
Download this ZIP and save the executed notebook. Checkpoints remain in Drive.
Set `INCLUDE_CHECKPOINTS=True` for a larger archive containing trained models too.
No results or quality claims are fabricated when a run is incomplete.
"""), code('''INCLUDE_CHECKPOINTS = False
command = [sys.executable, "export_results.py", "--output", str(OUTPUT)]
if INCLUDE_CHECKPOINTS:
    command.append("--include-checkpoints")
subprocess.run(command, check=True)
archive = OUTPUT.parent / (OUTPUT.name + "_results.zip")
print("Download:", archive)
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
'''), md("""## Report completion
Use `comparison.csv`, `REVIEW_RESULTS.md`, histories, generated images and diagnostic
figures in the final report. Include the numeric student ID and an accessible,
frozen code/data link. Do not present smoke metrics as formal experiments or
compare GAN BCE to autoregressive NLL. Discuss observed failures and seed limits.

References: [MADE](https://proceedings.mlr.press/v37/germain15.html),
[GAN](https://arxiv.org/abs/1406.2661),
[original MNIST mirror/checksums](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
""")]
notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"name": "MNIST_Colab.ipynb"}})
nbf.validate(notebook)
nbf.write(notebook, ROOT / "MNIST_Colab.ipynb")
print("Wrote MNIST_Colab.ipynb with", len(cells), "cells")
