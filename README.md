> **This public repository is the marker-accessible frozen copy of the assignment code.** Everything below is the original README of the frozen code archive `CS5494_MNIST_code_60444432.zip`. The `formal_results/` directory additionally contains the measured outputs of the six formal runs (tables, figures, histories; no checkpoints).

# CS5494 MNIST generation — upload-ready implementation

Three models: fully visible sigmoid belief network (FVSBN), masked neural autoregressor (MADE), and a GAN on original grayscale MNIST. Authored for Zian LI (EID zli934, student ID **60444432**). The final individual report filename is **60444432.pdf**. **Formal GPU training results and the final report remain pending execution.** Local smoke results verify the workflow only.

## Run in Colab

Upload **`MNIST_Colab.ipynb`**, select a GPU runtime, and run its cells in order. The notebook embeds all source files: no local ZIP, private GitHub access or manual dataset upload is required. It downloads the public original MNIST files and checks their published MD5 hashes. The setup uses Colab's existing PyTorch, preserving CUDA support.

The default stores checkpoints under `MyDrive/CS5494_MNIST/runs`; the notebook asks Colab to mount your Drive so interrupted work survives runtime deletion. Set `USE_DRIVE=False` to keep results in temporary `/content` and download them before disconnecting. The code uses your Drive only for that task directory.

1. Run setup, configuration and mathematical checks.
2. Run the three main models. Every completed epoch is checkpointed. Rerun the same training cell to resume; completed experiments are skipped.
3. Run the three ablations with unchanged configuration.
4. Run auxiliary diagnostics and export the results ZIP. Download the ZIP and the executed notebook for report preparation. Checkpoints stay in Drive; the export cell can optionally include them.

The notebook prints real epoch times. Full runtime depends on the assigned GPU; no unmeasured training duration is promised. A CPU smoke configuration is also included. If changing any hyperparameter, use a new output directory to preserve existing experiments; configuration mismatches are rejected.

## Local commands

```sh
python -m pip install -r requirements.txt
python check_models.py
python train.py --suite all --smoke --device cpu --output smoke_runs
python diagnostics.py --output smoke_runs --smoke --device cpu
python export_results.py --output smoke_runs

# Formal runs on a CUDA machine:
python train.py --suite main --device cuda --output runs
python train.py --suite ablations --device cuda --output runs
python diagnostics.py --device cuda --output runs
python export_results.py --output runs
```

## Fixed experiment design

The official 60,000 training images are split once with seed 5494 into 50,000 training and 10,000 validation images. The official 10,000 test images are reserved for final AR likelihood evaluation. Labels are unused for generative training. The auxiliary diagnostic classifier uses training labels only and reports validation accuracy.

| Experiment | Training images | Epochs | Learning rate | Architecture / changed factor |
|---|---:|---:|---:|---|
| FVSBN main | 50,000 | 30 | 0.001 | Strictly lower triangular logistic conditionals |
| MADE main | 50,000 | 40 | 0.001 | 784 → 512 → 784 with causal masks |
| GAN main | 50,000 | 50 | 0.0002 | G: 100 → 256 → 512 → 784; D: 784 → 512 → 256 → 1 |
| MADE depth | 50,000 | 40 | 0.001 | Two 512-unit hidden layers |
| MADE data | 10,000 | 40 | 0.001 | Same main model, nested subset of training indices |
| GAN learning rate | 50,000 | 50 | 0.0001 | Same main architecture and seed |

All use batch size 256 and Adam. AR uses default betas (0.9, 0.999); GAN uses (0.5, 0.999) and one discriminator update per generator update. AR checkpoint selection uses validation NLL. GAN uses the final fixed epoch, avoiding undocumented visual cherry-picking. Each run saves 64 unfiltered samples with one declared seed, and five training stages × four samples with fixed random values.

For AR, threshold grayscale at 0.5 and model raster-order binary pixels. Training is parallel across outputs but sampling is sequential. FVSBN uses an exact incremental sampling implementation, checked against naive ancestral sampling. MADE hidden degree masks use non-strict inequalities internally and strict inequalities at the output. Current and future pixels cannot influence their own conditional probabilities.

For GAN, retain grayscale intensities scaled to [-1,1], use tanh at the generator output, logits at the discriminator output, and BCEWithLogits for both discriminator and non-saturating generator objectives. No binarized training images are passed to the GAN. Final saved images are rescaled to [0,1].

## Evidence and limitations

- `check_models.py` checks Jacobian causality, input perturbations, exact small-dimensional density normalization, finite-difference gradients, zero masked-weight gradients, sampling consistency, and GAN gradient routing.
- `last.pt` includes current model, optimizer, RNG, history and best AR state. Loading is limited to checkpoints this script produces; do not use untrusted checkpoint files.
- Loss curves, full histories, model configurations, device/version information and real timings are saved. CUDA timings synchronize before and after measurement.
- NLL is nats/image; bits/dimension is NLL/(784 log 2). GAN losses are objectives, not likelihoods, and should not be numerically compared to AR NLL.
- Data-size ablations fix epochs rather than gradient updates; discuss that distinction. One seed per formal configuration does not establish statistical significance.
- Diagnostic class counts are classifier predictions on generated images, not ground-truth coverage. Pixel distances and nearest neighbours help inspection but do not prove sample quality or rule out memorization.

## Files

`models.py` contains the three model definitions. `data.py` verifies and preprocesses MNIST. `train.py` runs resumable experiments. `diagnostics.py` generates auxiliary diversity evidence. `export_results.py` packages the evidence. `build_notebook.py` regenerates the self-contained Colab notebook after source changes. `REPORT_PLAN.md` maps the report to the grading criteria.

## Code access for the marker

The coursework Git repository is private. A Git commit link alone does not give the marker access. `python package_code.py` creates **`CS5494_MNIST_code_60444432.zip`**, containing source, the self-contained notebook, instructions and verification records, with hashes for every file. Upload the frozen archive to a location the marker can access and include that link in the final report. Check the link from an account without your private repository access. Preserve the linked version after submission. The archive contains code and local correctness evidence; it is separate from the formal experiment results ZIP and `60444432.pdf`.

References: [MADE paper](https://proceedings.mlr.press/v37/germain15.html), [GAN paper](https://arxiv.org/abs/1406.2661), [MNIST checksums and mirror in torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
