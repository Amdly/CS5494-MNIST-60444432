# MNIST experiment review packet

Zian LI — student ID 60444432 (EID zli934). Final report filename: 60444432.pdf.

This packet contains measured outputs. It is not the final assignment report.

## fvsbn_main

Configuration: `{"name": "fvsbn_main", "model": "fvsbn", "epochs": 30, "lr": 0.001, "n_train": 50000, "batch_size": 256, "hidden": 512, "depth": 1, "seed": 5494, "smoke_test": false, "split_seed": 5494, "binary_threshold": 0.5, "schema_version": 1}`.

Selected epoch: 30 (minimum validation NLL). Training optimization time: 136.59 s.

Test NLL: 82.882 nats/image; 0.15252 bits/dimension.

![Loss curves](fvsbn_main/learning_curves.png)

![Fixed-randomness progress](fvsbn_main/generation_progress.png)

![Unfiltered samples](fvsbn_main/samples.png)

## gan_lr1e4

Configuration: `{"name": "gan_lr1e4", "model": "gan", "epochs": 50, "lr": 0.0001, "n_train": 50000, "batch_size": 256, "hidden": 512, "depth": 1, "seed": 5494, "smoke_test": false, "split_seed": 5494, "binary_threshold": 0.5, "schema_version": 1}`.

Selected epoch: 50 (final fixed epoch). Training optimization time: 510.77 s.

![Loss curves](gan_lr1e4/learning_curves.png)

![Fixed-randomness progress](gan_lr1e4/generation_progress.png)

![Unfiltered samples](gan_lr1e4/samples.png)

## gan_main

Configuration: `{"name": "gan_main", "model": "gan", "epochs": 50, "lr": 0.0002, "n_train": 50000, "batch_size": 256, "hidden": 512, "depth": 1, "seed": 5494, "smoke_test": false, "split_seed": 5494, "binary_threshold": 0.5, "schema_version": 1}`.

Selected epoch: 50 (final fixed epoch). Training optimization time: 539.41 s.

![Loss curves](gan_main/learning_curves.png)

![Fixed-randomness progress](gan_main/generation_progress.png)

![Unfiltered samples](gan_main/samples.png)

## made_10k

Configuration: `{"name": "made_10k", "model": "made", "epochs": 40, "lr": 0.001, "n_train": 10000, "batch_size": 256, "hidden": 512, "depth": 1, "seed": 5494, "smoke_test": false, "split_seed": 5494, "binary_threshold": 0.5, "schema_version": 1}`.

Selected epoch: 40 (minimum validation NLL). Training optimization time: 43.68 s.

Test NLL: 99.735 nats/image; 0.18353 bits/dimension.

![Loss curves](made_10k/learning_curves.png)

![Fixed-randomness progress](made_10k/generation_progress.png)

![Unfiltered samples](made_10k/samples.png)

## made_depth2

Configuration: `{"name": "made_depth2", "model": "made", "epochs": 40, "lr": 0.001, "n_train": 50000, "batch_size": 256, "hidden": 512, "depth": 2, "seed": 5494, "smoke_test": false, "split_seed": 5494, "binary_threshold": 0.5, "schema_version": 1}`.

Selected epoch: 40 (minimum validation NLL). Training optimization time: 337.64 s.

Test NLL: 86.934 nats/image; 0.15997 bits/dimension.

![Loss curves](made_depth2/learning_curves.png)

![Fixed-randomness progress](made_depth2/generation_progress.png)

![Unfiltered samples](made_depth2/samples.png)

## made_main

Configuration: `{"name": "made_main", "model": "made", "epochs": 40, "lr": 0.001, "n_train": 50000, "batch_size": 256, "hidden": 512, "depth": 1, "seed": 5494, "smoke_test": false, "split_seed": 5494, "binary_threshold": 0.5, "schema_version": 1}`.

Selected epoch: 40 (minimum validation NLL). Training optimization time: 213.58 s.

Test NLL: 82.519 nats/image; 0.15185 bits/dimension.

![Loss curves](made_main/learning_curves.png)

![Fixed-randomness progress](made_main/generation_progress.png)

![Unfiltered samples](made_main/samples.png)

## Interpretation checklist

- Compare FVSBN and MADE likelihood only on the same fixed binary test set. GAN has no reported tractable likelihood.
- Compare depth and data ablations against MADE main; the data-size experiment fixes epochs, so fewer samples also mean fewer updates.
- Compare GAN learning-rate curves, fixed-noise progress and unfiltered samples. Loss alone cannot establish visual quality or absence of mode collapse.
- Report actual GPU model, training time, sample count and inference time. AR and GAN have different sampling computation.
- Inspect any classifier diagnostics with their validation accuracy and generated-image distribution shift in mind.
- Use author Zian LI, student ID 60444432, filename 60444432.pdf, and a frozen accessible code/data link in the final report.
