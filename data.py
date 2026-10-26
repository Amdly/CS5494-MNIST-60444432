"""Download and verify the original MNIST IDX files, then make a fixed split."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np
import torch

# Same public mirror and original checksums used by torchvision.datasets.MNIST.
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load_data(directory, split_seed=5494):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    arrays = {}
    for name, checksum in FILES.items():
        assert len(checksum) == 32
        path = directory / name
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != checksum:
            print("Downloading", name, flush=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(MIRROR + name, timeout=90) as response:
                temporary.write_bytes(response.read())
            if hashlib.md5(temporary.read_bytes()).hexdigest() != checksum:
                raise ValueError("MNIST checksum mismatch: " + name)
            temporary.replace(path)
        content = gzip.decompress(path.read_bytes())
        magic, count = struct.unpack(">II", content[:8])
        if magic == 2051:
            height, width = struct.unpack(">II", content[8:16])
            assert (height, width) == (28, 28)
            arrays[name] = np.frombuffer(content, dtype=np.uint8, offset=16).copy().reshape(count, 784)
        elif magic == 2049:
            arrays[name] = np.frombuffer(content, dtype=np.uint8, offset=8).copy()
            assert len(arrays[name]) == count
        else:
            raise ValueError("Invalid IDX magic")
        records[name] = {"url": MIRROR + name, "md5": checksum,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    images = torch.from_numpy(arrays["train-images-idx3-ubyte.gz"])
    labels = torch.from_numpy(arrays["train-labels-idx1-ubyte.gz"])
    indices = np.random.default_rng(split_seed).permutation(60000)
    training, validation = indices[:50000], indices[50000:]
    (directory / "manifest.json").write_text(json.dumps({"files": records, "split_seed": split_seed,
                 "train_count": 50000, "validation_count": 10000, "test_count": 10000,
                 "training_index_sha256": hashlib.sha256(training.tobytes()).hexdigest()}, indent=2))
    np.savez_compressed(directory / "split_indices.npz", train=training, validation=validation)
    return {"train": images[training], "validation": images[validation],
            "test": torch.from_numpy(arrays["t10k-images-idx3-ubyte.gz"]),
            "train_labels": labels[training], "validation_labels": labels[validation],
            "test_labels": torch.from_numpy(arrays["t10k-labels-idx1-ubyte.gz"])}


def preprocess(images, binary):
    # 128/255 >= .5; original grayscale is retained for GAN inputs.
    return (images >= 128).float() if binary else images.float().div(127.5).sub(1)
