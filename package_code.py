"""Freeze a self-contained code archive for the report's accessible code link."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ["models.py", "data.py", "check_models.py", "train.py", "diagnostics.py",
         "export_results.py", "build_notebook.py", "validate_pipeline.py", "package_code.py",
         "MNIST_Colab.ipynb", "README.md", "requirements.txt", "model_checks.json",
         "pipeline_validation.json"]


def main():
    paths = [ROOT / name for name in FILES]
    assert all(path.is_file() for path in paths)
    manifest = {"author": "Zian LI", "student_id": "60444432", "eid": "zli934",
                "purpose": "Source code and local workflow checks; formal training results are separate.",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    archive = ROOT / "CS5494_MNIST_code_60444432.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.name)
        z.writestr("code_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    record = {"file": archive.name, "bytes": archive.stat().st_size,
              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "verified_files": len(paths), "public_access": "User must share this archive with the marker; the coursework Git repository is private."}
    (ROOT / "code_archive_validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
