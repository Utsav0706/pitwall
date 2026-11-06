"""Fetch the two llama.cpp pieces the GGUF export needs, pinned and checked.

- convert_hf_to_gguf.py (Python, from the source at a fixed tag; commit verified)
- llama-quantize (a prebuilt binary from the same release; sha256 verified), so nothing
  is compiled on Colab

The release is pinned (b11838, two weeks old when chosen) and every download is checked
against a known hash before it is used: a build tool is code that runs on your machine.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

TAG = "b11838"
COMMIT = "4231b23dc2c46661d01157aac5245bdcf03ce4b1"
REPO = "https://github.com/ggml-org/llama.cpp"
ASSETS = {
    # platform.system() -> (asset name, sha256 from the release's asset digests)
    "Linux": (
        f"llama-{TAG}-bin-ubuntu-x64.tar.gz",
        "507e2f4fe3553499b518efe74a4244e35d709154a4e51c67e3affeef228ceb30",
    ),
    "Windows": (
        f"llama-{TAG}-bin-win-cpu-x64.zip",
        "45462c0f6b0e9f9b841e3d2bc2a086b29925a9691d866458bbbf90cf564ae2b7",
    ),
}


class LlamaCppError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlamaCpp:
    source: Path  # the cloned source tree (convert script, gguf-py)
    bin_dir: Path  # where llama-quantize and its shared libraries are

    @property
    def convert_script(self) -> Path:
        return self.source / "convert_hf_to_gguf.py"

    @property
    def quantize(self) -> Path:
        name = "llama-quantize.exe" if platform.system() == "Windows" else "llama-quantize"
        return self.bin_dir / name

    def env(self) -> dict[str, str]:
        """Environment for running the binary: its shared libraries sit next to it."""
        env = dict(os.environ)
        key = "PATH" if platform.system() == "Windows" else "LD_LIBRARY_PATH"
        env[key] = str(self.bin_dir) + os.pathsep + env.get(key, "")
        return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone(dest: Path) -> Path:
    source = dest / "src"
    if not source.exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", TAG, REPO, str(source)],
            check=True,
        )
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != COMMIT:
        raise LlamaCppError(f"llama.cpp {TAG} is at {head}, expected {COMMIT}")
    return source


def _download_binaries(dest: Path) -> Path:
    system = platform.system()
    if system not in ASSETS:
        raise LlamaCppError(f"no pinned llama.cpp binary for {system}")
    name, expected = ASSETS[system]
    archive = dest / name
    if not archive.exists():
        urllib.request.urlretrieve(f"{REPO}/releases/download/{TAG}/{name}", archive)
    actual = sha256_file(archive)
    if actual != expected:
        archive.unlink()
        raise LlamaCppError(f"{name} sha256 {actual[:12]} != pinned {expected[:12]}; deleted")
    out = dest / "bin"
    if not out.exists():
        out.mkdir(parents=True)
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(out)
        else:
            with tarfile.open(archive) as t:
                t.extractall(out, filter="data")
    # The archive may nest files in a folder; find the binary wherever it is.
    target = "llama-quantize.exe" if system == "Windows" else "llama-quantize"
    found = sorted(out.rglob(target))
    if not found:
        raise LlamaCppError(f"{target} not found in {name}")
    found[0].chmod(0o755)
    return found[0].parent


def fetch(dest: Path) -> LlamaCpp:
    dest.mkdir(parents=True, exist_ok=True)
    return LlamaCpp(source=_clone(dest), bin_dir=_download_binaries(dest))
