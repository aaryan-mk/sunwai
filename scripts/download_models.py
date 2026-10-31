"""Explicitly download model weights (never done implicitly on import).

Usage:
    python scripts/download_models.py latentsync --dest third_party/LatentSync [--repo ByteDance/LatentSync-1.6]
    python scripts/download_models.py check --dest third_party/LatentSync

Requires the ``huggingface_hub`` package (installed with the ``pipeline`` extra). The Hugging Face repository id of the
LatentSync checkpoints is configurable because it is owned by the upstream project; verify it on the LatentSync README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "ByteDance/LatentSync-1.6"
REQUIRED = (
    "checkpoints/latentsync_unet.pt",
    "checkpoints/whisper/tiny.pt",
    "checkpoints/auxiliary/syncnet_v2.model",
    "configs/unet/stage2_512.yaml",
)


def missing(dest: Path) -> list[str]:
    """Return the required LatentSync files that are absent under ``dest``."""
    return [rel for rel in REQUIRED if not (dest / rel).exists()]


def download_latentsync(dest: Path, repo: str) -> None:
    """Download the LatentSync checkpoints from Hugging Face into ``dest/checkpoints``."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, local_dir=str(dest / "checkpoints"))


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["latentsync", "check"])
    parser.add_argument("--dest", type=Path, required=True, help="LatentSync checkout directory")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repository id of the checkpoints")
    args = parser.parse_args(argv)
    if args.action == "latentsync":
        download_latentsync(args.dest, args.repo)
    absent = missing(args.dest)
    if absent:
        print("missing:", *absent, sep="\n  ")
        return 1
    print("LatentSync layout complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
