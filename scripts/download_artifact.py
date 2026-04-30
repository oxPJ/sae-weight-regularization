from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


RESULTS_REPO_ID = "anonsaereg/SAE-REG-results"
RESULTS_REPO_TYPE = "dataset"

MODELS_REPO_ID = "anonsaereg/SAE-REG-models"
MODELS_REPO_TYPE = "model"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download_results(root: Path) -> None:
    print(f"Downloading results from {RESULTS_REPO_ID}...")

    snapshot_download(
        repo_id=RESULTS_REPO_ID,
        repo_type=RESULTS_REPO_TYPE,
        local_dir=root,
        allow_patterns=["data_results/**"],
    )

    print(f"Results downloaded to {root / 'data_results'}")


def download_model_weights(root: Path) -> None:
    print(f"Downloading model weights from {MODELS_REPO_ID}...")

    model_dir = root / "data_model_weights"
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODELS_REPO_ID,
        repo_type=MODELS_REPO_TYPE,
        local_dir=model_dir,
    )

    print(f"Model weights downloaded to {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download artifacts required to reproduce the SAE regularization experiments."
    )
    parser.add_argument(
        "--skip-results",
        action="store_true",
        help="Do not download data_results artifacts.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not download data_model_weights artifacts.",
    )
    args = parser.parse_args()

    root = repo_root()

    if not args.skip_results:
        download_results(root)

    if not args.skip_models:
        download_model_weights(root)

    print("Done.")


if __name__ == "__main__":
    main()