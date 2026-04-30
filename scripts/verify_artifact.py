from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


RESULTS_REPO_ID = "anonsaereg/SAE-REG-results"
RESULTS_REPO_TYPE = "dataset"
RESULTS_PREFIX = "data_results"

MODELS_REPO_ID = "anonsaereg/SAE-REG-models"
MODELS_REPO_TYPE = "model"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def file_count_and_size(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in files)
    return len(files), total_size


def check_local_directory(path: Path) -> bool:
    if not path.exists():
        print(f"Missing: {path}")
        return False

    if not path.is_dir():
        print(f"Expected directory but found file: {path}")
        return False

    count, total_size = file_count_and_size(path)
    if count == 0:
        print(f"No files found in: {path}")
        return False

    print(f"Found {count} files in {path} ({total_size / 1e9:.2f} GB)")
    return True


def remote_files(repo_id: str, repo_type: str) -> list[str]:
    api = HfApi()
    return [
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        if not path.endswith("/")
    ]


def verify_results(root: Path, check_remote: bool) -> bool:
    local_dir = root / RESULTS_PREFIX
    ok = check_local_directory(local_dir)

    if check_remote:
        expected = [
            path
            for path in remote_files(RESULTS_REPO_ID, RESULTS_REPO_TYPE)
            if path.startswith(f"{RESULTS_PREFIX}/")
        ]

        missing = [
            path
            for path in expected
            if not (root / path).is_file()
        ]

        if missing:
            print(f"Missing {len(missing)} result files compared with Hugging Face.")
            for path in missing[:20]:
                print(f"  - {path}")
            if len(missing) > 20:
                print("  ...")
            ok = False
        else:
            print(f"All {len(expected)} result files from Hugging Face are present locally.")

    return ok


def verify_model_weights(root: Path, check_remote: bool) -> bool:
    local_dir = root / "data_model_weights"
    ok = check_local_directory(local_dir)

    if check_remote:
        expected = remote_files(MODELS_REPO_ID, MODELS_REPO_TYPE)

        missing = [
            path
            for path in expected
            if not (local_dir / path).is_file()
        ]

        if missing:
            print(f"Missing {len(missing)} model files compared with Hugging Face.")
            for path in missing[:20]:
                print(f"  - data_model_weights/{path}")
            if len(missing) > 20:
                print("  ...")
            ok = False
        else:
            print(f"All {len(expected)} model files from Hugging Face are present locally.")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that downloaded SAE regularization artifacts are present locally."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only check local directories; do not compare against Hugging Face.",
    )
    args = parser.parse_args()

    root = repo_root()
    check_remote = not args.offline

    results_ok = verify_results(root, check_remote=check_remote)
    models_ok = verify_model_weights(root, check_remote=check_remote)

    if results_ok and models_ok:
        print("Artifact verification passed.")
    else:
        raise SystemExit("Artifact verification failed.")


if __name__ == "__main__":
    main()