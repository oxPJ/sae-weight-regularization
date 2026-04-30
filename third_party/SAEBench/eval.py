#!/usr/bin/env python3
"""
Evaluation script for dictionary_learning SAEs using SAEBench.

Usage:
    python eval_dictionary_learning_saes.py --sae_dir <path_to_saes> [--model_name <model>] [--eval_types <types>]

Example:
    python eval_dictionary_learning_saes.py --sae_dir saes_EleutherAI_pythia-70m-deduped_top_k_l2/resid_post_layer_3/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch

import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes
import sae_bench.evals.core.main as core
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.graphing_utils as graphing_utils
from sae_bench.custom_saes.base_sae import BaseSAE
from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
from sae_bench.custom_saes.relu_sae import ReluSAE
from sae_bench.custom_saes.topk_sae import TopKSAE


# =============================================================================
# Trainer class pattern matching
# =============================================================================

TOPK_TRAINER_PATTERNS = ["TopK", "topk", "top_k", "BatchTopK", "Matryoshka"]
RELU_TRAINER_PATTERNS = ["Standard", "PAnneal", "standard", "p_anneal"]
GATED_TRAINER_PATTERNS = ["Gated", "gated"]
JUMPRELU_TRAINER_PATTERNS = ["JumpRelu", "jumprelu", "JumpReLU"]


def is_topk_trainer(trainer_class: str) -> bool:
    """Check if trainer class is a TopK variant."""
    return any(pattern in trainer_class for pattern in TOPK_TRAINER_PATTERNS)


def is_relu_trainer(trainer_class: str) -> bool:
    """Check if trainer class is a ReLU/Standard variant."""
    return any(pattern in trainer_class for pattern in RELU_TRAINER_PATTERNS)


def is_gated_trainer(trainer_class: str) -> bool:
    """Check if trainer class is a Gated variant."""
    return any(pattern in trainer_class for pattern in GATED_TRAINER_PATTERNS)


def is_jumprelu_trainer(trainer_class: str) -> bool:
    """Check if trainer class is a JumpReLU variant."""
    return any(pattern in trainer_class for pattern in JUMPRELU_TRAINER_PATTERNS)


def get_trainer_type(trainer_class: str) -> str:
    """
    Determine the trainer type from the trainer class string.
    
    Raises ValueError if the trainer class is not recognized.
    """
    if is_topk_trainer(trainer_class):
        return "topk"
    elif is_relu_trainer(trainer_class):
        return "relu"
    elif is_gated_trainer(trainer_class):
        return "gated"
    elif is_jumprelu_trainer(trainer_class):
        return "jumprelu"
    else:
        raise ValueError(
            f"Unknown trainer class '{trainer_class}'. "
            f"Cannot determine which SAE architecture to use.\n"
            f"Known patterns:\n"
            f"  TopK-like: {TOPK_TRAINER_PATTERNS}\n"
            f"  ReLU-like: {RELU_TRAINER_PATTERNS}\n"
            f"  Gated-like: {GATED_TRAINER_PATTERNS}\n"
            f"  JumpReLU-like: {JUMPRELU_TRAINER_PATTERNS}\n"
            f"Please add your trainer class pattern to the appropriate list."
        )


# =============================================================================
# Weight loading utilities
# =============================================================================

def _get_key(
    state: dict,
    possible_keys: list[str],
    config_path: Path,
    transpose_if: Optional[str] = None,
) -> torch.Tensor:
    """
    Extract a tensor from state dict, trying multiple possible key names.
    
    Args:
        state: The state dictionary
        possible_keys: List of possible key names to try
        config_path: Path to config file (for error messages)
        transpose_if: If provided and the matching key equals this, transpose the tensor
        
    Returns:
        The tensor value
        
    Raises:
        KeyError: If none of the possible keys are found
    """
    for key in possible_keys:
        if key in state:
            tensor = state[key]
            if transpose_if and key == transpose_if:
                tensor = tensor.T
            return tensor

    raise KeyError(
        f"Could not find any of {possible_keys} in state dict for {config_path}. "
        f"Available keys: {list(state.keys())}"
    )


# =============================================================================
# SAE Loading
# =============================================================================

def load_dictionary_learning_saes(
    root_dir: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    config_name: str = "config.json",
    weights_name: str = "ae.pt",
) -> list[tuple[str, BaseSAE]]:
    """
    Load SAEs trained with the dictionary_learning library.
    
    Args:
        root_dir: Root directory containing SAE subdirectories
        model_name: Name of the model (e.g., "pythia-70m-deduped")
        device: Device to load SAEs to
        dtype: Data type for SAE weights
        config_name: Name of the config file in each SAE directory
        weights_name: Name of the weights file in each SAE directory
        
    Returns:
        List of (sae_id, sae) tuples
        
    Raises:
        ValueError: If trainer class is unknown or required config values are missing
        KeyError: If required weights are not found in state dict
    """
    root = Path(root_dir)
    selected_saes: list[tuple[str, BaseSAE]] = []

    if not root.exists():
        raise FileNotFoundError(f"SAE directory does not exist: {root}")

    config_files = list(root.rglob(config_name))
    if not config_files:
        raise FileNotFoundError(
            f"No config files named '{config_name}' found in {root} or subdirectories"
        )

    print(f"Found {len(config_files)} config file(s) in {root}")
    print("=" * 60)

    for config_path in config_files:
        weights_path = config_path.with_name(weights_name)
        if not weights_path.exists():
            print(f"WARNING: Skipping {config_path.parent}: missing {weights_name}")
            continue

        with open(config_path) as f:
            cfg_json = json.load(f)

        # Extract trainer config
        trainer_cfg = cfg_json.get("trainer", {})
        layer = trainer_cfg.get("layer")
        trainer_class = trainer_cfg.get("trainer_class", "")

        if layer is None:
            raise ValueError(f"Missing 'layer' in trainer config: {config_path}")

        if not trainer_class:
            raise ValueError(
                f"Missing 'trainer_class' in config {config_path}. "
                "Cannot determine SAE architecture."
            )

        # Determine trainer type (will raise if unknown)
        trainer_type = get_trainer_type(trainer_class)

        # Load state dict
        state = torch.load(weights_path, map_location="cpu", weights_only=False)

        print(f"\nLoading: {config_path.parent}")
        print(f"  trainer_class: {trainer_class}")
        print(f"  detected type: {trainer_type}")
        print(f"  state dict keys: {list(state.keys())}")

        # Extract weights with flexible key naming
        b_dec = _get_key(
            state, ["b_dec", "bias", "decoder.bias"], config_path
        )
        b_enc = _get_key(
            state, ["b_enc", "encoder.bias"], config_path
        )
        W_enc = _get_key(
            state, ["W_enc", "encoder.weight"], config_path, transpose_if="encoder.weight"
        )
        W_dec = _get_key(
            state, ["W_dec", "decoder.weight"], config_path, transpose_if="decoder.weight"
        )

        renamed = {
            "W_enc": W_enc,
            "W_dec": W_dec,
            "b_enc": b_enc,
            "b_dec": b_dec,
        }

        d_in = renamed["b_dec"].shape[0]
        d_sae = renamed["b_enc"].shape[0]

        # Create the appropriate SAE type
        if trainer_type == "topk":
            k = trainer_cfg.get("k")
            if k is None:
                raise ValueError(
                    f"TopK-style trainer '{trainer_class}' requires 'k' in config, "
                    f"but none found in {config_path}"
                )

            renamed["k"] = torch.tensor(k, dtype=torch.int)

            sae = TopKSAE(
                d_in=d_in,
                d_sae=d_sae,
                k=k,
                model_name=model_name,
                hook_layer=layer,
                device=device,
                dtype=dtype,
            )
            sae.load_state_dict(renamed)  # type: ignore[arg-type]
            sae = sae.to(device=device, dtype=dtype)
            architecture = "topk"
            print(f"  -> Loaded as TopKSAE with k={k}")

        elif trainer_type == "relu":
            sae = ReluSAE(
                d_in=d_in,
                d_sae=d_sae,
                model_name=model_name,
                hook_layer=layer,
                device=device,
                dtype=dtype,
            )
            sae.load_state_dict(renamed)  # type: ignore[arg-type]
            sae = sae.to(device=device, dtype=dtype)
            architecture = "relu"
            print(f"  -> Loaded as ReluSAE")

        elif trainer_type == "gated":
            # TODO: Add GatedSAE support when available in sae_bench
            raise NotImplementedError(
                f"Gated SAE loading not yet implemented for '{trainer_class}'. "
                "Please add support or use a different trainer type."
            )

        elif trainer_type == "jumprelu":
            # TODO: Add JumpReLU SAE support when available in sae_bench
            raise NotImplementedError(
                f"JumpReLU SAE loading not yet implemented for '{trainer_class}'. "
                "Please add support or use a different trainer type."
            )

        else:
            # This shouldn't happen if get_trainer_type is working correctly
            raise RuntimeError(f"Unexpected trainer type: {trainer_type}")

        # Configure SAE metadata
        sae.cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=sae.cfg.hook_name,
            hook_layer=layer,
        )
        sae.cfg.context_size = trainer_cfg.get("context_size", 128)
        sae.cfg.dtype = general_utils.dtype_to_str(dtype)
        sae.cfg.architecture = architecture
        sae.cfg.training_tokens = trainer_cfg.get("n_tokens")

        # Generate SAE ID from relative path
        relative_id = config_path.parent.relative_to(root).as_posix()
        sae_id = relative_id.replace("/", "_") if relative_id != "." else config_path.parent.name
        selected_saes.append((sae_id, sae))
        print(f"  -> SAE ID: {sae_id}")

    print("=" * 60)
    print(f"Successfully loaded {len(selected_saes)} SAE(s)")
    
    return selected_saes


# =============================================================================
# Evaluation functions
# =============================================================================

def run_core_eval(
    selected_saes: list[tuple[str, BaseSAE]],
    output_folder: str = "eval_results/core",
    dataset: str = "Skylion007/openwebtext",
    context_size: int = 128,
    n_batches: int = 200,
    batch_size: int = 32,
    dtype: str = "float32",
) -> None:
    """Run core evaluation metrics."""
    print("\n" + "=" * 60)
    print("Running Core Evaluation")
    print("=" * 60)
    
    _ = core.multiple_evals(
        selected_saes=selected_saes,
        n_eval_reconstruction_batches=n_batches,
        n_eval_sparsity_variance_batches=n_batches,
        eval_batch_size_prompts=batch_size,
        compute_featurewise_density_statistics=True,
        compute_featurewise_weight_based_metrics=True,
        exclude_special_tokens_from_reconstruction=True,
        dataset=dataset,
        context_size=context_size,
        output_folder=output_folder,
        verbose=True,
        dtype=dtype,
    )


def run_sparse_probing_eval(
    selected_saes: list[tuple[str, BaseSAE]],
    model_name: str,
    device: torch.device,
    output_folder: str = "eval_results/sparse_probing",
    llm_batch_size: int = 512,
    dtype: str = "float32",
    random_seed: int = 42,
) -> None:
    """Run sparse probing evaluation."""
    print("\n" + "=" * 60)
    print("Running Sparse Probing Evaluation")
    print("=" * 60)
    
    dataset_names = ["LabHC/bias_in_bios_class_set1"]
    
    _ = sparse_probing.run_eval(
        sparse_probing.SparseProbingEvalConfig(
            model_name=model_name,
            random_seed=random_seed,
            llm_batch_size=llm_batch_size,
            llm_dtype=dtype,
            dataset_names=dataset_names,
        ),
        selected_saes,
        device,
        output_folder,
        force_rerun=False,
        clean_up_activations=True,
        save_activations=False,
    )


def run_all_evals(
    selected_saes: list[tuple[str, BaseSAE]],
    model_name: str,
    device: torch.device,
    eval_types: list[str],
    llm_batch_size: int = 512,
    dtype: str = "float32",
) -> None:
    """Run all specified evaluation types."""
    print("\n" + "=" * 60)
    print(f"Running evaluations: {eval_types}")
    print("=" * 60)
    
    _ = run_all_evals_custom_saes.run_evals(
        model_name,
        selected_saes,
        llm_batch_size,
        dtype,
        device,
        eval_types,
        api_key=None,
        force_rerun=False,
        save_activations=False,
    )


# =============================================================================
# Plotting utilities
# =============================================================================

def generate_plots(
    results_folder: str = "./eval_results",
    image_folder: str = "./images",
    eval_types: Optional[list[str]] = None,
) -> None:
    """Generate plots from evaluation results."""
    if eval_types is None:
        eval_types = ["core", "sparse_probing", "scr", "tpp"]
    
    os.makedirs(image_folder, exist_ok=True)
    
    trainer_markers = {
        "standard": "o",
        "jumprelu": "X",
        "topk": "^",
        "p_anneal": "*",
        "gated": "d",
    }

    trainer_colors = {
        "standard": "blue",
        "jumprelu": "orange",
        "topk": "green",
        "p_anneal": "red",
        "gated": "purple",
    }

    core_folders = [f"{results_folder}/core"]
    core_filenames = graphing_utils.find_eval_results_files(core_folders)

    for eval_type in eval_types:
        if eval_type == "core":
            continue  # Core results are used for x-axis (L0), not plotted separately
            
        eval_folders = [f"{results_folder}/{eval_type}"]
        eval_filenames = graphing_utils.find_eval_results_files(eval_folders)

        if eval_filenames and core_filenames:
            print(f"\nGenerating plots for {eval_type}...")
            image_base_name = os.path.join(image_folder, eval_type)
            try:
                graphing_utils.plot_results(
                    eval_filenames,
                    core_filenames,
                    eval_type,
                    image_base_name,
                    k=10,
                    trainer_markers=trainer_markers,
                    trainer_colors=trainer_colors,
                )
                print(f"  -> Saved to {image_base_name}_*.png")
            except Exception as e:
                print(f"  -> Error: {e}")


# =============================================================================
# Main function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dictionary_learning SAEs using SAEBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all default evaluations
  python eval_dictionary_learning_saes.py --sae_dir ./my_saes/

  # Run only core evaluation
  python eval_dictionary_learning_saes.py --sae_dir ./my_saes/ --eval_types core

  # Run multiple evaluation types
  python eval_dictionary_learning_saes.py --sae_dir ./my_saes/ --eval_types core sparse_probing scr

  # Specify model and batch size
  python eval_dictionary_learning_saes.py --sae_dir ./my_saes/ --model_name pythia-160m-deduped --batch_size 256
        """,
    )
    
    parser.add_argument(
        "--sae_dir",
        type=str,
        required=True,
        help="Directory containing SAE config.json and ae.pt files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia-70m-deduped",
        help="Model name (default: pythia-70m-deduped)",
    )
    parser.add_argument(
        "--eval_types",
        type=str,
        nargs="+",
        default=["core", "sparse_probing", "scr", "tpp"],
        choices=["core", "sparse_probing", "scr", "tpp", "absorption", "unlearning"],
        help="Evaluation types to run (default: core sparse_probing scr tpp)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="LLM batch size (default: 512)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: float32)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results (default: ./eval_results)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./images",
        help="Output directory for plots (default: ./images)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Setup
    device = general_utils.setup_environment()
    print(f"Using device: {device}")

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Load SAEs
    print(f"\nLoading SAEs from: {args.sae_dir}")
    try:
        selected_saes = load_dictionary_learning_saes(
            root_dir=args.sae_dir,
            model_name=args.model_name,
            device=device,
            dtype=torch_dtype,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nERROR: {e}")
        return 1

    if not selected_saes:
        print("\nERROR: No SAEs were loaded. Check your directory structure.")
        return 1

    # Run evaluations
    if "core" in args.eval_types:
        run_core_eval(
            selected_saes=selected_saes,
            output_folder=f"{args.output_dir}/core",
            dtype=args.dtype,
        )

    # Run other evaluations via the unified interface
    other_eval_types = [et for et in args.eval_types if et != "core"]
    if other_eval_types:
        run_all_evals(
            selected_saes=selected_saes,
            model_name=args.model_name,
            device=device,
            eval_types=other_eval_types,
            llm_batch_size=args.batch_size,
            dtype=args.dtype,
        )

    # Generate plots
    if not args.no_plot:
        print("\n" + "=" * 60)
        print("Generating plots")
        print("=" * 60)
        generate_plots(
            results_folder=args.output_dir,
            image_folder=args.image_dir,
            eval_types=args.eval_types,
        )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    if not args.no_plot:
        print(f"Plots saved to: {args.image_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
