"""
Hyperparameter Search for Steering Experiments

Systematically explores different steering hyperparameters:
1. SAE_CONFIG: "k40_l2w0" (no L2 reg) or "k40_l2w0001" (with L2 reg)
2. STEERING_SEED: Which random seed's SAE to use (0, 1, 2)
3. DETERMINISTIC: True/False - controls sampling vs greedy decoding
4. APPLY_ONLY_GENERATION_STEPS: True/False - apply steering only during generation
5. STEERING_STRENGTH_MODE: "fixed", "resid_rms", "target_feat_preact_delta"
6. Strength values (mode-dependent):
   - fixed: STEERING_STRENGTH values
   - resid_rms: STEERING_N_RESID_RMS values
   - target_feat_preact_delta: TARGET_FEATURE_PREACT_DELTA values

Usage: python steering_hparam_search.py
"""

import csv
import json
import os
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
from transformer_lens import HookedTransformer

# Local imports
from shared_feature_analysis import (
    K40_CONFIGS,
    DATA_RESULTS_DIR,
    SHARED_FEATURE_CACHE_DIR,
    load_or_compute_shared_features,
)
from steering_shared_features import (
    MODEL_NAME,
    HOOK_NAME,
    NEUTRAL_PROMPTS,
    OPENAI_API_KEY_FILE,
    set_seed,
    estimate_resid_rms_at_hook,
    get_sae_path,
    load_autointerp_results,
    load_sae_state,
    generate_with_steering,
    select_features_by_sharedness,
    evaluate_steering_with_llm,
    judge_with_embeddings,
)

# Available SAE configurations
AVAILABLE_SAE_CONFIGS = list(K40_CONFIGS.keys())  # ["k40_l2w0", "k40_l2w0001"]
AVAILABLE_SEEDS = [0, 1, 2]  # Seeds available for each config

# Default SAE config (for backward compatibility)
DEFAULT_SAE_CONFIG = "k40_l2w0"
DEFAULT_STEERING_SEED = 0

# =============================================================================
# Hyperparameter Search Space
# =============================================================================

# Grid of hyperparameters to explore
HPARAM_GRID = {
    # SAE model configuration
    "sae_configs": ["k40_l2w0", "k40_l2w0001"],  # No L2 reg vs with L2 reg
    
    # SAE random seed (which seed's SAE to use for steering)
    "steering_seeds": [0],  # Default to seed 0; can expand to [0, 1, 2]
    
    # Decoding strategy
    "deterministic": [True, False],
    
    # Whether to apply steering only during generation (vs also prompt processing)
    "apply_only_generation_steps": [True, False],
    
    # Steering strength modes and their corresponding strength values
    "steering_modes": {
        "fixed": {
            "strength_param": "steering_strength",
            "values": [5.0, 10.0, 20.0],
        },
        "resid_rms": {
            "strength_param": "steering_n_resid_rms",
            "values": [0.6, 1.0, 3.0],
        },
        "target_feat_preact_delta": {
            "strength_param": "target_feat_preact_delta",
            "values": [1.0, 5.0, 10.0],
        },
    },
}

# Reduced feature count for faster hyperparameter search
N_FEATURES_PER_CATEGORY_HPARAM = 50  # Fewer features for speed
N_PROMPTS_PER_FEATURE = 3  # Number of prompts per feature

# Output settings
OUTPUT_DIR = os.path.join(DATA_RESULTS_DIR, "hparam_search_results")
RANDOM_SEED = 42
MAX_NEW_TOKENS = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HparamConfig:
    """A single hyperparameter configuration."""
    deterministic: bool
    apply_only_generation_steps: bool
    steering_mode: str
    strength_value: float
    sae_config: str = DEFAULT_SAE_CONFIG  # "k40_l2w0" or "k40_l2w0001"
    steering_seed: int = DEFAULT_STEERING_SEED  # 0, 1, or 2
    
    @property
    def config_name(self) -> str:
        det = "det" if self.deterministic else "sample"
        gen = "genonly" if self.apply_only_generation_steps else "allsteps"
        return f"{self.sae_config}_s{self.steering_seed}_{self.steering_mode}_{self.strength_value}_{det}_{gen}"
    
    @property
    def config_name_short(self) -> str:
        """Shorter name without SAE config (for when comparing within same SAE)."""
        det = "det" if self.deterministic else "sample"
        gen = "genonly" if self.apply_only_generation_steps else "allsteps"
        return f"{self.steering_mode}_{self.strength_value}_{det}_{gen}"
    
    def to_dict(self) -> dict:
        return {
            "sae_config": self.sae_config,
            "steering_seed": self.steering_seed,
            "deterministic": self.deterministic,
            "apply_only_generation_steps": self.apply_only_generation_steps,
            "steering_mode": self.steering_mode,
            "strength_value": self.strength_value,
            "config_name": self.config_name,
        }


@dataclass
class HparamSearchResult:
    """Results for a single hyperparameter configuration."""
    config: HparamConfig
    n_samples: int
    mean_llm_score: float | None
    std_llm_score: float | None
    mean_embed_cosine: float | None
    std_embed_cosine: float | None
    frac_llm_high_change: float | None  # Fraction with LLM score >= 4
    frac_embed_match: float | None
    by_category: dict  # Per-sharedness-category metrics
    raw_results: list = field(default_factory=list)


# =============================================================================
# Hyperparameter Grid Generation
# =============================================================================

def generate_hparam_configs(
    grid: dict = HPARAM_GRID,
    subset_modes: list[str] | None = None,
    sae_configs: list[str] | None = None,
    steering_seeds: list[int] | None = None,
) -> list[HparamConfig]:
    """
    Generate all hyperparameter configurations from the grid.
    
    Args:
        grid: Hyperparameter grid dictionary
        subset_modes: Optional list of modes to include (e.g., ["fixed", "resid_rms"])
        sae_configs: Optional list of SAE configs to include (e.g., ["k40_l2w0"])
        steering_seeds: Optional list of seeds to include (e.g., [0, 1])
    
    Returns:
        List of HparamConfig objects
    """
    configs = []
    
    steering_modes = grid["steering_modes"]
    if subset_modes:
        steering_modes = {k: v for k, v in steering_modes.items() if k in subset_modes}
    
    # SAE configs to use
    sae_config_list = sae_configs if sae_configs else grid.get("sae_configs", [DEFAULT_SAE_CONFIG])
    seed_list = steering_seeds if steering_seeds else grid.get("steering_seeds", [DEFAULT_STEERING_SEED])
    
    for sae_cfg in sae_config_list:
        for seed in seed_list:
            for deterministic in grid["deterministic"]:
                for apply_gen_only in grid["apply_only_generation_steps"]:
                    for mode_name, mode_config in steering_modes.items():
                        for strength_val in mode_config["values"]:
                            configs.append(HparamConfig(
                                deterministic=deterministic,
                                apply_only_generation_steps=apply_gen_only,
                                steering_mode=mode_name,
                                strength_value=strength_val,
                                sae_config=sae_cfg,
                                steering_seed=seed,
                            ))
    
    return configs


def generate_reduced_configs(
    deterministic_values: list[bool] = [True, False],
    apply_gen_only_values: list[bool] = [True],
    modes_and_strengths: dict | None = None,
    sae_configs: list[str] | None = None,
    steering_seeds: list[int] | None = None,
) -> list[HparamConfig]:
    """
    Generate a reduced set of configs for faster exploration.
    
    Default focuses on key comparisons:
    - Both deterministic options
    - Only generation-step steering (most common)
    - Fewer strength values per mode
    - Single SAE config (can expand)
    """
    if modes_and_strengths is None:
        modes_and_strengths = {
            "fixed": [3.0, 10.0],
            "resid_rms": [0.6, 2.0],
            "target_feat_preact_delta": [5.0, 15.0],
        }
    
    if sae_configs is None:
        sae_configs = [DEFAULT_SAE_CONFIG]
    
    if steering_seeds is None:
        steering_seeds = [DEFAULT_STEERING_SEED]
    
    configs = []
    for sae_cfg in sae_configs:
        for seed in steering_seeds:
            for det in deterministic_values:
                for apply_gen in apply_gen_only_values:
                    for mode, strengths in modes_and_strengths.items():
                        for strength in strengths:
                            configs.append(HparamConfig(
                                deterministic=det,
                                apply_only_generation_steps=apply_gen,
                                steering_mode=mode,
                                strength_value=strength,
                                sae_config=sae_cfg,
                                steering_seed=seed,
                            ))
    return configs


# =============================================================================
# Steering with Configurable Hparams
# =============================================================================

def run_steering_with_hparams(
    model: HookedTransformer,
    W_dec: torch.Tensor,
    W_enc: torch.Tensor | None,
    feature_idx: int,
    prompt: str,
    config: HparamConfig,
    resid_rms_scale: float = 1.0,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, str]:
    """
    Run steering with specific hyperparameter configuration.
    
    Returns (original_text, steered_text).
    """
    # Map config to generate_with_steering parameters
    steering_strength = config.strength_value if config.steering_mode == "fixed" else 1.0
    
    return generate_with_steering(
        model=model,
        W_dec=W_dec,
        W_enc=W_enc,
        feature_idx=feature_idx,
        prompt=prompt,
        steering_strength=steering_strength,
        max_new_tokens=max_new_tokens,
        deterministic=config.deterministic,
        normalize=True,
        apply_only_generation_steps=config.apply_only_generation_steps,
        steering_strength_mode=config.steering_mode,
        resid_rms_scale=resid_rms_scale if config.steering_mode == "resid_rms" else None,
        target_feat_preact_delta=config.strength_value if config.steering_mode == "target_feat_preact_delta" else 5.0,
    )


# =============================================================================
# Evaluation Helpers
# =============================================================================

def compute_metrics_from_results(results: list[dict]) -> dict:
    """Compute aggregate metrics from a list of result dicts."""
    llm_scores = [r["llm_match_score"] for r in results if r.get("llm_match_score") is not None]
    embed_cosines = [r["embed_cosine"] for r in results if r.get("embed_cosine") is not None]
    embed_matches = [r["embed_match"] for r in results if r.get("embed_match") is not None]
    
    metrics = {
        "n_samples": len(results),
        "mean_llm_score": float(np.mean(llm_scores)) if llm_scores else None,
        "std_llm_score": float(np.std(llm_scores)) if llm_scores else None,
        "mean_embed_cosine": float(np.mean(embed_cosines)) if embed_cosines else None,
        "std_embed_cosine": float(np.std(embed_cosines)) if embed_cosines else None,
        "frac_llm_high_change": sum(s >= 4 for s in llm_scores) / len(llm_scores) if llm_scores else None,
        "frac_embed_match": sum(embed_matches) / len(embed_matches) if embed_matches else None,
    }
    return metrics


def compute_by_category_metrics(results: list[dict]) -> dict:
    """Compute metrics broken down by sharedness category."""
    by_cat = defaultdict(list)
    for r in results:
        cat = r.get("sharedness_category", "unknown")
        by_cat[cat].append(r)
    
    cat_metrics = {}
    for cat, cat_results in by_cat.items():
        cat_metrics[cat] = compute_metrics_from_results(cat_results)
    
    return cat_metrics


# =============================================================================
# Main Hyperparameter Search
# =============================================================================

def _load_sae_resources(
    sae_config: str,
    steering_seed: int,
    n_features_per_category: int,
    verbose: bool = True,
) -> dict:
    """
    Load all resources needed for a specific SAE configuration.
    
    Returns dict with: shared_result, feature_explanations, sae_weights, selected_features
    """
    if verbose:
        print(f"\n  Loading resources for {sae_config} seed {steering_seed}...")
    
    # Load shared feature analysis
    cache_path = os.path.join(SHARED_FEATURE_CACHE_DIR, f"{sae_config}_analysis.json")
    shared_result = load_or_compute_shared_features(
        config_name=sae_config,
        config=K40_CONFIGS[sae_config],
        cache_path=cache_path,
        verbose=verbose
    )
    
    # Load auto-interp results
    feature_explanations = load_autointerp_results(
        config_name=sae_config,
        seed=steering_seed,
    )
    
    if not feature_explanations and verbose:
        print(f"    WARNING: No auto-interp results for {sae_config} seed {steering_seed}")
    
    # Load SAE weights
    sae_path = get_sae_path(sae_config, steering_seed)
    sae = load_sae_state(sae_path)
    
    # Select features
    try:
        selected_features = select_features_by_sharedness(
            shared_result,
            n_per_category=n_features_per_category,
            seed=RANDOM_SEED,
            feature_explanations=feature_explanations,
            require_explanation=bool(feature_explanations),
        )
    except ValueError as e:
        if verbose:
            print(f"    Warning: {e}")
            print("    Proceeding without explanation requirement...")
        selected_features = select_features_by_sharedness(
            shared_result,
            n_per_category=n_features_per_category,
            seed=RANDOM_SEED,
            feature_explanations=feature_explanations,
            require_explanation=False,
        )
    
    # Flatten features
    all_features = []
    for category, features in selected_features.items():
        all_features.extend(features)
    
    return {
        "shared_result": shared_result,
        "feature_explanations": feature_explanations,
        "W_enc": sae["W_enc"],
        "W_dec": sae["W_dec"],
        "selected_features": selected_features,
        "all_features": all_features,
    }


def run_hparam_search(
    configs: list[HparamConfig] | None = None,
    n_features_per_category: int = N_FEATURES_PER_CATEGORY_HPARAM,
    n_prompts: int = N_PROMPTS_PER_FEATURE,
    use_llm_judge: bool = True,
    use_embed_judge: bool = True,
    output_dir: str = OUTPUT_DIR,
    verbose: bool = True,
) -> list[HparamSearchResult]:
    """
    Run hyperparameter search across all configurations.
    
    Args:
        configs: List of HparamConfig to test (or None for default grid)
        n_features_per_category: Features to sample per sharedness category
        n_prompts: Number of prompts to test per feature
        use_llm_judge: Whether to use LLM-based evaluation
        use_embed_judge: Whether to use embedding-based evaluation
        output_dir: Directory for output files
        verbose: Print progress
    
    Returns:
        List of HparamSearchResult for each configuration
    """
    set_seed(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate configs if not provided
    if configs is None:
        configs = generate_hparam_configs()
    
    print("="*70)
    print("Hyperparameter Search for Steering")
    print("="*70)
    print(f"Testing {len(configs)} configurations")
    print(f"Features per category: {n_features_per_category}")
    print(f"Prompts per feature: {n_prompts}")
    print(f"Device: {DEVICE}")
    
    # Identify unique SAE configs needed
    unique_sae_keys = set((cfg.sae_config, cfg.steering_seed) for cfg in configs)
    print(f"SAE configs to load: {len(unique_sae_keys)}")
    for sae_cfg, seed in sorted(unique_sae_keys):
        print(f"  - {sae_cfg} seed {seed}")
    
    # Load model (shared across all configs)
    print(f"\n[1] Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    
    resid_scale = estimate_resid_rms_at_hook(model, prompts=NEUTRAL_PROMPTS[:3], hook_name=HOOK_NAME)
    print(f"    Estimated resid RMS: {resid_scale:.4f}")
    
    # Pre-load all SAE resources (caching)
    print("\n[2] Loading SAE resources...")
    sae_resources = {}
    for sae_cfg, seed in unique_sae_keys:
        key = (sae_cfg, seed)
        sae_resources[key] = _load_sae_resources(
            sae_config=sae_cfg,
            steering_seed=seed,
            n_features_per_category=n_features_per_category,
            verbose=verbose,
        )
        n_features = len(sae_resources[key]["all_features"])
        print(f"    {sae_cfg} seed {seed}: {n_features} features")
    
    # Select prompts
    prompts_to_use = NEUTRAL_PROMPTS[:n_prompts]
    
    # Load API key for LLM judge
    api_key = None
    if use_llm_judge and os.path.exists(OPENAI_API_KEY_FILE):
        with open(OPENAI_API_KEY_FILE) as f:
            api_key = f.read().strip()
    
    # Run search
    print("\n[3] Running hyperparameter search...")
    all_search_results = []
    
    for cfg_idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {cfg_idx+1}/{len(configs)}: {config.config_name}")
        print(f"{'='*60}")
        print(f"  sae_config: {config.sae_config}")
        print(f"  steering_seed: {config.steering_seed}")
        print(f"  deterministic: {config.deterministic}")
        print(f"  apply_only_generation_steps: {config.apply_only_generation_steps}")
        print(f"  steering_mode: {config.steering_mode}")
        print(f"  strength_value: {config.strength_value}")
        
        # Get resources for this SAE config
        res_key = (config.sae_config, config.steering_seed)
        resources = sae_resources[res_key]
        W_enc = resources["W_enc"]
        W_dec = resources["W_dec"]
        all_features = resources["all_features"]
        
        config_results = []
        
        for feat_info in all_features:
            feature_idx = feat_info['feature_idx']
            category = feat_info['sharedness_category']
            expected_concept = feat_info.get('expected_concept', 'N/A')
            
            for prompt in prompts_to_use:
                try:
                    original, steered = run_steering_with_hparams(
                        model=model,
                        W_dec=W_dec,
                        W_enc=W_enc,
                        feature_idx=feature_idx,
                        prompt=prompt,
                        config=config,
                        resid_rms_scale=resid_scale,
                    )
                    
                    config_results.append({
                        'feature_idx': feature_idx,
                        'sharedness_category': category,
                        'share_count': feat_info['share_count'],
                        'expected_concept': expected_concept,
                        'prompt': prompt,
                        'original_text': original,
                        'steered_text': steered,
                        'config_name': config.config_name,
                        **config.to_dict(),
                    })
                except Exception as e:
                    print(f"    Error with feature {feature_idx}: {e}")
        
        print(f"  Generated {len(config_results)} steering samples")
        
        # Run evaluations
        if use_llm_judge and api_key:
            print("  Running LLM evaluation...")
            config_results = evaluate_steering_with_llm(config_results, api_key=api_key)
        
        if use_embed_judge:
            print("  Running embedding evaluation...")
            config_results = judge_with_embeddings(config_results)
        
        # Compute metrics
        overall_metrics = compute_metrics_from_results(config_results)
        by_category = compute_by_category_metrics(config_results)
        
        search_result = HparamSearchResult(
            config=config,
            n_samples=overall_metrics["n_samples"],
            mean_llm_score=overall_metrics["mean_llm_score"],
            std_llm_score=overall_metrics["std_llm_score"],
            mean_embed_cosine=overall_metrics["mean_embed_cosine"],
            std_embed_cosine=overall_metrics["std_embed_cosine"],
            frac_llm_high_change=overall_metrics["frac_llm_high_change"],
            frac_embed_match=overall_metrics["frac_embed_match"],
            by_category=by_category,
            raw_results=config_results,
        )
        all_search_results.append(search_result)
        
        # Print summary for this config
        print("\n  Results:")
        if overall_metrics["mean_llm_score"] is not None:
            print(f"    LLM score: {overall_metrics['mean_llm_score']:.2f} (±{overall_metrics['std_llm_score']:.2f})")
            print(f"    High change (≥4): {overall_metrics['frac_llm_high_change']*100:.1f}%")
        if overall_metrics["mean_embed_cosine"] is not None:
            print(f"    Embed cosine: {overall_metrics['mean_embed_cosine']:.3f} (±{overall_metrics['std_embed_cosine']:.3f})")
        
        # Save intermediate results
        _save_config_results(config, config_results, output_dir)
    
    # Save final summary
    _save_search_summary(all_search_results, output_dir)
    
    # Print final comparison
    print_search_comparison(all_search_results)
    
    return all_search_results


def _save_config_results(config: HparamConfig, results: list[dict], output_dir: str):
    """Save results for a single configuration."""
    filename = os.path.join(output_dir, f"{config.config_name}_results.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def _save_search_summary(results: list[HparamSearchResult], output_dir: str):
    """Save summary of all configurations."""
    summary = []
    for r in results:
        summary.append({
            **r.config.to_dict(),
            "n_samples": r.n_samples,
            "mean_llm_score": r.mean_llm_score,
            "std_llm_score": r.std_llm_score,
            "mean_embed_cosine": r.mean_embed_cosine,
            "std_embed_cosine": r.std_embed_cosine,
            "frac_llm_high_change": r.frac_llm_high_change,
            "frac_embed_match": r.frac_embed_match,
            "by_category": r.by_category,
        })
    
    # JSON
    json_path = os.path.join(output_dir, "hparam_search_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {json_path}")
    
    # CSV
    csv_path = os.path.join(output_dir, "hparam_search_summary.csv")
    fieldnames = [
        "config_name", "sae_config", "steering_seed", "steering_mode", "strength_value",
        "deterministic", "apply_only_generation_steps", "n_samples", "mean_llm_score",
        "std_llm_score", "mean_embed_cosine", "std_embed_cosine", "frac_llm_high_change",
        "frac_embed_match"
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for s in summary:
            writer.writerow(s)
    print(f"Saved CSV to {csv_path}")


def print_search_comparison(results: list[HparamSearchResult]):
    """Print a comparison table of all configurations."""
    print("\n" + "="*100)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*100)
    
    # Sort by mean LLM score (descending) if available, else by embed cosine
    sorted_results = sorted(
        results,
        key=lambda r: (r.mean_llm_score or 0, r.mean_embed_cosine or 0),
        reverse=True
    )
    
    print(f"\n{'Config':<55} {'LLM Score':<12} {'High Δ%':<10} {'Embed Cos':<12}")
    print("-"*100)
    
    for r in sorted_results:
        llm_str = f"{r.mean_llm_score:.2f}±{r.std_llm_score:.2f}" if r.mean_llm_score else "N/A"
        high_change = f"{r.frac_llm_high_change*100:.1f}%" if r.frac_llm_high_change else "N/A"
        embed_str = f"{r.mean_embed_cosine:.3f}±{r.std_embed_cosine:.3f}" if r.mean_embed_cosine else "N/A"
        
        print(f"{r.config.config_name:<55} {llm_str:<12} {high_change:<10} {embed_str:<12}")
    
    # Best by SAE config (if multiple)
    sae_configs = set(r.config.sae_config for r in results)
    if len(sae_configs) > 1:
        print("\n" + "-"*100)
        print("BEST CONFIG PER SAE MODEL:")
        print("-"*100)
        
        for sae_cfg in sorted(sae_configs):
            sae_results = [r for r in sorted_results if r.config.sae_config == sae_cfg]
            if sae_results:
                best = sae_results[0]
                score_str = f"LLM={best.mean_llm_score:.2f}" if best.mean_llm_score else "N/A"
                print(f"  {sae_cfg}: {best.config.config_name_short} ({score_str})")
    
    # Best by mode
    print("\n" + "-"*100)
    print("BEST CONFIG PER STEERING MODE:")
    print("-"*100)
    
    modes = set(r.config.steering_mode for r in results)
    for mode in sorted(modes):
        mode_results = [r for r in sorted_results if r.config.steering_mode == mode]
        if mode_results:
            best = mode_results[0]
            score_str = f"LLM={best.mean_llm_score:.2f}" if best.mean_llm_score else "N/A"
            print(f"  {mode}: {best.config.config_name} ({score_str})")
    
    # By-category comparison for best config
    if sorted_results and sorted_results[0].by_category:
        print("\n" + "-"*100)
        print(f"BY-CATEGORY BREAKDOWN (best config: {sorted_results[0].config.config_name}):")
        print("-"*100)
        
        by_cat = sorted_results[0].by_category
        for cat, metrics in by_cat.items():
            llm = metrics.get("mean_llm_score")
            embed = metrics.get("mean_embed_cosine")
            llm_str = f"{llm:.2f}" if llm else "N/A"
            embed_str = f"{embed:.3f}" if embed else "N/A"
            print(f"  {cat:<20}: LLM={llm_str}, Embed={embed_str} (n={metrics['n_samples']})")


# =============================================================================
# Quick Exploration Modes
# =============================================================================

def run_quick_mode_comparison(sae_config: str = DEFAULT_SAE_CONFIG, seed: int = DEFAULT_STEERING_SEED):
    """Run a quick comparison of steering modes with default strengths."""
    print(f"Running quick mode comparison (SAE: {sae_config}, seed: {seed})...")
    configs = [
        HparamConfig(True, True, "fixed", 5.0, sae_config, seed),
        HparamConfig(True, True, "resid_rms", 0.6, sae_config, seed),
        HparamConfig(True, True, "target_feat_preact_delta", 5.0, sae_config, seed),
    ]
    return run_hparam_search(
        configs=configs,
        n_features_per_category=5,
        n_prompts=2,
        output_dir=os.path.join(DATA_RESULTS_DIR, f"hparam_search_quick_{sae_config}"),
    )


def run_strength_sweep(
    mode: str = "resid_rms",
    sae_config: str = DEFAULT_SAE_CONFIG,
    seed: int = DEFAULT_STEERING_SEED,
):
    """Sweep strength values for a single mode."""
    print(f"Running strength sweep for mode: {mode} (SAE: {sae_config}, seed: {seed})")
    
    mode_config = HPARAM_GRID["steering_modes"][mode]
    strengths = mode_config["values"]
    
    configs = [
        HparamConfig(True, True, mode, s, sae_config, seed)
        for s in strengths
    ]
    
    return run_hparam_search(
        configs=configs,
        n_features_per_category=10,
        n_prompts=3,
        output_dir=os.path.join(DATA_RESULTS_DIR, f"hparam_search_{mode}_sweep_{sae_config}"),
    )


def run_deterministic_comparison(sae_config: str = DEFAULT_SAE_CONFIG, seed: int = DEFAULT_STEERING_SEED):
    """Compare deterministic vs sampling across modes."""
    print(f"Running deterministic vs sampling comparison (SAE: {sae_config}, seed: {seed})...")
    
    configs = []
    for det in [True, False]:
        configs.extend([
            HparamConfig(det, True, "fixed", 5.0, sae_config, seed),
            HparamConfig(det, True, "resid_rms", 0.6, sae_config, seed),
            HparamConfig(det, True, "target_feat_preact_delta", 5.0, sae_config, seed),
        ])
    
    return run_hparam_search(
        configs=configs,
        n_features_per_category=10,
        n_prompts=3,
        output_dir=os.path.join(DATA_RESULTS_DIR, f"hparam_search_deterministic_{sae_config}"),
    )


def run_generation_step_comparison(sae_config: str = DEFAULT_SAE_CONFIG, seed: int = DEFAULT_STEERING_SEED):
    """Compare applying steering to all steps vs generation only."""
    print(f"Running generation step comparison (SAE: {sae_config}, seed: {seed})...")
    
    configs = []
    for apply_gen in [True, False]:
        configs.extend([
            HparamConfig(True, apply_gen, "fixed", 5.0, sae_config, seed),
            HparamConfig(True, apply_gen, "resid_rms", 0.6, sae_config, seed),
            HparamConfig(True, apply_gen, "target_feat_preact_delta", 5.0, sae_config, seed),
        ])
    
    return run_hparam_search(
        configs=configs,
        n_features_per_category=10,
        n_prompts=3,
        output_dir=os.path.join(DATA_RESULTS_DIR, f"hparam_search_genstep_{sae_config}"),
    )


def run_sae_config_comparison(seed: int = DEFAULT_STEERING_SEED):
    """Compare different SAE configurations (L2 reg vs no L2 reg)."""
    print(f"Running SAE config comparison (seed: {seed})...")
    
    configs = []
    for sae_cfg in AVAILABLE_SAE_CONFIGS:
        configs.extend([
            HparamConfig(True, True, "fixed", 5.0, sae_cfg, seed),
            HparamConfig(True, True, "resid_rms", 0.6, sae_cfg, seed),
            HparamConfig(True, True, "target_feat_preact_delta", 5.0, sae_cfg, seed),
        ])
    
    return run_hparam_search(
        configs=configs,
        n_features_per_category=10,
        n_prompts=3,
        output_dir=os.path.join(DATA_RESULTS_DIR, "hparam_search_sae_comparison"),
    )


def run_seed_comparison(sae_config: str = DEFAULT_SAE_CONFIG):
    """Compare different random seeds for the same SAE config."""
    print(f"Running seed comparison for SAE: {sae_config}...")
    
    configs = []
    for seed in AVAILABLE_SEEDS:
        configs.extend([
            HparamConfig(True, True, "fixed", 5.0, sae_config, seed),
            HparamConfig(True, True, "resid_rms", 0.6, sae_config, seed),
            HparamConfig(True, True, "target_feat_preact_delta", 5.0, sae_config, seed),
        ])
    
    return run_hparam_search(
        configs=configs,
        n_features_per_category=10,
        n_prompts=3,
        output_dir=os.path.join(DATA_RESULTS_DIR, f"hparam_search_seed_comparison_{sae_config}"),
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter search for steering")
    parser.add_argument("--mode", choices=[
        "full", "quick", "strength", "deterministic", "genstep", "reduced",
        "sae-compare", "seed-compare"
    ], default="reduced", help="Search mode")
    parser.add_argument("--sweep-mode", choices=["fixed", "resid_rms", "target_feat_preact_delta"],
                        default="resid_rms", help="Mode for strength sweep")
    parser.add_argument("--sae-config", choices=AVAILABLE_SAE_CONFIGS,
                        default=DEFAULT_SAE_CONFIG, help="SAE configuration to use")
    parser.add_argument("--sae-configs", nargs="+", choices=AVAILABLE_SAE_CONFIGS,
                        default=None, help="Multiple SAE configs for grid search")
    parser.add_argument("--seed", type=int, choices=AVAILABLE_SEEDS,
                        default=DEFAULT_STEERING_SEED, help="SAE random seed to use")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=None, help="Multiple seeds for grid search (e.g., --seeds 0 1 2)")
    parser.add_argument("--n-features", type=int, default=N_FEATURES_PER_CATEGORY_HPARAM,
                        help="Features per sharedness category")
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS_PER_FEATURE,
                        help="Prompts per feature")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM evaluation")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding evaluation")
    
    args = parser.parse_args()
    
    # Validate seed values if provided
    if args.seeds:
        for s in args.seeds:
            if s not in AVAILABLE_SEEDS:
                parser.error(f"Invalid seed {s}. Available seeds: {AVAILABLE_SEEDS}")
    
    if args.mode == "quick":
        run_quick_mode_comparison(sae_config=args.sae_config, seed=args.seed)
    elif args.mode == "strength":
        run_strength_sweep(mode=args.sweep_mode, sae_config=args.sae_config, seed=args.seed)
    elif args.mode == "deterministic":
        run_deterministic_comparison(sae_config=args.sae_config, seed=args.seed)
    elif args.mode == "genstep":
        run_generation_step_comparison(sae_config=args.sae_config, seed=args.seed)
    elif args.mode == "sae-compare":
        run_sae_config_comparison(seed=args.seed)
    elif args.mode == "seed-compare":
        run_seed_comparison(sae_config=args.sae_config)
    elif args.mode == "reduced":
        # Use specified SAE configs and seeds, or defaults
        sae_configs = args.sae_configs if args.sae_configs else [args.sae_config]
        seeds = args.seeds if args.seeds else [args.seed]
        
        configs = generate_reduced_configs(sae_configs=sae_configs, steering_seeds=seeds)
        run_hparam_search(
            configs=configs,
            n_features_per_category=args.n_features,
            n_prompts=args.n_prompts,
            use_llm_judge=not args.no_llm,
            use_embed_judge=not args.no_embed,
            output_dir=os.path.join(DATA_RESULTS_DIR, f"hparam_search_reduced_{sae_configs[0]}"),
        )
    else:  # full
        # Use specified SAE configs and seeds, or defaults from grid
        sae_configs = args.sae_configs if args.sae_configs else None
        seeds = args.seeds if args.seeds else None
        
        configs = generate_hparam_configs(sae_configs=sae_configs, steering_seeds=seeds)
        run_hparam_search(
            configs=configs,
            n_features_per_category=args.n_features,
            n_prompts=args.n_prompts,
            use_llm_judge=not args.no_llm,
            use_embed_judge=not args.no_embed,
        )


if __name__ == "__main__":
    main()
