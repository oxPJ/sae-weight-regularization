"""
Auto-Interp Evaluation: All Alive Features (Ignoring Sharedness)

This script runs auto-interp evaluation on a random sample of ALL alive features,
regardless of their sharedness status. This provides a baseline for comparison
with the sharedness-stratified analysis.

"Alive" features are those with non-zero encoder/decoder cosine similarity,
indicating the feature has learned something (as opposed to dead features).

Usage: nohup python autointerp_all_features.py > autointerp_all.log 2>&1 &
"""

import json
import os
import torch
import numpy as np
from dataclasses import dataclass, asdict
import random

import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.sae_bench_utils.general_utils as general_utils

# Local imports
from shared_feature_analysis import (
    K40_CONFIGS,
    MODEL_BASE_DIR,
    load_or_compute_shared_features,
    load_sae_weights,
)


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "pythia-70m-deduped"

# Which k=40 configuration to use
SAE_CONFIG_NAME = "k40_l2w0001"  # or "k40_l2w0001"

# Which seed to use for the SAE
STEERING_SEED = 0

# Auto-interp parameters
LLM_BATCH_SIZE = 512
LLM_DTYPE = "float16"
RANDOM_SEED = 42

# Number of features to evaluate (sampled from ALL alive features)
N_FEATURES_TO_SAMPLE = 300

OUTPUT_DIR = "eval_results/autointerp_all_features"
LOGS_DIR = "eval_results/autointerp_all_features/logs"


# =============================================================================
# Alive Feature Detection
# =============================================================================

def compute_feature_cosine_similarities(sae_path: str) -> tuple[dict[int, float], int]:
    """
    Compute cosine similarity between encoder and decoder for each feature.
    
    Features with cosine_sim == 0 are "dead" (either encoder or decoder has zero norm).
    
    Returns:
        Tuple of (dict mapping feature_idx -> cosine_sim, total n_features)
    """
    weights = load_sae_weights(sae_path)
    
    # Get encoder and decoder weights
    if 'encoder.weight' in weights:
        We = weights['encoder.weight'].cpu().float().numpy()
        Wd = weights['decoder.weight'].cpu().float().numpy()
    elif 'W_enc' in weights:
        We = weights['W_enc'].cpu().float().numpy()
        Wd = weights['W_dec'].cpu().float().numpy()
    else:
        raise KeyError(f"Could not find encoder weights. Keys: {weights.keys()}")
    
    # Detect layout - encoder.weight for nn.Linear is [out_features, in_features]
    # For SAE: encoder maps activation_dim -> dict_size, so shape [dict_size, activation_dim]
    if We.shape[0] > We.shape[1]:
        # encoder.weight has shape [dict_size, activation_dim] (standard nn.Linear)
        n_feats = We.shape[0]
        is_transposed = False
    else:
        # Transposed layout
        n_feats = We.shape[1]
        is_transposed = True
    
    cos_sims = {}
    
    for i in range(n_feats):
        if is_transposed:
            enc_vec = We[:, i]
            dec_vec = Wd[i, :]
        else:
            enc_vec = We[i, :]
            dec_vec = Wd[:, i]
        
        norm_e = np.linalg.norm(enc_vec)
        norm_d = np.linalg.norm(dec_vec)
        
        if norm_e == 0 or norm_d == 0:
            cos_sims[i] = 0.0
        else:
            cos_sims[i] = float(np.dot(enc_vec, dec_vec) / (norm_e * norm_d))
    
    return cos_sims, n_feats


def get_alive_feature_indices(cos_sims: dict[int, float]) -> list[int]:
    """Get indices of all alive features (non-zero cosine similarity)."""
    return [idx for idx, cs in cos_sims.items() if cs != 0.0]


# =============================================================================
# SAE Loading (adapted for TopK)
# =============================================================================

def load_local_topk_sae(
    sae_path: str,
    model_name: str,
    device: str,
    dtype: torch.dtype,
) -> base_sae.BaseSAE:
    """Load a TopK SAE from local path."""
    config_path = os.path.join(sae_path, "config.json")
    weights_path = os.path.join(sae_path, "ae.pt")
    
    with open(config_path) as f:
        config = json.load(f)
    
    pt_params = torch.load(weights_path, map_location=torch.device("cpu"))
    
    layer = config["trainer"]["layer"]
    k = config["trainer"]["k"]
    
    print(f"Loading SAE from {sae_path}")
    print(f"  Layer: {layer}, K: {k}")
    
    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
        "k": "k",
    }
    
    # Remove threshold if present
    if "threshold" in pt_params:
        del pt_params["threshold"]
    
    renamed_params = {key_mapping.get(k_name, k_name): v for k_name, v in pt_params.items()}
    
    # Transpose weight matrices (due to nn.Linear convention)
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T
    
    sae = topk_sae.TopKSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        k=k,
        model_name=model_name,
        hook_layer=layer,
        device=torch.device(device),
        dtype=dtype,
        use_threshold=False,
    )
    
    sae.load_state_dict(renamed_params)
    sae.to(device=device, dtype=dtype)
    sae.cfg.architecture = "topk"
    
    return sae


def get_sae_path(config_name: str, seed: int, base_dir: str = MODEL_BASE_DIR) -> str:
    """Get full path to SAE for a given config and seed."""
    config = K40_CONFIGS[config_name]
    rel_path = config['seeds'][seed]
    return os.path.join(base_dir, rel_path)


# =============================================================================
# Feature Selection
# =============================================================================

def select_random_alive_features(
    cos_sims: dict[int, float],
    n_samples: int,
    seed: int = 42,
    shared_result=None,  # Optional, for tracking sharedness info
) -> list[dict]:
    """
    Randomly sample from all alive features.
    
    Args:
        cos_sims: dict mapping feature_idx -> cosine_similarity
        n_samples: number of features to sample
        seed: random seed
        shared_result: Optional SharedFeatureAnalysisResult for annotating sharedness
    
    Returns:
        List of feature info dicts with idx, cos_sim, and optional sharedness info
    """
    np.random.seed(seed)
    random.seed(seed)
    
    alive_indices = get_alive_feature_indices(cos_sims)
    n_alive = len(alive_indices)
    
    print(f"  Total features: {len(cos_sims)}")
    print(f"  Alive features: {n_alive}")
    print(f"  Dead features: {len(cos_sims) - n_alive}")
    
    n_sample = min(n_samples, n_alive)
    sampled_indices = list(np.random.choice(alive_indices, n_sample, replace=False))
    
    # Build sharedness lookup if provided
    sharedness_lookup = {}
    if shared_result is not None:
        for idx in shared_result.always_shared_indices:
            sharedness_lookup[idx] = ("always_shared", shared_result.n_seed_pairs)
        for idx in shared_result.sometimes_shared_indices:
            count = shared_result.feature_share_counts[idx]
            sharedness_lookup[idx] = ("sometimes_shared", count)
        for idx in shared_result.never_shared_indices:
            sharedness_lookup[idx] = ("never_shared", 0)
    
    features = []
    for idx in sampled_indices:
        feat_info = {
            'feature_idx': int(idx),
            'cosine_similarity': cos_sims[idx],
        }
        
        if shared_result is not None:
            cat, count = sharedness_lookup.get(idx, ("unknown", 0))
            feat_info['sharedness_category'] = cat
            feat_info['share_count'] = count
            feat_info['share_fraction'] = count / shared_result.n_seed_pairs if shared_result.n_seed_pairs > 0 else 0
        
        features.append(feat_info)
    
    print(f"  Sampled {len(features)} features")
    
    # Print sharedness distribution if available
    if shared_result is not None:
        cat_counts = {}
        for f in features:
            cat = f.get('sharedness_category', 'unknown')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"  Sharedness distribution in sample:")
        for cat, count in sorted(cat_counts.items()):
            print(f"    {cat}: {count} ({count/len(features)*100:.1f}%)")
    
    return features


# =============================================================================
# Result Storage
# =============================================================================

@dataclass
class AllFeaturesAutoInterpResult:
    """Results for all-features auto-interp analysis."""
    sae_config: str
    sae_seed: int
    n_features_total: int
    n_alive_features: int
    n_sampled: int
    n_evaluated: int
    mean_score: float
    std_score: float
    median_score: float
    # Breakdown by sharedness (for comparison)
    always_shared_mean: float
    sometimes_shared_mean: float
    never_shared_mean: float
    always_shared_n: int
    sometimes_shared_n: int
    never_shared_n: int
    feature_results: list


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print("=" * 60)
    print("Auto-Interp Evaluation: All Alive Features")
    print("=" * 60)
    
    # Setup
    device = general_utils.setup_environment()
    dtype = general_utils.str_to_dtype(LLM_DTYPE)
    
    # Load API key
    try:
        with open("openai_api_key.txt") as f:
            api_key = f.read().strip()
        print("✓ OpenAI API key loaded")
    except FileNotFoundError:
        raise Exception("Please create openai_api_key.txt with your API key")
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Step 1: Load shared feature analysis (for annotating results, not filtering)
    print(f"\n[1] Loading shared feature analysis for {SAE_CONFIG_NAME}...")
    cache_path = f"shared_feature_cache/{SAE_CONFIG_NAME}_analysis.json"
    shared_result = load_or_compute_shared_features(
        config_name=SAE_CONFIG_NAME,
        config=K40_CONFIGS[SAE_CONFIG_NAME],
        cache_path=cache_path,
        verbose=True
    )
    
    print(f"\n    Always shared: {shared_result.n_always_shared} features")
    print(f"    Sometimes shared: {shared_result.n_sometimes_shared} features")
    print(f"    Never shared: {len(shared_result.never_shared_indices)} features")
    
    # Step 2: Get SAE path and compute alive features
    print(f"\n[2] Computing alive features...")
    sae_path = get_sae_path(SAE_CONFIG_NAME, STEERING_SEED)
    cos_sims, n_features = compute_feature_cosine_similarities(sae_path)
    
    # Step 3: Sample random alive features
    print(f"\n[3] Sampling {N_FEATURES_TO_SAMPLE} random alive features...")
    selected_features = select_random_alive_features(
        cos_sims=cos_sims,
        n_samples=N_FEATURES_TO_SAMPLE,
        seed=RANDOM_SEED,
        shared_result=shared_result,
    )
    
    feature_indices = [f['feature_idx'] for f in selected_features]
    
    # Step 4: Load SAE
    print(f"\n[4] Loading SAE (seed={STEERING_SEED})...")
    sae = load_local_topk_sae(
        sae_path=sae_path,
        model_name=MODEL_NAME,
        device=device,
        dtype=dtype,
    )
    sae_name = f"{SAE_CONFIG_NAME}_seed{STEERING_SEED}_all_features"
    
    # Step 5: Run auto-interp evaluation
    print(f"\n[5] Running auto-interp on {len(feature_indices)} features...")
    
    config = autointerp.AutoInterpEvalConfig(
        model_name=MODEL_NAME,
        llm_batch_size=LLM_BATCH_SIZE,
        llm_dtype=LLM_DTYPE,
        random_seed=RANDOM_SEED,
        n_latents=None,
        override_latents=[int(i) for i in feature_indices],
    )
    
    log_path = os.path.join(LOGS_DIR, f"{sae_name}_detailed.txt")
    
    results = autointerp.run_eval(
        config=config,
        selected_saes=[(sae_name, sae)],
        device=device,
        api_key=api_key,
        output_path=OUTPUT_DIR,
        force_rerun=True,
        save_logs_path=log_path,
    )
    
    # Step 6: Extract per-feature results
    result_key = list(results.keys())[0]
    raw_results = results[result_key].get("eval_result_unstructured", {})
    
    feature_results = []
    all_scores = []
    scores_by_category = {
        "always_shared": [],
        "sometimes_shared": [],
        "never_shared": [],
    }
    
    for feat_info in selected_features:
        feat_idx = feat_info['feature_idx']
        feat_data = raw_results.get(feat_idx, raw_results.get(str(feat_idx), {}))
        
        score = float(feat_data.get("score", 0.0))
        explanation = feat_data.get("explanation", "N/A")
        
        if explanation and explanation != "N/A":
            all_scores.append(score)
            cat = feat_info.get('sharedness_category', 'unknown')
            if cat in scores_by_category:
                scores_by_category[cat].append(score)
        
        feature_results.append({
            'feature_idx': feat_idx,
            'cosine_similarity': feat_info['cosine_similarity'],
            'sharedness_category': feat_info.get('sharedness_category', 'unknown'),
            'share_count': feat_info.get('share_count', 0),
            'autointerp_score': score,
            'explanation': explanation,
        })
    
    # Step 7: Compute statistics
    all_scores = np.array(all_scores)
    n_alive = len(get_alive_feature_indices(cos_sims))
    
    overall_result = AllFeaturesAutoInterpResult(
        sae_config=SAE_CONFIG_NAME,
        sae_seed=STEERING_SEED,
        n_features_total=n_features,
        n_alive_features=n_alive,
        n_sampled=len(feature_indices),
        n_evaluated=len(all_scores),
        mean_score=float(np.mean(all_scores)) if len(all_scores) > 0 else 0.0,
        std_score=float(np.std(all_scores)) if len(all_scores) > 0 else 0.0,
        median_score=float(np.median(all_scores)) if len(all_scores) > 0 else 0.0,
        always_shared_mean=float(np.mean(scores_by_category["always_shared"])) if scores_by_category["always_shared"] else 0.0,
        sometimes_shared_mean=float(np.mean(scores_by_category["sometimes_shared"])) if scores_by_category["sometimes_shared"] else 0.0,
        never_shared_mean=float(np.mean(scores_by_category["never_shared"])) if scores_by_category["never_shared"] else 0.0,
        always_shared_n=len(scores_by_category["always_shared"]),
        sometimes_shared_n=len(scores_by_category["sometimes_shared"]),
        never_shared_n=len(scores_by_category["never_shared"]),
        feature_results=feature_results,
    )
    
    # Step 8: Save results
    result_path = os.path.join(OUTPUT_DIR, f"{sae_name}_analysis.json")
    with open(result_path, "w") as f:
        json.dump(asdict(overall_result), f, indent=2)
    print(f"\n✓ Saved results to {result_path}")
    
    # Clean up
    del sae
    torch.cuda.empty_cache()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nConfiguration: {SAE_CONFIG_NAME}, Seed: {STEERING_SEED}")
    print(f"Total features: {n_features}")
    print(f"Alive features: {n_alive}")
    print(f"Sampled: {len(feature_indices)}")
    print(f"Evaluated: {len(all_scores)}")
    print(f"\nOverall Auto-Interp Score:")
    print(f"  Mean: {overall_result.mean_score:.4f} (±{overall_result.std_score:.4f})")
    print(f"  Median: {overall_result.median_score:.4f}")
    
    print(f"\n{'Category':<20} {'Mean Score':<12} {'N':<6}")
    print("-" * 40)
    print(f"{'always_shared':<20} {overall_result.always_shared_mean:<12.4f} {overall_result.always_shared_n:<6}")
    print(f"{'sometimes_shared':<20} {overall_result.sometimes_shared_mean:<12.4f} {overall_result.sometimes_shared_n:<6}")
    print(f"{'never_shared':<20} {overall_result.never_shared_mean:<12.4f} {overall_result.never_shared_n:<6}")
    
    # Compare to sharedness-stratified baseline
    shared_avg = (overall_result.always_shared_mean + overall_result.sometimes_shared_mean) / 2
    if overall_result.always_shared_n + overall_result.sometimes_shared_n > 0:
        shared_avg = (
            overall_result.always_shared_mean * overall_result.always_shared_n +
            overall_result.sometimes_shared_mean * overall_result.sometimes_shared_n
        ) / (overall_result.always_shared_n + overall_result.sometimes_shared_n)
    
    print(f"\nShared (weighted avg): {shared_avg:.4f}")
    print(f"Never shared: {overall_result.never_shared_mean:.4f}")
    diff = shared_avg - overall_result.never_shared_mean
    print(f"Difference (shared - never): {diff:+.4f}")


if __name__ == "__main__":
    main()
