"""
Shared Feature Analysis Module

Computes which features are consistently shared across SAEs trained with different
random seeds, following the methodology from Braun et al. (2024).

Key metrics:
- Mean Max Cosine Similarity: For each feature in SAE1, find max similarity with any feature in SAE2
- Fraction Paired (>0.7): Fraction of features with max similarity > 0.7
- Shared Features: Features where encoder AND decoder Hungarian matchings agree with sim > 0.7
"""

import os
import json
import torch
import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, asdict
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

# Shared paths are anchored at the repository root so these scripts work from any cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

DATA_RESULTS_DIR = os.path.join(_REPO_ROOT, "data_results")
SHARED_FEATURE_CACHE_DIR = os.path.join(DATA_RESULTS_DIR, "shared_feature_cache")

MODEL_BASE_DIR = os.path.join(
    _REPO_ROOT,
    "data_model_weights",
    "random_seeds_constrained_saes_EleutherAI_pythia-70m-deduped_top_k_l2",
)

# k=40 model trainers with different seeds
K40_CONFIGS = {
    "k40_l2w0": {
        "k": 40,
        "l2_w": 0,
        "seeds": {
            0: "resid_post_layer_3/trainer_0",
            1: "resid_post_layer_3/trainer_8",
            2: "resid_post_layer_3/trainer_16",
        }
    },
    "k40_l2w0001": {
        "k": 40,
        "l2_w": 0.0001,
        "seeds": {
            0: "resid_post_layer_3/trainer_1",
            1: "resid_post_layer_3/trainer_9",
            2: "resid_post_layer_3/trainer_17",
        }
    }
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureSharedInfo:
    """Information about a feature's sharedness across seeds."""
    feature_idx: int
    share_count: int  # How many seed pairs this feature is shared in
    avg_similarity: float  # Average similarity when shared
    partners: list  # List of matching partner indices
    is_always_shared: bool  # Shared in ALL seed pairs
    is_sometimes_shared: bool  # Shared in at least one seed pair


@dataclass
class SharedFeatureAnalysisResult:
    """Results of shared feature analysis for a configuration."""
    config_name: str
    k: int
    l2_w: float
    n_features: int
    n_seed_pairs: int
    n_always_shared: int
    n_sometimes_shared: int
    frac_always_shared: float
    frac_sometimes_shared: float
    mean_max_cos_avg: float
    frac_paired_dec_avg: float
    feature_share_counts: list  # List of share counts for each feature
    always_shared_indices: list
    sometimes_shared_indices: list
    never_shared_indices: list


# =============================================================================
# Feature Extraction
# =============================================================================

def load_sae_weights(sae_path: str) -> dict:
    """Load SAE weights from a path."""
    weights_path = os.path.join(sae_path, "ae.pt")
    weights = torch.load(weights_path, map_location="cpu")
    return weights


def get_decoder_features(weights: dict) -> np.ndarray:
    """Extract decoder features. Returns shape [dict_size, activation_dim]."""
    if 'decoder.weight' in weights:
        dec = weights['decoder.weight'].cpu().float().numpy()
    elif 'W_dec' in weights:
        dec = weights['W_dec'].cpu().float().numpy()
    else:
        raise KeyError(f"Could not find decoder weights. Keys: {weights.keys()}")
    
    # decoder.weight has shape [activation_dim, dict_size], transpose to get features as rows
    if dec.shape[0] < dec.shape[1]:
        return dec.T
    return dec


def get_encoder_features(weights: dict) -> np.ndarray:
    """Extract encoder features. Returns shape [dict_size, activation_dim]."""
    if 'encoder.weight' in weights:
        enc = weights['encoder.weight'].cpu().float().numpy()
    elif 'W_enc' in weights:
        enc = weights['W_enc'].cpu().float().numpy()
    else:
        raise KeyError(f"Could not find encoder weights. Keys: {weights.keys()}")
    
    # encoder.weight has shape [dict_size, activation_dim] for nn.Linear
    if enc.shape[0] < enc.shape[1]:
        return enc.T
    return enc


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to unit norm (per row)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norms + 1e-8)


def cosine_similarity_matrix(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between features."""
    a_norm = normalize_features(features_a)
    b_norm = normalize_features(features_b)
    return a_norm @ b_norm.T


def hungarian_matching(sim_matrix: np.ndarray):
    """Find optimal 1-to-1 matching using Hungarian algorithm."""
    cost_matrix = -sim_matrix  # minimize negative = maximize positive
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_sims = sim_matrix[row_ind, col_ind]
    return row_ind, col_ind, matched_sims


# =============================================================================
# Shared Feature Computation
# =============================================================================

def compute_pairwise_shared_features(
    weights_a: dict,
    weights_b: dict,
    threshold: float = 0.7
) -> dict:
    """
    Compute shared features between two SAEs.
    
    A feature is "shared" if:
    1. Hungarian matching on encoder agrees with matching on decoder
    2. Both similarities > threshold
    
    Returns dict with metrics and per-feature results.
    """
    # Get features
    dec_a = get_decoder_features(weights_a)
    dec_b = get_decoder_features(weights_b)
    enc_a = get_encoder_features(weights_a)
    enc_b = get_encoder_features(weights_b)
    
    # Compute similarity matrices (using absolute value to handle sign flips)
    dec_sim = np.abs(cosine_similarity_matrix(dec_a, dec_b))
    enc_sim = np.abs(cosine_similarity_matrix(enc_a, enc_b))
    
    # Mean Max Cosine Similarity
    max_sim_dec_a = np.max(dec_sim, axis=1)
    max_sim_dec_b = np.max(dec_sim, axis=0)
    max_sim_enc_a = np.max(enc_sim, axis=1)
    max_sim_enc_b = np.max(enc_sim, axis=0)
    
    mean_max_cos_dec = (np.mean(max_sim_dec_a) + np.mean(max_sim_dec_b)) / 2
    mean_max_cos_enc = (np.mean(max_sim_enc_a) + np.mean(max_sim_enc_b)) / 2
    
    # Fraction paired
    frac_paired_dec = np.mean(max_sim_dec_a > threshold)
    frac_paired_enc = np.mean(max_sim_enc_a > threshold)
    
    # Hungarian matching
    dec_row, dec_col, dec_matched_sims = hungarian_matching(dec_sim)
    enc_row, enc_col, enc_matched_sims = hungarian_matching(enc_sim)
    
    # Find shared features
    n_features = len(dec_row)
    shared_mask = np.zeros(n_features, dtype=bool)
    
    for i in range(n_features):
        dec_partner = dec_col[i]
        enc_partner = enc_col[i]
        
        if dec_partner == enc_partner:
            if dec_matched_sims[i] > threshold and enc_matched_sims[i] > threshold:
                shared_mask[i] = True
    
    frac_shared = np.mean(shared_mask)
    
    return {
        'mean_max_cos_dec': mean_max_cos_dec,
        'mean_max_cos_enc': mean_max_cos_enc,
        'mean_max_cos_avg': (mean_max_cos_dec + mean_max_cos_enc) / 2,
        'frac_paired_dec': frac_paired_dec,
        'frac_paired_enc': frac_paired_enc,
        'frac_shared': frac_shared,
        'shared_mask': shared_mask,
        'dec_matching': (dec_row, dec_col),
        'dec_matched_sims': dec_matched_sims,
        'enc_matched_sims': enc_matched_sims,
        'max_sims_dec': max_sim_dec_a,
        'max_sims_enc': max_sim_enc_a,
    }


def analyze_shared_features_across_seeds(
    config_name: str,
    config: dict,
    base_dir: str = MODEL_BASE_DIR,
    threshold: float = 0.7,
    verbose: bool = True
) -> SharedFeatureAnalysisResult:
    """
    Analyze which features are shared across all seed pairs for a configuration.
    
    Args:
        config_name: Name of the configuration (e.g., "k40_l2w0")
        config: Config dict with k, l2_w, and seeds
        base_dir: Base directory containing model folders
        threshold: Cosine similarity threshold for "shared" features
        verbose: Print progress
    
    Returns:
        SharedFeatureAnalysisResult with detailed analysis
    """
    k = config['k']
    l2_w = config['l2_w']
    seed_paths = config['seeds']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing: {config_name} (k={k}, l2_w={l2_w})")
        print(f"{'='*60}")
    
    # Load all SAE weights
    weights_by_seed = {}
    for seed, rel_path in seed_paths.items():
        full_path = os.path.join(base_dir, rel_path)
        if verbose:
            print(f"Loading seed {seed}: {rel_path}")
        weights_by_seed[seed] = load_sae_weights(full_path)
    
    seeds = list(weights_by_seed.keys())
    n_features = get_decoder_features(weights_by_seed[seeds[0]]).shape[0]
    
    # Compute pairwise shared features
    seed_pairs = list(combinations(seeds, 2))
    n_pairs = len(seed_pairs)
    
    # Track share counts for each feature
    feature_share_count = np.zeros(n_features)
    feature_avg_sim = np.zeros(n_features)
    feature_matchings = {i: [] for i in range(n_features)}
    
    # Aggregate metrics
    all_mean_max_cos = []
    all_frac_paired = []
    all_frac_shared = []
    
    for seed_a, seed_b in seed_pairs:
        metrics = compute_pairwise_shared_features(
            weights_by_seed[seed_a],
            weights_by_seed[seed_b],
            threshold=threshold
        )
        
        all_mean_max_cos.append(metrics['mean_max_cos_avg'])
        all_frac_paired.append(metrics['frac_paired_dec'])
        all_frac_shared.append(metrics['frac_shared'])
        
        shared_mask = metrics['shared_mask']
        dec_matching = metrics['dec_matching']
        dec_matched_sims = metrics['dec_matched_sims']
        
        for feat_idx in range(n_features):
            if shared_mask[feat_idx]:
                feature_share_count[feat_idx] += 1
                feature_avg_sim[feat_idx] += dec_matched_sims[feat_idx]
                partner_idx = dec_matching[1][feat_idx]
                feature_matchings[feat_idx].append({
                    'pair': (seed_a, seed_b),
                    'partner': int(partner_idx),
                    'sim': float(dec_matched_sims[feat_idx])
                })
        
        if verbose:
            print(f"  Seed {seed_a} vs {seed_b}: "
                  f"mean_max_cos={metrics['mean_max_cos_avg']:.4f}, "
                  f"frac_shared={metrics['frac_shared']:.2%}")
    
    # Normalize average similarity
    feature_avg_sim = np.divide(
        feature_avg_sim, 
        feature_share_count, 
        where=feature_share_count > 0,
        out=np.zeros_like(feature_avg_sim)
    )
    
    # Categorize features
    always_shared_mask = feature_share_count == n_pairs
    sometimes_shared_mask = (feature_share_count > 0) & (feature_share_count < n_pairs)
    never_shared_mask = feature_share_count == 0
    
    always_shared_indices = np.where(always_shared_mask)[0].tolist()
    sometimes_shared_indices = np.where(sometimes_shared_mask)[0].tolist()
    never_shared_indices = np.where(never_shared_mask)[0].tolist()
    
    n_always = len(always_shared_indices)
    n_sometimes = len(sometimes_shared_indices)
    
    if verbose:
        print("\n--- Shared Feature Summary ---")
        print(f"Always shared (all {n_pairs} pairs): {n_always} ({n_always/n_features:.2%})")
        print(f"Sometimes shared (1-{n_pairs-1} pairs): {n_sometimes} ({n_sometimes/n_features:.2%})")
        print(f"Never shared: {len(never_shared_indices)} ({len(never_shared_indices)/n_features:.2%})")
    
    return SharedFeatureAnalysisResult(
        config_name=config_name,
        k=k,
        l2_w=l2_w,
        n_features=n_features,
        n_seed_pairs=n_pairs,
        n_always_shared=n_always,
        n_sometimes_shared=n_sometimes,
        frac_always_shared=n_always / n_features,
        frac_sometimes_shared=n_sometimes / n_features,
        mean_max_cos_avg=float(np.mean(all_mean_max_cos)),
        frac_paired_dec_avg=float(np.mean(all_frac_paired)),
        feature_share_counts=feature_share_count.tolist(),
        always_shared_indices=always_shared_indices,
        sometimes_shared_indices=sometimes_shared_indices,
        never_shared_indices=never_shared_indices,
    )


def load_or_compute_shared_features(
    config_name: str,
    config: dict,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
    **kwargs
) -> SharedFeatureAnalysisResult:
    """
    Load shared feature analysis from cache or compute it.
    
    Args:
        config_name: Configuration name
        config: Configuration dict
        cache_path: Path to save/load cached results
        force_recompute: If True, always recompute even if cache exists
        **kwargs: Additional arguments for analyze_shared_features_across_seeds
    
    Returns:
        SharedFeatureAnalysisResult
    """
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached results from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        return SharedFeatureAnalysisResult(**data)
    
    result = analyze_shared_features_across_seeds(config_name, config, **kwargs)
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Saved results to {cache_path}")
    
    return result


def get_features_by_sharedness_category(
    result: SharedFeatureAnalysisResult,
    category: str
) -> list:
    """
    Get feature indices by sharedness category.
    
    Args:
        result: SharedFeatureAnalysisResult
        category: One of "always", "sometimes", "never", "shared" (always+sometimes)
    
    Returns:
        List of feature indices
    """
    if category == "always":
        return result.always_shared_indices
    elif category == "sometimes":
        return result.sometimes_shared_indices
    elif category == "never":
        return result.never_shared_indices
    elif category == "shared":
        return result.always_shared_indices + result.sometimes_shared_indices
    else:
        raise ValueError(f"Unknown category: {category}")


# =============================================================================
# Main (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    print("Shared Feature Analysis Module")
    print("=" * 60)
    
    # Analyze both k=40 configurations
    for config_name, config in K40_CONFIGS.items():
        cache_path = os.path.join(SHARED_FEATURE_CACHE_DIR, f"{config_name}_analysis.json")
        result = load_or_compute_shared_features(
            config_name=config_name,
            config=config,
            cache_path=cache_path,
            verbose=True
        )
        
        print(f"\nResults for {config_name}:")
        print(f"  Always shared: {result.n_always_shared} ({result.frac_always_shared:.2%})")
        print(f"  Sometimes shared: {result.n_sometimes_shared} ({result.frac_sometimes_shared:.2%})")
        print(f"  Mean Max Cos: {result.mean_max_cos_avg:.4f}")
        print(f"  Frac Paired: {result.frac_paired_dec_avg:.2%}")
