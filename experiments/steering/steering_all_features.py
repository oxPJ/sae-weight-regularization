"""
Steering Experiment: All Alive Features (Ignoring Sharedness)

This experiment runs steering evaluation on a random sample of ALL alive features,
regardless of their sharedness status. This provides a baseline for comparison
with the sharedness-stratified steering analysis.

"Alive" features are those with non-zero encoder/decoder cosine similarity,
indicating the feature has learned something (as opposed to dead features).

Usage: nohup python steering_all_features.py > steering_all.log 2>&1 &
"""

import csv
import json
import os
import random
import time
from typing import Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# Local import for shared feature analysis
from shared_feature_analysis import (
    K40_CONFIGS,
    MODEL_BASE_DIR,
    DATA_RESULTS_DIR,
    SHARED_FEATURE_CACHE_DIR,
    load_or_compute_shared_features,
    load_sae_weights,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "pythia-70m-deduped"
HOOK_LAYER = 3
HOOK_NAME = f"blocks.{HOOK_LAYER}.hook_resid_post"

# Which k=40 configuration to use for steering
SAE_CONFIG_NAME = "k40_l2w0001"  # Change as needed

# Which seed to use for the SAE in steering experiments
STEERING_SEED = 0

# Steering parameters
STEERING_STRENGTH_MODE = "fixed"
STEERING_STRENGTH = 5.0
STEERING_N_RESID_RMS = 0.60
TARGET_FEATURE_PREACT_DELTA = 5.0
MAX_NEW_TOKENS = 30

# Feature selection
N_FEATURES_TO_SAMPLE = 300  # Number of features to sample from all alive features

# Reproducibility
DETERMINISTIC = False
RANDOM_SEED = 42



# LLM judge params
OPENAI_API_KEY_FILE = "openai_api_key.txt"  # optional
LLM_MODEL = "gpt-5.1"                       # change if needed
LLM_RETRIES = 3
LLM_MAX_COMPLETION_TOKENS = 20

# Embeddings judge params (optional)
USE_EMBEDDINGS_JUDGE = True
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_SIMILARITY_THRESHOLD = 0.45
APPLY_ONLY_GENERATION_STEPS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NEUTRAL_PROMPTS = [
    "The most important thing is",
    "A common example of",
    "Many people believe that",
    "In recent years,",
    "The main reason for",
    "It is well known that",
    "According to experts,",
    "One thing that matters is",
]


# Auto-interp results paths (to get expected concepts/explanations)
AUTOINTERP_RESULTS_DIRS = {
    "k40_l2w0": os.path.join(DATA_RESULTS_DIR, "autointerp_all_features_l2_w0", "autointerp_all_features_l2_w0"),
    "k40_l2w0001": os.path.join(DATA_RESULTS_DIR, "autointerp_all_features_l2_w0001", "autointerp_all_features_l2_w0001"),
}

# Output files
OUT_DIR = os.path.join(DATA_RESULTS_DIR, "steering_all_features_results")
OUT_JSON = os.path.join(OUT_DIR, f"steering_all_features_results_{SAE_CONFIG_NAME}.json")
OUT_CSV = os.path.join(OUT_DIR, f"steering_all_features_results_{SAE_CONFIG_NAME}.csv")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureSteeringResult:
    """Result of steering with a single feature."""
    feature_idx: int
    sharedness_category: str  # "always", "sometimes", "never", or "unknown"
    share_count: int  # How many seed pairs this feature is shared in
    cosine_similarity: float  # Encoder-decoder cosine similarity
    expected_concept: str  # Auto-interp explanation for the feature
    prompt: str
    original_text: str
    steered_text: str
    llm_match_score: int | None
    embed_cosine: float | None
    embed_match: bool | None


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resid_rms(x: torch.Tensor) -> float:
    """Root-mean-square of a vector."""
    return float(torch.sqrt(torch.mean(x.float() ** 2)).item())


def estimate_resid_rms_at_hook(model: HookedTransformer, prompts: list[str], hook_name: str = HOOK_NAME) -> float:
    """Estimate typical residual-stream scale (RMS) at hook_name."""
    vals: list[float] = []

    def grab(activation, hook):
        if isinstance(activation, torch.Tensor) and activation.dim() == 3:
            v = activation[0, -1, :].detach()
            vals.append(resid_rms(v))
        return activation

    model.eval()
    with torch.no_grad():
        for p in prompts:
            toks = model.to_tokens(p).to(DEVICE)
            model.run_with_hooks(toks, fwd_hooks=[(hook_name, grab)])

    if not vals:
        return 1.0
    return float(np.mean(vals))


def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """Return a copy of v normalized to unit norm."""
    v = v.clone().detach()
    norm = v.norm()
    if norm.item() == 0.0:
        return v
    return v / (norm + 1e-12)


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
    
    # Detect layout
    if We.shape[0] > We.shape[1]:
        n_feats = We.shape[0]
        is_transposed = False
    else:
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
# SAE Loading
# =============================================================================

def get_sae_path(config_name: str, seed: int, base_dir: str = MODEL_BASE_DIR) -> str:
    """Get full path to SAE for a given config and seed."""
    config = K40_CONFIGS[config_name]
    rel_path = config['seeds'][seed]
    return os.path.join(base_dir, rel_path)


def load_autointerp_results(config_name: str, seed: int, results_dir: str | None = None) -> dict[int, dict]:
    """
    Load auto-interp results to get feature explanations (expected concepts).
    
    Returns a dict mapping feature_idx -> {'explanation': str, 'score': float}
    """
    if results_dir is None:
        results_dir = AUTOINTERP_RESULTS_DIRS.get(config_name)
        if results_dir is None:
            print(f"  Warning: No auto-interp results directory configured for config '{config_name}'")
            return {}
    
    sae_name = f"{config_name}_seed{seed}_all_features"
    result_path = os.path.join(results_dir, f"{sae_name}_analysis.json")
    
    feature_explanations = {}
    
    if not os.path.exists(result_path):
        print(f"  Warning: No auto-interp results found at {result_path}")
        return feature_explanations
    
    try:
        with open(result_path) as f:
            data = json.load(f)
        
        # Extract explanations from feature_results
        feature_results = data.get("feature_results", [])
        
        for feat in feature_results:
            feat_idx = int(feat.get("feature_idx", -1))
            explanation = feat.get("explanation", "N/A")
            score = float(feat.get("autointerp_score", 0.0))
            
            if feat_idx >= 0:
                feature_explanations[feat_idx] = {
                    "explanation": explanation,
                    "score": score,
                }
        
        print(f"  Loaded {len(feature_explanations)} feature explanations from {result_path}")
        
    except Exception as e:
        print(f"  Warning: Failed to load auto-interp results: {e}")
    
    return feature_explanations


def load_sae_state(sae_path: str) -> dict[str, torch.Tensor | None]:
    """Load SAE weights from path."""
    weights = load_sae_weights(sae_path)
    
    if 'encoder.weight' in weights:
        W_enc = weights['encoder.weight']
        W_dec = weights['decoder.weight']
        b_enc = weights.get('encoder.bias', None)
        b_dec = weights.get('decoder.bias', weights.get('bias', None))
    elif 'W_enc' in weights:
        W_enc = weights['W_enc']
        W_dec = weights['W_dec']
        b_enc = weights.get('b_enc', None)
        b_dec = weights.get('b_dec', None)
    else:
        raise ValueError(f"Unexpected keys in SAE weights: {list(weights.keys())[:10]}")
    
    return {
        "W_enc": W_enc.float(),
        "W_dec": W_dec.float(),
        "b_enc": (b_enc.float() if isinstance(b_enc, torch.Tensor) else None),
        "b_dec": (b_dec.float() if isinstance(b_dec, torch.Tensor) else None),
    }


# =============================================================================
# Steering & Generation
# =============================================================================

def make_steering_hook(steering_vector: torch.Tensor, steering_strength: float, apply_only_generation_steps: bool = True):
    """Create a steering hook function."""
    steering_vector = steering_vector.to(DEVICE)

    def hook_fn(activation, hook):
        if not isinstance(activation, torch.Tensor):
            return activation

        if apply_only_generation_steps and activation.dim() == 3 and activation.shape[1] != 1:
            return activation

        if activation.dim() == 3:
            act = activation.clone()
            act[:, -1, :] = act[:, -1, :] + steering_strength * steering_vector
            return act
        if activation.dim() == 2:
            return activation + steering_strength * steering_vector
        return activation

    return hook_fn


def calibrate_strength_target_feat_delta(
    W_enc: torch.Tensor,
    W_dec: torch.Tensor,
    feature_idx: int,
    target_preact_delta: float,
    normalize: bool = True,
) -> float:
    """Choose steering_strength so that encoder pre-activation increases by target."""
    v = W_dec[:, feature_idx].float()
    if normalize:
        v = v / (v.norm() + 1e-12)
    w = W_enc[feature_idx].float()
    proj = float(torch.dot(w, v).item())
    if abs(proj) < 1e-8:
        return 0.0
    return float(target_preact_delta / proj)


def generate_with_steering(
    model: HookedTransformer,
    W_dec: torch.Tensor,
    W_enc: torch.Tensor | None,
    feature_idx: int,
    prompt: str,
    steering_strength: float = 3.0,
    max_new_tokens: int = 30,
    deterministic: bool = True,
    normalize: bool = True,
    apply_only_generation_steps: bool = True,
    steering_strength_mode: str = "fixed",
    resid_rms_scale: float | None = None,
    target_feat_preact_delta: float = TARGET_FEATURE_PREACT_DELTA,
) -> tuple[str, str]:
    """Generate with and without steering. Returns original_text, steered_text."""

    tokens = model.to_tokens(prompt)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not deterministic,
        "temperature": 0.0 if deterministic else 0.7,
        "top_p": 1.0 if deterministic else 0.9,
    }

    # Original (no steering)
    with torch.no_grad():
        original_output = model.generate(tokens, **gen_kwargs)
        original_text = model.tokenizer.decode(original_output[0])

    # Steering vector
    steering_vector = W_dec[:, feature_idx].to(DEVICE)
    if normalize:
        steering_vector = normalize_vector(steering_vector)

    # Calibrate strength
    strength = float(steering_strength)
    if steering_strength_mode == "resid_rms":
        if resid_rms_scale is None:
            resid_rms_scale = 1.0
        strength = float(STEERING_N_RESID_RMS * resid_rms_scale)
    elif steering_strength_mode == "target_feat_preact_delta":
        if W_enc is not None:
            strength = calibrate_strength_target_feat_delta(
                W_enc=W_enc,
                W_dec=W_dec,
                feature_idx=feature_idx,
                target_preact_delta=float(target_feat_preact_delta),
                normalize=normalize,
            )

    # Add hook
    hook = make_steering_hook(steering_vector, strength, apply_only_generation_steps=apply_only_generation_steps)
    model.add_hook(HOOK_NAME, hook)

    try:
        with torch.no_grad():
            steered_output = model.generate(tokens, **gen_kwargs)
            steered_text = model.tokenizer.decode(steered_output[0])
    finally:
        model.reset_hooks()

    return original_text, steered_text


# =============================================================================
# Feature Selection
# =============================================================================

def has_valid_explanation(explanation: str | None) -> bool:
    """Check if an explanation is valid (not empty, not N/A)."""
    if explanation is None:
        return False
    explanation = explanation.strip()
    return explanation != "" and explanation.upper() != "N/A"


def select_random_alive_features(
    cos_sims: dict[int, float],
    n_samples: int,
    seed: int = 42,
    shared_result=None,
    feature_explanations: dict[int, dict] | None = None,
    require_explanation: bool = True,
) -> list[dict]:
    """
    Randomly sample from all alive features.
    
    Args:
        cos_sims: dict mapping feature_idx -> cosine_similarity
        n_samples: number of features to sample
        seed: random seed
        shared_result: Optional SharedFeatureAnalysisResult for annotating sharedness
        feature_explanations: Dict mapping feature_idx -> {'explanation': str, 'score': float}
        require_explanation: If True, only select features with valid explanations
    
    Returns:
        List of feature info dicts
    """
    np.random.seed(seed)
    random.seed(seed)
    
    if feature_explanations is None:
        feature_explanations = {}
    
    alive_indices = get_alive_feature_indices(cos_sims)
    n_alive = len(alive_indices)
    
    print(f"  Total features: {len(cos_sims)}")
    print(f"  Alive features: {n_alive}")
    print(f"  Dead features: {len(cos_sims) - n_alive}")
    
    # Filter to features with valid explanations if required
    if require_explanation:
        valid_indices = []
        for idx in alive_indices:
            feat_info = feature_explanations.get(int(idx), {})
            explanation = feat_info.get("explanation", "N/A")
            if has_valid_explanation(explanation):
                valid_indices.append(idx)
        
        print(f"  Features with valid explanations: {len(valid_indices)}")
        
        if len(valid_indices) == 0:
            raise ValueError(
                "No features with valid explanations!\n"
                "Please run autointerp_all_features.py first to generate feature explanations."
            )
        
        indices_to_sample = valid_indices
    else:
        indices_to_sample = alive_indices
    
    n_sample = min(n_samples, len(indices_to_sample))
    sampled_indices = list(np.random.choice(indices_to_sample, n_sample, replace=False))
    
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
        # Get explanation
        feat_info = feature_explanations.get(int(idx), {})
        explanation = feat_info.get("explanation", "N/A")
        autointerp_score = feat_info.get("score", 0.0)
        
        # Get sharedness
        cat, count = sharedness_lookup.get(idx, ("unknown", 0))
        share_fraction = count / shared_result.n_seed_pairs if shared_result and shared_result.n_seed_pairs > 0 else 0
        
        features.append({
            'feature_idx': int(idx),
            'cosine_similarity': cos_sims[idx],
            'sharedness_category': cat,
            'share_count': count,
            'share_fraction': share_fraction,
            'expected_concept': explanation,
            'autointerp_score': autointerp_score,
        })
    
    print(f"  Sampled {len(features)} features for steering")
    
    # Print sharedness distribution
    cat_counts = {}
    for f in features:
        cat = f.get('sharedness_category', 'unknown')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"  Sharedness distribution in sample:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count} ({count/len(features)*100:.1f}%)")
    
    return features


# =============================================================================
# Judges (LLM + Embeddings)
# =============================================================================

def evaluate_steering_with_llm(results: list[dict], api_key: str):
    """
    Annotate results with LLM-based match scores.
    """
    try:
        from openai import OpenAI
    except Exception:
        print("openai package not installed. Skipping LLM evaluation.")
        for r in results:
            r["llm_match_score"] = None
        return results

    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are a strict evaluator. You will be given:\n"
        "- an expected concept\n"
        "- an original (unsteered) text\n"
        "- a steered text\n\n"
        "Score ONLY whether the steered text is MORE related to the expected concept than the original text.\n"
        "Use this 1-5 scale:\n"
        "1 = steered is much LESS related than original\n"
        "2 = steered is slightly LESS related\n"
        "3 = no meaningful change\n"
        "4 = steered is slightly MORE related\n"
        "5 = steered is much MORE related\n\n"
        "Respond with exactly one integer (1-5) and nothing else."
    )

    import re

    for r in results:
        expected_concept = r.get('expected_concept', 'N/A')
        
        if expected_concept == 'N/A' or not expected_concept:
            prompt_text = (
                f"Original text: \"{r['original_text']}\"\n\n"
                f"Steered text: \"{r['steered_text']}\"\n\n"
                "Rate how different the steered text is from the original (1-5):"
            )
        else:
            prompt_text = (
                f"Expected concept: \"{expected_concept}\"\n\n"
                f"Original text: \"{r['original_text']}\"\n\n"
                f"Steered text: \"{r['steered_text']}\"\n\n"
                "Question: Compared to the original, is the steered text more related to the expected concept? Rate 1-5."
            )

        score = None
        attempt = 0
        while attempt < LLM_RETRIES:
            attempt += 1
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=0.0,
                    max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
                )

                text = resp.choices[0].message.content.strip()
                m = re.search(r"[1-5]", text or "")
                if m:
                    score = int(m.group(0))
                break

            except Exception as e:
                wait = 2 ** attempt
                print(f"LLM error (attempt {attempt}): {e}. Retrying in {wait}s.")
                time.sleep(wait)

        r["llm_match_score"] = score

    return results


def judge_with_embeddings(results: list[dict]):
    """
    Add embedding-based similarity scores.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        print("sentence-transformers not installed. Skipping embeddings judge.")
        for r in results:
            r["embed_cosine"] = None
            r["embed_match"] = None
        return results

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    has_concepts = any(r.get("expected_concept") and r["expected_concept"] != "N/A" for r in results)
    
    if has_concepts:
        concepts = []
        steered_texts = []
        valid_indices = []
        
        for i, r in enumerate(results):
            concept = r.get("expected_concept", "N/A")
            if concept and concept != "N/A":
                concepts.append(concept)
                steered_texts.append(r["steered_text"])
                valid_indices.append(i)
        
        if concepts:
            emb_concepts = embed_model.encode(concepts, convert_to_tensor=True)
            emb_steered = embed_model.encode(steered_texts, convert_to_tensor=True)
            cos_sims = util.cos_sim(emb_concepts, emb_steered).diagonal().cpu().numpy()
            
            for j, i in enumerate(valid_indices):
                cos = float(cos_sims[j])
                results[i]["embed_cosine"] = cos
                results[i]["embed_match"] = cos >= EMBEDDING_SIMILARITY_THRESHOLD
        
        for i, r in enumerate(results):
            if i not in valid_indices:
                r["embed_cosine"] = None
                r["embed_match"] = None
    else:
        originals = [r["original_text"] for r in results]
        steered = [r["steered_text"] for r in results]
        
        emb_orig = embed_model.encode(originals, convert_to_tensor=True)
        emb_steered = embed_model.encode(steered, convert_to_tensor=True)

        cos_sims = util.cos_sim(emb_orig, emb_steered).diagonal().cpu().numpy()

        for i, r in enumerate(results):
            cos = float(cos_sims[i])
            r["embed_cosine"] = cos
            r["embed_match"] = cos < (1 - EMBEDDING_SIMILARITY_THRESHOLD)

    return results


# =============================================================================
# Analysis Helpers
# =============================================================================

def summarize_by_category(results: list[dict]):
    """Print summary statistics by sharedness category."""
    from collections import defaultdict
    
    cat_scores = defaultdict(list)
    cat_embed = defaultdict(list)
    
    for r in results:
        cat = r.get("sharedness_category", "unknown")
        if r.get("llm_match_score") is not None:
            cat_scores[cat].append(r["llm_match_score"])
        if r.get("embed_cosine") is not None:
            cat_embed[cat].append(r["embed_cosine"])

    print("\n" + "="*70)
    print("SUMMARY BY SHAREDNESS CATEGORY")
    print("="*70)
    
    print("\nLLM 'Match' Scores (higher = steering more effective):")
    for cat in ["always_shared", "sometimes_shared", "never_shared", "unknown"]:
        scores = cat_scores.get(cat, [])
        if scores:
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            high_match = sum(s >= 4 for s in scores) / len(scores) * 100
            print(f"  {cat:20s}: mean={mean:.2f} (±{std:.2f}), high_match(≥4)={high_match:.1f}% (n={len(scores)})")

    print("\nEmbedding Similarity (concept vs steered text):")
    for cat in ["always_shared", "sometimes_shared", "never_shared", "unknown"]:
        embeds = cat_embed.get(cat, [])
        if embeds:
            mean = float(np.mean(embeds))
            std = float(np.std(embeds))
            print(f"  {cat:20s}: mean={mean:.3f} (±{std:.3f}) (n={len(embeds)})")


# =============================================================================
# Main Experiment
# =============================================================================

def run_steering_experiment():
    """Run the main steering experiment on all alive features."""
    set_seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("="*70)
    print("Steering Experiment: All Alive Features")
    print("="*70)
    print(f"\nConfiguration: {SAE_CONFIG_NAME}")
    print(f"Steering with SAE from seed: {STEERING_SEED}")
    print(f"Device: {DEVICE}")
    
    # Step 1: Compute/load shared feature analysis (for annotation, not filtering)
    print("\n[1] Computing shared feature analysis...")
    cache_path = os.path.join(SHARED_FEATURE_CACHE_DIR, f"{SAE_CONFIG_NAME}_analysis.json")
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
    print("\n[2] Computing alive features...")
    sae_path = get_sae_path(SAE_CONFIG_NAME, STEERING_SEED)
    cos_sims, n_features = compute_feature_cosine_similarities(sae_path)
    n_alive = len(get_alive_feature_indices(cos_sims))
    print(f"    Total features: {n_features}")
    print(f"    Alive features: {n_alive}")
    
    # Step 3: Load auto-interp results (for expected concepts)
    print("\n[3] Loading auto-interp results for expected concepts...")
    feature_explanations = load_autointerp_results(
        config_name=SAE_CONFIG_NAME,
        seed=STEERING_SEED,
    )
    
    if not feature_explanations:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"ERROR: No auto-interp results found!\n"
            f"{'='*60}\n"
            f"Steering evaluation requires expected concepts from auto-interp.\n"
            f"Please run autointerp_all_features.py first:\n"
            f"  python autointerp_all_features.py\n"
            f"{'='*60}"
        )
    
    # Step 4: Load model and SAE
    print(f"\n[4] Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    
    print(f"[5] Loading SAE weights (seed={STEERING_SEED})...")
    sae = load_sae_state(sae_path)
    W_enc, W_dec, b_enc = sae["W_enc"], sae["W_dec"], sae["b_enc"]
    
    resid_scale = estimate_resid_rms_at_hook(model, prompts=NEUTRAL_PROMPTS[:3], hook_name=HOOK_NAME)
    print(f"    Estimated resid RMS at {HOOK_NAME}: {resid_scale:.4f}")
    
    # Step 5: Select random alive features
    print(f"\n[6] Selecting {N_FEATURES_TO_SAMPLE} random alive features...")
    selected_features = select_random_alive_features(
        cos_sims=cos_sims,
        n_samples=N_FEATURES_TO_SAMPLE,
        seed=RANDOM_SEED,
        shared_result=shared_result,
        feature_explanations=feature_explanations,
        require_explanation=True,
    )
    
    # Step 6: Run steering experiments
    print(f"\n[7] Running steering experiment on {len(selected_features)} features...")
    results = []
    
    for i, feat_info in enumerate(selected_features):
        feature_idx = feat_info['feature_idx']
        share_count = feat_info.get('share_count', 0)
        sharedness_category = feat_info.get('sharedness_category', 'unknown')
        expected_concept = feat_info.get('expected_concept', 'N/A')
        cos_sim = feat_info.get('cosine_similarity', 0.0)
        
        concept_info = f", concept: \"{expected_concept[:40]}...\"" if expected_concept != 'N/A' else ""
        print(f"\n  [{i+1}/{len(selected_features)}] Feature {feature_idx} ({sharedness_category}, cos_sim={cos_sim:.3f}{concept_info})")
        
        for prompt in NEUTRAL_PROMPTS[:3]:  # Use fewer prompts for speed
            original, steered = generate_with_steering(
                model=model,
                W_dec=W_dec,
                W_enc=W_enc,
                feature_idx=feature_idx,
                prompt=prompt,
                steering_strength=STEERING_STRENGTH,
                max_new_tokens=MAX_NEW_TOKENS,
                deterministic=DETERMINISTIC,
                normalize=True,
                apply_only_generation_steps=APPLY_ONLY_GENERATION_STEPS,
                steering_strength_mode=STEERING_STRENGTH_MODE,
                resid_rms_scale=resid_scale,
                target_feat_preact_delta=TARGET_FEATURE_PREACT_DELTA,
            )
            
            # Print preview
            orig_preview = original[:120].replace("\n", " ")
            steered_preview = steered[:120].replace("\n", " ")
            print(f"    Prompt: \"{prompt}\"")
            print(f"    Original: {orig_preview}...")
            print(f"    Steered:  {steered_preview}...")
            
            results.append({
                'feature_idx': feature_idx,
                'sharedness_category': sharedness_category,
                'share_count': share_count,
                'share_fraction': feat_info.get('share_fraction', 0),
                'cosine_similarity': cos_sim,
                'expected_concept': expected_concept,
                'autointerp_score': feat_info.get('autointerp_score', 0.0),
                'prompt': prompt,
                'original_text': original,
                'steered_text': steered,
                'seed': RANDOM_SEED,
                'model_name': MODEL_NAME,
                'sae_config': SAE_CONFIG_NAME,
                'steering_seed': STEERING_SEED,
            })
    
    # Save initial results
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {OUT_JSON}")
    
    # Step 7: LLM evaluation (if API key available)
    api_key = None
    if os.path.exists(OPENAI_API_KEY_FILE):
        with open(OPENAI_API_KEY_FILE) as f:
            api_key = f.read().strip()

    if api_key:
        print("\n[8] Running LLM evaluation...")
        results = evaluate_steering_with_llm(results, api_key=api_key)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print("\n[8] No API key found, skipping LLM evaluation")

    # Step 8: Embeddings judge
    if USE_EMBEDDINGS_JUDGE:
        print("\n[9] Running embeddings-based evaluation...")
        results = judge_with_embeddings(results)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
    
    # Save CSV
    print("\n[10] Saving CSV summary...")
    keys = ["sharedness_category", "feature_idx", "share_count", "share_fraction", "cosine_similarity",
            "expected_concept", "autointerp_score", "prompt", "llm_match_score", "embed_cosine", "embed_match"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in keys}
            writer.writerow(row)
    print(f"Saved CSV to {OUT_CSV}")
    
    # Print summary
    summarize_by_category(results)
    
    return results


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    run_steering_experiment()
