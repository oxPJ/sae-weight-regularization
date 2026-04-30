import json

import torch

import sae_bench.evals.ravel.main as ravel
import sae_bench.sae_bench_utils.testing_utils as testing_utils
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.sae_bench_utils.sae_selection_utils import select_saes_multiple_patterns

results_filename = "tests/acceptance/test_data/ravel/ravel_expected_results.json"


def test_end_to_end_different_seed():
    """Estimated runtime: 1 hour"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = RAVELEvalConfig()

    test_config.entity_attribute_selection = {
        "city": ["Country", "Continent", "Language"],
    }

    test_config.model_name = "gemma-2-2b"
    tolerance = 0.04
    test_config.random_seed = 48

    test_config.llm_dtype = "bfloat16"
    test_config.llm_batch_size = 32

    sae_regex_patterns = [
        r"sae_bench_gemma-2-2b_topk_width-2pow14_date-1109",
    ]
    sae_block_pattern = [
        r"blocks.5.hook_resid_post__trainer_2",
    ]

    selected_saes = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    run_results = ravel.run_eval(
        test_config,
        selected_saes,
        device,
        output_path="evals/ravel/test_results/",
        force_rerun=True,
    )

    with open("test_data.json", "w") as f:
        json.dump(run_results, f, indent=4)

    with open(results_filename) as f:
        expected_results = json.load(f)

    sae_name = "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109_blocks.5.hook_resid_post__trainer_2"

    run_result_metrics = run_results[sae_name]["eval_result_metrics"]

    testing_utils.compare_dicts_within_tolerance(
        run_result_metrics,
        expected_results["eval_result_metrics"],
        tolerance,
        keys_to_compare=["disentanglement_score"],
    )
