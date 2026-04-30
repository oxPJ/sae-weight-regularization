from pathlib import Path
from unittest.mock import patch

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import run_eval_single_sae


def test_run_eval_single_sae_saves_tokens_to_artifacts_folder(
    gpt2_l4_sae: SAE,
    gpt2_model: HookedTransformer,
    gpt2_l4_sae_sparsity: torch.Tensor,
    tmp_path: Path,
):
    artifacts_folder = tmp_path / "artifacts"

    config = AutoInterpEvalConfig(
        model_name="gpt2",
        dataset_name="roneneldan/TinyStories",
        n_latents=2,
        total_tokens=255,
        llm_context_size=128,
    )
    with patch("sae_bench.evals.autointerp.main.AutoInterp.run", return_value={}):
        run_eval_single_sae(
            config=config,
            sae=gpt2_l4_sae,
            model=gpt2_model,
            device="cpu",
            artifacts_folder=str(artifacts_folder),
            api_key="fake_api_key",
            sae_sparsity=gpt2_l4_sae_sparsity,
        )

    assert (artifacts_folder / "gpt2_255_tokens_128_ctx.pt").exists()
    tokenized_dataset = torch.load(artifacts_folder / "gpt2_255_tokens_128_ctx.pt")
    assert tokenized_dataset.shape == (2, 128)


def test_run_eval_single_sae_saves_handles_slash_in_model_name(
    gpt2_l4_sae: SAE,
    gpt2_model: HookedTransformer,
    gpt2_l4_sae_sparsity: torch.Tensor,
    tmp_path: Path,
):
    artifacts_folder = tmp_path / "artifacts"

    config = AutoInterpEvalConfig(
        model_name="openai/gpt2",
        dataset_name="roneneldan/TinyStories",
        n_latents=2,
        total_tokens=255,
        llm_context_size=128,
    )
    with patch("sae_bench.evals.autointerp.main.AutoInterp.run", return_value={}):
        run_eval_single_sae(
            config=config,
            sae=gpt2_l4_sae,
            model=gpt2_model,
            device="cpu",
            artifacts_folder=str(artifacts_folder),
            api_key="fake_api_key",
            sae_sparsity=gpt2_l4_sae_sparsity,
        )

    assert (artifacts_folder / "openai_gpt2_255_tokens_128_ctx.pt").exists()
