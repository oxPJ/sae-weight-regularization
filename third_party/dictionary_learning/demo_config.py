#this file is a modified version of demo_config.py in the dictionary_learning_demo repo: https://github.com/adamkarvonen/dictionary_learning_demo


from dataclasses import dataclass, asdict, field
from typing import Optional, Type, Any
from enum import Enum
import torch as t
import itertools

from dictionary_learning.trainers.standard import (
    StandardTrainer,
    StandardTrainerAprilUpdate,
)
from dictionary_learning.trainers.top_k import (
    TopKTrainer,
    AutoEncoderTopK,
)
from dictionary_learning.trainers.batch_top_k import (
    BatchTopKTrainer,
    BatchTopKSAE,
)
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer,
    MatryoshkaBatchTopKSAE,
)
from dictionary_learning.trainers.top_k_l2 import TopKTrainerL2
from dictionary_learning.trainers.top_k_l1 import TopKTrainerL1
from dictionary_learning.trainers.batch_top_k_l1 import BatchTopKTrainerL1
from dictionary_learning.trainers.matryoshka_batch_top_k_l1 import MatryoshkaBatchTopKTrainerL1
from dictionary_learning.trainers.batch_top_k_l2 import BatchTopKTrainerL2
from dictionary_learning.trainers.matryoshka_batch_top_k_l2 import MatryoshkaBatchTopKTrainerL2

from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"
    Matryoshka_BATCH_TOP_K = "matryoshka_batch_top_k"
    TOP_K_L2="top_k_l2"
    TOP_K_L1="top_k_l1"
    BATCH_TOP_K_L1="batch_top_k_l1"
    Matryoshka_BATCH_TOP_K_L1="matryoshka_batch_top_k_l1"
    BATCH_TOP_K_L2="batch_top_k_l2"
    Matryoshka_BATCH_TOP_K_L2="matryoshka_batch_top_k_l2"


@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype


@dataclass
class SparsityPenalties:
    standard: list[float]
    standard_new: list[float]
    p_anneal: list[float]
    gated: list[float]


num_tokens = 50_000_000

print(f"NOTE: Training on {num_tokens} tokens")

eval_num_inputs = 200
random_seeds = [0, 1, 2]
dictionary_widths = [2**14]
# dictionary_widths = [2**14]

WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 5000
DECAY_START_FRACTION = 0.8
K_ANNEAL_END_FRACTION = 0.01
remove_bos = True
max_activation_norm_multiple = 10

learning_rates = [5e-5]


wandb_project = "qwen-8b-sweep"

LLM_CONFIG = {
    "EleutherAI/pythia-70m-deduped": LLMConfig(
        llm_batch_size=64, context_length=1024, sae_batch_size=2048, dtype=t.float32
    ),
    "EleutherAI/pythia-160m-deduped": LLMConfig(
        llm_batch_size=32, context_length=1024, sae_batch_size=2048, dtype=t.float32
    ),
    "google/gemma-2-2b": LLMConfig(
        llm_batch_size=4, context_length=1024, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen2.5-Coder-32B-Instruct": LLMConfig(
        llm_batch_size=4, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen3-8B": LLMConfig(
        llm_batch_size=16, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen3-14B": LLMConfig(
        llm_batch_size=8, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen3-32B": LLMConfig(
        llm_batch_size=2, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
}

SPARSITY_PENALTIES = SparsityPenalties(
    standard=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    standard_new=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    p_anneal=[0.006, 0.008, 0.01, 0.015, 0.02, 0.025],
    gated=[0.012, 0.018, 0.024, 0.04, 0.06, 0.08],
)

WEIGHT_L2_GRID = [0, 0.0001] 
WEIGHT_L1_GRID = [0, 0.0001] 
TARGET_L0s = [40, 80, 150, 320]



@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str
    warmup_steps: int
    steps: int
    decay_start: Optional[int]


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    resample_steps: Optional[int] = None


@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    initial_sparsity_penalty: float
    sparsity_warmup_steps: Optional[int]
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: Optional[int] = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold
    k_anneal_steps: Optional[int] = None

@dataclass
class TopKTrainerConfigWeightPenalty(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold
    k_anneal_steps: Optional[int] = None
    weight_l_penalty: float = 0.0  


@dataclass
class MatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = field(
        default_factory=lambda: [
            (1 / 32),
            (1 / 16),
            (1 / 8),
            (1 / 4),
            ((1 / 2) + (1 / 32)),
        ]
    )
    group_weights: Optional[list[float]] = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold
    k_anneal_steps: Optional[int] = None


@dataclass
class MatryoshkaBatchTopKTrainerConfigWeightPenalty(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = field(
        default_factory=lambda: [
            (1 / 32),
            (1 / 16),
            (1 / 8),
            (1 / 4),
            ((1 / 2) + (1 / 32)),
        ]
    )
    group_weights: Optional[list[float]] = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold
    k_anneal_steps: Optional[int] = None
    weight_l_penalty: float = 0.0  



@dataclass
class GatedTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: Optional[int]
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001


def get_trainer_configs(
    architectures: list[str],
    learning_rates: list[float],
    seeds: list[int],
    activation_dim: int,
    dict_sizes: list[int],
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
    anneal_end_fraction=K_ANNEAL_END_FRACTION,
) -> list[dict]:
    decay_start = int(steps * decay_start_fraction)
    anneal_end = int(steps * anneal_end_fraction)

    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }
    if TrainerType.P_ANNEAL.value in architectures:
        for seed, dict_size, learning_rate, sparsity_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.p_anneal
        ):
            config = PAnnealTrainerConfig(
                **base_config,
                trainer=PAnnealTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                initial_sparsity_penalty=sparsity_penalty,
                wandb_name=f"PAnnealTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard
        ):
            config = StandardTrainerConfig(
                **base_config,
                trainer=StandardTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD_NEW.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard_new
        ):
            config = StandardNewTrainerConfig(
                **base_config,
                trainer=StandardTrainerAprilUpdate,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainerNew-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.GATED.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.gated
        ):
            config = GatedTrainerConfig(
                **base_config,
                trainer=GatedSAETrainer,
                dict_class=GatedAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"GatedTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=TopKTrainer,
                dict_class=AutoEncoderTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                k_anneal_steps=anneal_end,
                wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
    if TrainerType.TOP_K_L2.value in architectures:
            for seed, dict_size, learning_rate, k, lw_2 in itertools.product(
                seeds, dict_sizes, learning_rates, TARGET_L0s, WEIGHT_L2_GRID
            ):
                config = TopKTrainerConfigWeightPenalty(
                    **base_config,
                    trainer=TopKTrainerL2,
                    dict_class=AutoEncoderTopK,
                    lr=learning_rate,
                    dict_size=dict_size,
                    seed=seed,
                    k=k,
                    weight_l_penalty=lw_2,
                    k_anneal_steps=anneal_end,
                    wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
                )
                trainer_configs.append(asdict(config))
    if TrainerType.TOP_K_L1.value in architectures:
        for seed, dict_size, learning_rate, k, lw_1 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, WEIGHT_L1_GRID
        ):
            config = TopKTrainerConfigWeightPenalty(
                **base_config,
                trainer=TopKTrainerL1,
                dict_class=AutoEncoderTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                weight_l_penalty=lw_1,
                k_anneal_steps=anneal_end,
                wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=BatchTopKTrainer,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                k_anneal_steps=anneal_end,
                wandb_name=f"BatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K_L1.value in architectures:
        for seed, dict_size, learning_rate, k, lw_1 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, WEIGHT_L1_GRID
        ):
            config = TopKTrainerConfigWeightPenalty(
                **base_config,
                trainer=BatchTopKTrainerL1,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                weight_l_penalty=lw_1,
                k_anneal_steps=anneal_end,
                wandb_name=f"BatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K_L2.value in architectures:
        for seed, dict_size, learning_rate, k, lw_2 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, WEIGHT_L2_GRID
        ):
            config = TopKTrainerConfigWeightPenalty(
                **base_config,
                trainer=BatchTopKTrainerL2,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                weight_l_penalty=lw_2,
                k_anneal_steps=anneal_end,
                wandb_name=f"BatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.Matryoshka_BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = MatryoshkaBatchTopKTrainerConfig(
                **base_config,
                trainer=MatryoshkaBatchTopKTrainer,
                dict_class=MatryoshkaBatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                k_anneal_steps=anneal_end,
                wandb_name=f"MatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
    if TrainerType.Matryoshka_BATCH_TOP_K_L1.value in architectures:
        for seed, dict_size, learning_rate, k, lw_1 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, WEIGHT_L1_GRID
        ):
            config = MatryoshkaBatchTopKTrainerConfigWeightPenalty(
                **base_config,
                trainer=MatryoshkaBatchTopKTrainerL1,
                dict_class=MatryoshkaBatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                weight_l_penalty=lw_1,
                k_anneal_steps=anneal_end,
                wandb_name=f"MatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
    if TrainerType.Matryoshka_BATCH_TOP_K_L2.value in architectures:
        for seed, dict_size, learning_rate, k, lw_2 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, WEIGHT_L2_GRID
        ):
            config = MatryoshkaBatchTopKTrainerConfigWeightPenalty(
                **base_config,
                trainer=MatryoshkaBatchTopKTrainerL2,
                dict_class=MatryoshkaBatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                weight_l_penalty=lw_2,
                k_anneal_steps=anneal_end,
                wandb_name=f"MatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
    if TrainerType.JUMP_RELU.value in architectures:
        for seed, dict_size, learning_rate, target_l0 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = JumpReluTrainerConfig(
                **base_config,
                trainer=JumpReluTrainer,
                dict_class=JumpReluAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                target_l0=target_l0,
                wandb_name=f"JumpReluTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    return trainer_configs
