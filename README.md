# SAE Weight Regularization Artifact

This repository contains the code needed to reproduce the analyses for the paper
on weight regularization of sparse autoencoders.

The repository is organized around three workflow areas:

- analysis notebooks in `analysis/`
- experiment and reproduction code in `experiments/`
- artifact download/verification utilities in `scripts/`

Large result files and model weights are hosted on Hugging Face rather than
committed to this repository.

## Repository Layout

```text
.
├── analysis/
│   ├── mnist/
│   │   └── analyse_models.ipynb
│   └── pythia/
│       ├── pareto_radar_plot.ipynb
│       ├── decoder_orthogonality.ipynb
│       ├── Feature Consistency Analysis Across Random Seeds Pythia model.ipynb
│       ├── sae_weights_pythia_models copy.ipynb
│       └── auto_interp_steering_analysis_topk40_model copy.ipynb
├── experiments/
│   ├── mnist/
│   │   ├── MNIST_not_constrained.ipynb
│   │   └── MNIST_norm_constrained.ipynb
│   └── steering/
│       ├── shared_feature_analysis.py
│       ├── steering_all_features.py
│       ├── steering_shared_features.py
│       └── steering_shared_hparam_sweep.py
├── SAEBench_extension/
│   ├── autointerp_all_features.py
│   └── eval.py
├── scripts/
│   ├── download_artifact.py
│   └── verify_artifact.py
├── third_party/
│   └── dictionary_learning/
├── external/
├── requirements.txt
└── ATTRIBUTIONS.md
```

`third_party/dictionary_learning/` contains the vendored dictionary learning code
used for SAE training. `SAEBench_extension/` contains local scripts that build on
the `sae-bench` package installed from `requirements.txt`.

## Artifact Downloads

The artifacts are stored in two Hugging Face repositories:

- Results: `anonsaereg/SAE-REG-results`
- Model weights: `anonsaereg/SAE-REG-models`

The results repository is a Hugging Face dataset repository. The model weights
repository is a Hugging Face model repository.

To download the artifacts, install the repository dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python scripts/download_artifact.py
```

This recreates the expected local artifact directories:

```text
data_results/
data_model_weights/
```

To check that the artifacts are present, run:

```bash
python scripts/verify_artifact.py
```

If the Hugging Face repositories are private, first authenticate with an account
that has access:

```bash
huggingface-cli login
```

For a local-only check without querying Hugging Face, run:

```bash
python scripts/verify_artifact.py --offline
```

## Reproducing the Analyses

The fastest path is to reproduce the paper figures and analysis from the
precomputed artifacts:

1. Clone this repository.
2. Install the Python dependencies with `pip install -r requirements.txt`.
3. Download the artifacts with `python scripts/download_artifact.py`.
4. Verify the artifacts with `python scripts/verify_artifact.py`.
5. Run the analysis notebooks listed below.

### MNIST

The MNIST training experiments are in `experiments/mnist/`, and the analysis
notebook is in `analysis/mnist/`.

- `experiments/mnist/MNIST_not_constrained.ipynb` trains/evaluates the baseline
  MNIST SAE.
- `experiments/mnist/MNIST_norm_constrained.ipynb` trains/evaluates the
  norm-constrained or weight-regularized MNIST SAE.
- `analysis/mnist/analyse_models.ipynb` analyzes the saved MNIST runs and
  produces the MNIST plots used in the paper.

Run the training notebooks first if reproducing the MNIST models from scratch.
If using the downloaded artifacts, start with `analysis/mnist/analyse_models.ipynb`.

### Pythia SAE Analysis

The Pythia analysis notebooks are in `analysis/pythia/` and expect the
downloaded `data_results/` and `data_model_weights/` directories to be present
at the repository root.

Use these notebooks for the main Pythia SAE analyses:

- `analysis/pythia/pareto_radar_plot.ipynb`
- `analysis/pythia/decoder_orthogonality.ipynb`
- `analysis/pythia/Feature Consistency Analysis Across Random Seeds Pythia model.ipynb`
- `analysis/pythia/sae_weights_pythia_models copy.ipynb`

These notebooks load the saved evaluation outputs and SAE model weights, then
produce the comparison plots and summary statistics reported in the paper.

### Steering Analysis

The steering experiment code is in `experiments/steering/`.

The main scripts are:

- `experiments/steering/shared_feature_analysis.py`
- `experiments/steering/steering_all_features.py`
- `experiments/steering/steering_shared_features.py`
- `experiments/steering/steering_shared_hparam_sweep.py`

The paper-facing steering plots and summaries are produced in:

- `analysis/pythia/auto_interp_steering_analysis_topk40_model copy.ipynb`

The downloaded `data_results/` artifacts include the steering outputs needed by
this notebook, so rerunning the steering scripts is not required for the
standard analysis reproduction path.

## Suggested Review Path

For artifact reviewers, the recommended path is:

```bash
git clone <repo-url>
cd SAE-Regularization-Artifact
pip install -r requirements.txt
python scripts/download_artifact.py
python scripts/verify_artifact.py
```

Then open the notebooks in this order:

1. `analysis/mnist/analyse_models.ipynb`
2. `analysis/pythia/pareto_radar_plot.ipynb`
3. `analysis/pythia/decoder_orthogonality.ipynb`
4. `analysis/pythia/Feature Consistency Analysis Across Random Seeds Pythia model.ipynb`
5. `analysis/pythia/auto_interp_steering_analysis_topk40_model copy.ipynb`

This path reproduces the analyses from precomputed artifacts without requiring
reviewers to retrain all models from scratch.

## Notes on Full Reproduction

Full reproduction from scratch requires rerunning the MNIST training notebooks,
the Pythia SAE training/evaluation code, and the steering scripts. The vendored
`third_party/dictionary_learning/` code and `SAEBench_extension/` scripts are
included for that lower-level training/evaluation workflow. This is more
computationally expensive than the standard artifact path above. The downloaded
artifacts are provided so reviewers can reproduce the paper analyses directly
from the exact saved results and model weights.
