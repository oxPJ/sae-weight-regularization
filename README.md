# SAE Weight Regularization Artifact

This repository contains the code needed to reproduce the analyses for the paper
on weight regularization of sparse autoencoders.

The repository is organized around three workflows:

- MNIST experiments in `MNIST/`
- Pythia SAE analysis and plotting in `results_analysis_and_plotting_pythia/`
- Steering analysis code in `steering_analysis/`

Large result files and model weights are hosted on Hugging Face rather than
committed to this repository.

## Repository Layout

```text
.
├── MNIST/
│   ├── MNIST_not_constrained.ipynb
│   ├── MNIST_norm_constrained.ipynb
│   └── analyse_models.ipynb
├── results_analysis_and_plotting_pythia/
│   ├── pareto_radar_plot.ipynb
│   ├── decoder_orthogonality.ipynb
│   ├── Feature Consistency Analysis Across Random Seeds Pythia model.ipynb
│   ├── sae_weights_pythia_models copy.ipynb
│   └── auto_interp_steering_analysis_topk40_model copy.ipynb
├── steering_analysis/
│   ├── steering_shared_features.py
│   └── steering_shared_hparam_sweep.py
├── SAEBench/
├── dictionary_learning_demo/
├── download_artifact.py
└── verify_artifact.py
```

`SAEBench/` and `dictionary_learning_demo/` contain the external code used for
SAE evaluation and training.

## Artifact Downloads

The artifacts are stored in two Hugging Face repositories:

- Results: `anonsaereg/SAE-REG-results`
- Model weights: `anonsaereg/SAE-REG-models`

The results repository is a Hugging Face dataset repository. The model weights
repository is a Hugging Face model repository.

To download the artifacts, install the Hugging Face Hub client:

```bash
pip install huggingface_hub
```

Then run:

```bash
python download_artifact.py
```

This recreates the expected local artifact directories:

```text
data_results/
data_model_weights/
```

To check that the artifacts are present, run:

```bash
python verify_artifact.py
```

If the Hugging Face repositories are private, first authenticate with an account
that has access:

```bash
huggingface-cli login
```

For a local-only check without querying Hugging Face, run:

```bash
python verify_artifact.py --offline
```

## Reproducing the Analyses

The fastest path is to reproduce the paper figures and analysis from the
precomputed artifacts:

1. Clone this repository.
2. Install the Python dependencies used by the notebooks.
3. Download the artifacts with `python download_artifact.py`.
4. Verify the artifacts with `python verify_artifact.py`.
5. Run the analysis notebooks listed below.

### MNIST

The MNIST experiments are self-contained in `MNIST/`.

- `MNIST/MNIST_not_constrained.ipynb` trains/evaluates the baseline MNIST SAE.
- `MNIST/MNIST_norm_constrained.ipynb` trains/evaluates the norm-constrained or
  weight-regularized MNIST SAE.
- `MNIST/analyse_models.ipynb` analyzes the saved MNIST runs and produces the
  MNIST plots used in the paper.

Run the training notebooks first if reproducing the MNIST models from scratch.
If using the downloaded artifacts, start with `MNIST/analyse_models.ipynb`.

### Pythia SAE Analysis

The Pythia analysis notebooks are in `results_analysis_and_plotting_pythia/` and
expect the downloaded `data_results/` and `data_model_weights/` directories to
be present at the repository root.

Use these notebooks for the main Pythia SAE analyses:

- `results_analysis_and_plotting_pythia/pareto_radar_plot.ipynb`
- `results_analysis_and_plotting_pythia/decoder_orthogonality.ipynb`
- `results_analysis_and_plotting_pythia/Feature Consistency Analysis Across Random Seeds Pythia model.ipynb`
- `results_analysis_and_plotting_pythia/sae_weights_pythia_models copy.ipynb`

These notebooks load the saved evaluation outputs and SAE model weights, then
produce the comparison plots and summary statistics reported in the paper.

### Steering Analysis

The steering experiment code is in `steering_analysis/`.

The main scripts are:

- `steering_analysis/steering_shared_features.py`
- `steering_analysis/steering_shared_hparam_sweep.py`

The paper-facing steering plots and summaries are produced in:

- `results_analysis_and_plotting_pythia/auto_interp_steering_analysis_topk40_model copy.ipynb`

The downloaded `data_results/` artifacts include the steering outputs needed by
this notebook, so rerunning the steering scripts is not required for the
standard analysis reproduction path.

## Suggested Review Path

For artifact reviewers, the recommended path is:

```bash
git clone <repo-url>
cd SAE_Regularization_ICML
pip install huggingface_hub
python download_artifact.py
python verify_artifact.py
```

Then open the notebooks in this order:

1. `MNIST/analyse_models.ipynb`
2. `results_analysis_and_plotting_pythia/pareto_radar_plot.ipynb`
3. `results_analysis_and_plotting_pythia/decoder_orthogonality.ipynb`
4. `results_analysis_and_plotting_pythia/Feature Consistency Analysis Across Random Seeds Pythia model.ipynb`
5. `results_analysis_and_plotting_pythia/auto_interp_steering_analysis_topk40_model copy.ipynb`

This path reproduces the analyses from precomputed artifacts without requiring
reviewers to retrain all models from scratch.

## Notes on Full Reproduction

Full reproduction from scratch requires rerunning the MNIST training notebooks,
the Pythia SAE training/evaluation code, and the steering scripts. This is more
computationally expensive than the standard artifact path above. The downloaded
artifacts are provided so reviewers can reproduce the paper analyses directly
from the exact saved results and model weights.
