# Third-Party Code

## third_party/dictionary_learning (vendored)

- Original: https://github.com/saprmarks/dictionary_learning
- Original authors: Samuel Marks, Adam Karvonen, Aaron Mueller
- License: MIT (see third_party/dictionary_learning/LICENSE)
- Modifications: addition of trainer files with L1 and L2 penalties added to the loss function and related files updates so that a training sweep can be made on those new trainers.

## sae-bench (pip dependency)

- Source: https://github.com/adamkarvonen/SAEBench (PyPI: `sae-bench`)
- Authors: Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges,
  Joseph Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum McDougall,
  Kola Ayonrinde, Matthew Wearden, Arthur Conmy, Samuel Marks, Neel Nanda
- Citation: Karvonen et al., "SAEBench: A Comprehensive Benchmark for Sparse
  Autoencoders in Language Model Interpretability," ICML 2025.
- Used as: pip dependency (not redistributed in this repo). Custom
  extensions building on SAEBench are in `SAEBench_extensions/`.
