# BridgeFold

BridgeFold is an early research prototype exploring flow matching for protein backbone generation.

The current version focuses on a controlled first step: learning a velocity field for fixed length C alpha traces with a small Transformer. It is a single scale baseline used to validate the data pipeline and training objective before adding sampling and coarse to fine generation.

## Current status

Implemented:

- Loading and splitting fixed length C alpha backbones from an RCSB derived dataset
- Per structure centering and dataset level coordinate scaling
- A 3 layer Transformer encoder with coordinate, position, and time embeddings
- A conditional flow matching objective between centered Gaussian noise and protein coordinates
- A fixed batch overfitting check for the initial training baseline

Not implemented yet:

- ODE based sampling of new backbones
- Held out evaluation and structure quality metrics
- Coarse to fine bridge construction
- Explicit geometric constraints or rotation equivariant layers

This repository should therefore be read as a research prototype in progress, not as a complete protein generation system.

## Data

The included dataset contains 256 protein backbone segments of length 64. Each example contains the 3D coordinates of its C alpha atoms.

The data is separated by RCSB 30 percent sequence clusters and contains:

- 204 training structures
- 26 validation structures
- 26 test structures

The metadata in `data/rcsb_ca64_cluster30.json` records the source structure, chain, sequence, crop, split, and basic geometry checks for every example.

## Model

For an interpolated structure `x_t`, the model predicts the velocity that moves centered noise toward a real backbone. The network uses:

- coordinate embeddings for each 3D point
- sinusoidal embeddings for residue position and flow time
- 3 Transformer encoder layers
- 4 attention heads and a hidden size of 96
- a linear head that predicts one 3D velocity per residue

The model is intentionally small so that the first experiments remain easy to inspect and run.

## Run the current baseline

The prototype requires Python, PyTorch, and NumPy.

```bash
python train_single_scale.py
```

The script trains on one fixed batch for 501 steps. This is a sanity check that the model and objective can fit a small sample. It is not a full training or evaluation run.

## Roadmap

- [x] Curated fixed length backbone dataset
- [x] Single scale flow matching objective
- [x] Transformer velocity model
- [x] Fixed batch training sanity check
- [ ] Numerical sampler
- [ ] Validation metrics and generated structure analysis
- [ ] Multiscale coarse to fine generation
- [ ] Geometric inductive biases and chain constraints
