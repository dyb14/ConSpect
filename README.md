# ConSpect

**ConSpect: State-Adaptive Spectral Diffusion Network for Citywide Flow Prediction**

This repository contains the research implementation of ConSpect for citywide grid-based flow prediction on TaxiBJ and BikeNYC. The final model retains the three temporal branches used in the experiments: **closeness (C), period (P), and trend (T)**. Their features are fused before the shared ConSpect backbone, which combines local spatial modeling, state-adaptive spectral diffusion, and bounded local-diffusion fusion.

## Key repository structure

```text
ConSpect/
├── configs/                         # TaxiBJ and BikeNYC experiment configs
├── conspect/
│   ├── model.py                    # Main ConSpect model
│   ├── trainer.py                  # Validation-RMSE checkpoint selection
│   ├── training_protocol.py        # Smooth L1 + AdamW + OneCycleLR
│   ├── analysis/                   # Effective diffusion range analysis
│   ├── dataset/                    # TaxiBJ/BikeNYC data preparation
│   └── utils/
├── datasets/                        # Local dataset location; raw data are not tracked
├── tests/                           # Lightweight model tests
└── train.py                         # Main training entry point
```

## Environment

The formal experiments were designed for Python 3.12 and PyTorch with CUDA support. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

For a CUDA-specific PyTorch build, install the appropriate PyTorch wheel for the local CUDA environment first, then install the remaining dependencies.

## Dataset layout

Raw datasets are intentionally excluded from version control. Place the files at the paths expected by the configuration files.

### TaxiBJ

```text
datasets/TaxiBJ/
├── BJ13_M32x32_T30_InOut.h5
├── BJ14_M32x32_T30_InOut.h5
├── BJ15_M32x32_T30_InOut.h5
├── BJ16_M32x32_T30_InOut.h5
├── BJ_Holiday.txt
└── BJ_Meteorology.h5
```

### BikeNYC

```text
datasets/BikeNYC/
└── NYC14_M16x8_T60_NewEnd.h5
```

The repository does not redistribute third-party datasets. Obtain them from their original or authorized source and comply with their terms of use.

## Temporal input setting

The released model uses the final **C/P/T three-branch structure**:

- `len_closeness = 3`
- `len_period = 1`
- `len_trend = 1`

The corresponding branches are processed separately and fused before the shared spatial backbone. External features are incorporated by the model according to the dataset configuration.

## Training

All included configuration files use the same basic optimization budget:

```text
epochs = 100
batch_size = 32
```

TaxiBJ with four residual units:

```bash
python train.py configs/TaxiBJ/L4-R3/config.ini --seed 0
```

BikeNYC with four residual units:

```bash
python train.py configs/BikeNYC/L4-R3/config.ini --seed 0
```

## Reproducibility protocol

The included training protocol uses:

- 100 training epochs
- batch size 32
- Smooth L1 loss with `beta = 1`
- AdamW with learning rate `1e-3` and weight decay `0.05`
- OneCycleLR with maximum learning rate `2e-3` and warm-up fraction `0.10`
- best-checkpoint selection by validation RMSE
- deterministic random seeding where supported

Training outputs, logs, and checkpoints are written under `outputs/` and are excluded from Git.

## Smoke test and unit tests

A data-backed smoke test can be run after the datasets are placed locally:

```bash
python train.py configs/TaxiBJ/L4-R3/config.ini --smoke-test --epochs 1
```

Architecture-only tests do not require the datasets:

```bash
python -m unittest discover -s tests -v
```

## Spectral diffusion analysis

The `conspect.analysis` package contains the offline effective-diffusion analysis used to map diffusion time to spatial range statistics, including the Chebyshev-ring effective radius protocol.

## Outputs

Each training run stores its checkpoint, log, and machine-readable metrics in its run directory. Generated outputs are intentionally excluded from the repository so that the codebase remains source-only and reproducible.

## Citation

If this repository supports a paper submission, add the final bibliographic entry here after the manuscript metadata are fixed.

## License

No open-source license is included in this package. Add the license approved by the project authors or institution before public release if redistribution rights are intended.
