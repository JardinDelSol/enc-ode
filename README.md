# ENC-ODE

This is a standalone repository containing the core ENC-ODE implementation.

## Contents

- `main.py` - training and evaluation loop
- `model.py` - ENC-ODE model definition
- `ODEs.py` - ODE helper and solver code
- `dataloader.py` - dataset loading and padding logic
- `utils.py` - training utilities and logging helpers
- `randomseed_test.sh` - reproducibility script

## Notes

- This repository is intentionally minimal and excludes data, baseline experiments, and analysis notebooks.
- Dataset files should be provided separately; scripts assume a dataset path such as `../datasets/`.
- `ODEs.py` includes code adapted from `rtqichen/ffjord`.
# enc-ode
