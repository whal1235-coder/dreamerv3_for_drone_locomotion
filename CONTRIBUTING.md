# Contributing

Thanks for your interest in contributing!

## Setup

```bash
conda env create -f environment.yml
pip install -e .
```

## Pull requests

- Keep changes focused; one feature/fix per PR
- Run `python -m pyflakes dreamerv3 embodied` before submitting
- For new environments, add a config entry in `baselines.yaml` and register it in `embodied/envs/`

## Issues

- Bug reports: include the config used, MuJoCo/JAX versions, and the full traceback
- Feature requests: describe the environment or algorithm and why it fits DreamerV3
