
## Project Overview

This project is a fork of the `diffusers` library, adapted for running and experimenting with the [Wan 2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) model, a large-scale text-to-video diffusion model. The primary focus is on enabling efficient inference on TPUs, particularly Google's v6e TPUs, by leveraging technologies like JAX, Flax, and custom attention mechanisms such as Splash Attention.

The project includes scripts for running inference, comparing performance with different attention mechanisms (`compare_fa_sharding_consistency.py`), and detailed documentation on model architecture, memory usage, and multi-host execution.

## Dependencies

The project has several key dependencies:

- **Core:** `torch`, `jax[tpu]`, `transformers`, `accelerate`
- **Custom Backends:** `torchax` (a fork with specific changes), `maxdiffusion` (for VAE components)
- **General:** `flax`, `ftfy`, `imageio`, `imageio-ffmpeg`, `tensorflow`

Installation is managed via `pip` as detailed in the `README.md`.

## Key Files

- **`wan_tx.py`**: The main script for running the Wan-TX text-to-video generation pipeline.
- **`wan_tx_splash_attn.py`**: An alternative script that integrates JAX Pallas Splash Attention for potentially more efficient inference.
- **`compare_fa_sharding_consistency.py`**: A utility script to test and verify the consistency of Flash Attention implementations with specific tensor sharding strategies.
- **`README.md`**: Contains setup instructions, progress logs, model architecture details, and commands for multi-host execution.
- **`CONTRIBUTING.md`**: Standard contribution guidelines for the upstream `diffusers` project.
- **`pyproject.toml`**: Defines project metadata, dependencies, and tool configurations for `ruff` and `pytest`.

## Development Workflow

### Installation

Set up the development environment by installing the required packages:

```bash
pip install -e .
pip install transformers accelerate
pip install git+https://github.com/pytorch/xla.git@hanq_wan_changes#subdirectory=torchax
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jax[tpu]
pip install flax
pip install git+https://github.com/AI-Hypercomputer/maxdiffusion
```

### Running the Model

The main inference script is executed as follows:

```bash
python wan_tx.py
```

### Testing

The project uses `pytest`. To run the full test suite:

```bash
make test
```

To run specific tests:

```bash
pytest tests/<TEST_TO_RUN>.py
```

To include slow tests, set the `RUN_SLOW` environment variable:

```bash
RUN_SLOW=yes pytest tests/<TEST_TO_RUN>.py
```

### Code Style & Linting

The project uses `ruff` for formatting and linting. To apply styles and check for quality:

```bash
make style
make quality
```

Docstrings follow the Google style guide.

## Custom Commands

The `README.md` includes a detailed set of shell commands and functions (`run`, `SETUP_COMMAND`, `RUN_COMMAND`) for automating the setup and execution of the model on a multi-host TPU environment (e.g., a v6e-16 pod). These scripts handle environment setup, dependency installation, and distributed execution of the inference script.
