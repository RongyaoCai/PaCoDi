# Scripts

Scripts are grouped by purpose. The root of `scripts/` should contain only
this README; executable scripts live in exactly one functional folder.

| Folder | Purpose |
| --- | --- |
| `data/` | Dataset conversion and fixed synthetic dataset generation. |
| `experiments/` | Train + sample + evaluate launchers, one per task. |

Common commands:

```bash
# Dataset preparation
python scripts/data/prepare_npy_datasets.py
python scripts/data/generate_sines_dataset.py

# Main experiment scripts (one per task)
bash scripts/experiments/script_uncond.sh
bash scripts/experiments/script_conditional.sh
```

All experiment scripts accept environment-variable overrides for backbone,
dataset, sequence length, GPU, and sampling settings. See the top of each
script for the full list.
