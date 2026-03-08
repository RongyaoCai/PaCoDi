# PaCoDi

PaCoDi is a PyTorch project for time-series generation and text-conditioned generation. This repository includes training, sampling, and data building pipelines, and supports both unconditional and text-conditioned modes.


## Data

Download and unzip the dataset to the required location. See `./Data`.

## Training

Unconditional generation (example: air dataset):

```bash
python main.py --train --mode uncond --name air_run --config_file Config/uncond/air.yaml
```

Text-conditioned generation (example: air dataset):

```bash
python main.py --train --mode text --name air_text_run --config_file Config/text/air.yaml
```

## Sampling

Unconditional sampling:

```bash
python main.py --mode uncond --name air_run --config_file Config/uncond/air.yaml --milestone 10
```

Text-conditioned sampling (supports multiple runs):

```bash
python main.py --mode text --name air_text_run --config_file Config/text/air.yaml --milestone 10 --num_samples 5
```

## Outputs

Artifacts are saved to:

- `experiments/<mode>/<model_name>/<name>/checkpoint`
- `experiments/<mode>/<model_name>/<name>/log`
- `experiments/<mode>/<model_name>/<name>/samples`

## Configuration

Config files are located in `Config/uncond` and `Config/text`. You can edit:

- `model.target`/`model.params.seq_length` / `model.params.feature_size`
- `dataloader.train_dataset.params.data_root`
- `dataloader.batch_size` / `dataloader.sample_size`
