import torch
import copy

from utils.runtime import instantiate_from_config


def _build_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers=0,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "sampler": None,
        "drop_last": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return torch.utils.data.DataLoader(
        dataset,
        **loader_kwargs,
    )


def _with_output_dir(dataset_config, output_dir):
    if output_dir is not None:
        dataset_config["params"]["output_dir"] = output_dir
    return dataset_config


def build_dataloader(config, args=None):
    dataloader_config = config["dataloader"]
    dataset_config = dataloader_config["train_dataset"]
    output_dir = args.savesample_dir if args is not None else None
    dataset = instantiate_from_config(_with_output_dir(dataset_config, output_dir))

    dataloader = _build_loader(
        dataset=dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle"],
        num_workers=dataloader_config.get("num_workers", 0),
        pin_memory=dataloader_config.get("pin_memory", True),
        persistent_workers=dataloader_config.get("persistent_workers", True),
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
    )
    return {
        "dataloader": dataloader,
        "dataset": dataset,
    }


def build_dataloader_conditional(config, args=None, period="train"):
    if period not in {"train", "test"}:
        raise ValueError(f"Expected period to be 'train' or 'test', got {period!r}.")

    dataloader_config = config["dataloader"]
    dataset_key = f"{period}_dataset"
    dataset_config = dataloader_config[dataset_key]
    output_dir = None
    if args is not None:
        output_dir = args.savesample_dir if period == "train" else args.save_dir
    dataset = instantiate_from_config(_with_output_dir(dataset_config, output_dir))

    is_train = period == "train"
    batch_size = dataloader_config["batch_size"] if is_train else dataloader_config.get("sample_size", 16)
    dataloader = _build_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=dataloader_config.get("num_workers", 0),
        pin_memory=dataloader_config.get("pin_memory", True),
        persistent_workers=dataloader_config.get("persistent_workers", True),
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
    )
    return {
        "dataloader": dataloader,
        "dataset": dataset,
    }
