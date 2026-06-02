import importlib
import random
import sys
import warnings

import numpy as np
import torch
import yaml


def load_yaml_config(path):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.full_load(f)


def save_config_to_yaml(config, path):
    assert path.endswith(".yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def merge_opts_to_config(config, opts):
    def cast_value(current_value, value):
        if isinstance(current_value, bool):
            if isinstance(value, bool):
                return value
            value_lower = str(value).lower()
            if value_lower in {"true", "1", "yes", "y", "on"}:
                return True
            if value_lower in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError(f"Cannot parse boolean override value: {value}")
        return type(current_value)(value)

    def modify_dict(current, path, value):
        if len(path) == 1:
            current[path[0]] = cast_value(current[path[0]], value)
        else:
            current[path[0]] = modify_dict(current[path[0]], path[1:], value)
        return current

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "Each override should be a name/value pair."
        for i in range(len(opts) // 2):
            name = opts[2 * i]
            value = opts[2 * i + 1]
            config = modify_dict(config, name.split("."), value)
    return config


def instantiate_from_config(config):
    if config is None:
        return None
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", {}))


def normalize_to_neg_one_to_one(x):
    return x * 2 - 1


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


def seed_everything(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn(
            "You have chosen deterministic CUDNN. This can slow down training and "
            "may change restart behavior.",
            stacklevel=2,
        )


def write_args(args, path):
    args_dict = {name: getattr(args, name) for name in dir(args) if not name.startswith("_")}
    with open(path, "a", encoding="utf-8") as args_file:
        args_file.write(f"==> torch version: {torch.__version__}\n")
        args_file.write(f"==> cudnn version: {torch.backends.cudnn.version()}\n")
        args_file.write("==> Cmd:\n")
        args_file.write(str(sys.argv))
        args_file.write("\n==> args:\n")
        for key, value in sorted(args_dict.items()):
            args_file.write(f"  {key}: {value}\n")


def get_model_parameters_info(model):
    parameters = {"overall": {"trainable": 0, "non_trainable": 0, "total": 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {"trainable": 0, "non_trainable": 0}
        for _, param in child_module.named_parameters():
            if param.requires_grad:
                parameters[child_name]["trainable"] += param.numel()
            else:
                parameters[child_name]["non_trainable"] += param.numel()
        parameters[child_name]["total"] = (
            parameters[child_name]["trainable"] + parameters[child_name]["non_trainable"]
        )

        parameters["overall"]["trainable"] += parameters[child_name]["trainable"]
        parameters["overall"]["non_trainable"] += parameters[child_name]["non_trainable"]
        parameters["overall"]["total"] += parameters[child_name]["total"]

    def format_number(num):
        k = 2**10
        m = 2**20
        g = 2**30
        if num > g:
            unit = "G"
            num = round(float(num) / g, 2)
        elif num > m:
            unit = "M"
            num = round(float(num) / m, 2)
        elif num > k:
            unit = "K"
            num = round(float(num) / k, 2)
        else:
            unit = ""
        return f"{num}{unit}"

    def format_dict(values):
        for key, value in values.items():
            if isinstance(value, dict):
                format_dict(value)
            else:
                values[key] = format_number(value)

    format_dict(parameters)
    return parameters
