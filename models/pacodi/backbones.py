import inspect

from models.pacodi.dit_solver_v1 import DiTSolverV1


BACKBONE_REGISTRY = {
    "dit_solver_v1": DiTSolverV1,
}


def build_backbone(name, **kwargs):
    if name not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY))
        raise ValueError(f"Unknown backbone `{name}`. Available backbones: {available}")
    cls = BACKBONE_REGISTRY[name]
    accepted = inspect.signature(cls.__init__).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return cls(**filtered_kwargs)
