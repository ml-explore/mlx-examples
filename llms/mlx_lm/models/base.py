import inspect
from dataclasses import dataclass


@dataclass
class BaseModelArgs:
    train: bool = False

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
