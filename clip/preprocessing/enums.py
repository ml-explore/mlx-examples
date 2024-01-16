"""
Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/image_utils.py.
"""
from enum import Enum


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"
