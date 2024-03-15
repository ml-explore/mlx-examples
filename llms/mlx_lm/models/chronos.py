import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

## TODO: remove the following two lines
## import torch
## import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    PreTrainedModel,
)

from .base import BaseModelArgs
from .layers import LayerNorm, RMSNorm

##************************************************************************##
## INFO:
## - paper: https://arxiv.org/abs/2403.07815
## - model:
##   - https://huggingface.co/amazon/chronos-t5-large
##   - https://huggingface.co/elinas/chronos-13b
## - repo: https://github.com/amazon-science/chronos-forecasting
##
## Notes:
## - refer to olmo.py and hf-olmo model implementation for inspiration.
##   - https://github.com/allenai/OLMo
##   - https://huggingface.co/allenai/OLMo-1B
##************************************************************************##

try:
    import chronos
except ImportError:
    print("To run chronos, install chronos with: pip install chronos")
    exit(1)


def nansum(x: mx.array, axis: int = -1):
    return mx.sum(mx.where(mx.isnan(x), 0, x), axis=axis)


mx.nansum = nansum


@dataclass
class ModelArgs(BaseModelArgs):
    """
    This class holds all the configuration parameters to be used
    by ``ChronosTokenizer`` and ``ChronosModel``.
    """

    # TODO: assign default values
    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    context_length: int
    prediction_length: int
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert (
            self.pad_token_id < self.n_special_tokens
            and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"

    def create_tokenizer(self) -> "ChronosTokenizer":
        if self.tokenizer_class == "MeanScaleUniformBins":
            return MeanScaleUniformBins(**self.tokenizer_kwargs, config=self)
        raise ValueError


# Define ChronosConfig as in original implementation
ChronosConfig = ModelArgs


class ChronosTokenizer:
    """
    A ``ChronosTokenizer`` definines how time series are mapped into token IDs
    and back.

    For details, see the ``input_transform`` and ``output_transform`` methods,
    which concrete classes must implement.
    """

    def input_transform(self, context: mx.array) -> Tuple[mx.array, mx.array, Any]:
        """
        Turn a batch of time series into token IDs, attention map, and scale.

        Parameters
        ----------
        context
            A tensor shaped (batch_size, time_length), containing the
            timeseries to forecast. Use left-padding with ``torch.nan``
            to align time series of different lengths.

        Returns
        -------
        token_ids
            A tensor of integers, shaped (batch_size, time_length + 1)
            if ``config.use_eos_token`` and (batch_size, time_length)
            otherwise, containing token IDs for the input series.
        attention_mask
            A boolean tensor, same shape as ``token_ids``, indicating
            which input observations are not ``torch.nan`` (i.e. not
            missing nor padding).
        decoding_context
            An object that will be passed to ``output_transform``.
            Contains the relevant context to decode output samples into
            real values, such as location and scale parameters.
        """
        raise NotImplementedError()

    def output_transform(self, samples: mx.array, decoding_context: Any) -> mx.array:
        """
        Turn a batch of sample token IDs into real values.

        Parameters
        ----------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing token IDs of sample trajectories.
        decoding_context
            An object returned by ``input_transform`` containing
            relevant context to decode samples, such as location and scale.
            The nature of this depends on the specific tokenizer.

        Returns
        -------
        forecasts
            A real ``mx.array``, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        raise NotImplementedError()


# TODO: convert into mlx compatible code
class MeanScaleUniformBins(ChronosTokenizer):
    def __init__(
        self, low_limit: float, high_limit: float, config: ChronosConfig
    ) -> None:
        self.config = config
        self.centers = np.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        ## TODO: check if torch.concat can be replaced with mx.concat or np.concat
        self.boundaries = np.concat(
            (
                # TODO: check if torch.tensor <-> mx.array works or not
                #       mx.array DOESNOT have any "device" argument
                # DONE.
                mx.array([-1e20]),
                (self.centers[1:] + self.centers[:-1]) / 2,
                mx.array([1e20]),
            )
        )

    def input_transform(self, context: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        batch_size, length = context.shape

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]
        elif length < self.config.context_length:
            padding_size = (
                *context.shape[:-1],
                self.config.context_length - length,
            )
            # TODO: check if
            # - DONE | torch.full == mx.full (args: size --> shape, fill_value --> val)
            # - DONE | torch.nan == mx.nan (~same; dtype precision could be diferent)
            # - DONE | torch.cat == torch.concat torch.concatenate == mx.concatenate (args: dim --> axis)
            padding = mx.full(shape=padding_size, val=mx.nan)
            context = mx.concatenate((padding, context), axis=-1)

        # TODO: check if
        # - DONE | torch.isnan == mx.isnan (same)
        # - DONE | torch.abs == mx.abs
        # - DONE | torch.nansum == mx.nansum?? (np.nansum; created function mx.nansum following pytorch implementation)
        attention_mask = ~mx.isnan(context)
        scale = mx.nansum(mx.abs(context) * attention_mask, axis=-1) / mx.nansum(
            attention_mask, axis=-1
        )
        scale[~(scale > 0)] = 1.0
        # TODO: check if mx.array.unsqueeze(dim=-1) is valid or not
        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            # TODO: check if mx.array has something equivalent to torch.bucketize?
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )
        token_ids[~attention_mask] = self.config.pad_token_id

        # TODO: replace...
        # - DONE | torch.full --> mx.full
        # - DONE | torch.concat --> mx.concatenate
        if self.config.use_eos_token:
            eos_tokens = mx.full((batch_size, 1), val=self.config.eos_token_id)
            token_ids = mx.concatenate((token_ids, eos_tokens), axis=1)
            eos_mask = mx.full((batch_size, 1), val=True)
            attention_mask = mx.concatenate((attention_mask, eos_mask), axis=1)

        return token_ids, attention_mask, scale

    def output_transform(self, samples: mx.array, scale: mx.array) -> mx.array:
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        # TODO: replace
        # - DONE | torch.clamp --> mx.clip (args: min --> a_min, max --> a_max)
        indices = mx.clip(
            samples - self.config.n_special_tokens,
            a_min=0,
            a_max=len(self.centers) - 1,
        )
        return self.centers[indices] * scale_unsqueezed


# TODO: convert into mlx compatible code
class ChronosModel(nn.Module):
    """
    A ``ChronosModel`` wraps a ``PreTrainedModel`` object from ``transformers``
    and uses it to predict sample paths for time series tokens.

    Parameters
    ----------
    config
        The configuration to use.
    model
        The pre-trained model to use.
    """

    def __init__(self, config: ChronosConfig, model: PreTrainedModel) -> None:
        super().__init__()
        self.config = config
        self.model = model
        # TODO: is the device argument valid for mlx?
        self.device = model.device

    def forward(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> mx.array:
        """
        Predict future sample tokens for the given token sequences.

        Arguments ``prediction_length``, ``num_samples``, ``temperature``,
        ``top_k``, ``top_p`` can be used to customize the model inference,
        and default to the corresponding attributes in ``self.config`` if
        not provided.

        Returns
        -------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        if num_samples is None:
            num_samples = self.config.num_samples
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p

        preds = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(
                min_new_tokens=prediction_length,
                max_new_tokens=prediction_length,
                do_sample=True,
                num_return_sequences=num_samples,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=self.config.pad_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ),
        )

        if self.config.model_type == "seq2seq":
            preds = preds[..., 1:]  # remove the decoder start token
        else:
            assert self.config.model_type == "causal"
            # TODO: change torch.tesnor.size --> mx.array.shape
            # - DONE | input_ids.size --> input_ids.shape
            assert preds.size(-1) == input_ids.shape(-1) + prediction_length
            preds = preds[..., -prediction_length:]

        return preds.reshape(input_ids.shape(0), num_samples, -1)


# TODO: convert into mlx compatible code
def left_pad_and_stack_1D(tensors: List[mx.array]):
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, mx.array)
        assert c.ndim == 1
        # TODO: update
        # - DONE | torch.full --> mx.full (drop arg: devce; update arg: fill_value --> val)
        # - DONE | torch.concat --> mx.concatenate (update arg: dim --> axis)
        # - DONE | torch.nan --> mx.nan
        padding = mx.full(size=(max_len - len(c),), val=mx.nan)
        padded.append(mx.concatenate((padding, c), axis=-1))
    return mx.stack(padded)


# TODO: convert into mlx compatible code
class ChronosPipeline:
    """
    A ``ChronosPipeline`` uses the given tokenizer and model to forecast
    input time series.

    Use the ``from_pretrained`` class method to load serialized models.
    Use the ``predict`` method to get forecasts.

    Parameters
    ----------
    tokenizer
        The tokenizer object to use.
    model
        The model to use.
    """

    tokenizer: ChronosTokenizer
    model: ChronosModel

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def predict(
        self,
        context: Union[mx.array, List[mx.array]],
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = True,
    ) -> mx.array:
        """
        Get forecasts for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to what specified
            in ``self.model.config``.
        num_samples
            Number of sample paths to predict. Defaults to what
            specified in ``self.model.config``.
        temperature
            Temperature to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_k
            Top-k parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_p
            Top-p parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. True by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        samples
            Tensor of sample forecasts, of shape
            (batch_size, num_samples, prediction_length).
        """
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        if prediction_length is None:
            prediction_length = self.model.config.prediction_length

        if prediction_length > self.model.config.prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {self.model.config.prediction_length}. "
                f"The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            token_ids, attention_mask, scale = self.tokenizer.input_transform(context)
            samples = self.model(
                token_ids.to(self.model.device),
                attention_mask.to(self.model.device),
                min(remaining, self.model.config.prediction_length),
                num_samples,
                temperature,
                top_k,
                top_p,
            )
            prediction = self.tokenizer.output_transform(
                samples.to(scale.device), scale
            )

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            # TODO: check which one is better for torch.cat: mx.concatenate or mx.stack
            # - DONE | torch.concat --> mx.concatenate (update arg: dim --> axis)
            # - DONE*| torch.median(x, dim=1) --> np.median(x, axis=1)
            #   - 👉 NOTE: THERE IS NOT DIRECT OPTION IN mlx.nn
            #   - TODO: replace np.mdeian with mx.median once it is available in future.
            context = mx.concatenate(
                [context, np.median(prediction, axis=1).values], axis=-1
            )

        # TODO: check which one is better for torch.cat: mx.concatenate or mx.stack
        # - DONE | torch.concat --> mx.concatenate (update arg: dim --> axis)
        return mx.concatenate(predictions, axis=-1)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        config = AutoConfig.from_pretrained(*args, **kwargs)

        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        chronos_config = ChronosConfig(**config.chronos_config)

        if chronos_config.model_type == "seq2seq":
            inner_model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        else:
            assert config.model_type == "causal"
            inner_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        return cls(
            tokenizer=chronos_config.create_tokenizer(),
            model=ChronosModel(config=chronos_config, model=inner_model),
        )
