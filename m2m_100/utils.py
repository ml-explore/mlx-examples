"Utils for translation models"

import os
import time
from typing import Tuple, List

import numpy as np
import mlx.core as mx
from mlx_lm.sample_utils import top_p_sampling
import transformers
from transformers import PreTrainedTokenizerBase, NllbTokenizer, M2M100Config

from m2m_100_model import M2M100ForConditionalGeneration
from flores200_codes import FLORES_CODES

def convert(model_name: str, mlx_model: str) -> None:
    """
    Convert a model from Hugging Face to MLX format.

    Args:
        model_name (str): Hugging Face model name
        mlx_model (str): Path to save the MLX model
    """
    model = transformers.M2M100ForConditionalGeneration.from_pretrained(model_name)
    # save the tensors
    tensors = {
        key: tensor.numpy() for key, tensor in model.state_dict().items()
    }
    np.savez(mlx_model, **tensors)


def load_mlx_nllb_model(
    model_name: str, weights_path: str, source_language: str, target_language: str
) -> Tuple[M2M100ForConditionalGeneration, PreTrainedTokenizerBase, int]:
    """
    Load a nllb model and tokenizer from the given model name and weights path.
    """
    if source_language in FLORES_CODES and target_language in FLORES_CODES:
        source_language = FLORES_CODES[source_language]
        target_language = FLORES_CODES[target_language]

    if os.path.exists(weights_path):
        mlx_model = weights_path
    else:
        convert(model_name, weights_path)
        assert os.path.exists(weights_path), f"Model checkpoint not found at {weights_path}"

    config = M2M100Config.from_pretrained(model_name)

    model = M2M100ForConditionalGeneration(config)

    #setting to False because of the Positional Encoding layer
    model.load_weights(weights_path, strict=False)

    tokenizer = NllbTokenizer.from_pretrained(model_name,
        src_lang=source_language,
        tgt_lang=target_language
        )
    
    tgt_token_id =  tokenizer.convert_tokens_to_ids("yor_Latn")
    return model, tokenizer, tgt_token_id

def sample(logits: mx.array, temp: float, top_p: float) -> Tuple[mx.array, float]:
    """
    Sample a token from the logits.

    Args:
        logits (mx.array): Logits from the model
        temp (float): Temperature for sampling
        top_p (float): Top-p sampling value

    Returns:
        Tuple[mx.array, float]: Tuple of the token and its probability
    """
    softmax_logits = mx.softmax(logits)

    if temp == 0:
        token = mx.argmax(logits, axis=-1)
    else:
        if top_p > 0 and top_p < 1.0:
            token = top_p_sampling(logits, top_p, temp)
        else:
            token = mx.random.categorical(logits * (1 / temp))

    prob = softmax_logits[0, token]
    return token, prob


def run_translation_mlx(
    model: M2M100ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    input_sentences: List[str],
    target_language_token: int,
    max_generation_tokens: int,
    temp: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = False
    ) -> List[str]:

    """
    Run Translation using the MLX model.

    Returns:
         List[str]: List of translated sentences
    """

    bsz = len(input_sentences)
    tokens = tokenizer(input_sentences, return_tensors="np", padding="longest")

    print(tokens['input_ids'].shape)
    print(tokens['input_ids'])
    tokens = {key: mx.array(v) for key, v in tokens.items()}

    decoder_input_ids = mx.array([[model.config.eos_token_id, target_language_token]]*bsz)
    decoder_input_mask = mx.array([[1, 1]]*bsz)

    encoder_tokens = model.encode(tokens["input_ids"], tokens["attention_mask"])

    start_time = time.time()  # Start measuring time

    for _ in range(max_generation_tokens):
        outputs = model.decode(decoder_input_ids,decoder_input_mask, encoder_tokens, tokens["attention_mask"], None)
        logits = model.lm_head(outputs)[:,-1,:]

        next_token, _ = sample(logits, temp, top_p)

        decoder_input_ids = mx.concatenate([decoder_input_ids, next_token.reshape(-1, 1)], axis=1)
        decoder_input_mask = mx.concatenate([decoder_input_mask, mx.ones((bsz,1))], axis=1)

        if next_token[0] == model.config.eos_token_id:
            break


    end_time = time.time()  # Stop measuring time

    if verbose:
        total_tokens = decoder_input_ids.size - (bsz*2)
        total_time = end_time - start_time
        token_per_sec = total_tokens / total_time
        print(f"Generated {total_tokens} tokens in {total_time} seconds.")
        print(f"Token/s: {token_per_sec}")

    translated_sentences = tokenizer.batch_decode(np.array(decoder_input_ids), skip_special_tokens=True)

    return translated_sentences