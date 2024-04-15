"""
Test the M2M100ForConditionalGeneration model.
"""

import unittest

import os
import torch
from transformers import NllbTokenizer, M2M100ForConditionalGeneration, M2M100Config

import mlx.core as mx

from convert import convert
from m2m_100_model import M2M100ForConditionalGeneration as M2M100ForConditionalGenerationMLX


def forward_hf(
    model: M2M100ForConditionalGeneration,
    tokenizer: NllbTokenizer,
    sample_input: str):
    """
    Run Translation using the Torch HGF model.

    Args:
        model (M2M100ForConditionalGeneration): Huggingface model
        tokenizer (NllbTokenizer): NLLB tokenizer
        sample_input (str): Sample sentence to translate
    """

    input_tokens = tokenizer(sample_input, return_tensors="pt")

    with torch.no_grad():
        encoder_outputs = model.model.encoder(**input_tokens)


        decoder_input_ids = torch.tensor([[2, tokenizer.lang_code_to_id["fra_Latn"]]])
        decoder_input_mask = torch.tensor([[1, 1]])

        for _ in range(10):
            outputs = model.model.decoder(input_ids=decoder_input_ids,
                                        attention_mask=decoder_input_mask,
                                        encoder_hidden_states=encoder_outputs[0],
                                        encoder_attention_mask=input_tokens["attention_mask"])

            logits = model.lm_head(outputs[0])
            logits = logits[:, -1, :]

            next_token = torch.argmax(logits, dim=-1)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(0)], dim=-1)
            decoder_input_mask = torch.cat([decoder_input_mask,
                                            torch.ones_like(next_token).unsqueeze(0)], dim=-1)

            if next_token == model.config.eos_token_id:
                break

    return decoder_input_ids

def forward_mlx(
    model: M2M100ForConditionalGenerationMLX,
    tokenizer: NllbTokenizer,
    sample_input: str):
    """
    Run Translation using the MLX model.

    Args:
        model (M2M100ForConditionalGenerationMLX): MLX model
        tokenizer (NllbTokenizer): NLLB tokenizer
        sample_input (str): Sample sentence to translate
    """

    inputs = tokenizer(sample_input, return_tensors="np")

    decoder_input_ids = mx.array([[model.config.eos_token_id,
                            tokenizer.lang_code_to_id["fra_Latn"]]])

    decoder_input_mask = mx.array([[1, 1]])

    encoder_tokens = model.encode(inputs["input_ids"], inputs["attention_mask"])

    for _ in range(10):
        outputs = model.decode(decoder_input_ids,decoder_input_mask, encoder_tokens, inputs["attention_mask"], None)
        logits = model.lm_head(outputs)[:,-1,:]

        next_token = mx.argmax(logits, axis=-1)

        decoder_input_ids = mx.concatenate([decoder_input_ids, next_token.reshape(-1, 1)], axis=1)
        decoder_input_mask = mx.concatenate([decoder_input_mask, mx.ones((1,1))], axis=1)

        if next_token[0] == model.config.eos_token_id:
            break
    
    return decoder_input_ids

def load_hf_models(model_name: str):
    """
    Load the Huggingface models.

    Args:
        model_name (str): Model name
    """
    return M2M100ForConditionalGeneration.from_pretrained(model_name)

def load_mlx_models(model_name: str):
    """
    Load the MLX models.

    Args:
        model_name (str): Model name
    """
    current_directory = os.getcwd()
    weights_path = os.path.join(current_directory,
                                "weights",
                                model_name.replace("/", "-") + ".npz")

    convert(model_name, weights_path)

    config = M2M100Config.from_pretrained(model_name)
    model = M2M100ForConditionalGenerationMLX(config)

    model.load_weights(weights_path, strict=False)

    return model


class TestM2M100(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_name = "facebook/nllb-200-distilled-600M"
        cls.mlx_model = load_mlx_models(model_name)
        cls.hf_model = load_hf_models(model_name)

        cls.tokenizer = NllbTokenizer.from_pretrained(model_name,
                                        src_lang="eng_Latn",
                                        tgt_lang="fra_Latn"
                                        )

    def test_generation(self):
        sample_input = "what is your name?"

        hf_output = forward_hf(self.hf_model, self.tokenizer, sample_input)
        mlx_output = forward_mlx(self.mlx_model, self.tokenizer, sample_input)

        self.assertTrue(mx.allclose(hf_output.numpy(), mlx_output.asnumpy()))
    
if __name__ == "__main__":
    unittest.main()