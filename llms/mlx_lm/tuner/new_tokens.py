import mlx.nn as nn
import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper


def resize_embeddings(model: nn.Module, tokenizer: TokenizerWrapper) -> nn.Module:
    """
    Resizes model embeddings to accommodate new tokens
    """
    old_embedding = model.model.embed_tokens

    old_vocab_size = old_embedding.num_embeddings
    new_vocab_size = len(tokenizer._tokenizer)

    if old_vocab_size != new_vocab_size:
        if new_vocab_size < old_vocab_size:
            print(
                "Warning: New vocab size is smaller than original. Proceeding with trim."
            )

        # check if QuantizedEmbedding has required attributes for dequantization
        try:
            dequantized_weights = mx.dequantize(
                old_embedding.weight,
                scales=old_embedding.scales,
                biases=old_embedding.biases,
                group_size=old_embedding.group_size,
                bits=old_embedding.bits,
            )
        except AttributeError as e:
            print(f"Error: Cannot dequantize embed_tokens. Missing attributes: {e}")
            print("Falling back to random weights for embed_tokens.")
            dequantized_weights = mx.random.normal(
                (old_vocab_size, old_embedding.dims), loc=0.0, scale=0.02
            )

        # resize embed_tokens
        new_embedding = nn.Embedding(new_vocab_size, old_embedding.dims)
        new_weights = mx.zeros((new_vocab_size, old_embedding.dims))
        min_vocab_size = min(old_vocab_size, new_vocab_size)
        new_weights[:min_vocab_size] = dequantized_weights[:min_vocab_size]
        if new_vocab_size > old_vocab_size:
            new_weights[old_vocab_size:] = mx.random.normal(
                (new_vocab_size - old_vocab_size, old_embedding.dims),
                loc=0.0,
                scale=0.02,
            )
        new_embedding.weight = new_weights
        model.model.embed_tokens = new_embedding

        # attention layers handling
        if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
            model.model.embed_tokens.weight = new_weights
        elif hasattr(model, "lm_head"):
            old_lm_head = model.lm_head
            if isinstance(old_lm_head, nn.QuantizedLinear):
                # resize nn.QuantizedLinear
                output_dims, compressed_input_dims = old_lm_head.weight.shape
                bits = old_lm_head.bits
                input_dims = compressed_input_dims * (32 // bits)

                # dequantize lm_head weights
                try:
                    dequantized_lm_weights = mx.dequantize(
                        old_lm_head.weight,
                        scales=old_lm_head.scales,
                        biases=old_lm_head.biases,
                        group_size=old_lm_head.group_size,
                        bits=old_lm_head.bits,
                    )
                except AttributeError as e:
                    print(f"Error: Cannot dequantize lm_head. Missing attributes: {e}")
                    print("Falling back to random weights for lm_head.")
                    dequantized_lm_weights = mx.random.normal(
                        (output_dims, input_dims), loc=0.0, scale=0.02
                    )

                new_lm_head = nn.QuantizedLinear(
                    input_dims=input_dims,
                    output_dims=new_vocab_size,
                    bias="bias" in old_lm_head,
                    group_size=old_lm_head.group_size,
                    bits=old_lm_head.bits,
                )
                new_weights_lm = mx.zeros((new_vocab_size, input_dims))
                new_weights_lm[:min_vocab_size] = dequantized_lm_weights[
                    :min_vocab_size
                ]
                if new_vocab_size > output_dims:
                    new_weights_lm[output_dims:] = mx.random.normal(
                        (new_vocab_size - output_dims, input_dims), loc=0.0, scale=0.02
                    )
                new_lm_head.weight, new_lm_head.scales, new_lm_head.biases = (
                    mx.quantize(
                        new_weights_lm, new_lm_head.group_size, new_lm_head.bits
                    )
                )
                if "bias" in old_lm_head:
                    new_lm_head.bias = mx.zeros((new_vocab_size,))
                    new_lm_head.bias[:min_vocab_size] = old_lm_head.bias[
                        :min_vocab_size
                    ]
            else:
                # resize nn.Linear
                new_lm_head = nn.Linear(
                    old_lm_head.input_dims, new_vocab_size, bias="bias" in old_lm_head
                )
                new_weights_lm = mx.zeros((new_vocab_size, old_lm_head.input_dims))
                min_vocab_size = min(old_lm_head.weight.shape[0], new_vocab_size)
                new_weights_lm[:min_vocab_size] = old_lm_head.weight[:min_vocab_size]
                if new_vocab_size > old_lm_head.weight.shape[0]:
                    new_weights_lm[old_lm_head.weight.shape[0] :] = mx.random.normal(
                        (
                            new_vocab_size - old_lm_head.weight.shape[0],
                            old_lm_head.input_dims,
                        ),
                        loc=0.0,
                        scale=0.02,
                    )
                new_lm_head.weight = new_weights_lm
                # todo typechecking
                if "bias" in old_lm_head:
                    new_lm_head.bias = mx.zeros((new_vocab_size,))
                    new_lm_head.bias[:min_vocab_size] = old_lm_head.bias[
                        :min_vocab_size
                    ]

            model.lm_head = new_lm_head
    else:
        print("Vocab already sized right.")
    return model


def update_tokenizer(
    tokenizer: TokenizerWrapper, tokens: list[str], special: bool
) -> TokenizerWrapper:
    """
    Appends new tokens to the end of the tokenizer vocab
    """
    if special:
        # todo TokenizerWrapper access method
        tokenizer._tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        print(f"Tokenizer updated with special tokens: {tokens}")
        print(f"Tokenizer vocab size after append: {len(tokenizer._tokenizer)}")
    else:
        # todo add regular tokens
        pass
    return tokenizer


def implement_new_tokens(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    tokens: list[str],
    special: bool = False,
) -> tuple[nn.Module, TokenizerWrapper]:
    """
    Update model`s tokenizer and embeddings with new tokens accordingly
    """
    tokenizer = update_tokenizer(tokenizer=tokenizer, tokens=tokens, special=special)
    model = resize_embeddings(model=model, tokenizer=tokenizer)
    return model, tokenizer