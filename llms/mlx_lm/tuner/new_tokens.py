import mlx.nn as nn
import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper


def resize_embeddings(model: nn.Module, tokenizer: TokenizerWrapper) -> nn.Module:
    """
    Resizes model embeddings to accommodate new tokens, minimizing dequantization.
    """
    old_embedding = model.model.embed_tokens
    old_vocab_size = old_embedding.num_embeddings
    new_vocab_size = len(tokenizer._tokenizer)

    if old_vocab_size == new_vocab_size:
        print("Vocab already sized right.")
        return model

    if new_vocab_size < old_vocab_size:
        print("Warning: New vocab size is smaller than original. Proceeding with trim.")

    if (
        hasattr(old_embedding, "weight")
        and hasattr(old_embedding, "scales")
        and hasattr(old_embedding, "biases")
        and hasattr(old_embedding, "group_size")
        and hasattr(old_embedding, "bits")
    ):
        # quantized embedding case:  minimize dequantization

        new_embedding = nn.QuantizedEmbedding(
            new_vocab_size,
            old_embedding.dims,
            group_size=old_embedding.group_size,
            bits=old_embedding.bits,
        )
        if new_vocab_size > old_vocab_size:
            # Add new rows
            new_row_count = new_vocab_size - old_vocab_size
            new_rows = mx.random.normal((new_row_count, old_embedding.dims), scale=0.02)
            new_rows_q, new_rows_scales, new_rows_biases = mx.quantize(
                new_rows, old_embedding.group_size, old_embedding.bits
            )

            new_embedding.weight = mx.concatenate(
                [old_embedding.weight, new_rows_q], axis=0
            )
            new_embedding.scales = mx.concatenate(
                [old_embedding.scales, new_rows_scales], axis=0
            )
            new_embedding.biases = mx.concatenate(
                [old_embedding.biases, new_rows_biases], axis=0
            )

        else:  # new_vocab_size < old_vocab_size: Slice existing
            new_embedding.weight = old_embedding.weight[:new_vocab_size]
            new_embedding.scales = old_embedding.scales[:new_vocab_size]
            new_embedding.biases = old_embedding.biases[:new_vocab_size]

    else:
        # non-quantized embedding case (fallback, less efficient)
        # dequantize ONLY if necessary
        # should ideally be avoided entirely for quantized models.
        try:
            dequantized_weights = mx.dequantize(
                old_embedding.weight,
                scales=old_embedding.scales,
                biases=old_embedding.biases,
                group_size=old_embedding.group_size,
                bits=old_embedding.bits,
            )
        # handle missing quantization attributes
        except (AttributeError, TypeError):
            print("Falling back to random weights for embed_tokens.")
            dequantized_weights = mx.random.normal(
                (old_vocab_size, old_embedding.dims), loc=0.0, scale=0.02
            )

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

    # handle lm_head
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        if hasattr(new_embedding, "weight") and not isinstance(
            new_embedding, nn.QuantizedEmbedding
        ):
            model.model.embed_tokens.weight = new_embedding.weight

    elif hasattr(model, "lm_head"):
        old_lm_head = model.lm_head
        if isinstance(old_lm_head, nn.QuantizedLinear):
            output_dims, compressed_input_dims = old_lm_head.weight.shape
            bits = old_lm_head.bits
            input_dims = compressed_input_dims * (32 // bits)
            group_size = old_lm_head.group_size

            new_lm_head = nn.QuantizedLinear(
                input_dims=input_dims,
                output_dims=new_vocab_size,
                bias="bias" in old_lm_head,
                group_size=group_size,
                bits=bits,
            )

            if new_vocab_size > old_vocab_size:
                new_row_count = new_vocab_size - old_vocab_size
                new_rows = mx.random.normal((new_row_count, input_dims), scale=0.02)
                new_rows_q, new_rows_scales, new_rows_biases = mx.quantize(
                    new_rows, group_size, bits
                )
                new_lm_head.weight = mx.concatenate(
                    [old_lm_head.weight, new_rows_q], axis=0
                )
                new_lm_head.scales = mx.concatenate(
                    [old_lm_head.scales, new_rows_scales], axis=0
                )
                new_lm_head.biases = mx.concatenate(
                    [old_lm_head.biases, new_rows_biases], axis=0
                )
            else:
                new_lm_head.weight = old_lm_head.weight[:new_vocab_size]
                new_lm_head.scales = old_lm_head.scales[:new_vocab_size]
                new_lm_head.biases = old_lm_head.biases[:new_vocab_size]

            if "bias" in old_lm_head:
                if new_vocab_size > old_vocab_size:
                    new_bias = mx.concatenate(
                        [old_lm_head.bias, mx.zeros(new_vocab_size - old_vocab_size)]
                    )
                else:
                    new_bias = old_lm_head.bias[:new_vocab_size]
                new_lm_head.bias = new_bias
        # nn.Linear case
        else:
            new_lm_head = nn.Linear(
                old_lm_head.input_dims, new_vocab_size, bias="bias" in old_lm_head
            )
            new_weights_lm = mx.zeros((new_vocab_size, old_lm_head.input_dims))
            min_vocab_size = min(old_vocab_size, new_vocab_size)
            new_weights_lm[:min_vocab_size] = old_lm_head.weight[:min_vocab_size]
            if new_vocab_size > old_vocab_size:
                new_weights_lm[old_vocab_size:] = mx.random.normal(
                    (new_vocab_size - old_vocab_size, old_lm_head.input_dims),
                    loc=0.0,
                    scale=0.02,
                )
            new_lm_head.weight = new_weights_lm
            if "bias" in old_lm_head:
                new_lm_head.bias = mx.zeros((new_vocab_size,))
                new_lm_head.bias[:min_vocab_size] = old_lm_head.bias[:min_vocab_size]

        model.lm_head = new_lm_head

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
