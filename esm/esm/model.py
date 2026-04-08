import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .modules import ContactPredictionHead, RobertaLMHead, TransformerLayer
from .tokenizer import ProteinTokenizer


class ESM2(nn.Module):
    """
    ESM-2 protein language model in MLX.

    Args:
        num_layers (int): Number of transformer layers.
        embed_dim (int): Embedding dimension.
        attention_heads (int): Number of attention heads.
        tokenizer (Optional[ProteinTokenizer]): Tokenizer to use (created if None).
        token_dropout (bool): Apply token-dropout masking behavior.
    """

    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        tokenizer: Optional[ProteinTokenizer] = None,
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads

        # Initialize tokenizer
        if tokenizer is None:
            tokenizer = ProteinTokenizer()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)

        # Special token IDs / config
        self.padding_idx = tokenizer.pad_id
        self.mask_idx = tokenizer.mask_id
        self.cls_idx = tokenizer.cls_id
        self.eos_idx = tokenizer.eos_id
        self.prepend_bos = tokenizer.prepend_bos
        self.append_eos = tokenizer.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self) -> None:
        """Initialize embeddings, transformer stack, and output heads."""
        self.embed_scale = 1

        # Token embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.embed_dim)

        # Transformer layers (register each layer so MLX tracks parameters)
        self._layer_indices = list(range(self.num_layers))
        for i in self._layer_indices:
            layer = TransformerLayer(
                self.embed_dim,
                4 * self.embed_dim,  # FFN dimension = 4×embed_dim
                self.attention_heads,
            )
            setattr(self, f"layer_{i}", layer)

        # Contact prediction head (uses all layers × heads attentions)
        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

        # Final norm + LM head (tied weights)
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.vocab_size,
            weight=self.embed_tokens.weight,
        )

    def __call__(
        self,
        tokens: mx.array,
        repr_layers: List[int] = [],
        need_head_weights: bool = False,
        return_contacts: bool = False,
    ) -> Dict[str, mx.array]:
        """
        Forward pass through ESM-2.

        Args:
            tokens: Tensor of token IDs with shape (B, T).
            repr_layers: Layers to return hidden states from (0..num_layers).
            need_head_weights: If True, return attention weights.
            return_contacts: If True, compute residue-residue contact probabilities.

        Returns:
            dict with:
                logits: (B, T, V)
                representations: {layer_idx: (B, T, E)}
                attentions: (B, L, H, T, T) if requested
                contacts: (B, T', T') if requested
        """
        if return_contacts:
            need_head_weights = True

        # Ensure tokens is 2D (B, T)
        if tokens.ndim == 1:
            tokens = mx.expand_dims(tokens, axis=0)
        assert tokens.ndim == 2

        # Padding mask (B, T)
        padding_mask = mx.equal(tokens, self.padding_idx)

        # Embed tokens (B, T, E)
        x = self.embed_scale * self.embed_tokens(tokens)

        # Token dropout: zero masked tokens + rescale based on observed mask ratio
        if self.token_dropout:
            # x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            mask_positions = mx.equal(tokens, self.mask_idx)
            x = mx.where(mask_positions[:, :, None], 0.0, x)
            
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = mx.sum(~padding_mask, axis=-1)  # Shape: (B,)
            mask_ratio_observed = mx.sum(mask_positions, axis=-1).astype(x.dtype) / src_lengths  # Shape: (B,)
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        # Zero out padding positions
        if padding_mask.any():
            x = x * (1 - padding_mask[:, :, None].astype(x.dtype))

        # Track requested representations
        repr_layers = set(repr_layers)
        hidden_representations: Dict[int, mx.array] = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights: List[mx.array] = []

        # (B, T, E) -> (T, B, E) for transformer layers
        x = mx.swapaxes(x, 0, 1)

        # If no padding anywhere, skip the mask
        if not padding_mask.any():
            padding_mask = None

        # Transformer stack
        for layer_idx in self._layer_indices:
            layer = getattr(self, f"layer_{layer_idx}")
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )

            # Save hidden representation if requested (store back as (B, T, E))
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = mx.swapaxes(x, 0, 1)

            # Save per-layer attentions if requested (H, B, T, T) -> (B, H, T, T)
            if need_head_weights:
                attn_weights.append(mx.swapaxes(attn, 0, 1))

        # Final layer norm, back to (B, T, E)
        x = self.emb_layer_norm_after(x)
        x = mx.swapaxes(x, 0, 1)

        # Save final hidden if requested
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        # Language modeling logits
        x = self.lm_head(x)

        # Build result dict
        result: Dict[str, mx.array] = {
            "logits": x,
            "representations": hidden_representations,
        }

        # Collect attentions and optional contacts
        if need_head_weights:
            # Stack layers -> (B, L, H, T, T)
            attentions = mx.stack(attn_weights, axis=1)

            # Mask out padded positions if present
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.astype(attentions.dtype)
                attention_mask = mx.expand_dims(attention_mask, 1) * mx.expand_dims(
                    attention_mask, 2
                )
                attentions = attentions * attention_mask[:, None, None, :, :]

            result["attentions"] = attentions

        # Compute contacts if requested
        if return_contacts:
            contacts = self.contact_head(tokens, attentions)
            result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens: mx.array) -> mx.array:
        """
        Predict residue-residue contacts.

        Args:
            tokens: Tensor of shape (B, T).

        Returns:
            mx.array: Contact probabilities of shape (B, T', T').
        """
        return self(tokens, return_contacts=True)["contacts"]

    def extract_features(
        self,
        tokens: mx.array,
        repr_layers: Optional[List[int]] = None,
        return_all_hiddens: bool = False,
    ) -> Dict[int, mx.array]:
        """
        Extract hidden representations from selected layers.

        Args:
            tokens: Tensor of shape (B, T).
            repr_layers: Layer indices to return (default: last layer).
            return_all_hiddens: If True, return all layers (0..num_layers).

        Returns:
            dict: {layer_idx: (B, T, E)} for requested layers.
        """
        if return_all_hiddens:
            repr_layers = list(range(self.num_layers + 1))
        elif repr_layers is None:
            repr_layers = [self.num_layers]

        result = self(tokens, repr_layers=repr_layers)
        return result["representations"]

    def get_sequence_representations(
        self,
        tokens: mx.array,
        layer: int = -1,
    ) -> mx.array:
        """
        Average token representations into a per-sequence embedding.

        Args:
            tokens: Tensor of shape (B, T).
            layer: Layer index to use (-1 or num_layers for last).

        Returns:
            mx.array: Sequence embeddings of shape (B, E).
        """
        if layer == -1:
            layer = self.num_layers

        representations = self.extract_features(tokens, repr_layers=[layer])
        repr = representations[layer]

        # Mask: non-padding and not CLS; optionally not EOS
        mask = mx.logical_and(
            mx.not_equal(tokens, self.padding_idx),
            mx.not_equal(tokens, self.cls_idx),
        )
        if self.append_eos:
            mask = mx.logical_and(mask, mx.not_equal(tokens, self.eos_idx))

        # Mean over valid positions
        mask = mask[:, :, None].astype(repr.dtype)
        masked_repr = repr * mask
        seq_lens = mx.sum(mask, axis=1, keepdims=True)
        seq_repr = mx.sum(masked_repr, axis=1) / mx.maximum(seq_lens[:, :, 0], 1.0)

        return seq_repr

    @classmethod
    def from_pretrained(cls, model_path: str) -> Tuple[ProteinTokenizer, "ESM2"]:
        """
        Load model weights and config from a directory.

        Expects:
            - config.json
            - model.safetensors
            - vocab.txt (optional, will use default if not found)
            - special_tokens_map.json (optional, will use default if not found)

        Args:
            model_path: Path to directory with weights and config.

        Returns:
            (tokenizer, model): Initialized tokenizer and ESM2 model.
        """
        model_dir = Path(model_path)
        config_path = model_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check for vocab and special tokens files
        vocab_path = model_dir / "vocab.txt"
        special_tokens_path = model_dir / "special_tokens_map.json"

        if vocab_path.exists() and special_tokens_path.exists():
            tokenizer = ProteinTokenizer(
                vocab_file=str(vocab_path),
                special_tokens_map_file=str(special_tokens_path),
            )
        else:
            tokenizer = ProteinTokenizer()

        model = cls(
            num_layers=config["num_hidden_layers"],
            embed_dim=config["hidden_size"],
            attention_heads=config["num_attention_heads"],
            tokenizer=tokenizer,
            token_dropout=config["token_dropout"],
        )

        # Load safetensors as nested dict and update model params
        weights_path = model_dir / "model.safetensors"
        flat_weights = mx.load(str(weights_path))
        nested_weights: Dict[str, dict] = {}
        for key, value in flat_weights.items():
            parts = key.split(".")
            cur = nested_weights
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value

        model.update(nested_weights)
        return tokenizer, model
