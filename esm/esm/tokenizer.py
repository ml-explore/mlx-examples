import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import mlx.core as mx

# Canonical amino-acid tokens (IUPAC standard + uncommon variants)
PROTEIN_TOKENS = [
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
]

ArrayLike = Union[List[int], mx.array]


class ProteinTokenizer:
    """
    Protein sequence tokenizer compatible with ESM-2.

    This class converts protein sequences into token IDs and back, following
    the vocabulary, special tokens, and formatting rules used by ESM-2.
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        special_tokens_map_file: Optional[str] = None,
    ):
        """
        Initialize the ProteinTokenizer.

        Args:
            vocab_file: Optional path to a file containing the vocabulary,
                one token per line.
            special_tokens_map_file: Optional path to a JSON file defining
                special token names and values.

        If both files are provided, they override the default vocabulary and
        special token mappings. Otherwise, defaults are loaded.
        """

        # Load vocabulary from files if given, otherwise use built-in defaults
        if vocab_file and special_tokens_map_file:
            self._load_from_files(vocab_file, special_tokens_map_file)
        else:
            self._load_default_vocab()

        # Create token ↔ ID mappings
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

        # Cache special token IDs
        self.cls_id = self.token_to_id["<cls>"]
        self.pad_id = self.token_to_id["<pad>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.mask_id = self.token_to_id["<mask>"]

        # Behavior flags for ESM-2-style BOS/EOS
        self.prepend_bos = True
        self.append_eos = True

    def _load_from_files(self, vocab_file: str, special_tokens_map_file: str) -> None:
        """Load vocabulary and special tokens from the provided files."""
        # Vocabulary file: one token per line
        vocab_path = Path(vocab_file)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = [line.strip() for line in f if line.strip()]

        # Special tokens mapping file (JSON)
        special_tokens_path = Path(special_tokens_map_file)
        with open(special_tokens_path, "r", encoding="utf-8") as f:
            self.special_tokens_map = json.load(f)

    def _load_default_vocab(self) -> None:
        """Load the built-in ESM vocabulary and special token mapping."""
        # ESM convention: prepend special tokens, then amino acids, then <mask>
        prepend_toks = ["<cls>", "<pad>", "<eos>", "<unk>"]
        append_toks = ["<mask>"]

        self.vocab = prepend_toks + PROTEIN_TOKENS

        # Pad vocab size to multiple of 8 (original implementation detail)
        while len(self.vocab) % 8 != 0:
            self.vocab.append(f"<null_{len(self.vocab) - len(prepend_toks)}>")

        self.vocab.extend(append_toks)

        # Default special tokens map
        self.special_tokens_map = {
            "cls_token": "<cls>",
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "mask_token": "<mask>",
        }

    def encode(
        self,
        sequence: str,
        *,
        add_special_tokens: bool = True,
        return_batch_dim: bool = False,
        dtype=mx.int32,
    ) -> mx.array:
        """
        Convert a protein sequence into token IDs.

        Args:
            sequence: Protein sequence (case-insensitive).
            add_special_tokens: If True, add <cls> at the start and <eos> at the end.
            return_batch_dim: If True, output shape will be (1, L) instead of (L,).
            dtype: MLX dtype for the returned array.

        Returns:
            mx.array: Token IDs of shape (L,) or (1, L).
        """
        ids: List[int] = []

        if add_special_tokens and self.prepend_bos:
            ids.append(self.cls_id)

        # Map each residue to its ID (defaulting to <unk> if not in vocab)
        for ch in sequence.upper():
            ids.append(self.token_to_id.get(ch, self.unk_id))

        if add_special_tokens and self.append_eos:
            ids.append(self.eos_id)

        arr = mx.array(ids, dtype=dtype)
        return mx.expand_dims(arr, axis=0) if return_batch_dim else arr

    def batch_encode(
        self,
        sequences: Sequence[str],
        *,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        dtype=mx.int32,
    ) -> mx.array:
        """
        Encode multiple protein sequences into a padded batch.

        Args:
            sequences: List/sequence of protein strings.
            add_special_tokens: If True, add <cls> and <eos> tokens.
            max_length: If provided, truncate sequences to this length before padding.
            dtype: MLX dtype for the returned array.

        Returns:
            mx.array: Tensor of shape (B, L) with right-padding using <pad> IDs.
        """
        # Encode each sequence as (L,)
        encoded = [
            self.encode(s, add_special_tokens=add_special_tokens, dtype=dtype)
            for s in sequences
        ]
        encoded = [e if e.ndim == 1 else e[0] for e in encoded]

        if max_length is not None:
            encoded = [e[:max_length] for e in encoded]

        # Find the longest sequence and right-pad all others
        max_len = max((int(e.shape[0]) for e in encoded), default=0)
        padded = []
        for e in encoded:
            pad_len = max_len - int(e.shape[0])
            if pad_len > 0:
                pad = mx.full((pad_len,), self.pad_id, dtype=dtype)
                e = mx.concatenate([e, pad], axis=0)
            padded.append(e)

        return mx.stack(padded, axis=0) if padded else mx.array([], dtype=dtype)

    def decode(
        self,
        token_ids: ArrayLike,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Convert token IDs back into a protein sequence string.

        Args:
            token_ids: 1-D or 2-D array/list of IDs. If 2-D, only the first row is decoded.
            skip_special_tokens: If True, remove all special tokens from output.

        Returns:
            str: Protein sequence.
        """
        # Normalize to a 1-D MLX array
        if hasattr(token_ids, "shape") and hasattr(token_ids, "tolist"):
            ids = token_ids if token_ids.ndim == 1 else token_ids[0]
        else:
            ids = mx.array(token_ids, dtype=mx.int32)

        ids_list = [int(x) for x in ids.tolist()]
        toks: List[str] = []

        for i in ids_list:
            tok = self.id_to_token.get(i, "<unk>")
            if skip_special_tokens and tok in {
                "<cls>",
                "<pad>",
                "<eos>",
                "<unk>",
                "<mask>",
            }:
                continue
            toks.append(tok)

        return "".join(toks)

    def __len__(self) -> int:
        """Return the size of the tokenizer’s vocabulary."""
        return len(self.vocab)
