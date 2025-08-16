import unittest

import numpy as np
from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM

from esm import ESM2

# Paths for MLX and Hugging Face versions of ESM-2
MLX_PATH = "checkpoints/mlx-esm2_t12_35M_UR50D"
HF_PATH = "facebook/esm2_t12_35M_UR50D"


def load_mlx_model():
    """Load MLX ESM-2 model and tokenizer."""
    tokenizer, model = ESM2.from_pretrained(MLX_PATH)
    return tokenizer, model


def load_hf_model():
    """Load Hugging Face ESM-2 model and tokenizer with hidden states + attentions."""
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    config = EsmConfig.from_pretrained(
        HF_PATH, output_hidden_states=True, output_attentions=True
    )
    model = EsmForMaskedLM.from_pretrained(HF_PATH, config=config)
    return tokenizer, model


class TestESM2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load both MLX and HF models/tokenizers once for all tests
        cls.mlx_tokenizer, cls.mlx_model = load_mlx_model()
        cls.hf_tokenizer, cls.hf_model = load_hf_model()

    def test_tokenizer(self):
        """Verify MLX tokenizer matches Hugging Face tokenizer behavior."""
        self.assertEqual(len(self.mlx_tokenizer), len(self.hf_tokenizer))

        sequences = [
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        ]

        # Compare batched tokenization (padded sequences)
        mlx_batch = self.mlx_tokenizer.batch_encode(sequences)
        hf_batch = (
            self.hf_tokenizer(sequences, return_tensors="pt", padding=True)["input_ids"]
            .cpu()
            .numpy()
        )
        self.assertEqual(tuple(mlx_batch.shape), tuple(hf_batch.shape))
        self.assertTrue(
            np.array_equal(np.array(mlx_batch.tolist(), dtype=hf_batch.dtype), hf_batch)
        )

        # Compare single-sequence encode/decode
        for sequence in sequences:
            mlx_tokens = self.mlx_tokenizer.encode(sequence)
            hf_tokens = (
                self.hf_tokenizer(sequence, return_tensors="pt")["input_ids"]
                .cpu()
                .numpy()
                .tolist()[0]
            )
            self.assertTrue(np.array_equal(mlx_tokens, hf_tokens))
            self.assertEqual(
                self.mlx_tokenizer.decode(mlx_tokens),
                self.hf_tokenizer.decode(hf_tokens).replace(" ", ""),
            )
            self.assertEqual(
                self.mlx_tokenizer.decode(mlx_tokens, skip_special_tokens=True),
                self.hf_tokenizer.decode(hf_tokens, skip_special_tokens=True).replace(
                    " ", ""
                ),
            )

    def test_model(self):
        """Verify MLX and HF model outputs match (logits, hidden states, attentions)."""
        sequences = [
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        ]
        for sequence in sequences:
            # Tokenize
            mlx_tokens = self.mlx_tokenizer.encode(sequence, return_batch_dim=True)
            hf_tokens = self.hf_tokenizer(sequence, return_tensors="pt")["input_ids"]

            # Forward pass
            mlx_outputs = self.mlx_model(
                mlx_tokens,
                repr_layers=[self.mlx_model.num_layers],
                need_head_weights=True,
            )
            hf_outputs = self.hf_model(input_ids=hf_tokens)

            # Compare logits
            mlx_logits = np.array(mlx_outputs["logits"])
            hf_logits = hf_outputs["logits"].detach().cpu().numpy()
            self.assertTrue(np.allclose(mlx_logits, hf_logits, rtol=1e-4, atol=1e-4))

            # Compare final-layer hidden states
            final_layer = self.mlx_model.num_layers
            mlx_hidden_states = np.array(mlx_outputs["representations"][final_layer])
            hf_hidden_states = hf_outputs["hidden_states"][-1].detach().cpu().numpy()
            self.assertTrue(
                np.allclose(mlx_hidden_states, hf_hidden_states, rtol=1e-4, atol=1e-4)
            )

            # Compare attentions for final layer
            mlx_attentions = np.array(
                mlx_outputs["attentions"][:, final_layer - 1, :, :, :]
            )
            hf_attentions = hf_outputs["attentions"][-1].detach().cpu().numpy()
            self.assertTrue(
                np.allclose(mlx_attentions, hf_attentions, rtol=1e-4, atol=1e-4)
            )


if __name__ == "__main__":
    unittest.main()
