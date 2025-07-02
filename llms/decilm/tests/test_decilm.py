"""Tests for DeciLM implementation."""

import unittest
import mlx.core as mx
import mlx.nn as nn

import sys
sys.path.append('..')

from decilm import DeciLMArgs, DummyAttention, DummyFFN, VariableAttention, DeciLMBlock


class TestDeciLMComponents(unittest.TestCase):
    """Test DeciLM model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.args = DeciLMArgs(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=11008,
        )
        
    def test_dummy_attention(self):
        """Test dummy attention passthrough."""
        dummy_attn = DummyAttention(self.args)
        
        # Test input
        x = mx.random.normal((2, 10, 4096))
        
        # Should return input unchanged
        output = dummy_attn(x)
        self.assertTrue(mx.array_equal(output, x))
        
    def test_dummy_ffn(self):
        """Test dummy FFN passthrough."""
        dummy_ffn = DummyFFN(self.args)
        
        # Test input
        x = mx.random.normal((2, 10, 4096))
        
        # Should return input unchanged
        output = dummy_ffn(x)
        self.assertTrue(mx.array_equal(output, x))
        
    def test_variable_attention(self):
        """Test variable attention with different KV heads."""
        # Test with 4 KV heads (less than Q heads)
        var_attn = VariableAttention(self.args, n_kv_heads=4)
        
        x = mx.random.normal((2, 10, 4096))
        output = var_attn(x)
        
        # Output shape should match input
        self.assertEqual(output.shape, (2, 10, 4096))
        
    def test_decilm_block_dummy(self):
        """Test DeciLM block with dummy components."""
        # Config with dummy attention and FFN
        block_config = {
            "attention": {"no_op": True},
            "ffn": {"no_op": True}
        }
        
        block = DeciLMBlock(self.args, block_config)
        x = mx.random.normal((2, 10, 4096))
        
        output = block(x)
        
        # With both dummy, output should be close to input
        # (only layer norms applied)
        self.assertEqual(output.shape, x.shape)
        
    def test_decilm_block_mixed(self):
        """Test DeciLM block with mixed dummy/active components."""
        # Config with active attention but dummy FFN
        block_config = {
            "attention": {"no_op": False, "n_heads_in_group": 8},
            "ffn": {"no_op": True}
        }
        
        block = DeciLMBlock(self.args, block_config)
        x = mx.random.normal((2, 10, 4096))
        
        output = block(x)
        self.assertEqual(output.shape, x.shape)
        
    def test_block_config_variations(self):
        """Test various block configurations."""
        configs = [
            # Standard block
            {
                "attention": {"no_op": False, "n_heads_in_group": 8},
                "ffn": {"no_op": False, "ffn_mult": 2.5}
            },
            # Variable FFN multiplier
            {
                "attention": {"no_op": False, "n_heads_in_group": 8},
                "ffn": {"no_op": False, "ffn_mult": 1.5}
            },
            # Different KV heads
            {
                "attention": {"no_op": False, "n_heads_in_group": 4},
                "ffn": {"no_op": False, "ffn_mult": 2.5}
            },
        ]
        
        x = mx.random.normal((1, 5, 4096))
        
        for config in configs:
            block = DeciLMBlock(self.args, config)
            output = block(x)
            self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()