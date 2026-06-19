"""Cosmos 3 Mixture-of-Transformers model for MLX.

Dual-pathway architecture: AR reasoner (text understanding) and
bidirectional diffuser (video/audio generation).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import Cosmos3Attention


@dataclass
class Cosmos3Config:
    """Configuration for Cosmos 3 Nano."""

    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    max_position_embeddings: int = 262144


class Cosmos3MLP(nn.Module):
    """GLU/SiLU gated feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Cosmos3DecoderLayer(nn.Module):
    """Single MoT decoder layer with understanding + generation pathways."""

    def __init__(self, config: Cosmos3Config):
        super().__init__()

        # Understanding pathway norms
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Dual-pathway attention
        self.self_attn = Cosmos3Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            mrope_section=config.mrope_section,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )

        # Shared MLP (understanding pathway)
        self.mlp = Cosmos3MLP(config.hidden_size, config.intermediate_size)

        # Generation pathway norms and MLP
        self.input_layernorm_moe_gen = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_moe_gen = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_moe_gen = Cosmos3MLP(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass for understanding pathway only.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [3, batch, seq_len]
            cache: optional KV cache

        Returns:
            (output, updated_cache)
        """
        # Pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention (understanding pathway only)
        attn_out, _, new_cache, _ = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            understanding_mask=None,
            generation_tokens=None,
            cache=cache,
        )

        # Residual
        hidden_states = residual + attn_out

        # Post-attention norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache

    def forward_with_generation(
        self,
        und_hidden: mx.array,
        gen_hidden: mx.array,
        position_ids: mx.array,
    ) -> Tuple[mx.array, mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass with both understanding and generation pathways.

        Args:
            und_hidden: [batch, und_len, hidden_size] understanding tokens
            gen_hidden: [batch, gen_len, hidden_size] generation tokens
            position_ids: [3, batch, und_len + gen_len]

        Returns:
            (und_output, gen_output, und_kv) where und_kv is cached (keys, values)
        """
        # Understanding pre-norm
        und_residual = und_hidden
        und_normed = self.input_layernorm(und_hidden)

        # Generation pre-norm
        gen_residual = gen_hidden
        gen_normed = self.input_layernorm_moe_gen(gen_hidden)

        # Dual-pathway attention
        und_attn, gen_attn, _, und_kv = self.self_attn(
            hidden_states=und_normed,
            position_ids=position_ids,
            understanding_mask=None,
            generation_tokens=gen_normed,
        )

        # Understanding residual + MLP
        und_hidden = und_residual + und_attn
        und_residual = und_hidden
        und_hidden = self.post_attention_layernorm(und_hidden)
        und_hidden = self.mlp(und_hidden)
        und_hidden = und_residual + und_hidden

        # Generation residual + MLP
        gen_hidden = gen_residual + gen_attn
        gen_residual = gen_hidden
        gen_hidden = self.post_attention_layernorm_moe_gen(gen_hidden)
        gen_hidden = self.mlp_moe_gen(gen_hidden)
        gen_hidden = gen_residual + gen_hidden

        return und_hidden, gen_hidden, und_kv

    def forward_generation_cached(
        self,
        gen_hidden: mx.array,
        und_kv: Tuple[mx.array, mx.array],
        position_ids: mx.array,
        und_len: int,
    ) -> mx.array:
        """Forward pass for generation pathway only, using cached understanding K/V.

        Skips the entire understanding pathway. Used for denoising steps 1+ when
        the text tokens haven't changed.
        """
        gen_residual = gen_hidden
        gen_normed = self.input_layernorm_moe_gen(gen_hidden)

        gen_attn = self.self_attn.generation_only_forward(
            gen_normed, und_kv, position_ids, und_len,
        )

        gen_hidden = gen_residual + gen_attn
        gen_residual = gen_hidden
        gen_hidden = self.post_attention_layernorm_moe_gen(gen_hidden)
        gen_hidden = self.mlp_moe_gen(gen_hidden)
        gen_hidden = gen_residual + gen_hidden

        return gen_hidden


class Cosmos3Model(nn.Module):
    """Cosmos 3 Mixture-of-Transformers model.

    Dual-pathway: AR reasoner for text understanding, bidirectional
    diffuser for video/audio generation.
    """

    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = [
            Cosmos3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Final norms (understanding + generation)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Generation pathway: latent projections
        # patch_latent_dim = latent_channel(48) * latent_patch_size(2)^2 = 192
        # These are loaded from weights; default matches Cosmos3-Nano
        self.proj_in = nn.Linear(192, config.hidden_size, bias=True)
        self.proj_out = nn.Linear(config.hidden_size, 192, bias=True)

        # Audio projections
        sound_dim = 64
        self.audio_proj_in = nn.Linear(sound_dim, config.hidden_size, bias=True)
        self.audio_proj_out = nn.Linear(config.hidden_size, sound_dim, bias=True)

        # Timestep embedder
        from .timestep import TimestepEmbedding
        self.time_embedder = TimestepEmbedding(config.hidden_size)

        # Modality embeddings
        self.audio_modality_embed = mx.zeros((config.hidden_size,))
        self.action_modality_embed = mx.zeros((config.hidden_size,))

    def diffusion_forward(
        self,
        input_ids: mx.array,
        gen_tokens: mx.array,
        timestep: mx.array,
        grid_t: int = 1,
        grid_h: int = 1,
        grid_w: int = 1,
        audio_tokens: Optional[mx.array] = None,
        noisy_frame_indexes: Optional[list[int]] = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Forward pass for diffusion generation (one denoising step).

        Runs both understanding (text) and generation (latent) pathways
        through the dual-pathway MoT transformer. Returns velocity prediction
        for the generation tokens, and optionally for audio tokens.

        Args:
            input_ids: [batch, text_len] text token IDs
            gen_tokens: [batch, num_patches, patch_latent_dim] patchified latents
            timestep: [batch] current diffusion timestep
            grid_t: temporal grid size (number of latent frames)
            grid_h: height grid size (latent height / patch_size)
            grid_w: width grid size (latent width / patch_size)
            audio_tokens: optional [batch, sound_len, sound_dim] audio latents
            noisy_frame_indexes: which temporal frames are noisy (get timestep
                embedding). None = all frames are noisy (t2v default). For i2v
                with frame 0 conditioned: [1, 2, 3, ...].

        Returns:
            If audio_tokens is None:
                [batch, num_patches, patch_latent_dim] velocity prediction
            If audio_tokens is provided:
                (vision_velocity, audio_velocity) tuple
        """
        batch = input_ids.shape[0]
        text_len = input_ids.shape[1]
        num_patches = gen_tokens.shape[1]

        # Embed text tokens
        und_h = self.embed_tokens(input_ids)

        # Project generation tokens into hidden space
        gen_h = self.proj_in(gen_tokens)

        # Add timestep embedding only to noisy frame tokens.
        # HF Cosmos3OmniTransformer._apply_timestep_embeds_to_noisy_tokens uses
        # scatter_add to selectively add timestep to noisy positions only.
        # Conditioned frames (e.g. frame 0 in i2v) get no timestep signal.
        scaled_t = timestep * 0.001
        t_emb = self.time_embedder(scaled_t)  # [batch, hidden_size]

        if noisy_frame_indexes is None:
            # All frames noisy (t2v): add timestep to everything
            gen_h = gen_h + mx.expand_dims(t_emb, 1)
        else:
            # Selective: build a mask for noisy token positions
            spatial_tokens = grid_h * grid_w
            noisy_mask = mx.zeros((num_patches,), dtype=gen_h.dtype)
            for fi in noisy_frame_indexes:
                start = fi * spatial_tokens
                end = start + spatial_tokens
                noisy_mask = noisy_mask.at[start:end].add(mx.ones((spatial_tokens,), dtype=gen_h.dtype))
            # [1, num_patches, 1] mask * [1, 1, hidden_size] timestep
            gen_h = gen_h + noisy_mask.reshape(1, num_patches, 1) * mx.expand_dims(t_emb, 1)

        # Build 3D mRoPE position IDs
        # Text tokens: all 3 axes share monotonically increasing IDs
        # HF uses get_3d_mrope_ids_text_tokens which produces float positions
        text_pos = mx.arange(text_len, dtype=mx.float32)[None, :]  # [1, text_len]
        text_position_ids = mx.stack([text_pos, text_pos, text_pos])  # [3, 1, text_len]

        # Generation tokens: FPS-modulated temporal positions
        # HF: scaled_t = frame_index / tps * base_tps + temporal_offset
        # where tps = fps / temporal_compression_factor, base_tps = base_fps / base_tcf
        temporal_margin = 15000
        temporal_offset = float(text_len + temporal_margin)

        # Video FPS modulation: fps=24, temporal_compression_factor=4, base_fps=24
        fps = 24.0
        video_tcf = 4  # temporal compression factor for video VAE
        base_fps = 24.0
        tps = fps / video_tcf  # 6.0 tokens per second
        base_tps = base_fps / video_tcf  # 6.0
        frame_indices = mx.arange(grid_t, dtype=mx.float32)
        scaled_t = frame_indices / tps * base_tps + temporal_offset
        t_idx = mx.broadcast_to(scaled_t.reshape(-1, 1), (grid_t, grid_h * grid_w)).reshape(1, -1)

        h_idx = mx.arange(grid_h, dtype=mx.float32).reshape(1, -1, 1)
        h_idx = mx.broadcast_to(h_idx, (grid_t, grid_h, grid_w)).reshape(1, -1)

        w_idx = mx.arange(grid_w, dtype=mx.float32).reshape(1, 1, -1)
        w_idx = mx.broadcast_to(w_idx, (grid_t, grid_h, grid_w)).reshape(1, -1)

        gen_position_ids = mx.stack([t_idx, h_idx, w_idx])  # [3, 1, num_patches]

        # Handle audio tokens
        if audio_tokens is not None:
            sound_len = audio_tokens.shape[1]

            # Project audio tokens + add modality embedding + timestep
            audio_h = self.audio_proj_in(audio_tokens)
            audio_h = audio_h + self.audio_modality_embed
            audio_h = audio_h + mx.expand_dims(t_emb, 1)

            # Audio mRoPE: temporal siblings with video, grid_h=1, grid_w=1
            # Audio: temporal_compression_factor=1, so tps = fps/1 = 24
            # base_tps for audio uses the audio compression factor (1), not video's (4)
            audio_tps = fps / 1.0  # 24 tokens per second
            audio_base_tps = base_fps / 1.0  # 24.0 (base_fps / audio_tcf)
            audio_frame_indices = mx.arange(sound_len, dtype=mx.float32)
            audio_scaled_t = audio_frame_indices / audio_tps * audio_base_tps + temporal_offset
            audio_t_idx = audio_scaled_t.reshape(1, -1)
            audio_h_idx = mx.zeros((1, sound_len), dtype=mx.float32)
            audio_w_idx = mx.zeros((1, sound_len), dtype=mx.float32)
            audio_position_ids = mx.stack([audio_t_idx, audio_h_idx, audio_w_idx])

            # Concatenate video + audio in generation pathway
            gen_h = mx.concatenate([gen_h, audio_h], axis=1)
            gen_position_ids = mx.concatenate([gen_position_ids, audio_position_ids], axis=2)

        # Concatenate text + generation position IDs
        position_ids = mx.concatenate([text_position_ids, gen_position_ids], axis=2)

        # Forward through all layers with both pathways
        und_kv_cache = []
        for layer in self.layers:
            und_h, gen_h, und_kv = layer.forward_with_generation(
                und_h, gen_h, position_ids
            )
            und_kv_cache.append(und_kv)

        # Apply generation final norm
        gen_h = self.norm_moe_gen(gen_h)

        # Split and project back
        if audio_tokens is not None:
            vision_h = gen_h[:, :num_patches, :]
            audio_h = gen_h[:, num_patches:, :]
            vision_velocity = self.proj_out(vision_h)
            audio_velocity = self.audio_proj_out(audio_h)
            return (vision_velocity, audio_velocity), und_kv_cache
        else:
            velocity = self.proj_out(gen_h)
            return velocity, und_kv_cache

    def diffusion_forward_cached(
        self,
        gen_tokens: mx.array,
        timestep: mx.array,
        und_kv_cache: list,
        position_ids: mx.array,
        text_len: int,
        audio_tokens: Optional[mx.array] = None,
        grid_t: int = 1,
        grid_h: int = 1,
        grid_w: int = 1,
        noisy_frame_indexes: Optional[list[int]] = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Cached diffusion forward — generation pathway only.

        Reuses precomputed understanding K/V from step 0.
        Only recomputes generation pathway (proj_in, timestep, attention, MLP, proj_out).
        """
        num_patches = gen_tokens.shape[1]

        # Project generation tokens + selective timestep embedding
        gen_h = self.proj_in(gen_tokens)
        scaled_t = timestep * 0.001
        t_emb = self.time_embedder(scaled_t)

        if noisy_frame_indexes is None:
            gen_h = gen_h + mx.expand_dims(t_emb, 1)
        else:
            spatial_tokens = grid_h * grid_w
            noisy_mask = mx.zeros((num_patches,), dtype=gen_h.dtype)
            for fi in noisy_frame_indexes:
                start = fi * spatial_tokens
                end = start + spatial_tokens
                noisy_mask = noisy_mask.at[start:end].add(mx.ones((spatial_tokens,), dtype=gen_h.dtype))
            gen_h = gen_h + noisy_mask.reshape(1, num_patches, 1) * mx.expand_dims(t_emb, 1)

        # Handle audio tokens
        if audio_tokens is not None:
            audio_h = self.audio_proj_in(audio_tokens)
            audio_h = audio_h + self.audio_modality_embed
            audio_h = audio_h + mx.expand_dims(t_emb, 1)
            gen_h = mx.concatenate([gen_h, audio_h], axis=1)

        # Forward through layers using cached understanding K/V
        for layer, und_kv in zip(self.layers, und_kv_cache):
            gen_h = layer.forward_generation_cached(
                gen_h, und_kv, position_ids, text_len,
            )

        # Apply generation final norm
        gen_h = self.norm_moe_gen(gen_h)

        # Split and project back
        if audio_tokens is not None:
            vision_h = gen_h[:, :num_patches, :]
            audio_h = gen_h[:, num_patches:, :]
            vision_velocity = self.proj_out(vision_h)
            audio_velocity = self.audio_proj_out(audio_h)
            return vision_velocity, audio_velocity
        else:
            velocity = self.proj_out(gen_h)
            return velocity
