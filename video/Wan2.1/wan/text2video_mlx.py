import glob
import gc
import logging
import math
import os
import random
import sys
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .modules.model_mlx import WanModel
from .modules.t5_mlx import T5EncoderModel
from .modules.vae_mlx import WanVAE
from .utils.fm_solvers_mlx import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from .utils.fm_solvers_unipc_mlx import FlowUniPCMultistepScheduler
from .wan_model_io import load_wan_from_safetensors


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
        """
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = mx.float16 if config.param_dtype == 'float16' else mx.float32

        # Initialize T5 text encoder - with automatic conversion
        t5_checkpoint_path = os.path.join(checkpoint_dir, config.t5_checkpoint)
        mlx_t5_path = t5_checkpoint_path.replace('.safetensors', '_mlx.safetensors')
        if not os.path.exists(mlx_t5_path):
            # Check if it's a .pth file that needs conversion
            pth_path = t5_checkpoint_path.replace('.safetensors', '.pth')
            if os.path.exists(pth_path):
                logging.info(f"Converting T5 PyTorch model to safetensors: {pth_path}")
                from .t5_torch_to_sf import convert_pickle_to_safetensors
                convert_pickle_to_safetensors(pth_path, t5_checkpoint_path, load_method="weights_only")
                # Convert torch safetensors to MLX safetensors
                from .t5_model_io import convert_safetensors_to_mlx_weights
                convert_safetensors_to_mlx_weights(t5_checkpoint_path, mlx_t5_path, float16=(self.param_dtype == mx.float16))
                
            else:
                raise FileNotFoundError(f"T5 checkpoint not found: {t5_checkpoint_path} or {pth_path}")

        t5_checkpoint_path = mlx_t5_path  # Use the MLX version
        logging.info(f"Loading T5 text encoder... from {t5_checkpoint_path}")
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            checkpoint_path=t5_checkpoint_path,
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer))

        # Initialize VAE
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        
        # Initialize VAE - with automatic conversion
        vae_path = os.path.join(checkpoint_dir, config.vae_checkpoint)
        if not os.path.exists(vae_path):
            # Check for PyTorch VAE file to convert
            pth_vae_path = vae_path.replace('_mlx.safetensors', '.pth')
            if not os.path.exists(pth_vae_path):
                # Try alternative naming
                pth_vae_path = os.path.join(checkpoint_dir, 'Wan2.1_VAE.pth')
            
            if os.path.exists(pth_vae_path):
                logging.info(f"Converting VAE PyTorch model to MLX: {pth_vae_path}")
                from .vae_model_io import convert_pytorch_to_mlx
                convert_pytorch_to_mlx(pth_vae_path, vae_path, float16=(self.param_dtype == mx.float16))
            else:
                raise FileNotFoundError(f"VAE checkpoint not found: {vae_path} or {pth_vae_path}")

        logging.info("Loading VAE...")
        self.vae = WanVAE(vae_pth=vae_path)

        # Initialize WanModel
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        
        # Create model with config parameters
        self.model = WanModel(
            model_type='t2v',
            patch_size=config.patch_size,
            text_len=config.text_len,
            in_dim=16,
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=4096,
            out_dim=16,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=getattr(config, 'window_size', (-1, -1)),
            qk_norm=getattr(config, 'qk_norm', True),
            cross_attn_norm=getattr(config, 'cross_attn_norm', True),
            eps=getattr(config, 'eps', 1e-6)
        )
        
        # In WanT2V.__init__, replace the model loading section with:

        # Load pretrained weights - with automatic conversion
        model_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model_mlx.safetensors")
        if not os.path.exists(model_path):
            # Check for directory with multiple files (14B model)
            pattern = os.path.join(checkpoint_dir, "diffusion_mlx_model*.safetensors")
            mlx_files = glob.glob(pattern)
            
            if not mlx_files:
                # No MLX files found, look for PyTorch files to convert
                pytorch_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
                pytorch_pattern = os.path.join(checkpoint_dir, "diffusion_pytorch_model-*.safetensors")
                pytorch_files = glob.glob(pytorch_pattern)
                
                if os.path.exists(pytorch_path):
                    logging.info(f"Converting single PyTorch model to MLX: {pytorch_path}")
                    from .wan_model_io import convert_safetensors_to_mlx_weights
                    convert_safetensors_to_mlx_weights(
                        pytorch_path, 
                        model_path,
                        float16=(self.param_dtype == mx.float16)
                    )
                elif pytorch_files:
                    logging.info(f"Converting {len(pytorch_files)} PyTorch model files to MLX")
                    from .wan_model_io import convert_multiple_safetensors_to_mlx
                    convert_multiple_safetensors_to_mlx(
                        checkpoint_dir,
                        float16=(self.param_dtype == mx.float16)
                    )
                else:
                    raise FileNotFoundError(f"No PyTorch model files found in {checkpoint_dir}")

        # Load the model (now MLX format exists)
        if os.path.exists(model_path):
            # Single file (1.3B)
            logging.info(f"Loading model weights from {model_path}")
            self.model = load_wan_from_safetensors(model_path, self.model, float16=(self.param_dtype == mx.float16))
        else:
            # Multiple files (14B)
            logging.info(f"Loading model weights from directory {checkpoint_dir}")
            self.model = load_wan_from_safetensors(checkpoint_dir, self.model, float16=(self.param_dtype == mx.float16))
        
        # Set model to eval mode
        self.model.eval()
        
        self.sp_size = 1  # No sequence parallelism in MLX version
        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tuple[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save memory

        Returns:
            mx.array:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames
                - H: Frame height (from size)
                - W: Frame width (from size)
        """
        # Preprocess
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2]
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.patch_size[1] * self.patch_size[2]) *
            target_shape[1] / self.sp_size
        ) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        
        # Set random seed
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        mx.random.seed(seed)

        # Encode text prompts
        logging.info("Encoding text prompts...")
        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])

        # Generate initial noise
        noise = [
            mx.random.normal(
                shape=target_shape,
                dtype=mx.float32
            )
        ]

        # Initialize scheduler
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(
                sampling_steps, shift=shift
            )
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")

        # Sample videos
        latents = noise
        
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        logging.info(f"Generating video with {len(timesteps)} steps...")
        
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = mx.array([t])

            # Model predictions
            noise_pred_cond = self.model(
                latent_model_input, t=timestep, **arg_c
            )[0]
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep, **arg_null
            )[0]

            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Scheduler step
            temp_x0 = sample_scheduler.step(
                mx.expand_dims(noise_pred, 0),
                t,
                mx.expand_dims(latents[0], 0),
                return_dict=False
            )[0]
            latents = [mx.squeeze(temp_x0, 0)]
            mx.eval(latents)

        x0 = latents
        
        # Decode latents to video
        logging.info("Decoding latents to video...")
        
        videos = self.vae.decode(x0)

        # Memory cleanup
        del noise, latents, sample_scheduler
        if offload_model:
            mx.eval(videos)  # Ensure computation is complete
            gc.collect()

        return videos[0]