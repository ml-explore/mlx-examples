# MLX implementation of text2video.py

import gc
import glob
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from functools import partial
from typing import Optional, Tuple, List, Dict, Any, Union

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from .wan_model_io import convert_wan_2_2_safetensors_to_mlx, convert_multiple_wan_2_2_safetensors_to_mlx, load_wan_2_2_from_safetensors

class WanT2V:
    def __init__(
        self,
        config,
        checkpoint_dir: str,
        device_id: int = 0,
        convert_model_dtype: bool = False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components for MLX.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`, *optional*, defaults to 0):
                Device id (kept for compatibility, MLX handles device automatically)
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
        """
        self.config = config
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        # Convert PyTorch dtype to MLX dtype
        if str(config.param_dtype) == 'torch.bfloat16':
            self.param_dtype = mx.bfloat16
        elif str(config.param_dtype) == 'torch.float16':
            self.param_dtype = mx.float16
        elif str(config.param_dtype) == 'torch.float32':
            self.param_dtype = mx.float32
        else:
            self.param_dtype = mx.float32  # default
        
        # Initialize T5 text encoder
        print(f"checkpoint_dir is: {checkpoint_dir}")
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
        self.vae = Wan2_1_VAE(vae_pth=vae_path)
        
        # Load low and high noise models
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        
        # Helper function to load model with automatic conversion
        def load_model_with_conversion(checkpoint_dir, subfolder, config, param_dtype):
            """Load model with automatic PyTorch to MLX conversion if needed."""
            
            # Look for existing MLX files
            mlx_single = os.path.join(checkpoint_dir, subfolder, "diffusion_pytorch_model_mlx.safetensors")
            mlx_pattern = os.path.join(checkpoint_dir, subfolder, "diffusion_mlx_model*.safetensors")
            mlx_files = glob.glob(mlx_pattern)
            
            # If no MLX files, convert PyTorch files
            if not os.path.exists(mlx_single) and not mlx_files:
                pytorch_single = os.path.join(checkpoint_dir, subfolder, "diffusion_pytorch_model.safetensors")
                pytorch_pattern = os.path.join(checkpoint_dir, subfolder, "diffusion_pytorch_model-*.safetensors")
                pytorch_files = glob.glob(pytorch_pattern)
                
                if os.path.exists(pytorch_single):
                    logging.info(f"Converting PyTorch model to MLX: {pytorch_single}")
                    convert_wan_2_2_safetensors_to_mlx(
                        pytorch_single, 
                        mlx_single,
                        float16=(param_dtype == mx.float16)
                    )
                elif pytorch_files:
                    logging.info(f"Converting {len(pytorch_files)} PyTorch files to MLX")
                    convert_multiple_wan_2_2_safetensors_to_mlx(
                        os.path.join(checkpoint_dir, subfolder),
                        float16=(param_dtype == mx.float16)
                    )
                    mlx_files = glob.glob(mlx_pattern)  # Update file list
                else:
                    raise FileNotFoundError(f"No model files found in {os.path.join(checkpoint_dir, subfolder)}")
            
            # Create model
            model = WanModel(
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
            
            # Load weights
            if os.path.exists(mlx_single):
                logging.info(f"Loading single MLX file: {mlx_single}")
                model = load_wan_2_2_from_safetensors(mlx_single, model, float16=(param_dtype == mx.float16))
            else:
                logging.info(f"Loading multiple MLX files from: {os.path.join(checkpoint_dir, subfolder)}")
                model = load_wan_2_2_from_safetensors(
                    os.path.join(checkpoint_dir, subfolder), 
                    model, 
                    float16=(param_dtype == mx.float16)
                )
            
            return model

        # Load both models
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        logging.info("Loading low noise model")
        self.low_noise_model = load_model_with_conversion(
            checkpoint_dir, 
            config.low_noise_checkpoint, 
            self.config, 
            self.param_dtype
        )
        self.low_noise_model = self._configure_model(self.low_noise_model, convert_model_dtype)

        logging.info("Loading high noise model")
        self.high_noise_model = load_model_with_conversion(
            checkpoint_dir, 
            config.high_noise_checkpoint, 
            self.config, 
            self.param_dtype
        )
        self.high_noise_model = self._configure_model(self.high_noise_model, convert_model_dtype)
        
        self.sp_size = 1  # No sequence parallel in single device
        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model: nn.Module, convert_model_dtype: bool) -> nn.Module:
        """
        Configures a model object for MLX.

        Args:
            model (nn.Module):
                The model instance to configure.
            convert_model_dtype (`bool`):
                Convert model parameters dtype to 'config.param_dtype'.

        Returns:
            nn.Module:
                The configured model.
        """
        model.eval()
        
        if convert_model_dtype:
            # In MLX, we would need to manually convert parameters
            # This would be implemented in the actual model class
            pass
        
        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        """
        Prepares and returns the required model for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        
        # MLX doesn't need the CPU offloading logic, just return the right model
        return getattr(self, required_model_name)

    def generate(
        self,
        input_prompt: str,
        size: Tuple[int, int] = (1280, 720),
        frame_num: int = 81,
        shift: float = 5.0,
        sample_solver: str = 'unipc',
        sampling_steps: int = 50,
        guide_scale: Union[float, Tuple[float, float]] = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True
    ) -> Optional[mx.array]:
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps.
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion.
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                Not used in MLX version (kept for compatibility)

        Returns:
            mx.array:
                Generated video frames tensor. Dimensions: (C, N, H, W)
        """
        # Preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        
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
        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])
        
        # Generate initial noise
        noise = [
            mx.random.normal(
                shape=target_shape,
                dtype=mx.float32
            )
        ]
        
        # Set boundary
        boundary = self.boundary * self.num_train_timesteps
        
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
            raise NotImplementedError("Unsupported solver.")
        
        # Sample videos
        latents = noise
        
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        mx.eval(latents)
        
        # Denoising loop
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = mx.array([t])
            
            # Select model based on timestep
            model = self._prepare_model_for_timestep(
                t, boundary, offload_model
            )
            sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]
            
            # Model predictions
            noise_pred_cond = model(
                latent_model_input, t=timestep, **arg_c
            )[0]
            mx.eval(noise_pred_cond)  # Force evaluation

            noise_pred_uncond = model(
                latent_model_input, t=timestep, **arg_null
            )[0]
            mx.eval(noise_pred_uncond)  # Force evaluation
            
            # Classifier-free guidance
            noise_pred = noise_pred_uncond + sample_guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            mx.eval(noise_pred)  # Force evaluation
            
            # Scheduler step
            temp_x0 = sample_scheduler.step(
                mx.expand_dims(noise_pred, axis=0),
                t,
                mx.expand_dims(latents[0], axis=0),
                return_dict=False
            )[0]
            latents = [mx.squeeze(temp_x0, axis=0)]

            mx.eval(latents)
        
        # Decode final latents
        x0 = latents
        videos = self.vae.decode(x0)
        
        # Cleanup
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
        
        return videos[0]