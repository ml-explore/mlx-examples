import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))
    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SchedulerOutput:
    """Output class for scheduler step results."""
    def __init__(self, prev_sample: mx.array):
        self.prev_sample = prev_sample


class FlowDPMSolverMultistepScheduler:
    """
    MLX implementation of FlowDPMSolverMultistepScheduler.
    A fast dedicated high-order solver for diffusion ODEs.
    """
    
    order = 1
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        use_dynamic_shifting: bool = False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: Optional[str] = "zero",
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        invert_sigmas: bool = False,
    ):
        # Store configuration
        self.config = {
            'num_train_timesteps': num_train_timesteps,
            'solver_order': solver_order,
            'prediction_type': prediction_type,
            'shift': shift,
            'use_dynamic_shifting': use_dynamic_shifting,
            'thresholding': thresholding,
            'dynamic_thresholding_ratio': dynamic_thresholding_ratio,
            'sample_max_value': sample_max_value,
            'algorithm_type': algorithm_type,
            'solver_type': solver_type,
            'lower_order_final': lower_order_final,
            'euler_at_final': euler_at_final,
            'final_sigmas_type': final_sigmas_type,
            'lambda_min_clipped': lambda_min_clipped,
            'variance_type': variance_type,
            'invert_sigmas': invert_sigmas,
        }
        
        # Validate algorithm type
        if algorithm_type not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            if algorithm_type == "deis":
                self.config['algorithm_type'] = "dpmsolver++"
            else:
                raise NotImplementedError(f"{algorithm_type} is not implemented")
        
        # Validate solver type
        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.config['solver_type'] = "midpoint"
            else:
                raise NotImplementedError(f"{solver_type} is not implemented")
        
        # Initialize scheduling
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = mx.array(sigmas, dtype=mx.float32)
        
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        
        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        
        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])
    
    @property
    def step_index(self):
        return self._step_index
    
    @property
    def begin_index(self):
        return self._begin_index
    
    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index
    
    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, None] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        """Sets the discrete timesteps used for the diffusion chain."""
        if self.config['use_dynamic_shifting'] and mu is None:
            raise ValueError(
                "you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`"
            )
        
        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]
        
        if self.config['use_dynamic_shifting']:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config['shift']
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        
        if self.config['final_sigmas_type'] == "sigma_min":
            sigma_last = self.sigma_min
        elif self.config['final_sigmas_type'] == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config['final_sigmas_type']}"
            )
        
        timesteps = sigmas * self.config['num_train_timesteps']
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(timesteps, dtype=mx.int64)
        
        self.num_inference_steps = len(timesteps)
        
        self.model_outputs = [None] * self.config['solver_order']
        self.lower_order_nums = 0
        
        self._step_index = None
        self._begin_index = None
    
    def _threshold_sample(self, sample: mx.array) -> mx.array:
        """Dynamic thresholding method."""
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        
        # Flatten sample for quantile calculation
        sample_flat = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        
        abs_sample = mx.abs(sample_flat)
        
        # Compute quantile
        s = mx.quantile(
            abs_sample, 
            self.config['dynamic_thresholding_ratio'], 
            axis=1, 
            keepdims=True
        )
        s = mx.clip(s, 1, self.config['sample_max_value'])
        
        # Threshold and normalize
        sample_flat = mx.clip(sample_flat, -s, s) / s
        
        sample = sample_flat.reshape(batch_size, channels, *remaining_dims)
        return sample.astype(dtype)
    
    def _sigma_to_t(self, sigma):
        return sigma * self.config['num_train_timesteps']
    
    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma
    
    def time_shift(self, mu: float, sigma: float, t: mx.array):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1)**sigma)
    
    def convert_model_output(
        self,
        model_output: mx.array,
        sample: mx.array,
        **kwargs,
    ) -> mx.array:
        """Convert model output to the corresponding type the algorithm needs."""
        # DPM-Solver++ needs to solve an integral of the data prediction model
        if self.config['algorithm_type'] in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config['prediction_type'] == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config['prediction_type']} must be "
                    f"'flow_prediction' for the FlowDPMSolverMultistepScheduler."
                )
            
            if self.config['thresholding']:
                x0_pred = self._threshold_sample(x0_pred)
            
            return x0_pred
        
        # DPM-Solver needs to solve an integral of the noise prediction model
        elif self.config['algorithm_type'] in ["dpmsolver", "sde-dpmsolver"]:
            if self.config['prediction_type'] == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config['prediction_type']} must be "
                    f"'flow_prediction' for the FlowDPMSolverMultistepScheduler."
                )
            
            if self.config['thresholding']:
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred
            
            return epsilon
    
    def dpm_solver_first_order_update(
        self,
        model_output: mx.array,
        sample: mx.array,
        noise: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """One step for the first-order DPMSolver (equivalent to DDIM)."""
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s = mx.log(alpha_s) - mx.log(sigma_s)
        
        h = lambda_t - lambda_s
        
        if self.config['algorithm_type'] == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (mx.exp(-h) - 1.0)) * model_output
        elif self.config['algorithm_type'] == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (mx.exp(h) - 1.0)) * model_output
        elif self.config['algorithm_type'] == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * mx.exp(-h)) * sample +
                (alpha_t * (1 - mx.exp(-2.0 * h))) * model_output +
                sigma_t * mx.sqrt(1.0 - mx.exp(-2 * h)) * noise
            )
        elif self.config['algorithm_type'] == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample -
                2.0 * (sigma_t * (mx.exp(h) - 1.0)) * model_output +
                sigma_t * mx.sqrt(mx.exp(2 * h) - 1.0) * noise
            )
        
        return x_t
    
    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[mx.array],
        sample: mx.array,
        noise: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """One step for the second-order multistep DPMSolver."""
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        lambda_s1 = mx.log(alpha_s1) - mx.log(sigma_s1)
        
        m0, m1 = model_output_list[-1], model_output_list[-2]
        
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        
        if self.config['algorithm_type'] == "dpmsolver++":
            if self.config['solver_type'] == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample -
                    (alpha_t * (mx.exp(-h) - 1.0)) * D0 -
                    0.5 * (alpha_t * (mx.exp(-h) - 1.0)) * D1
                )
            elif self.config['solver_type'] == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample -
                    (alpha_t * (mx.exp(-h) - 1.0)) * D0 +
                    (alpha_t * ((mx.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.config['algorithm_type'] == "dpmsolver":
            if self.config['solver_type'] == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample -
                    (sigma_t * (mx.exp(h) - 1.0)) * D0 -
                    0.5 * (sigma_t * (mx.exp(h) - 1.0)) * D1
                )
            elif self.config['solver_type'] == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample -
                    (sigma_t * (mx.exp(h) - 1.0)) * D0 -
                    (sigma_t * ((mx.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.config['algorithm_type'] == "sde-dpmsolver++":
            assert noise is not None
            if self.config['solver_type'] == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * mx.exp(-h)) * sample +
                    (alpha_t * (1 - mx.exp(-2.0 * h))) * D0 +
                    0.5 * (alpha_t * (1 - mx.exp(-2.0 * h))) * D1 +
                    sigma_t * mx.sqrt(1.0 - mx.exp(-2 * h)) * noise
                )
            elif self.config['solver_type'] == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * mx.exp(-h)) * sample +
                    (alpha_t * (1 - mx.exp(-2.0 * h))) * D0 +
                    (alpha_t * ((1.0 - mx.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1 +
                    sigma_t * mx.sqrt(1.0 - mx.exp(-2 * h)) * noise
                )
        elif self.config['algorithm_type'] == "sde-dpmsolver":
            assert noise is not None
            if self.config['solver_type'] == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample -
                    2.0 * (sigma_t * (mx.exp(h) - 1.0)) * D0 -
                    (sigma_t * (mx.exp(h) - 1.0)) * D1 +
                    sigma_t * mx.sqrt(mx.exp(2 * h) - 1.0) * noise
                )
            elif self.config['solver_type'] == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample -
                    2.0 * (sigma_t * (mx.exp(h) - 1.0)) * D0 -
                    2.0 * (sigma_t * ((mx.exp(h) - 1.0) / h - 1.0)) * D1 +
                    sigma_t * mx.sqrt(mx.exp(2 * h) - 1.0) * noise
                )
        
        return x_t
    
    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[mx.array],
        sample: mx.array,
        **kwargs,
    ) -> mx.array:
        """One step for the third-order multistep DPMSolver."""
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
            self.sigmas[self.step_index - 2],
        )
        
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)
        
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        lambda_s1 = mx.log(alpha_s1) - mx.log(sigma_s1)
        lambda_s2 = mx.log(alpha_s2) - mx.log(sigma_s2)
        
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        
        if self.config['algorithm_type'] == "dpmsolver++":
            x_t = (
                (sigma_t / sigma_s0) * sample -
                (alpha_t * (mx.exp(-h) - 1.0)) * D0 +
                (alpha_t * ((mx.exp(-h) - 1.0) / h + 1.0)) * D1 -
                (alpha_t * ((mx.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.config['algorithm_type'] == "dpmsolver":
            x_t = (
                (alpha_t / alpha_s0) * sample -
                (sigma_t * (mx.exp(h) - 1.0)) * D0 -
                (sigma_t * ((mx.exp(h) - 1.0) / h - 1.0)) * D1 -
                (sigma_t * ((mx.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            )
        
        return x_t
    
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        
        indices = mx.where(schedule_timesteps == timestep)[0]
        pos = 1 if len(indices) > 1 else 0
        
        return int(indices[pos])
    
    def _init_step_index(self, timestep):
        """Initialize the step_index counter for the scheduler."""
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index
    
    def step(
        self,
        model_output: mx.array,
        timestep: Union[int, mx.array],
        sample: mx.array,
        generator=None,
        variance_noise: Optional[mx.array] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample from the previous timestep."""
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        if self.step_index is None:
            self._init_step_index(timestep)
        
        # Improve numerical stability for small number of steps
        lower_order_final = (
            (self.step_index == len(self.timesteps) - 1) and
            (self.config['euler_at_final'] or
             (self.config['lower_order_final'] and len(self.timesteps) < 15) or
             self.config['final_sigmas_type'] == "zero")
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and
            self.config['lower_order_final'] and
            len(self.timesteps) < 15
        )
        
        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config['solver_order'] - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output
        
        # Upcast to avoid precision issues
        sample = sample.astype(mx.float32)
        
        # Generate noise if needed for SDE variants
        if self.config['algorithm_type'] in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
            noise = mx.random.normal(model_output.shape, dtype=mx.float32)
        elif self.config['algorithm_type'] in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.astype(mx.float32)
        else:
            noise = None
        
        if self.config['solver_order'] == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=sample, noise=noise
            )
        elif self.config['solver_order'] == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=sample, noise=noise
            )
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, sample=sample
            )
        
        if self.lower_order_nums < self.config['solver_order']:
            self.lower_order_nums += 1
        
        # Cast sample back to expected dtype
        prev_sample = prev_sample.astype(model_output.dtype)
        
        # Increase step index
        self._step_index += 1
        
        if not return_dict:
            return (prev_sample,)
        
        return SchedulerOutput(prev_sample=prev_sample)
    
    def scale_model_input(self, sample: mx.array, *args, **kwargs) -> mx.array:
        """Scale model input - no scaling needed for this scheduler."""
        return sample
    
    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """Add noise to original samples."""
        sigmas = self.sigmas.astype(original_samples.dtype)
        schedule_timesteps = self.timesteps
        
        # Get step indices
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps)
                for t in timesteps
            ]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]
        
        sigma = sigmas[step_indices]
        while len(sigma.shape) < len(original_samples.shape):
            sigma = mx.expand_dims(sigma, -1)
        
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples
    
    def __len__(self):
        return self.config['num_train_timesteps']