import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np


class SchedulerOutput:
    """Output class for scheduler step results."""
    def __init__(self, prev_sample: mx.array):
        self.prev_sample = prev_sample


class FlowUniPCMultistepScheduler:
    """
    MLX implementation of UniPCMultistepScheduler.
    A training-free framework designed for the fast sampling of diffusion models.
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
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
        solver_p = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: Optional[str] = "zero",
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
            'predict_x0': predict_x0,
            'solver_type': solver_type,
            'lower_order_final': lower_order_final,
            'disable_corrector': disable_corrector,
            'solver_p': solver_p,
            'timestep_spacing': timestep_spacing,
            'steps_offset': steps_offset,
            'final_sigmas_type': final_sigmas_type,
        }
        
        # Validate solver type
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.config['solver_type'] = "bh2"
            else:
                raise NotImplementedError(
                    f"{solver_type} is not implemented for {self.__class__}"
                )
        
        self.predict_x0 = predict_x0
        # setable values
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = mx.array(sigmas, dtype=mx.float32)
        
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        
        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None
        
        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])
    
    @property
    def step_index(self):
        """The index counter for current timestep."""
        return self._step_index
    
    @property
    def begin_index(self):
        """The index for the first timestep."""
        return self._begin_index
    
    def set_begin_index(self, begin_index: int = 0):
        """Sets the begin index for the scheduler."""
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
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)
        
        # add an index counter for schedulers
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
        sample: mx.array = None,
        **kwargs,
    ) -> mx.array:
        """Convert the model output to the corresponding type the UniPC algorithm needs."""
        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        
        if self.predict_x0:
            if self.config['prediction_type'] == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config['prediction_type']} must be 'flow_prediction' "
                    f"for the UniPCMultistepScheduler."
                )
            
            if self.config['thresholding']:
                x0_pred = self._threshold_sample(x0_pred)
            
            return x0_pred
        else:
            if self.config['prediction_type'] == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config['prediction_type']} must be 'flow_prediction' "
                    f"for the UniPCMultistepScheduler."
                )
            
            if self.config['thresholding']:
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred
            
            return epsilon
    
    def multistep_uni_p_bh_update(
        self,
        model_output: mx.array,
        sample: mx.array = None,
        order: int = None,
        **kwargs,
    ) -> mx.array:
        """One step for the UniP (B(h) version)."""
        model_output_list = self.model_outputs
        
        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample
        
        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t
        
        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        
        h = lambda_t - lambda_s0
        
        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = mx.log(alpha_si) - mx.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        
        rks.append(1.0)
        rks = mx.array(rks)
        
        R = []
        b = []
        
        hh = -h if self.predict_x0 else h
        h_phi_1 = mx.exp(hh) - 1  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1
        
        factorial_i = 1
        
        if self.config['solver_type'] == "bh1":
            B_h = hh
        elif self.config['solver_type'] == "bh2":
            B_h = mx.exp(hh) - 1
        else:
            raise NotImplementedError()
        
        for i in range(1, order + 1):
            R.append(mx.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        
        R = mx.stack(R)
        b = mx.array(b)
        
        if len(D1s) > 0:
            D1s = mx.stack(D1s, axis=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = mx.array([0.5], dtype=x.dtype)
            else:
                rhos_p = mx.linalg.solve(R[:-1, :-1], b[:-1], stream=mx.cpu).astype(x.dtype)
        else:
            D1s = None
        
        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = mx.sum(rhos_p[:, None, None, None] * D1s, axis=0)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = mx.sum(rhos_p[:, None, None, None] * D1s, axis=0)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res
        
        x_t = x_t.astype(x.dtype)
        return x_t
    
    def multistep_uni_c_bh_update(
        self,
        this_model_output: mx.array,
        last_sample: mx.array = None,
        this_sample: mx.array = None,
        order: int = None,
        **kwargs,
    ) -> mx.array:
        """One step for the UniC (B(h) version)."""
        model_output_list = self.model_outputs
        
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output
        
        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        
        h = lambda_t - lambda_s0
        
        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = mx.log(alpha_si) - mx.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        
        rks.append(1.0)
        rks = mx.array(rks)
        
        R = []
        b = []
        
        hh = -h if self.predict_x0 else h
        h_phi_1 = mx.exp(hh) - 1
        h_phi_k = h_phi_1 / hh - 1
        
        factorial_i = 1
        
        if self.config['solver_type'] == "bh1":
            B_h = hh
        elif self.config['solver_type'] == "bh2":
            B_h = mx.exp(hh) - 1
        else:
            raise NotImplementedError()
        
        for i in range(1, order + 1):
            R.append(mx.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        
        R = mx.stack(R)
        b = mx.array(b)
        
        if len(D1s) > 0:
            D1s = mx.stack(D1s, axis=1)
        else:
            D1s = None
        
        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = mx.array([0.5], dtype=x.dtype)
        else:
            rhos_c = mx.linalg.solve(R, b, stream=mx.cpu).astype(x.dtype)
        
        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = mx.sum(rhos_c[:-1, None, None, None] * D1s, axis=0)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = mx.sum(rhos_c[:-1, None, None, None] * D1s, axis=0)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        
        x_t = x_t.astype(x.dtype)
        return x_t
    
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        condition = schedule_timesteps == timestep
        indices = mx.argmax(condition.astype(mx.int32))
        
        # Convert scalar to int and return
        return int(indices)
    
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
        return_dict: bool = True,
        generator=None
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample from the previous timestep."""
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        if self.step_index is None:
            self._init_step_index(timestep)
        
        use_corrector = (
            self.step_index > 0 and
            self.step_index - 1 not in self.disable_corrector and
            self.last_sample is not None
        )
        
        model_output_convert = self.convert_model_output(
            model_output, sample=sample
        )
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )
        
        for i in range(self.config['solver_order'] - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep
        
        if self.config['lower_order_final']:
            this_order = min(
                self.config['solver_order'],
                len(self.timesteps) - self.step_index
            )
        else:
            this_order = self.config['solver_order']
        
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0
        
        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )
        
        if self.lower_order_nums < self.config['solver_order']:
            self.lower_order_nums += 1
        
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