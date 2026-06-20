"""UniPC Multistep Scheduler for Cosmos 3 diffusion generation.

Implements the unified predictor-corrector (UniPC) framework for
iterative denoising with flow matching. Matches the HuggingFace
UniPCMultistepScheduler with flow_prediction type.

Key formulas (flow matching path):
- sigma_t is the noise level (0 = clean, 1 = pure noise)
- alpha_t = 1 - sigma_t
- x_t = alpha_t * x_0 + sigma_t * noise
- Model predicts velocity v, x_0 recovered as: x_0 = x_t - sigma_t * v
- Stepping uses log-SNR (lambda) space exponential integrators

The full UniPC algorithm runs a predictor (P) step followed by a corrector (C)
step at each iteration. The corrector uses the current model output to refine
the sample predicted by the previous predictor step.
"""

from typing import Optional

import mlx.core as mx
import numpy as np


class UniPCScheduler:
    """UniPC multi-step scheduler for flow matching denoising.

    Matches HF UniPCMultistepScheduler config:
    - prediction_type: flow_prediction
    - use_flow_sigmas: True
    - predict_x0: True
    - solver_order: 2
    - solver_type: bh2
    - lower_order_final: True
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        flow_shift: float = 1.0,
        solver_order: int = 2,
        use_karras_sigmas: bool = True,
        sigma_min: float = 0.147,
        sigma_max: float = 200.0,
        lower_order_final: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.flow_shift = flow_shift
        self.solver_order = solver_order
        self.use_karras_sigmas = use_karras_sigmas
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.lower_order_final = lower_order_final

        self.sigmas = None
        self.timesteps = None
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = None
        self.step_index = 0

    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps for inference.

        Uses Karras sigma schedule (matching HF config: use_karras_sigmas=True,
        use_flow_sigmas=True) converted to flow-matching space.
        """
        self.num_inference_steps = num_inference_steps

        if self.use_karras_sigmas:
            rho = 7.0
            ramp = np.linspace(0, 1, num_inference_steps)
            min_inv_rho = self.sigma_min ** (1 / rho)
            max_inv_rho = self.sigma_max ** (1 / rho)
            karras_sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            sigmas = karras_sigmas / (karras_sigmas + 1)
        else:
            sigmas = np.linspace(1, 1 / self.num_train_timesteps, num_inference_steps + 1)[:-1]
            if self.flow_shift != 1.0:
                sigmas = self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)

        eps = 1e-6
        if abs(sigmas[0] - 1.0) < eps:
            sigmas[0] -= eps

        sigmas = np.append(sigmas, 0.0)

        self.sigmas = mx.array(sigmas.astype(np.float32))
        self.timesteps = mx.array((sigmas[:-1] * self.num_train_timesteps).astype(np.float32))

        # Reset state
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = None
        self.step_index = 0

    def _convert_model_output(self, model_output: mx.array, sample: mx.array) -> mx.array:
        """Convert flow velocity prediction to x0 prediction."""
        sigma = float(self.sigmas[self.step_index].item())
        return sample - sigma * model_output

    def _compute_rks_and_D1s(self, order: int, h: float, lambda_s0: float,
                             m0: mx.array, step_offset: int = 0):
        """Compute rk ratios and D1 differences for multistep methods."""
        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i - step_offset
            mi = self.model_outputs[-(i + 1)]
            if mi is None:
                break
            sigma_si = float(self.sigmas[si].item())
            alpha_si = 1.0 - sigma_si
            lambda_si = np.log(alpha_si / max(sigma_si, 1e-10))
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        return rks, D1s

    def _compute_bh2_coefficients(self, h: float, order: int):
        """Compute R matrix and b vector for BH2 solver."""
        hh = -h  # predict_x0 mode
        h_phi_1 = np.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = np.expm1(hh)

        b = []
        for i in range(1, order + 1):
            b.append(float(h_phi_k * factorial_i / B_h))
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        return h_phi_1, B_h, b

    def _uni_p_bh_update(self, sample: mx.array, order: int) -> mx.array:
        """UniP predictor step (B(h) version).

        Predicts x_{t+1} from x_t using stored model outputs (x0 predictions).
        """
        m0 = self.model_outputs[-1]

        sigma_t = float(self.sigmas[self.step_index + 1].item())
        sigma_s0 = float(self.sigmas[self.step_index].item())
        alpha_t = 1.0 - sigma_t
        alpha_s0 = 1.0 - sigma_s0

        if sigma_t == 0.0:
            return m0

        lambda_t = np.log(max(alpha_t, 1e-10) / max(sigma_t, 1e-10))
        lambda_s0 = np.log(alpha_s0 / max(sigma_s0, 1e-10))
        h = lambda_t - lambda_s0

        h_phi_1, B_h, b = self._compute_bh2_coefficients(h, order)

        # First-order base step
        x_t_ = (sigma_t / sigma_s0) * sample - alpha_t * h_phi_1 * m0

        if order == 1 or self.model_outputs[-2] is None:
            return x_t_

        rks, D1s = self._compute_rks_and_D1s(order, h, lambda_s0, m0)
        if not D1s:
            return x_t_

        # For order 2, HF hardcodes rho_p = 0.5
        rho_p = 0.5
        pred_res = rho_p * D1s[0]

        x_t = x_t_ - alpha_t * B_h * pred_res
        return x_t

    def _uni_c_bh_update(
        self, this_model_output: mx.array, last_sample: mx.array,
        this_sample: mx.array, order: int,
    ) -> mx.array:
        """UniC corrector step (B(h) version).

        Corrects the predictor output using the model evaluation at the predicted point.
        """
        m0 = self.model_outputs[-1]

        sigma_t = float(self.sigmas[self.step_index].item())
        sigma_s0 = float(self.sigmas[self.step_index - 1].item())
        alpha_t = 1.0 - sigma_t
        alpha_s0 = 1.0 - sigma_s0

        lambda_t = np.log(max(alpha_t, 1e-10) / max(sigma_t, 1e-10))
        lambda_s0 = np.log(alpha_s0 / max(sigma_s0, 1e-10))
        h = lambda_t - lambda_s0

        h_phi_1, B_h, b_list = self._compute_bh2_coefficients(h, order)

        # Base step using last_sample (the sample before the predictor)
        x_t_ = (sigma_t / sigma_s0) * last_sample - alpha_t * h_phi_1 * m0

        # Compute rks and D1s from history (offset by 1 for corrector indexing)
        rks, D1s = self._compute_rks_and_D1s(order, h, lambda_s0, m0, step_offset=1)

        # Build R matrix and solve for rhos_c
        # rks_full includes the historical rks plus a trailing 1.0
        rks_np = np.array([rk for rk in rks] + [1.0])

        R = np.stack([np.power(rks_np, i) for i in range(order)])  # [order, order]
        b = np.array(b_list[:order])

        if order == 1:
            rhos_c = np.array([0.5])
        else:
            rhos_c = np.linalg.solve(R, b)

        # Apply correction
        D1_t = this_model_output - m0
        if D1s:
            corr_res = sum(rhos_c[k] * D1s[k] for k in range(len(D1s)))
        else:
            corr_res = 0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
    ) -> mx.array:
        """Perform one denoising step using UniPC predictor-corrector.

        Matches HF UniPCMultistepScheduler.step() exactly:
        1. Convert model output to x0 prediction
        2. Corrector step (refine previous predictor output using current model eval)
        3. Update history
        4. Predictor step (predict next sample)
        """
        # Convert model output to x0 prediction
        model_output_x0 = self._convert_model_output(model_output, sample)

        # Corrector step: refine the current sample using the new model output
        use_corrector = (
            self.step_index > 0
            and self.last_sample is not None
        )
        if use_corrector:
            sample = self._uni_c_bh_update(
                this_model_output=model_output_x0,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # Shift history buffers
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_x0
        self.timestep_list[-1] = float(timestep.item()) if hasattr(timestep, 'item') else float(timestep)

        # Determine effective order for this step
        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order

        # Warmup: don't use higher order until we have enough history
        this_order = min(this_order, self.lower_order_nums + 1)
        assert this_order > 0
        self.this_order = this_order

        # Save sample for next corrector step
        self.last_sample = sample

        # Predictor step
        prev_sample = self._uni_p_bh_update(sample=sample, order=this_order)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        self.step_index += 1
        return prev_sample

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        """Add noise to clean samples at given timestep.

        Flow matching: X_t = (1 - sigma) * data + sigma * noise
        """
        sigma = timestep / self.num_train_timesteps
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        while sigma.ndim < original_samples.ndim:
            sigma = mx.expand_dims(sigma, -1)

        return (1 - sigma) * original_samples + sigma * noise
