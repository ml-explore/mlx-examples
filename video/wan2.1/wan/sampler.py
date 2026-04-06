# Copyright © 2026 Apple Inc.

# Ported from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_unipc_multistep.py
# Converted for flow matching.

"""
FlowUniPCMultistepScheduler for Wan2.1 denoising.

UniPC multistep solver adapted for flow matching prediction.
"""

from functools import partial
from typing import List, Optional

import mlx.core as mx


def _lambda64(alpha: mx.array, sigma: mx.array) -> mx.array:
    # log(alpha/sigma) needs float64 for numerical stability; Metal GPU doesn't support float64.
    with mx.stream(mx.cpu):
        result = mx.log(alpha.astype(mx.float64)) - mx.log(sigma.astype(mx.float64))
        return result.astype(mx.float32)


class FlowUniPCMultistepScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: Optional[List[int]] = None,
        final_sigmas_type: str = "zero",
    ):
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                solver_type = "bh2"
            else:
                raise NotImplementedError(f"{solver_type} not implemented")

        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.shift = shift
        self.predict_x0 = predict_x0
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = (
            disable_corrector if disable_corrector is not None else []
        )
        self.final_sigmas_type = final_sigmas_type

        sigmas = (
            1.0 - mx.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1]
        )
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigmas = sigmas.astype(mx.float32)

        self.sigma_min = float(sigmas[-1].item())
        self.sigma_max = float(sigmas[0].item())

        self.sigmas = None
        self.timesteps = None
        self.num_inference_steps = None
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None

    @property
    def step_index(self):
        return self._step_index

    def set_timesteps(self, num_inference_steps, shift=None):
        sigmas = mx.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)[
            :-1
        ]
        if shift is None:
            shift = self.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        sigma_last = 0
        timesteps = sigmas * self.num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.array([sigma_last])]).astype(mx.float32)

        self.sigmas = sigmas
        self.timesteps = timesteps.astype(mx.int32)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def convert_model_output(self, model_output, sample):
        sigma_t = self.sigmas[self.step_index]
        if self.predict_x0:
            return sample - sigma_t * model_output
        else:
            return sample - (1 - sigma_t) * model_output

    def multistep_uni_p_bh_update(self, model_output, sample, order):
        """Predictor step of the UniPC multistep solver.

        Key variables:
            rks: Ratios of lambda differences between past and current steps
            D1s: First-order finite differences of model outputs
            R, b: Linear system for polynomial coefficient computation
            h_phi_k: Exponential integrator phi functions
            B_h: Scale factor -- expm1(h) for bh2 solver type
            rhos_p: Polynomial coefficients from solving R*rhos = b
        """
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = _lambda64(alpha_t, sigma_t)
        lambda_s0 = _lambda64(alpha_s0, sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = _lambda64(alpha_si, sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(mx.array(1.0, dtype=mx.float32))
        rks = mx.stack(rks)

        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = mx.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        else:
            B_h = mx.expm1(hh)

        for i in range(1, order + 1):
            R.append(mx.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = mx.stack(R)
        b = mx.stack(b)

        if len(D1s) > 0:
            D1s = mx.stack(D1s, axis=1)
            if order == 2:
                rhos_p = mx.array([0.5], dtype=x.dtype)
            else:
                # Run on CPU for numerical stability (float64 not supported on Metal GPU),
                # matching the reference implementation.
                with mx.stream(mx.cpu):
                    rhos_p = mx.linalg.solve(R[:-1, :-1], b[:-1]).astype(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = mx.sum(
                    rhos_p.reshape(-1, *([1] * (D1s.ndim - 1))) * D1s, axis=1
                )
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = mx.sum(
                    rhos_p.reshape(-1, *([1] * (D1s.ndim - 1))) * D1s, axis=1
                )
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        return x_t.astype(x.dtype)

    def multistep_uni_c_bh_update(
        self, this_model_output, last_sample, this_sample, order
    ):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = _lambda64(alpha_t, sigma_t)
        lambda_s0 = _lambda64(alpha_s0, sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = _lambda64(alpha_si, sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(mx.array(1.0, dtype=mx.float32))
        rks = mx.stack(rks)

        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = mx.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        else:
            B_h = mx.expm1(hh)

        for i in range(1, order + 1):
            R.append(mx.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = mx.stack(R)
        b = mx.stack(b)

        if len(D1s) > 0:
            D1s = mx.stack(D1s, axis=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = mx.array([0.5], dtype=x.dtype)
        else:
            # Run on CPU for numerical stability (float64 not supported on Metal GPU),
            # matching the reference implementation.
            with mx.stream(mx.cpu):
                rhos_c = mx.linalg.solve(R, b).astype(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = mx.sum(
                    rhos_c[:-1].reshape(-1, *([1] * (D1s.ndim - 1))) * D1s, axis=1
                )
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = mx.sum(
                    rhos_c[:-1].reshape(-1, *([1] * (D1s.ndim - 1))) * D1s, axis=1
                )
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t.astype(x.dtype)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        if isinstance(timestep, mx.array):
            timestep_val = timestep
        else:
            timestep_val = mx.array(timestep, dtype=schedule_timesteps.dtype)
        diff = mx.abs(schedule_timesteps - timestep_val)
        first_idx = mx.argmin(diff)
        num_matches = int((diff == 0).sum().item())
        if num_matches > 1:
            return int(first_idx.item()) + 1
        return int(first_idx.item())

    def _init_step_index(self, timestep):
        self._step_index = self.index_for_timestep(timestep)

    def step(self, model_output, timestep, sample):
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps before step()")
        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0
            and self.step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        self._step_index += 1
        return prev_sample


@partial(mx.compile, shapeless=True)
def _euler_step(model_output, sample, sigma, sigma_next):
    return sample + model_output * (sigma_next - sigma)


class FlowEulerDiscreteScheduler:
    """Simple Euler flow-matching scheduler for step-distilled models.

    Unlike UniPC, this uses a single-step Euler update matching how
    step-distilled models were trained. Timestep selection uses indexed
    positions from the full 1000-step schedule (via denoising_step_list)
    rather than linear interpolation.
    """

    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = 0

    def set_timesteps(self, denoising_step_list, shift=5.0):
        """Build schedule by indexing into the full shifted schedule.

        Args:
            denoising_step_list: e.g. [1000, 750, 500, 250]. Each value V
                maps to index (num_train_timesteps - V) in the shifted schedule.
            shift: Noise schedule shift factor (default 5.0 for distilled).
        """
        sigmas = mx.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps

        indices = mx.array([self.num_train_timesteps - x for x in denoising_step_list])
        self.sigmas = sigmas[indices].astype(mx.float32)
        self.timesteps = timesteps[indices].astype(mx.float32)
        self.num_inference_steps = len(denoising_step_list)

    def step(self, model_output, timestep, sample):
        """Euler flow-matching update: x_new = x + f * (sigma_next - sigma)."""
        t_val = timestep.item() if isinstance(timestep, mx.array) else timestep
        step_index = int(
            mx.argmin(mx.abs(self.timesteps.astype(mx.float32) - t_val)).item()
        )

        sigma = self.sigmas[step_index]
        sigma_next = (
            self.sigmas[step_index + 1]
            if step_index < self.num_inference_steps - 1
            else mx.array(0.0)
        )
        return _euler_step(
            model_output.astype(mx.float32),
            sample.astype(mx.float32),
            sigma,
            sigma_next,
        ).astype(sample.dtype)
