from typing import Optional, Tuple, Dict, Literal
import torch
import numpy as np
from tqdm import tqdm
from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model import ControlLDM
from torch.distributions import Normal

def make_ddim_timesteps(
    ddim_discr_method: str,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
    verbose: bool = True,
) -> np.ndarray:
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )

    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(
    alphacums: np.ndarray, ddim_timesteps: np.ndarray, eta: float, verbose: bool = True
) -> Tuple[np.ndarray]:
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    if verbose:
        print(
            f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}"
        )
        print(
            f"For the chosen value of eta, which is {eta}, "
            f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


class DDIMSampler(Sampler):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
        eta: float,
    ) -> "DDIMSampler":
        super().__init__(betas, parameterization, rescale_cfg)
        self.eta = eta

    def make_schedule(
        self,
        ddim_num_steps,
        ddim_discretize="uniform",
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.num_timesteps,
            verbose=False,
        )
        original_alphas_cumprod = self.training_alphas_cumprod
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=original_alphas_cumprod,
            ddim_timesteps=self.ddim_timesteps,
            eta=self.eta,
            verbose=False,
        )
        self.register("ddim_sigmas", ddim_sigmas)
        self.register("ddim_alphas", ddim_alphas)
        self.register("ddim_sqrt_alphas", np.sqrt(ddim_alphas))
        self.register("ddim_alphas_prev", ddim_alphas_prev)
        self.register("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))

    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.ddim_sqrt_alphas, t, x_start.shape) * x_start
            + extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x_start.shape)
            * noise
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.ddim_sqrt_alphas, t, x_t.shape) * v
            + extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x_t.shape) * x_t
        )

    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
        cldm=None,
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.0:
            model_output = model(x, model_t, cond)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([model_t] * 2)
            cond_in = {k: torch.cat([cond[k], uncond[k]]) for k in cond.keys()}
            model_cond, model_uncond = model(x_in, t_in, cond_in).chunk(2)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        if self.parameterization == "eps":
            e_t = model_output
        else:
            e_t = self.predict_eps_from_z_and_v(x, t, model_output)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        a_t = extract_into_tensor(alphas, t, x.shape)
        a_prev = extract_into_tensor(alphas_prev, t, x.shape)
        sigma_t = extract_into_tensor(sigmas, t, x.shape)
        sqrt_one_minus_at = extract_into_tensor(sqrt_one_minus_alphas, t, x.shape)

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(x)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
        cldm=None,
    ) -> torch.Tensor:
        self.make_schedule(ddim_num_steps=steps)
        self.to(device)
        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x = x_T
        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        iterator = tqdm(
            time_range,
            desc="DDIM Sampler",
            total=total_steps,
            disable=not progress,
        )
        bs = x_size[0]

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            x = self.p_sample(model, x, model_t, t, cond, uncond, cur_cfg_scale, cldm)

        return x


    @torch.no_grad()
    def sample_with_logprob(
    self, model: ControlLDM, device: str, steps: int, x_size: Tuple[int],
    cond: Dict[str, torch.Tensor], uncond: Optional[Dict[str, torch.Tensor]], cfg_scale: float,
    x_T: torch.Tensor | None = None,
):
        self.make_schedule(ddim_num_steps=steps)
        self.to(device)
        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        latents_trajectory = [x_T]
        log_probs_trajectory = []
        timesteps_trajectory = []
        all_kl = []

        x = x_T
        time_range = np.flip(self.ddim_timesteps) # timestep values in descending order (e.g. 999, 970, ...)
        total_steps = len(self.ddim_timesteps)
        bs = x_size[0]

        for i, step_value in enumerate(tqdm(time_range, desc="DDIM Sampler with logprob")):

            current_ddim_schedule_index = torch.full((bs,), total_steps - 1 - i, device=device, dtype=torch.long)
            model_t_input = torch.full((bs,), step_value, device=device, dtype=torch.long)
            timesteps_trajectory.append(model_t_input)
            if uncond is None or cfg_scale == 1.0:
                model_output_current_policy = model(x, model_t_input, cond)
            else:
                x_in, t_in = torch.cat([x] * 2), torch.cat([model_t_input] * 2)
                cond_in = {k: torch.cat([cond[k], uncond[k]]) for k in cond.keys()}
                model_cond, model_uncond = model(x_in, t_in, cond_in).chunk(2)
                model_output_current_policy = model_uncond + cfg_scale * (model_cond - model_uncond)

            if self.parameterization == "eps":
                e_t_current_policy = model_output_current_policy
            else:
                e_t_current_policy = self.predict_eps_from_z_and_v(x, current_ddim_schedule_index, model_output_current_policy)
            a_t = extract_into_tensor(self.ddim_alphas, current_ddim_schedule_index, x.shape)
            a_prev = extract_into_tensor(self.ddim_alphas_prev, current_ddim_schedule_index, x.shape)
            sqrt_one_minus_at = extract_into_tensor(self.ddim_sqrt_one_minus_alphas, current_ddim_schedule_index, x.shape)
            sigmas_t = extract_into_tensor(self.ddim_sigmas, current_ddim_schedule_index, x.shape)
            pred_x0_current_policy = (x - sqrt_one_minus_at * e_t_current_policy) / a_t.sqrt()
            dir_xt_current_policy = (1.0 - a_prev - sigmas_t**2).sqrt() * e_t_current_policy
            x_prev_mean_current_policy = a_prev.sqrt() * pred_x0_current_policy + dir_xt_current_policy
            noise = sigmas_t * torch.randn_like(x)
            x_prev = x_prev_mean_current_policy + noise
            latents  = x_prev.clone()
            latents_trajectory.append(latents)
            dist = Normal(x_prev_mean_current_policy, sigmas_t)
            log_prob = dist.log_prob(x_prev).mean(dim=list(range(1, len(x.shape))))
            log_probs_trajectory.append(log_prob)

            x = x_prev
            all_kl.append(torch.zeros(len(x_prev), device=x_prev.device))

        final_image = model.vae_decode(x)
        latents = torch.stack(latents_trajectory, dim=1)
        log_probs = torch.stack(log_probs_trajectory, dim=1)
        timesteps = torch.stack(timesteps_trajectory, dim=1) # Stack the timesteps

        all_kl = torch.stack(all_kl, dim=1)

        return final_image, latents, log_probs, timesteps, all_kl

    def compute_log_prob_from_state(
            self,
            model: "ControlLDM",
            x_t: torch.Tensor,
            current_timestep_value: torch.Tensor,
            condition: Dict[str, torch.Tensor],
            x_prev_observed: torch.Tensor,
            accelerator,
            is_reference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the log-probability of transitioning from the current state x_t to
        x_prev_observed under the given model. This is the core step used in GRPO training.

        Args:
            model (ControlLDM): The current policy model.
            x_t (torch.Tensor): Current state in the trajectory, shape (B, C, H, W).
            current_timestep_value (torch.Tensor): Actual timestep values, shape (B,) (e.g., 999, 970).
            condition (Dict[str, torch.Tensor]): Conditional inputs for the model.
            x_prev_observed (torch.Tensor): Observed next state in the trajectory, shape (B, C, H, W).
            accelerator: Accelerator handle (not used directly here).
            is_reference (bool): Whether to use the reference model (or a reference
                path inside the model) for the forward pass.

        Returns:
            Tuple[
                torch.Tensor,  # log_prob_new: log-probability under the new policy, shape (B,)
                torch.Tensor,  # model_output_new: raw model output at this step
                torch.Tensor,  # x_prev_mean_new: predicted mean of the next state
                torch.Tensor,  # sigmas_t: standard deviation at the current timestep
            ]
        """
        bs = x_t.shape[0]
        device = x_t.device

        model_output_new = model(
            x_t,
            current_timestep_value,
            condition,
            use_reference_controlnet=is_reference,
        )

        schedule_timesteps = torch.from_numpy(self.ddim_timesteps).to(device)
        t_schedule_indices = (
                schedule_timesteps.unsqueeze(0) == current_timestep_value.unsqueeze(1)
        ).nonzero()[:, 1]

        if self.parameterization == "eps":
            e_t_new = model_output_new
        else:
            e_t_new = self.predict_eps_from_z_and_v(x_t, t_schedule_indices, model_output_new)

        alphas_t = extract_into_tensor(self.ddim_alphas, t_schedule_indices, x_t.shape)
        alphas_prev = extract_into_tensor(self.ddim_alphas_prev, t_schedule_indices, x_t.shape)
        sqrt_one_minus_alphas_t = extract_into_tensor(
            self.ddim_sqrt_one_minus_alphas, t_schedule_indices, x_t.shape
        )
        # Ensure sigmas_t shape matches for broadcasting with tensors below.
        sigmas_t = extract_into_tensor(self.ddim_sigmas, t_schedule_indices, x_t.shape)

        pred_x0_new = (x_t - sqrt_one_minus_alphas_t * e_t_new) / alphas_t.sqrt()
        dir_xt_new = (1.0 - alphas_prev - sigmas_t ** 2).sqrt() * e_t_new
        x_prev_mean_new = alphas_prev.sqrt() * pred_x0_new + dir_xt_new

        dist = Normal(x_prev_mean_new, sigmas_t)
        log_prob_new = dist.log_prob(x_prev_observed.detach())
        log_prob_new = log_prob_new.mean(dim=list(range(1, x_prev_observed.ndim)))

        return log_prob_new, model_output_new, x_prev_mean_new, sigmas_t

