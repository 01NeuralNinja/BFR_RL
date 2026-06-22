import os
from argparse import ArgumentParser
from collections import defaultdict
import lpips
import torch
import json
from torch.distributions import Normal
from concurrent import futures
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to
from diffbir.sampler import DDIMSampler
import diffbir.rewards
from diffbir.stat_tracking import PerPromptStatTracker
from torch.utils.data import Sampler
import numpy as np

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the ratio of zero standard deviation for each unique prompt's reward values.

    Parameters:
        prompts (list): List of prompts.
        gathered_rewards (dict): Dictionary containing reward values, must include the 'ori_avg' key.

    Returns:
        zero_std_ratio (float): The ratio of prompts with zero standard deviation.
        prompt_std_devs (numpy.ndarray): Array of standard deviations for each unique prompt.
    """
    # Convert the list of prompts to a NumPy array
    prompt_array = np.array(prompts)

    # Get unique prompts and their grouping information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array,
        return_inverse=True,
        return_counts=True
    )

    # Group the rewards based on the prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    # Calculate the standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])

    # Calculate the ratio of zero standard deviations
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio


class DistributedKRepeatSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        k,
        num_replicas,
        rank,
        seed=0,
        sequential: bool = True,
        shuffle_within_global_batch: bool = True,
        start_index: int = 0,
    ):
        """
        Initialize the DistributedKRepeatSampler.

        Parameters:
            dataset (torch.utils.data.Dataset): The dataset to sample from.
            batch_size (int): The batch size.
            k (int): The number of repetitions for each sample.
            num_replicas (int): The total number of replicas (workers).
            rank (int): The rank of the current replica (worker).
            seed (int, optional): The seed for randomness (default is 0).
            sequential (bool, optional): Whether to sample sequentially (default is True).
            shuffle_within_global_batch (bool, optional): Whether to shuffle the global batch (default is True).
            start_index (int, optional): The starting index for sequential sampling (default is 0).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, \
            f"k must divide n*b. k={k}, num_replicas={num_replicas}, batch_size={batch_size}"
        self.m = self.total_samples // self.k  # The number of distinct samples in each "global batch"

        if len(self.dataset) == 0:
            raise ValueError("Empty dataset is not allowed.")

        self.sequential = sequential
        self.shuffle_within_global_batch = shuffle_within_global_batch
        self.epoch = 0

        # Sequential pointer: Continues across iterations, not reset by set_epoch
        self.ptr = start_index % len(self.dataset)

    def __iter__(self):
        while True:
            # 1) Get m distinct sample indices for the global batch
            if self.sequential:
                indices = []
                for _ in range(self.m):
                    indices.append(self.ptr)
                    self.ptr += 1
                    if self.ptr >= len(self.dataset):
                        self.ptr = 0
            else:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # 2) Repeat each sample k times to form the global batch
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            # 3) Optionally shuffle the global batch
            if self.shuffle_within_global_batch:
                g2 = torch.Generator()
                g2.manual_seed(self.seed + self.epoch)
                perm = torch.randperm(len(repeated_indices), generator=g2).tolist()
                repeated_indices = [repeated_indices[i] for i in perm]

            # 4) Split and return the current rank's portion
            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield repeated_indices[start:end]

    def set_epoch(self, epoch: int):
        """
        Set the epoch for the sampler.

        Parameters:
            epoch (int): The current epoch number.
        """
        # In sequential mode, only store the epoch for reproducibility of shuffling; do not reset ptr
        self.epoch = epoch


def main(args) -> None:
    cfg = OmegaConf.load(args.config)
    with open('./configs/train/cldm_ds_config.json', 'r') as f:
        deepspeed_config = json.load(f)

    # NOTE: this value only sets the Accelerator's gradient-sync cadence (i.e. how often a
    # logging step is counted). The *actual* optimizer-update frequency is governed by the
    # DeepSpeed config (cldm_ds_config.json), which intentionally leaves
    # `gradient_accumulation_steps` unset -> DeepSpeed defaults to 1 (one optimizer update per
    # micro-step). This pairing is what reproduces the paper. It is validated ONLY with
    # accelerate==1.7.0 (see requirements.txt); other accelerate versions change how the two
    # layers interact and will alter the training dynamics. Do NOT add
    # `gradient_accumulation_steps` to cldm_ds_config.json.
    num_train_timesteps = int(cfg.train.sample_steps * cfg.train.timestep_fraction)
    final_accumulation_steps = (cfg.train.sample_num_batches_per_epoch // 2)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config)

    accelerator = Accelerator(
        gradient_accumulation_steps=final_accumulation_steps,
        log_with=cfg.train.log_record,
        deepspeed_plugin=deepspeed_plugin,
    )
    accelerator.init_trackers("lrpo", config=OmegaConf.to_container(cfg, resolve=True))
    device = accelerator.device

    exp_dir = cfg.train.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Main device: {device}")
    print(f"Experiment directory: {exp_dir}")

    # Instantiate and load pre-trained models
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu", weights_only=False)["state_dict"]
    cldm.load_pretrained_sd(sd)
    control_weight = torch.load(cfg.train.controlnet_path, map_location="cpu", weights_only=False)
    cldm.load_controlnet_from_ckpt(control_weight)

    # SwinIR model (for inference, frozen)
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd_swinir = torch.load(cfg.train.swinir_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd_swinir:
        sd_swinir = sd_swinir["state_dict"]
    sd_swinir = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd_swinir.items()}
    swinir.load_state_dict(sd_swinir, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    swinir.eval().to(device)

    # Diffusion model
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    # CLDM optimizer
    opt_cldm = torch.optim.AdamW(
        [p for p in cldm.controlnet.parameters() if p.requires_grad],
        lr=cfg.train.learning_rate
    )

    # Dataset and sampler initialization
    dataset = instantiate_from_config(cfg.dataset.train)
    train_sampler = DistributedKRepeatSampler(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        k=cfg.train.num_image_per_lr,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
        sequential=True,
    )

    # Note: The batch_size passed to DataLoader is the per-process micro batch size
    # The train_micro_batch_size_per_gpu in the DeepSpeed JSON should match this
    loader = DataLoader(
        dataset=dataset, batch_sampler=train_sampler,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )


    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Reward function for evaluation
    reward_fn = getattr(diffbir.rewards, 'multi_score')(accelerator.device, {"face_reward_remote": 1.0, })

    # Thread pool executor for parallel operations
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # --- 3. Prepare CLDM model and optimizer with DeepSpeed ---
    if accelerator.is_main_process:
        print("Preparing CLDM model and optimizer with DeepSpeed...")

    # Prepare the model, optimizer, and data loader for distributed training
    cldm, opt_cldm, prepared_loader = accelerator.prepare(cldm, opt_cldm, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    # Initialize training parameters
    global_step = 0
    start_epoch = 0
    stat_tracker = PerPromptStatTracker(global_std=True)

    # Prepare DDIM sampler for the diffusion model
    sampler = DDIMSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False, eta=1.0)
    sampler.make_schedule(ddim_num_steps=cfg.train.sample_steps)
    _a = sampler.ddim_sigmas
    _w = torch.flip(_a, dims=[0])
    _wm = torch.mean(_w)
    _w = (torch.ones_like(_w) if _wm < 1e-6 else _w / _wm).to(accelerator.device)

    # Initialize the training iteration
    train_iter = iter(prepared_loader)

    for epoch in range(start_epoch, cfg.train.num_epochs):

        #################### 1. Sampling Phase (SAMPLING) ####################
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}/{cfg.train.num_epochs}: Starting sampling...")
        cldm.eval()
        samples = []
        all_captions_for_epoch = []

        for i in tqdm(range(cfg.train.sample_num_batches_per_epoch), disable=not accelerator.is_main_process,
                      desc="Sampling batches"):
            train_sampler.set_epoch(epoch * cfg.train.sample_num_batches_per_epoch + i)
            dataset.set_epoch(epoch * cfg.train.sample_num_batches_per_epoch + i)
            batch_data = next(train_iter)
            gt_orig, lq_orig, prompt, caption = batch_transform(batch_data)

            gt = rearrange(gt_orig, "b h w c -> b c h w").contiguous().float()
            lq_for_swinir = rearrange(lq_orig, "b h w c -> b c h w").contiguous().float()

            if epoch < 2:
                continue
            with torch.no_grad():
                clean = swinir(lq_for_swinir)
                cond = cldm.prepare_condition(clean, prompt)
                cldm.eval()
                final_images, latents, log_probs, timesteps, all_kl = sampler.sample_with_logprob(
                    model=cldm,
                    device=accelerator.device,
                    steps=cfg.train.sample_steps,
                    x_size=(len(lq_orig), 4, 64, 64),
                    cond=cond,
                    uncond=None,
                    cfg_scale=1.0,
                )

                gt_latents_trajectory = []  # (B, steps+1, C, H, W), from noise to clean
                gt_x = pure_cldm.vae_encode(gt)  # GT at t=0 (clean latents)

                gt_current = gt_x  #
                time_range = np.flip(sampler.ddim_timesteps)
                for i, step_value in enumerate(time_range):
                    t_idx = torch.full((len(gt),), i, device=device, dtype=torch.long)
                    noise = torch.randn_like(gt_current)
                    gt_current = sampler.q_sample(gt_current, t_idx, noise)  # Adding noise at each step
                    gt_latents_trajectory.append(gt_current)
                gt_latents_trajectory = torch.stack(gt_latents_trajectory[::-1], dim=1)

            # Asynchronously calculate rewards
            if accelerator.is_main_process:
                print(f"Process {accelerator.process_index} is synchronously calling reward function...")
            prompt_metadata = {}
            prompt_metadata['gt_images'] = gt  # Values range should be consistent with final_images
            rewards = executor.submit(reward_fn, final_images, caption, prompt_metadata, only_strict=True)
            samples.append({
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "log_probs": log_probs,
                "condition": cond,
                "rewards": rewards,
                "kl": all_kl,
                "gt_latents": gt_latents_trajectory[:, :-1],  # x_t sequence (excluding the last clean)
                "gt_next_latents": gt_latents_trajectory[:, 1:],  # x_{t-1} sequence
            })
            all_captions_for_epoch.extend(caption)  # Collect all captions for this epoch

        if epoch < 2:
            continue
        # Wait for all rewards to be computed
        if accelerator.is_main_process:
            print("Waiting for reward calculations...")
        for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not accelerator.is_local_main_process,
                position=0,
        ):

            all_scores_from_server, _ = sample["rewards"].result()
            processed_rewards = {}
            for key, value in all_scores_from_server.items():
                processed_rewards[key] = torch.as_tensor(value, device=accelerator.device).float()

            # Assign the processed rewards to sample["rewards"]
            sample["rewards"] = processed_rewards

        #################### Data Processing Phase (DATA PROCESSING) ####################
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1) - cfg.train.kl_beta * samples["kl"]

        # Gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        if cfg.train.per_prompt_stat_tracking:
            gathered_captions = accelerator.gather_for_metrics(all_captions_for_epoch)  #
            advantages = stat_tracker.update(gathered_captions, gathered_rewards['avg'])

            if accelerator.is_local_main_process:
                print("len(captions)", len(gathered_captions))
                print("len unique captions", len(set(gathered_captions)))
            stat_tracker.clear()
        else:
            mean_reward = gathered_rewards['avg'].mean()
            std_reward = gathered_rewards['avg'].std() + 1e-4
            advantages = (gathered_rewards['avg'] - mean_reward) / std_reward

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())
            print("kl: ", samples["kl"].mean())

        del samples["rewards"]

        # Filter out samples with zero advantages

        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        num_batches = cfg.train.sample_num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        accelerator.log(
            {
                "actual_batch_size": mask.sum().item() // cfg.train.sample_num_batches_per_epoch,
            },
            step=global_step,
        )

        # Create a new dictionary to store the filtered samples
        filtered_samples = {}
        for k, v in samples.items():
            if isinstance(v, dict):
                # If the value is a dictionary, iterate over its tensors and apply the mask
                filtered_samples[k] = {sub_k: sub_v[mask] for sub_k, sub_v in v.items()}
            else:
                # Otherwise, directly apply the mask
                filtered_samples[k] = v[mask]
        samples = filtered_samples  # Update samples with the filtered dictionary

        #################### 3. Training Phase (TRAINING) ####################
        if accelerator.is_main_process:
            print("Starting model training...")
        for inner_epoch in range(cfg.train.train_num_inner_epochs):
            num_total_samples_in_buffer = len(samples["advantages"])  # Use the filtered actual sample size
            perm = torch.randperm(num_total_samples_in_buffer, device=accelerator.device)
            cldm.train()
            info = defaultdict(list)
            # Important: Gradient accumulation context manager starts here
            for i in tqdm(range(0, num_total_samples_in_buffer, cfg.train.batch_size),
                          disable=not accelerator.is_main_process, desc=f"Inner training {inner_epoch + 1}"):
                batch_indices = perm[i:i + cfg.train.batch_size]
                sample_batch = {
                    "latents": samples["latents"][batch_indices],
                    "next_latents": samples["next_latents"][batch_indices],
                    "log_probs": samples["log_probs"][batch_indices],
                    "advantages": samples["advantages"][batch_indices],
                    "kl": samples["kl"][batch_indices],
                    "timesteps": samples["timesteps"][batch_indices],
                    "condition": {k: v[batch_indices] for k, v in samples["condition"].items()},
                    "gt_latents": samples["gt_latents"][batch_indices],
                    "gt_next_latents": samples["gt_next_latents"][batch_indices],
                }
                for j in range(num_train_timesteps):
                    # Place accumulate outside the innermost loop
                    with accelerator.accumulate(cldm):
                        current_timestep_value = sample_batch["timesteps"][:, j]
                        log_prob_new, model_output_new, x_prev_mean_current, sigmas_t_current = \
                            sampler.compute_log_prob_from_state(
                                model=cldm,
                                x_t=sample_batch["latents"][:, j],
                                current_timestep_value=current_timestep_value,
                                condition=sample_batch["condition"],
                                x_prev_observed=sample_batch["next_latents"][:, j],
                                accelerator=accelerator,
                                is_reference=False
                            )
                        _g = sample_batch["gt_next_latents"][:, j]
                        _reg = x_prev_mean_current.new_zeros(())
                        if j >= num_train_timesteps - 5:
                            _d = Normal(x_prev_mean_current, sigmas_t_current)
                            _reg = (1.0 - torch.exp(_d.log_prob(_g.detach()).mean(dim=list(range(1, _g.ndim))) / 40.0)).mean()

                        log_prob_old = sample_batch["log_probs"][:, j]
                        advantage_j = torch.clamp(
                            sample_batch["advantages"][:, j] * _w[j],
                            -cfg.train.adv_clip_max,
                            cfg.train.adv_clip_max,
                        )

                        ratio = torch.exp(log_prob_new - log_prob_old)
                        unclipped_loss = -advantage_j * ratio
                        clipped_loss = -advantage_j * torch.clamp(ratio, 1.0 - cfg.train.clip_range,
                                                                  1.0 + cfg.train.clip_range)
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        loss_total = policy_loss + 0.001 * _reg

                        info["approx_kl"].append(0.5 * torch.mean((log_prob_new - log_prob_old) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > cfg.train.clip_range).float()))
                        info["policy_loss"].append(policy_loss)
                        info["likelihood_loss"].append(0.001 * _reg)

                        accelerator.backward(loss_total)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(cldm.parameters(), cfg.train.max_grad_norm)
                        opt_cldm.step()
                        opt_cldm.zero_grad()

                    if accelerator.sync_gradients:
                        info_averaged = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info_averaged = accelerator.reduce(info_averaged, reduction="mean")
                        info_averaged.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        info_averaged["likelihood_loss"] = torch.mean(torch.stack(info["likelihood_loss"])) if info[
                            "likelihood_loss"] else 0
                        accelerator.log(info_averaged, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

                        # Save inference-ready ControlNet weights every `ckpt_every` optimizer steps.
                        # ZeRO-2 keeps model params replicated (only optimizer states are sharded),
                        # so unwrapping on the main process yields the full ControlNet state_dict,
                        # directly loadable by inference.py via --diffbir_refl_path.
                        if cfg.train.ckpt_every and global_step % cfg.train.ckpt_every == 0:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                ctrl_sd = accelerator.unwrap_model(cldm).controlnet.state_dict()
                                save_path = os.path.join(ckpt_dir, f"controlnet_epoch_{epoch}_step_{global_step}.pth")
                                torch.save(ctrl_sd, save_path)
                                print(f"Saved ControlNet weights to: {save_path}")
                            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            B = cfg.train.sample_num_batches_per_epoch
            m = getattr(train_sampler, "m",
                        (accelerator.num_processes * cfg.train.batch_size) // cfg.train.num_image_per_lr)
            approx_used = min(len(dataset), (epoch + 1) * B * m)
            accelerator.log({
                "approx/unique_images_scheduled_cum": approx_used,
                "approx/unique_images_ratio": approx_used / len(dataset),
                "approx/epoch_index": epoch,
            }, step=global_step)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)