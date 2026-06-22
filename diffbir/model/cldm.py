from typing import Tuple, Set, List, Dict

import torch
from torch import nn
from peft import get_peft_model, LoraConfig

from .controlnet import ControlledUnetModel, ControlNet
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor
    ):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)  # trainable ControlNet
        self.fixed_controlnet = ControlNet(**controlnet_cfg)  # frozen ControlNet
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13
        self.reg_lambda = 1e-4  # regularization strength

    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        # Check if state dict contains controlnet parameters
        has_controlnet = any(k.startswith('controlnet.') for k in sd.keys())

        if has_controlnet:
            # Filter only controlnet parameters and remove 'controlnet.' prefix
            filtered_sd = {k.replace('controlnet.', ''): v for k, v in sd.items() if k.startswith('controlnet.')}
            unused, missing = self.controlnet.load_state_dict(filtered_sd, strict=True)
            self.fixed_controlnet.load_state_dict(filtered_sd, strict=True)
        else:
            # Load directly if no controlnet prefix found
            unused, missing = self.controlnet.load_state_dict(sd, strict=True)
            self.fixed_controlnet.load_state_dict(sd, strict=True)

        self.fixed_controlnet.eval()
        self.fixed_controlnet.train = disabled_train
        for p in self.fixed_controlnet.parameters():
            p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        self.fixed_controlnet.load_state_dict(init_sd, strict=True)
        self.fixed_controlnet.eval()
        self.fixed_controlnet.train = disabled_train
        for p in self.fixed_controlnet.parameters():
            p.requires_grad = False
        return init_with_new_zero, init_with_scratch

    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z

    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)

    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(
                cond_img * 2 - 1,
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ),
        )

    def forward(self, x_noisy, t, cond, use_reference_controlnet=False):
        """
        Forward pass.

        Args:
            x_noisy (torch.Tensor): Noisy latent.
            t (torch.Tensor): Timestep.
            cond (dict): Conditional inputs.
            use_reference_controlnet (bool): If True, use the frozen fixed_controlnet;
                                             otherwise use the trainable controlnet.
        """
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]

        # Select which ControlNet to use based on the flag.
        if use_reference_controlnet:
            # Use the frozen, untrained reference ControlNet.
            control_net_to_use = self.fixed_controlnet
        else:
            # Use the trainable ControlNet being optimized.
            control_net_to_use = self.controlnet

        # Compute the ControlNet outputs.
        control = control_net_to_use(x=x_noisy, hint=c_img, timesteps=t, context=c_txt)
        control = [c * scale for c, scale in zip(control, self.control_scales)]

        # Compute the U-Net output.
        eps = self.unet(
            x=x_noisy,
            timesteps=t,
            context=c_txt,
            control=control,
            only_mid_control=False,
        )
        return eps

    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        self.unet.dtype = dtype
        self.controlnet.dtype = dtype
        self.fixed_controlnet.dtype = dtype
        # convert unet blocks to dtype
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)
        # convert controlnet blocks and zero-convs to dtype
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)
        for module in [
            self.fixed_controlnet.input_blocks,
            self.fixed_controlnet.zero_convs,
            self.fixed_controlnet.middle_block,
            self.fixed_controlnet.middle_block_out,
        ]:
            module.type(dtype)

        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # GroupNorm32 only works with float32
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.fixed_controlnet.input_blocks,
            self.fixed_controlnet.zero_convs,
            self.fixed_controlnet.middle_block,
            self.fixed_controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)