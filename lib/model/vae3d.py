from collections.abc import Iterable
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from diffusers.models.activations import get_activation


def auto_grad_checkpoint(module, *args, **kwargs):
    if not isinstance(module, Iterable):
        return checkpoint.checkpoint(module, *args, **kwargs, use_reentrant=False)
    gc_step = module[0].grad_checkpointing_step
    return checkpoint.checkpoint_sequential(
        module, gc_step, *args, **kwargs, use_reentrant=False
    )


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE latent space.

    Args:
        parameters (torch.Tensor): Parameters tensor containing mean and logvar
        deterministic (bool, optional): If True, distribution is deterministic. Defaults to False.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device, dtype=self.parameters.dtype
        )
        return x

    def kl(
        self, other: Optional["DiagonalGaussianDistribution"] = None
    ) -> torch.Tensor:
        """Calculate KL divergence with another distribution."""
        if self.deterministic:
            return torch.Tensor([0.0])

        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )

        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class STVAEDownsample3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = x.shape

            # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
            x = x.permute(0, 3, 4, 1, 2).reshape(
                batch_size * height * width, channels, frames
            )

            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(
                    0, 3, 4, 1, 2
                )
            else:
                # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(
                    0, 3, 4, 1, 2
                )

        # Pad the tensor
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, height, width = x.shape
        # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frames, channels, height, width
        )
        x = self.conv(x)
        # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(
            0, 2, 1, 3, 4
        )
        return x


class STVAEUpsample3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.compress_time = compress_time

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            # if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
            if inputs.shape[2] > 1:
                # split first frame
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

                x_first = F.interpolate(x_first, scale_factor=2.0)
                x_rest = F.interpolate(x_rest, scale_factor=2.0)
                x_first = x_first[:, :, None, :, :]
                inputs = torch.cat([x_first, x_rest], dim=2)
            # elif inputs.shape[2] > 1:
            #     inputs = F.interpolate(inputs, scale_factor=2.0)
            else:
                inputs = inputs.squeeze(2)
                inputs = F.interpolate(inputs, scale_factor=2.0)
                inputs = inputs[:, :, None, :, :]
        else:
            # only interpolate 2D
            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = F.interpolate(inputs, scale_factor=2.0)
            inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)

        return inputs


class STVAECausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.conv_cache = None

    def padding_head(self, inputs: torch.Tensor) -> torch.Tensor:
        dim = self.temporal_dim
        kernel_size = self.time_kernel_size
        if kernel_size == 1:
            return inputs
        inputs = inputs.transpose(0, dim)
        inputs = torch.cat([inputs[:1]] * (kernel_size - 1) + [inputs], dim=0)
        inputs = inputs.transpose(0, dim).contiguous()
        return inputs

    def forward(self, inputs: torch.Tensor):
        input_parallel = self.padding_head(inputs)
        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        input_parallel = F.pad(input_parallel, padding_2d, mode="constant", value=0)
        output_parallel = self.conv(input_parallel)
        output = output_parallel
        return output


class STVAESpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_channels=f_channels, num_groups=32, eps=1e-6, affine=True
        )
        self.conv_y = STVAECausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1
        )
        self.conv_b = STVAECausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1
        )

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])

        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class STVAEResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut

        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(
                num_channels=in_channels, num_groups=groups, eps=eps
            )
            self.norm2 = nn.GroupNorm(
                num_channels=out_channels, num_groups=groups, eps=eps
            )
        else:
            self.norm1 = STVAESpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
            )
            self.norm2 = STVAESpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
            )

        self.conv1 = STVAECausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(
                in_features=temb_channels, out_features=out_channels
            )

        self.dropout = nn.Dropout(dropout)
        self.conv2 = STVAECausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = STVAECausalConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    pad_mode=pad_mode,
                )
            else:
                self.conv_shortcut = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(
        self,
        inputs: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = inputs
        if zq is not None:
            hidden_states = self.norm1(hidden_states, zq)
        else:
            hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = (
                hidden_states
                + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]
            )

        if zq is not None:
            hidden_states = self.norm2(hidden_states, zq)
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inputs = self.conv_shortcut(inputs)
            else:
                inputs = self.conv_shortcut(inputs)

        output_tensor = inputs + hidden_states
        return output_tensor


class STVAEDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                STVAEResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = None

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    STVAEDownsample3D(
                        out_channels,
                        out_channels,
                        padding=downsample_padding,
                        compress_time=compress_time,
                    )
                ]
            )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            # if self.training and self.gradient_checkpointing:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = auto_grad_checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, zq
                )
            else:
                hidden_states = resnet(hidden_states, temb, zq)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class STVAEMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                STVAEResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            # if self.training and self.gradient_checkpointing:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = auto_grad_checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, zq
                )
            else:
                hidden_states = resnet(hidden_states, temb, zq)

        return hidden_states


class STVAEUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time: bool = False,
        pad_mode: str = "first",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                STVAEResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    STVAEUpsample3D(
                        out_channels,
                        out_channels,
                        padding=upsample_padding,
                        compress_time=compress_time,
                    )
                ]
            )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            # if self.training and self.gradient_checkpointing:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = auto_grad_checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, zq
                )
            else:
                hidden_states = resnet(hidden_states, temb, zq)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class STVAEEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "STVAEDownBlock3D",
            "STVAEDownBlock3D",
            "STVAEDownBlock3D",
            "STVAEDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = STVAECausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode
        )
        self.down_blocks = nn.ModuleList([])

        # down blocks
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if down_block_type == "STVAEDownBlock3D":
                down_block = STVAEDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    compress_time=compress_time,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                raise ValueError(
                    "Invalid `down_block_type` encountered. Must be `STVAEDownBlock3D`"
                )

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = STVAEMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = STVAECausalConv3d(
            block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode
        )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self, sample: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.conv_in(sample)

        # if self.training and self.gradient_checkpointing:
        if self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 1. Down
            for down_block in self.down_blocks:
                hidden_states = auto_grad_checkpoint(
                    create_custom_forward(down_block), hidden_states, temb, None
                )

            # 2. Mid
            hidden_states = auto_grad_checkpoint(
                create_custom_forward(self.mid_block), hidden_states, temb, None
            )
        else:
            # 1. Down
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states, temb, None)

            # 2. Mid
            hidden_states = self.mid_block(hidden_states, temb, None)

        # 3. Post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class STVAEDecoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "STVAEUpBlock3D",
            "STVAEUpBlock3D",
            "STVAEUpBlock3D",
            "STVAEUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = STVAECausalConv3d(
            in_channels,
            reversed_block_out_channels[0],
            kernel_size=3,
            pad_mode=pad_mode,
        )

        # mid block
        self.mid_block = STVAEMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
            gradient_checkpointing=gradient_checkpointing,
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])

        output_channel = reversed_block_out_channels[0]
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if up_block_type == "STVAEUpBlock3D":
                up_block = STVAEUpBlock3D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=in_channels,
                    add_upsample=not is_final_block,
                    compress_time=compress_time,
                    pad_mode=pad_mode,
                    gradient_checkpointing=gradient_checkpointing,
                )
                prev_output_channel = output_channel
            else:
                raise ValueError(
                    "Invalid `up_block_type` encountered. Must be `STVAEUpBlock3D`"
                )

            self.up_blocks.append(up_block)

        self.norm_out = STVAESpatialNorm3D(reversed_block_out_channels[-1], in_channels)
        self.conv_act = nn.SiLU()
        self.conv_out = STVAECausalConv3d(
            reversed_block_out_channels[-1],
            out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self, sample: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""The forward method of the `STVAEDecoder3D` class."""
        hidden_states = self.conv_in(sample)

        # if self.training and self.gradient_checkpointing:
        if self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 1. Mid
            hidden_states = auto_grad_checkpoint(
                create_custom_forward(self.mid_block), hidden_states, temb, sample
            )

            # 2. Up
            for up_block in self.up_blocks:
                hidden_states = auto_grad_checkpoint(
                    create_custom_forward(up_block), hidden_states, temb, sample
                )
        else:
            # 1. Mid
            hidden_states = self.mid_block(hidden_states, temb, sample)

            # 2. Up
            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states, temb, sample)

        # 3. Post-process
        hidden_states = self.norm_out(hidden_states, sample)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderKLSTVAE(nn.Module):
    """Autoencoder with KL divergence for spatiotemporal data.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        down_block_types (tuple): Types of downsampling blocks
        up_block_types (tuple): Types of upsampling blocks
        block_out_channels (tuple): Number of output channels for each block
        latent_channels (int): Number of latent channels
        layers_per_block (int): Number of layers in each block
        act_fn (str): Activation function
        norm_eps (float): Epsilon for normalization
        norm_num_groups (int): Number of groups for group norm
        temporal_compression_ratio (float): Temporal compression ratio
        use_quant_conv (bool): Whether to use quantization convolution
        use_post_quant_conv (bool): Whether to use post quantization convolution
        tile_sample_min_size (int): Minimum tile size for sampling
        tile_sample_min_size_t (int): Minimum temporal tile size
        load_path (str, optional): Path to load pretrained weights
        use_checkpoint (bool): Whether to use gradient checkpointing
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "STVAEDownBlock3D",
            "STVAEDownBlock3D",
            "STVAEDownBlock3D",
            "STVAEDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "STVAEUpBlock3D",
            "STVAEUpBlock3D",
            "STVAEUpBlock3D",
            "STVAEUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: int = 4,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        tile_sample_min_size: int = 320,
        tile_sample_min_size_t: int = 81,
        load_path: Optional[str] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        # Input validation
        if len(down_block_types) != len(up_block_types):
            raise ValueError("Number of down blocks must match number of up blocks")
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                "Number of block output channels must match number of blocks"
            )

        self.latent_channels = latent_channels

        # Initialize encoder and decoder
        self.encoder = STVAEEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
            gradient_checkpointing=use_checkpoint,
        )
        self.decoder = STVAEDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
            gradient_checkpointing=use_checkpoint,
        )

        # Initialize conv layers
        self.quant_conv = (
            nn.Conv3d(2 * out_channels, 2 * out_channels, 1) if use_quant_conv else None
        )
        self.post_quant_conv = (
            nn.Conv3d(out_channels, out_channels, 1) if use_post_quant_conv else None
        )

        # Tiling parameters
        self._init_tiling_params(
            tile_sample_min_size, tile_sample_min_size_t, len(block_out_channels)
        )

        if load_path is not None:
            self._load_pretrained(load_path)

        # Freeze model
        self._freeze_model()

    def _init_tiling_params(
        self, tile_sample_min_size: int, tile_sample_min_size_t: int, num_blocks: int
    ):
        """Initialize parameters for tiled processing."""
        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = tile_sample_min_size // (2 ** (num_blocks - 1))
        self.tile_overlap_factor = 0.25
        self.tile_sample_min_size_t = tile_sample_min_size_t
        self.tile_latent_min_size_t = ((tile_sample_min_size_t - 1) // (2**2)) + 1

    def _freeze_model(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _load_pretrained(self, load_path: str):
        """Load pretrained weights from path."""
        print(f"Loading model from {load_path}")
        state_dict = torch.load(load_path, map_location="cpu")["state_dict"]
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {load_path}")

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size
            or x.shape[-2] > self.tile_sample_min_size
            or x.shape[-3] > self.tile_sample_min_size_t
        ):
            posterior = self.tiled_encode(x)
        else:
            h = self.encoder(x)
            if self.quant_conv is not None:
                h = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(h)
        return posterior

    @torch.no_grad()
    def decode(self, z: torch.FloatTensor):
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
            or z.shape[-3] > self.tile_latent_min_size_t
        ):
            dec = self.tiled_decode(z)
        else:
            if self.post_quant_conv is not None:
                z = self.post_quant_conv(z)
            dec = self.decoder(z)
        return dec

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
    ):
        posterior = self.encode(sample)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_sample_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [
                [t_chunk_idx[i], t_chunk_idx[i + 1] + 1]
                for i in range(len(t_chunk_idx) - 1)
            ]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        moments = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start:end]
            moment = self.tiled_encode2d(chunk_x, return_moments=True)
            moments.append(moment if idx == 0 else moment[:, :, 1:])
        moments = torch.cat(moments, dim=2)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def tiled_decode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_latent_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [
                [t_chunk_idx[i], t_chunk_idx[i + 1] + 1]
                for i in range(len(t_chunk_idx) - 1)
            ]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        dec_ = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start:end]
            if idx != 0:
                dec = self.tiled_decode2d(chunk_x)[:, :, 1:]
            else:
                dec = self.tiled_decode2d(chunk_x)
            dec_.append(dec)
        dec_ = torch.cat(dec_, dim=2)
        return dec_

    def tiled_encode2d(self, x, return_moments=False):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                if self.quant_conv is not None:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)
        if return_moments:
            return moments
        return posterior

    def tiled_decode2d(self, z):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                if self.post_quant_conv is not None:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec
