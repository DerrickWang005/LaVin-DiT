from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from einops import rearrange
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from lib.model.blocks import Block, FinalLayer, TimestepEmbedder


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs, use_reentrant=False)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(
            module, gc_step, *args, **kwargs, use_reentrant=False
        )
    return module(*args, **kwargs)


class LaVinDiT(nn.Module):
    def __init__(
        self,
        patch_size: list[int] = [2, 2],  # Add type hints
        in_channels: int = 16,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        kv_group: int = 1,
        drop_path: float = 0.0,
        uncond_ratio: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.depth = depth
        self.uncond_ratio = uncond_ratio

        self.cond_embedder = nn.Sequential(
            nn.Linear(
                in_features=patch_size[0] * patch_size[1] * in_channels,
                out_features=hidden_size,
                bias=True,
            ),
            LayerNorm(hidden_size),
        )
        self.x_embedder = nn.Sequential(
            nn.Linear(
                in_features=patch_size[0] * patch_size[1] * in_channels,
                out_features=hidden_size,
                bias=True,
            ),
            LayerNorm(hidden_size),
        )
        # type embeddings: prompt, answer
        self.prompt_embed = nn.Parameter(torch.randn(1, hidden_size))
        self.answer_embed = nn.Parameter(torch.randn(1, hidden_size))

        # timestep embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)

        # blocks
        drop_path = np.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(hidden_size, num_heads, kv_group=kv_group, drop_path=drop_path[i])
                for i in range(self.depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size, patch_size[0] * patch_size[1], self.out_channels
        )
        self.initialize_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.cond_embedder[0].weight)
        nn.init.xavier_uniform_(self.x_embedder[0].weight)

        nn.init.normal_(self.prompt_embed.data, std=0.02)
        nn.init.normal_(self.answer_embed.data, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.cond_modulation[-1].weight, 0)
            nn.init.constant_(block.cond_modulation[-1].bias, 0)
            nn.init.constant_(block.x_modulation[-1].weight, 0)
            nn.init.constant_(block.x_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, **kwargs
    ) -> list[torch.Tensor]:
        B, T_c, H, W, C = cond.shape
        cond = rearrange(cond, "B T_c H W C -> B (T_c H W) C")
        _, T_x, _, _, _ = x.shape
        x = rearrange(x, "B T_x H W C -> B (T_x H W) C")

        cond = self.cond_embedder(cond)  # B, L_c, D
        x = self.x_embedder(x)  # B, L_x, D
        t = self.t_embedder(t)  # (B, D)
        t_cond = t + self.prompt_embed
        t_x = t + self.answer_embed

        joint_seq_shape = (T_c + T_x, H, W)

        for i, block in enumerate(self.blocks):
            cond, x = auto_grad_checkpoint(
                block,
                cond,
                x,
                t_cond,
                t_x,
                joint_seq_shape,
            )

        x = self.final_layer(x, t_x)  # (N, patch_size ** 2 * out_channels)

        x = rearrange(
            x,
            "B (T_x H W) (ph pw c) -> B c T_x (H ph) (W pw)",
            T_x=T_x,
            H=H,
            W=W,
            ph=self.patch_size[0],
            pw=self.patch_size[1],
        )
        return x.float()

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conds: list[torch.Tensor],
        cfg_scale: float,
        **kwargs
    ) -> torch.Tensor:
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        x = rearrange(
            x,
            "B C F (H pH) (W pW) -> B F H W (pH pW C)",
            pH=self.patch_size[0],
            pW=self.patch_size[1],
        )
        x, x_null = x.chunk(2, dim=0)
        cond, cond_null = conds
        cond = rearrange(
            cond,
            "B C F (H pH) (W pW) -> B F H W (pH pW C)",
            pH=self.patch_size[0],
            pW=self.patch_size[1],
        )
        cond_null = rearrange(
            cond_null,
            "B C F (H pH) (W pW) -> B F H W (pH pW C)",
            pH=self.patch_size[0],
            pW=self.patch_size[1],
        )
        t, t_null = t.chunk(2, dim=0)
        cond = self.forward(x, t, cond, **kwargs)
        uncond = self.forward(x_null, t_null, cond_null, **kwargs)

        half_eps = uncond + cfg_scale * (cond - uncond)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return eps


def LaVinDiT_3B_2(**kwargs):  # GQA, 3.4B
    return LaVinDiT(
        patch_size=(2, 2),
        depth=22,
        hidden_size=2304,
        num_heads=32,
        kv_group=4,
        **kwargs
    )
