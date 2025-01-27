import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedRMSNorm as RMSNorm
from einops import rearrange
from timm.models.layers import DropPath
from flash_attn import flash_attn_func


class RotaryEmbedding3D(nn.Module):
    def __init__(self, dim: List[int], base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        # self.inv_freqs = 1.0 / (base ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))
        dim_t, dim_h, dim_w = dim
        self.inv_freqs_t = 1.0 / (
            base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t)
        )
        self.inv_freqs_h = 1.0 / (
            base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h)
        )
        self.inv_freqs_w = 1.0 / (
            base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w)
        )
        self._cache = {}

    def extra_repr(self) -> str:
        info = f"dim={self.dim} base={self.base}"
        return info

    @torch.cuda.amp.autocast(enabled=False)
    def compute_freqs(self, seq_len: List, dtype: torch.dtype, device: torch.device):
        T, H, W = seq_len
        key = f"{T}_{H}_{W}_{dtype}_{device}"

        if key in self._cache and T != 1:
            return self._cache[key]

        coords_t, coords_h, coords_w = torch.meshgrid(
            torch.arange(T, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        if T == 1:
            t_offset = torch.randint(0, 32, (1,), dtype=dtype, device=device)
            coords_t = coords_t + t_offset

        freqs_t = coords_t.reshape(-1, 1) * self.inv_freqs_t.to(device)[None, :]
        freqs_h = coords_h.reshape(-1, 1) * self.inv_freqs_h.to(device)[None, :]
        freqs_w = coords_w.reshape(-1, 1) * self.inv_freqs_w.to(device)[None, :]

        cos_t = torch.cos(freqs_t).to(dtype)
        sin_t = torch.sin(freqs_t).to(dtype)
        cos_h = torch.cos(freqs_h).to(dtype)
        sin_h = torch.sin(freqs_h).to(dtype)
        cos_w = torch.cos(freqs_w).to(dtype)
        sin_w = torch.sin(freqs_w).to(dtype)

        if T != 1:
            self._cache[key] = (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)

        if len(self._cache) > 8 and T != 1:
            self._cache.pop(next(iter(self._cache)))

        return cos_t, sin_t, cos_h, sin_h, cos_w, sin_w

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)
        return (x * cos + self.rotate_half(x) * sin).contiguous()

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: List):
        """3D-RoPE forward

        Args:
            q (torch.Tensor): query tensor [B, L, num_head, head_dim]
            k (torch.Tensor): key tensor [B, L, num_head, head_dim]
            seq_len (tuple): spatial dimensions (T, H, W) where T*H*W = L
        """
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = self.compute_freqs(
            seq_len, q.dtype, q.device
        )

        q_t, q_h, q_w = q.split(self.dim, dim=-1)
        k_t, k_h, k_w = k.split(self.dim, dim=-1)
        # T axial
        q_t = self.apply_rotary_emb(q_t, cos_t, sin_t)
        k_t = self.apply_rotary_emb(k_t, cos_t, sin_t)
        # H axial
        q_h = self.apply_rotary_emb(q_h, cos_h, sin_h)
        k_h = self.apply_rotary_emb(k_h, cos_h, sin_h)
        # W axial
        q_w = self.apply_rotary_emb(q_w, cos_w, sin_w)
        k_w = self.apply_rotary_emb(k_w, cos_w, sin_w)
        q = torch.cat([q_t, q_h, q_w], dim=-1)
        k = torch.cat([k_t, k_h, k_w], dim=-1)
        return q, k


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != self.dtype:
            t_freq = t_freq.to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core Model                                    #
#################################################################################
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kv_group: int = 1,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_heads // kv_group
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_drop = attn_drop

        self.cond_q = nn.Linear(dim, self.num_heads * self.head_dim, bias=qkv_bias)
        self.cond_k = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.cond_v = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.cond_qnorm = RMSNorm(self.num_heads * self.head_dim, eps=1e-6)
        self.cond_knorm = RMSNorm(self.num_kv_heads * self.head_dim, eps=1e-6)
        self.cond_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=qkv_bias)

        self.x_q = nn.Linear(dim, self.num_heads * self.head_dim, bias=qkv_bias)
        self.x_k = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.x_v = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.x_qnorm = RMSNorm(self.num_heads * self.head_dim, eps=1e-6)
        self.x_knorm = RMSNorm(self.num_kv_heads * self.head_dim, eps=1e-6)
        self.x_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=qkv_bias)

        self.proj_drop = nn.Dropout(proj_drop)
        self.rope3d = RotaryEmbedding3D(
            dim=[self.head_dim // 3] * 3,
            base=10000,
        )

    def forward(
        self,
        cond: torch.Tensor,
        x: torch.Tensor,
        joint_seq_shape: List,
    ) -> torch.Tensor:
        B, N, C = cond.shape
        B, M, C = x.shape

        # for cond
        cond_q = self.cond_q(cond)
        cond_k = self.cond_k(cond)
        cond_v = self.cond_v(cond)
        cond_q = self.cond_qnorm(cond_q)
        cond_k = self.cond_knorm(cond_k)
        cond_q = rearrange(
            cond_q, "b n (nh hd) -> b n nh hd", nh=self.num_heads, hd=self.head_dim
        ).contiguous()
        cond_k = rearrange(
            cond_k, "b n (nh hd) -> b n nh hd", nh=self.num_kv_heads, hd=self.head_dim
        ).contiguous()
        cond_v = rearrange(
            cond_v, "b n (nh hd) -> b n nh hd", nh=self.num_kv_heads, hd=self.head_dim
        ).contiguous()

        # for x
        x_q = self.x_q(x)
        x_k = self.x_k(x)
        x_v = self.x_v(x)
        x_q = self.x_qnorm(x_q)
        x_k = self.x_knorm(x_k)
        x_q = rearrange(
            x_q, "b m (nh hd) -> b m nh hd", nh=self.num_heads, hd=self.head_dim
        ).contiguous()
        x_k = rearrange(
            x_k, "b m (nh hd) -> b m nh hd", nh=self.num_kv_heads, hd=self.head_dim
        ).contiguous()
        x_v = rearrange(
            x_v, "b m (nh hd) -> b m nh hd", nh=self.num_kv_heads, hd=self.head_dim
        ).contiguous()

        # concate cond and x
        q = torch.cat([cond_q, x_q], dim=1)
        k = torch.cat([cond_k, x_k], dim=1)
        v = torch.cat([cond_v, x_v], dim=1)

        # apply rope
        q, k = self.rope3d(q, k, joint_seq_shape)

        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )
        out = out.reshape(B, N + M, C)

        cond, x = torch.split(out, [N, M], dim=1)
        cond = self.cond_proj(cond)
        x = self.x_proj(x)
        cond = self.proj_drop(cond)
        x = self.proj_drop(x)

        return cond, x


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
        intermediate_size = int((4 * hidden_size * 2 / 3) / 64) * 64
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, kv_group=1, drop_path=0.0):
        super().__init__()

        # joint attention
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.joint_attn = Attention(
            hidden_size, num_heads=num_heads, kv_group=kv_group, qkv_bias=False
        )
        self.post_norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # mlp
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cond_mlp = SwiGLU(hidden_size)
        self.x_mlp = SwiGLU(hidden_size)
        self.post_norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cond_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )
        self.x_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

    def modulate(self, x, scale):
        x = (1.0 + scale) * x
        return x

    def forward(self, cond, x, t_cond, t_x, joint_seq_shape):
        B = t_cond.size(0)

        scale_msa_cond, gate_msa_cond, scale_mlp_cond, gate_mlp_cond = (
            item.contiguous()
            for item in self.cond_modulation(t_cond).reshape(B, 4, -1).chunk(4, dim=1)
        )
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = (
            item.contiguous()
            for item in self.x_modulation(t_x).reshape(B, 4, -1).chunk(4, dim=1)
        )
        gate_msa_cond = gate_msa_cond.tanh()
        gate_mlp_cond = gate_mlp_cond.tanh()
        gate_msa_x = gate_msa_x.tanh()
        gate_mlp_x = gate_mlp_x.tanh()

        # for joint attention
        cond_mod = self.modulate(self.norm1(cond), scale_msa_cond)
        x_mod = self.modulate(self.norm1(x), scale_msa_x)
        cond_mod, x_mod = self.joint_attn(cond_mod, x_mod, joint_seq_shape)
        cond = cond + self.drop_path(gate_msa_cond * self.post_norm1(cond_mod))
        x = x + self.drop_path(gate_msa_x * self.post_norm1(x_mod))

        # for mlp
        cond_mod = self.modulate(self.norm2(cond), scale_mlp_cond)
        x_mod = self.modulate(self.norm2(x), scale_mlp_x)
        cond_mod = self.cond_mlp(cond_mod)
        x_mod = self.x_mlp(x_mod)
        cond = cond + self.drop_path(gate_mlp_cond * self.post_norm2(cond_mod))
        x = x + self.drop_path(gate_mlp_x * self.post_norm2(x_mod))

        return cond, x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.out_channels = out_channels

    def modulate(self, x, scale):
        x = (1.0 + scale) * x
        return x

    def forward(self, x, t):
        scale = self.adaLN_modulation(t).unsqueeze(1).contiguous()
        x = self.modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x
