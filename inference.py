import os
import numpy as np
import random
from contextlib import contextmanager
from glob import glob
import click
import torch
from safetensors.torch import load_file
from torchvision.transforms import v2
from torchvision.io import write_png
from torchvision.utils import make_grid
from PIL import Image
from lib.model.dit import LaVinDiT_3B_2
from lib.model.vae3d import AutoencoderKLSTVAE
from lib.transport import Sampler, create_transport


@contextmanager
def suspend_nn_inits():
    skip = lambda *args, **kwargs: None
    saved_inits = (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
        torch.nn.init.xavier_uniform_,
    )
    torch.nn.init.xavier_uniform_ = torch.nn.init.kaiming_uniform_ = (
        torch.nn.init.uniform_
    ) = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
            torch.nn.init.xavier_uniform_,
        ) = saved_inits  # restoring


class InferenceEngine:
    def __init__(
        self, vae_path, dit_path, timesteps: int, use_compile=True, use_bf16=True
    ):
        self.vae = AutoencoderKLSTVAE(
            tile_sample_min_size=512,
            tile_sample_min_size_t=129,
            load_path=vae_path,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.cuda()
        if use_bf16:
            self.vae.to(torch.bfloat16)

        with suspend_nn_inits():
            self.dit = LaVinDiT_3B_2()
        dit_ckpt = load_file(dit_path, device="cpu")
        m, u = self.dit.load_state_dict(dit_ckpt, strict=False)
        print(f"missing keys: {m}")
        print(f"unexpected keys: {u}")
        self.dit.requires_grad_(False)
        self.dit.eval()
        self.dit.cuda()
        if use_bf16:
            self.dit.to(torch.bfloat16)
        if use_compile:
            self.dit = torch.compile(self.dit, mode="reduce-overhead")

        transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            loss_weight="velocity",
            train_eps=0.2,
            sample_eps=0.1,
            snr_type="lognorm",
        )
        ode_params = {
            "sampling_method": "euler",
            "num_steps": timesteps,
            "atol": 1e-6,
            "rtol": 1e-3,
            "reverse": False,
        }
        transport_params = {"kind": "ode", "params": ode_params}
        sampler = Sampler(transport)
        self.sample_fn = sampler.sample_ode(**transport_params["params"])

    @property
    def vae_dtype(self):
        return next(self.vae.parameters()).dtype

    @property
    def dit_dtype(self):
        return next(self.dit.parameters()).dtype

    def get_task_prompt(self, task_dir: str):
        input_files = glob(os.path.join(task_dir, "*_input.png"))
        target_files = glob(os.path.join(task_dir, "*_target.png"))
        input_files.sort()
        target_files.sort()

        pair_list = []
        for input_file, target_file in zip(input_files, target_files):
            input_name = os.path.basename(input_file).split("_")[0]
            target_name = os.path.basename(target_file).split("_")[0]
            assert input_name == target_name
            pair_list.append((input_file, target_file))
        return pair_list

    def preprocess_prompt(self, prompts, height, width):
        rst = []
        for input_file, target_file in prompts:
            input_image = Image.open(input_file).convert("RGB")
            input_image = torch.from_numpy(np.array(input_image)).permute(2, 0, 1)
            input_image = v2.functional.resize(
                input_image,
                (height, width),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            )
            input_image = (input_image / 255.0 - 0.5) * 2

            target_image = Image.open(target_file).convert("RGB")
            target_image = torch.from_numpy(np.array(target_image)).permute(2, 0, 1)
            target_image = v2.functional.resize(
                target_image,
                (height, width),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            )
            target_image = (target_image / 255.0 - 0.5) * 2

            pair = torch.stack([input_image, target_image], dim=1)  # (3, 2, H, W)
            rst.append(pair)
        rst = torch.cat(rst, dim=1).unsqueeze(0)  # (1, 3, N, H, W)
        return rst

    def prepare_query(self, query, height, width):
        query_image = Image.open(query).convert("RGB")
        query_image = torch.from_numpy(np.array(query_image)).permute(2, 0, 1)
        query_image = v2.functional.resize(
            query_image,
            (height, width),
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True,
        )
        query_image = (query_image / 255.0 - 0.5) * 2
        query_image = query_image.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
        return query_image

    def encode_latents(self, x):
        x = x.to(self.vae_dtype)
        latent = self.vae.encode(x).sample()
        latent.sub_(0.032470703125).mul_(1.08936170213)
        return latent

    def decode_latents(self, latents):
        latents = latents.to(self.vae_dtype)
        latents.div_(1.08936170213).add_(0.032470703125)
        image = self.vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1).float()
        image = (image * 255.0).to(torch.uint8).cpu()
        return image

    def encode_cond(self, cond: torch.Tensor):
        cond = cond.cuda(non_blocking=True)
        cond = cond.permute(2, 1, 0, 3, 4)  # (T, C, 1, H, W)
        cond = self.encode_latents(cond)
        cond = cond.permute(2, 1, 0, 3, 4)  # (1, C, T, h, w)

        # uncondition
        B, C, T, h, w = cond.shape
        cond_null = torch.zeros(B, C, 1, h, w, device=cond.device, dtype=cond.dtype)
        cond_null = torch.cat([cond_null, cond[:, :, -1:]], dim=2)  # (1, C, 2, h, w)

        return (cond, cond_null)

    def generate(
        self,
        query: str,
        task_dir: str,
        height: int = 512,
        width: int = 512,
        num_prompt: int = 8,
        cfg_scale: float = 1.0,
    ):
        # prepare task prompts
        pair_list = self.get_task_prompt(task_dir)
        prompts_list = random.choices(pair_list, k=num_prompt)
        prompts = self.preprocess_prompt(prompts_list, height, width)  # (1, 3, N, H, W)

        # prepare query
        query_image = self.prepare_query(query, height, width)  # (1, 3, 1, H, W)

        cond = torch.cat([prompts, query_image], dim=2)  # (1, 3, N + 1, H, W)
        conds = self.encode_cond(cond)

        x = torch.randn(1, self.dit.in_channels, 1, height // 8, width // 8).repeat(
            2, 1, 1, 1, 1
        )
        x = x.cuda(non_blocking=True).to(self.dit_dtype)
        model_kwargs = dict(
            conds=conds,
            cfg_scale=cfg_scale,
        )
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16 if self.dit_dtype == torch.bfloat16 else torch.float32
        ):
            samples = self.sample_fn(x, self.dit.forward_with_cfg, **model_kwargs)[-1]
        samples = samples[:1]
        images = self.decode_latents(samples)
        query_image = (query_image + 1.0) * 127.5
        query_image = query_image.to(torch.uint8).cpu()
        return images, query_image


@click.command()
@click.option(
    "--vae_path",
    type=str,
    default="",
    help="Path to the vae checkpoint",
)
@click.option(
    "--dit_path",
    type=str,
    default="",
    help="Path to the dit checkpoint",
)
@click.option(
    "--timesteps", type=int, default=20, help="Number of timesteps for inference"
)
@click.option("--query", type=str, default="", help="Path to the query image")
@click.option("--output", type=str, default="", help="Path to the output image")
@click.option("--task_dir", type=str, default="", help="Path to the task directory")
@click.option("--height", type=int, default=512, help="Height of the image")
@click.option("--width", type=int, default=512, help="Width of the image")
@click.option("--num_prompt", type=int, default=4, help="Number of prompts")
@click.option("--cfg_scale", type=float, default=1.5, help="cfg_scale for inference")
def main(
    vae_path,
    dit_path,
    timesteps,
    query,
    output,
    task_dir,
    height,
    width,
    num_prompt,
    cfg_scale,
):
    engine = InferenceEngine(vae_path, dit_path, timesteps)
    images, query_image = engine.generate(
        query, task_dir, height, width, num_prompt, cfg_scale
    )
    result = torch.cat([query_image, images], dim=0).squeeze()  # (2, 3, H, W)
    result = make_grid(result, nrow=2, padding=4, normalize=False)
    write_png(result, output, compression_level=9)


if __name__ == "__main__":
    main()
