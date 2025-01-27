import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from tqdm import tqdm


class sde:
    """SDE solver class"""

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return (
            xhat + 0.5 * self.dt * (K1 + K2),
            xhat,
        )  # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_shifting_factor=None,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # self.t = th.linspace(t0, t1, num_steps)
        # if time_shifting_factor:
        #     self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
        if num_steps == 20:
            self.t = th.tensor(
                [
                    0.0000,
                    0.0056,
                    0.0111,
                    0.0167,
                    0.0222,
                    0.0278,
                    0.0333,
                    0.0389,
                    0.0444,
                    0.0500,
                    0.0524,
                    0.0730,
                    0.1148,
                    0.1777,
                    0.2618,
                    0.3671,
                    0.4936,
                    0.6412,
                    0.8100,
                    1.0000,
                ]
            )
        if num_steps == 30:
            self.t = th.tensor(
                [
                    0.0000,
                    0.0036,
                    0.0071,
                    0.0107,
                    0.0143,
                    0.0179,
                    0.0214,
                    0.0250,
                    0.0286,
                    0.0321,
                    0.0357,
                    0.0393,
                    0.0429,
                    0.0464,
                    0.0500,
                    0.0511,
                    0.0600,
                    0.0779,
                    0.1049,
                    0.1410,
                    0.1862,
                    0.2403,
                    0.3036,
                    0.3759,
                    0.4573,
                    0.5477,
                    0.6472,
                    0.7557,
                    0.8733,
                    1.0000,
                ]
            )
        if num_steps == 40:
            self.t = th.tensor(
                [
                    0.0000,
                    0.0026,
                    0.0053,
                    0.0079,
                    0.0105,
                    0.0132,
                    0.0158,
                    0.0184,
                    0.0211,
                    0.0237,
                    0.0263,
                    0.0289,
                    0.0316,
                    0.0342,
                    0.0368,
                    0.0395,
                    0.0421,
                    0.0447,
                    0.0474,
                    0.0500,
                    0.0506,
                    0.0555,
                    0.0655,
                    0.0804,
                    0.1004,
                    0.1253,
                    0.1553,
                    0.1902,
                    0.2302,
                    0.2752,
                    0.3251,
                    0.3801,
                    0.4401,
                    0.5051,
                    0.5750,
                    0.6500,
                    0.7300,
                    0.8150,
                    0.9050,
                    1.0000,
                ]
            )

        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):

        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples
