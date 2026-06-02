import torch
import torch.nn.functional as F
from typing import Optional


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def gather_with_terminal(consts: torch.Tensor, t: torch.Tensor, terminal_value: float):
    safe_t = torch.clamp(t, min=0)
    values = gather(consts, safe_t)
    terminal = torch.full_like(values, terminal_value)
    return torch.where(t.reshape(-1, 1, 1) < 0, terminal, values)


class DDPM:
    def __init__(self, total_steps: int, device: torch.device):
        super().__init__()
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, total_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.total_steps = total_steps
        self.sigma2 = self.beta #sigma^2 = beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean.to(self.device), var.to(self.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> [torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(x0).to(self.device)
        mean, var = self.q_xt_x0(x0, t)
        return (mean + (var**0.5)*eps).to(self.device)

    def p_sample(self, xt: torch.Tensor, n_xt: torch.Tensor, t: torch.Tensor, noise_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        # n_xt = pred_noise_xt = eps_theta
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * n_xt)

        if (t == 0).all():
            return mean

        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        if noise_scale is not None:
            eps = eps * noise_scale
        return mean + (var ** .5) * eps

    def ddim_sample(
        self,
        xt: torch.Tensor,
        eps_theta: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        noise_scale: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        alpha_bar_t = gather(self.alpha_bar, t)
        alpha_bar_next = gather_with_terminal(self.alpha_bar, t_next, terminal_value=1.0)

        pred_x0 = (xt - torch.sqrt(1.0 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)

        eta = float(eta)
        if eta > 0:
            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t), min=0.0)
            ) * torch.sqrt(torch.clamp(1.0 - alpha_bar_t / alpha_bar_next, min=0.0))
        else:
            sigma = torch.zeros_like(alpha_bar_t)

        eps_coef = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma.pow(2), min=0.0))
        x_next = torch.sqrt(alpha_bar_next) * pred_x0 + eps_coef * eps_theta

        if eta > 0 and not (t_next < 0).all():
            noise = torch.randn_like(xt)
            if noise_scale is not None:
                noise = noise * noise_scale
            x_next = x_next + sigma * noise

        return x_next

    def loss(self, n_gt: torch.Tensor, n_xt: torch.Tensor):
        # noise_gt : x_1 - x_0
        loss = F.mse_loss(n_xt, n_gt)
        return loss
