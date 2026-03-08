import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from Models.PaCoDi.CoupledTransformer import CoupledTransformer


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

class PaCoDi_sde(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            timesteps=1000,
            sampling_timesteps=250,
            beta_min=0.1,
            beta_max=20.0,
            cutoff_ratio=0.5,
            emb_size=128,
            patch_size=2,
            num_layers=4,
            device='cuda',
            **kwargs
    ):
        super().__init__()
        self.device = device
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.timesteps = float(timesteps)

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sampling_timesteps = int(default(sampling_timesteps, timesteps))
        self.cutoff_ratio = cutoff_ratio

        self.model = CoupledTransformer(
            feature_size=feature_size,
            emb_size=emb_size,
            patch_size=patch_size,
            num_layers=num_layers
        ).to(device)

        self.dc_predictor = nn.Sequential(
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, feature_size)
        )

        self.text_projection = nn.Linear(128, emb_size)

    def get_beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def get_marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean[:, None, None], std[:, None, None]

    def _apply_imag_constraints(self, x_imag):
        x_imag[:, 0, :] = 0
        if self.seq_length % 2 == 0:
            x_imag[:, -1, :] = 0
        return x_imag

    def _build_struct_weights(self, L_f, device):
        w_r = torch.ones(L_f, device=device)
        w_r[1:-1] = 0.5
        w_i = w_r.clone()
        w_i[0] = 0.0
        w_i[-1] = 0.0
        return w_r.view(1, -1, 1), w_i.view(1, -1, 1)

    def _build_keep_indices(self, L_f, device):
        if self.seq_length not in (128, 256):
            return torch.arange(L_f, device=device)

        cutoff = max(1, int(L_f * self.cutoff_ratio))
        keep = torch.zeros(L_f, dtype=torch.bool, device=device)
        keep[:cutoff] = True
        keep[-1] = True
        return torch.nonzero(keep, as_tuple=False).squeeze(1)

    def _gather_freq(self, x, keep_idx):
        return x.index_select(1, keep_idx)

    def _scatter_freq(self, x_cut, L_f, keep_idx):
        x_full = torch.zeros(x_cut.size(0), L_f, x_cut.size(2), device=x_cut.device)
        x_full[:, keep_idx, :] = x_cut
        return x_full


    def forward(self, data, **kwargs):
        device = self.device
        text_input = None
        if isinstance(data, (list, tuple)):
            x = data[0].to(device).float()
            text_emb = data[1].to(device).float()
        else:
            x = data.to(device).float()
            text_emb = None

        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.shape[1] > self.seq_length:
            x = x[:, :self.seq_length, :]

        loss_dc = 0.0
        if text_emb is not None:
            real_dc = x.mean(dim=1, keepdim=True)
            text_emb = self.text_projection(text_emb)
            pred_dc = self.dc_predictor(text_emb).unsqueeze(1)
            loss_dc = F.mse_loss(pred_dc, real_dc)
            x = x - real_dc

        x_freq = torch.fft.rfft(x, dim=1)
        x_real = x_freq.real.float()
        x_imag = x_freq.imag.float()

        L_full = x_real.shape[1]
        keep_idx = self._build_keep_indices(L_full, device)
        x_real = self._gather_freq(x_real, keep_idx)
        x_imag = self._gather_freq(x_imag, keep_idx)

        t = torch.rand(x_real.shape[0], device=device) * (1. - 1e-5) + 1e-5
        mean, std = self.get_marginal_prob(t)

        L_f = x_real.shape[1]
        w_r, w_i = self._build_struct_weights(L_f, device)

        noise_r = torch.randn_like(x_real) * torch.sqrt(w_r)
        noise_i = torch.randn_like(x_imag) * torch.sqrt(w_i)
        noise_i = self._apply_imag_constraints(noise_i)

        xt_r = x_real * mean + noise_r * std
        xt_i = x_imag * mean + noise_i * std

        if text_emb is not None:
            text_input = text_emb if torch.rand(1).item() >= 0.1 else None

        t_model = t * (self.timesteps - 1)
        pred_r, pred_i = self.model(
            input_r=xt_r,
            input_i=xt_i,
            t=t_model,
            text_input=text_input
        )

        pred_i = self._apply_imag_constraints(pred_i)

        L_f = pred_r.shape[1]
        w_r, w_i = self._build_struct_weights(L_f, device)

        inv_w_r = 1.0 / (w_r + 1e-7)
        inv_w_i = torch.zeros_like(w_i)
        mask_i = w_i > 0
        inv_w_i[mask_i] = 1.0 / w_i[mask_i]

        loss_r = (inv_w_r * (pred_r - noise_r) ** 2).mean()
        loss_i = (inv_w_i * (pred_i - noise_i) ** 2).mean()
        loss = loss_r + loss_i

        if text_emb is not None:
            loss = loss + loss_dc
        return loss

    @torch.no_grad()
    def _euler_maruyama_step(self, x_r, x_i, t, dt, text_input=None, cond=None, cfg_scale=0.0):
        batch = x_r.shape[0]
        t_batch = torch.full((batch,), t, device=self.device)
        mean, std = self.get_marginal_prob(t_batch)
        beta_t = self.get_beta(t_batch)[:, None, None]

        t_model = t_batch * (self.timesteps - 1)

        if cfg_scale > 0 and cond is not None:
            eps_uncond_r, eps_uncond_i = self.model(input_r=x_r, input_i=x_i, t=t_model, text_input=None)
            eps_cond_r, eps_cond_i = self.model(input_r=x_r, input_i=x_i, t=t_model, text_input=cond)
            eps_r = eps_uncond_r + cfg_scale * (eps_cond_r - eps_uncond_r)
            eps_i = eps_uncond_i + cfg_scale * (eps_cond_i - eps_uncond_i)
        else:
            eps_r, eps_i = self.model(input_r=x_r, input_i=x_i, t=t_model, text_input=text_input)

        eps_i = self._apply_imag_constraints(eps_i)

        L_f = x_r.shape[1]
        w_r, w_i = self._build_struct_weights(L_f, self.device)

        inv_w_r = 1.0 / (w_r + 1e-7)
        inv_w_i = torch.zeros_like(w_i)
        mask_i = w_i > 0
        inv_w_i[mask_i] = 1.0 / w_i[mask_i]

        std_safe = torch.clamp(std, min=1e-4)

        # drift_r = -0.5 * beta_t * x_r + 0.5 * beta_t * inv_w_r * (eps_r / ( std_safe + 1e-7))
        # drift_i = -0.5 * beta_t * x_i + 0.5 * beta_t * inv_w_i * (eps_i / ( std_safe + 1e-7))

        drift_r = -0.5 * beta_t * x_r + beta_t * (eps_r / (std_safe + 1e-7))
        drift_i = -0.5 * beta_t * x_i + beta_t * (eps_i / (std_safe + 1e-7))

        dt_abs = torch.tensor(abs(dt), device=self.device)
        diffusion_scale = torch.sqrt(beta_t)

        dw_r = torch.randn_like(x_r) * torch.sqrt(dt_abs) * torch.sqrt(w_r)
        dw_i = torch.randn_like(x_i) * torch.sqrt(dt_abs) * torch.sqrt(w_i)
        dw_i = self._apply_imag_constraints(dw_i)

        prev_x_r = x_r + drift_r * dt + diffusion_scale * dw_r
        prev_x_i = x_i + drift_i * dt + diffusion_scale * dw_i
        return prev_x_r, prev_x_i

    @torch.no_grad()
    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        L_full = self.seq_length // 2 + 1
        keep_idx = self._build_keep_indices(L_full, self.device)
        L_f = keep_idx.numel()

        w_r, w_i = self._build_struct_weights(L_f, self.device)

        xt_real = torch.randn(batch_size, L_f, self.feature_size, device=self.device) * torch.sqrt(w_r)
        xt_imag = torch.randn(batch_size, L_f, self.feature_size, device=self.device) * torch.sqrt(w_i)
        xt_imag = self._apply_imag_constraints(xt_imag)

        num_steps = self.sampling_timesteps
        step_size = 1.0 / num_steps
        time_steps = np.linspace(1.0, 1e-5, num_steps)

        for t in tqdm(time_steps, desc='SDE Sampling'):
            dt = -step_size
            xt_real, xt_imag = self._euler_maruyama_step(xt_real, xt_imag, t, dt)
            xt_imag = self._apply_imag_constraints(xt_imag)

        x_freq_final = torch.complex(
            self._scatter_freq(xt_real, L_full, keep_idx),
            self._scatter_freq(xt_imag, L_full, keep_idx)
        )

        x_freq_final = torch.complex(xt_real, xt_imag)

        x_final = torch.fft.irfft(x_freq_final, n=self.seq_length, dim=1)
        return x_final



    @torch.no_grad()
    def generate_text(self, batch_size=16, x_raw=None, cond=None, sampling_steps=50, cfg_scale=7.5):
        num_steps = sampling_steps
        L_full = self.seq_length // 2 + 1
        keep_idx = self._build_keep_indices(L_full, self.device)
        L_f = keep_idx.numel()

        w_r, w_i = self._build_struct_weights(L_f, self.device)

        cond = self.text_projection(cond)

        xt_real = torch.randn(batch_size, L_f, self.feature_size, device=self.device) * torch.sqrt(w_r)
        xt_imag = torch.randn(batch_size, L_f, self.feature_size, device=self.device) * torch.sqrt(w_i)
        xt_imag = self._apply_imag_constraints(xt_imag)

        step_size = 1.0 / num_steps
        time_steps = np.linspace(1.0, 1e-5, num_steps)

        for t in tqdm(time_steps, desc='Text-Guided SDE'):
            dt = -step_size
            xt_real, xt_imag = self._euler_maruyama_step(
                xt_real, xt_imag, t, dt, text_input=None, cond=cond, cfg_scale=cfg_scale
            )
            xt_imag = self._apply_imag_constraints(xt_imag)

        x_freq_final = torch.complex(
            self._scatter_freq(xt_real, L_full, keep_idx),
            self._scatter_freq(xt_imag, L_full, keep_idx)
        )
        x_ac = torch.fft.irfft(x_freq_final, n=self.seq_length, dim=1)

        if cond is not None:
            pred_dc = self.dc_predictor(cond).unsqueeze(1)
            x_ac = x_ac + pred_dc
        return x_ac, x_raw
