import sys

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from models.pacodi.backbones import build_backbone
from models.pacodi.normalization import InstanceNormalizationMixin


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


class PaCoDi_sde(nn.Module, InstanceNormalizationMixin):
    def __init__(
        self,
        seq_length,
        channels,
        max_step=1000,
        sampling_steps=250,
        beta_min=0.1,
        beta_max=20.0,
        condition_size=128,
        backbone="dit_solver_v1",
        real_imag_interaction=True,
        instance_norm=False,
        instance_norm_eps=1e-5,
        frequency_keep_ratio=1.0,
        frequency_keep_bins=None,
        emb_size=128,
        patch_size=1,
        num_layers=4,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.seq_length = seq_length
        self.channels = channels
        self.max_step = float(max_step)
        self.sampling_steps = int(default(sampling_steps, max_step))
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.instance_norm = instance_norm
        self.instance_norm_eps = instance_norm_eps
        
        self.full_spectrum_bins = self.seq_length // 2 + 1
        if frequency_keep_bins is None:
            frequency_keep_bins = round(self.full_spectrum_bins * float(frequency_keep_ratio))
        self.frequency_keep_bins = max(1, min(self.full_spectrum_bins, int(frequency_keep_bins)))
        self.keep_nyquist = self.seq_length % 2 == 0 and self.frequency_keep_bins < self.full_spectrum_bins
        self.modeled_spectrum_bins = self.frequency_keep_bins + int(self.keep_nyquist)

        self.model = build_backbone(
            backbone,
            channels=channels,
            emb_size=emb_size,
            patch_size=patch_size,
            num_layers=num_layers,
            real_imag_interaction=real_imag_interaction,
            **kwargs,
        )
        self.condition_projection = nn.Linear(condition_size, emb_size)
        self.to(device)

    def project_condition(self, condition):
        condition = condition.to(self.device).float()
        if condition.ndim > 2:
            condition = condition.reshape(condition.shape[0], -1)
        return self.condition_projection(condition)

    def get_beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def get_marginal_prob(self, t):
        log_mean = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        # mean = sqrt(alpha_bar_t), noise_scale = sqrt(1 - alpha_bar_t)
        mean = torch.exp(log_mean)
        noise_scale = torch.sqrt(1.0 - torch.exp(2.0 * log_mean))
        return mean[:, None, None], noise_scale[:, None, None]

    def _spectral_stats(self, num_bins, device, dtype=torch.float32, has_nyquist=None):
        if has_nyquist is None:
            has_nyquist = self.seq_length % 2 == 0 and num_bins == self.full_spectrum_bins

        endpoint = torch.zeros(num_bins, dtype=torch.bool, device=device)
        endpoint[0] = True
        if has_nyquist:
            endpoint[-1] = True

        var_r = torch.full((1, num_bins, 1), 0.5, device=device, dtype=dtype)
        var_i = torch.full((1, num_bins, 1), 0.5, device=device, dtype=dtype)
        var_r[:, endpoint, :] = 1.0
        var_i[:, endpoint, :] = 0.0

        inv_r = 1.0 / torch.clamp(var_r, min=1e-7)
        inv_i = torch.zeros_like(var_i)
        inv_i[var_i > 0] = 1.0 / var_i[var_i > 0]
        return {
            "var_r": var_r,
            "var_i": var_i,
            "inv_r": inv_r,
            "inv_i": inv_i,
            "full_bins": self.full_spectrum_bins,
            "modeled_bins": num_bins,
            "low_bins": num_bins if num_bins == self.full_spectrum_bins else num_bins - int(has_nyquist),
            "has_nyquist": has_nyquist,
        }

    def _to_spectrum(self, x):
        spectrum = torch.fft.rfft(x, dim=1, norm="ortho")
        if self.keep_nyquist:
            spectrum = torch.cat(
                [spectrum[:, : self.frequency_keep_bins, :], spectrum[:, -1:, :]],
                dim=1,
            )
        else:
            spectrum = spectrum[:, : self.frequency_keep_bins, :]
        stats = self._spectral_stats(
            spectrum.shape[1],
            spectrum.device,
            spectrum.real.dtype,
            has_nyquist=self.keep_nyquist or spectrum.shape[1] == self.full_spectrum_bins,
        )
        return spectrum.real.float(), spectrum.imag.float(), stats

    def _pad_spectrum(self, x_r, x_i, stats):
        full_bins = stats["full_bins"]
        if x_r.shape[1] == full_bins:
            return x_r, x_i

        padded_r = torch.zeros(x_r.shape[0], full_bins, x_r.shape[2], device=x_r.device, dtype=x_r.dtype)
        padded_i = torch.zeros_like(padded_r)
        low_bins = stats["low_bins"]
        padded_r[:, :low_bins, :] = x_r[:, :low_bins, :]
        padded_i[:, :low_bins, :] = x_i[:, :low_bins, :]
        if stats["has_nyquist"]:
            padded_r[:, -1:, :] = x_r[:, -1:, :]
        full_stats = self._spectral_stats(full_bins, x_r.device, x_r.dtype)
        return padded_r, padded_i * (full_stats["var_i"] > 0)

    def _to_time(self, x_r, x_i, stats):
        x_r, x_i = self._pad_spectrum(x_r, x_i, stats)
        stats = self._spectral_stats(x_r.shape[1], x_r.device, x_r.dtype)
        spectrum = torch.complex(x_r, x_i * (stats["var_i"] > 0))
        return torch.fft.irfft(spectrum, n=self.seq_length, dim=1, norm="ortho")

    def _spectral_noise_like(self, ref, stats):
        noise_r = torch.randn_like(ref) * torch.sqrt(stats["var_r"])
        noise_i = torch.randn_like(ref) * torch.sqrt(stats["var_i"])
        return noise_r, noise_i

    def _predict_noise(self, x_r, x_i, t, stats, condition=None, cfg_scale=0.0):
        t_model = t * (self.max_step - 1)
        if cfg_scale > 0 and condition is not None:
            uncond_r, uncond_i = self.model(x_r, x_i, t_model, condition=None)
            cond_r, cond_i = self.model(x_r, x_i, t_model, condition=condition)
            pred_r = uncond_r + cfg_scale * (cond_r - uncond_r)
            pred_i = uncond_i + cfg_scale * (cond_i - uncond_i)
        else:
            pred_r, pred_i = self.model(x_r, x_i, t_model, condition=condition)
        return pred_r, pred_i * (stats["var_i"] > 0)

    def _noise_loss(self, pred_r, pred_i, noise_r, noise_i, stats):
        loss_r = (stats["inv_r"] * (pred_r - noise_r) ** 2).mean()
        loss_i = (stats["inv_i"] * (pred_i - noise_i) ** 2).mean()
        return loss_r + loss_i

    def forward(self, data, **kwargs):
        if isinstance(data, (list, tuple)):
            x, condition = data[0], data[1]
        else:
            x, condition = data, None

        x = x.to(self.device).float()

        cond_emb = None
        if condition is not None:
            cond_emb = self.project_condition(condition)

        if self.instance_norm:
            x, _ = self.instance_normalize(x)

        x_r, x_i, stats = self._to_spectrum(x)
        t = torch.rand(x_r.shape[0], device=self.device) * (1.0 - 1e-5) + 1e-5
        mean, noise_scale = self.get_marginal_prob(t)

        noise_r, noise_i = self._spectral_noise_like(x_r, stats)
        xt_r = x_r * mean + noise_r * noise_scale
        xt_i = x_i * mean + noise_i * noise_scale

        condition = cond_emb if cond_emb is not None and torch.rand(1).item() >= 0.1 else None
        pred_r, pred_i = self._predict_noise(xt_r, xt_i, t, stats, condition=condition)
        return self._noise_loss(pred_r, pred_i, noise_r, noise_i, stats)

    @torch.no_grad()
    def _euler_maruyama_step(self, x_r, x_i, t_value, dt, stats, condition=None, cfg_scale=0.0):
        batch = x_r.shape[0]
        t = torch.full((batch,), float(t_value), device=self.device)
        _, noise_scale = self.get_marginal_prob(t)
        beta_t = self.get_beta(t)[:, None, None]
        g_t = torch.sqrt(beta_t)

        eps_r, eps_i = self._predict_noise(
            x_r,
            x_i,
            t,
            stats,
            condition=condition,
            cfg_scale=cfg_scale,
        )
        noise_scale = torch.clamp(noise_scale, min=1e-4)

        f_r = -0.5 * beta_t * x_r
        f_i = -0.5 * beta_t * x_i
        # Heteroscedastic score-noise identity:
        # score = -Sigma^{-1} eps / noise_scale, so Sigma @ score cancels to -eps / noise_scale.
        # The spectral covariance still appears in dW through var_r/var_i below.
        sigma_score_r = -eps_r / (noise_scale + 1e-7)
        sigma_score_i = -eps_i / (noise_scale + 1e-7)
        reverse_drift_r = f_r - g_t.pow(2) * sigma_score_r
        reverse_drift_i = f_i - g_t.pow(2) * sigma_score_i

        dt_scale = torch.sqrt(torch.tensor(abs(dt), device=self.device))
        dw_r = torch.randn_like(x_r) * dt_scale * torch.sqrt(stats["var_r"])
        dw_i = torch.randn_like(x_i) * dt_scale * torch.sqrt(stats["var_i"])

        next_r = x_r + reverse_drift_r * dt + g_t * dw_r
        next_i = x_i + reverse_drift_i * dt + g_t * dw_i
        return next_r, next_i * (stats["var_i"] > 0)

    @torch.no_grad()
    def _sample_spectrum(self, batch_size, num_steps, condition=None, cfg_scale=0.0, desc="SDE Sampling"):
        num_bins = self.modeled_spectrum_bins
        stats = self._spectral_stats(
            num_bins,
            self.device,
            has_nyquist=self.keep_nyquist or num_bins == self.full_spectrum_bins,
        )
        ref = torch.empty(batch_size, num_bins, self.channels, device=self.device)
        x_r, x_i = self._spectral_noise_like(ref, stats)

        time_steps = torch.linspace(1.0, 1e-5, int(num_steps) + 1, device=self.device)
        for step in tqdm(range(int(num_steps)), desc=desc, disable=not sys.stderr.isatty()):
            t_value = float(time_steps[step].item())
            dt = float((time_steps[step + 1] - time_steps[step]).item())
            x_r, x_i = self._euler_maruyama_step(
                x_r,
                x_i,
                t_value,
                dt,
                stats,
                condition=condition,
                cfg_scale=cfg_scale,
            )
        return x_r, x_i, stats

    @torch.no_grad()
    def generate_unconditional(self, batch_size=16, instance_stats=None):
        x_r, x_i, stats = self._sample_spectrum(
            batch_size=batch_size,
            num_steps=self.sampling_steps,
            desc="SDE Sampling",
        )
        x = self._to_time(x_r, x_i, stats)
        if self.instance_norm:
            x = self.instance_normalize(x, stats=instance_stats, inverse=True)
        return x

    @torch.no_grad()
    def generate_conditional(self, batch_size=16, cond=None, sampling_steps=50, cfg_scale=7.5, instance_stats=None):
        if cond is None:
            raise ValueError("generate_conditional requires a condition tensor.")
        cond_emb = self.project_condition(cond)
        x_r, x_i, stats = self._sample_spectrum(
            batch_size=batch_size,
            num_steps=sampling_steps,
            condition=cond_emb,
            cfg_scale=cfg_scale,
            desc="Conditional SDE",
        )
        x = self._to_time(x_r, x_i, stats)
        if self.instance_norm:
            x = self.instance_normalize(x, stats=instance_stats, inverse=True)
        return x
