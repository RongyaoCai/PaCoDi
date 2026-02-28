import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from Models.PaCoDi.CoupledTransformer import CoupledTransformer
from Models.PaCoDi.DDPM import DDPM

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

class PaCoDi_ddpm(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            timesteps=1000,
            sampling_timesteps=1000,
            beta_schedule='cosine',
            emb_size=128,
            patch_size=2,
            num_layers=4,
            eta=0.,
            use_ff=True,
            cutoff_ratio=0.5,
            device='cuda',
            **kwargs
    ):
        super().__init__()
        self.device = device
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.timesteps = timesteps
        self.num_timesteps = int(timesteps)

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

        self.ddpm = DDPM(timesteps, device)

        self.text_projection = nn.Linear(128, emb_size)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.fast_sampling = self.sampling_timesteps < timesteps
        self.cutoff_ratio = cutoff_ratio

    def _build_struct_weights(self, L_f, device):
        w_r = torch.ones(L_f, device=device)
        w_r[1:-1] = 0.5
        w_i = w_r.clone()
        w_i[0] = 0.0
        if self.seq_length % 2 == 0:
            w_i[-1] = 0.0
        inv_w_r = 1.0 / w_r
        inv_w_i = torch.zeros_like(w_i)
        mask_i = w_i > 0
        inv_w_i[mask_i] = 1.0 / w_i[mask_i]

        return inv_w_r.view(1, -1, 1), inv_w_i.view(1, -1, 1)

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

    def forward(self, data,**kwargs):
        device = self.device
        text_input = None
        if isinstance(data, (list, tuple)):
            x = data[0].to(device).float()  # raw time-series data
            text_emb = data[1].to(device).float()  # text_Embedding
        else:
            x = data.to(device)
            text_emb = None

        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if x.shape[1] > self.seq_length:
            x = x[:, :self.seq_length, :]

        B, L, M = x.shape

        loss_dc = 0.0
        if text_emb is not None:
            real_dc = x.mean(dim=1, keepdim=True)  # shape: [B, 1, M]
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

        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise_r = torch.randn_like(x_real).to(device)
        noise_i = torch.randn_like(x_imag).to(device)

        noise_i[:, 0, :] = 0  # DC
        if self.seq_length % 2 == 0:
            noise_i[:, -1, :] = 0  # Nyquist

        xt_r = self.ddpm.q_sample(x_real, t, noise_r)
        xt_i = self.ddpm.q_sample(x_imag, t, noise_i)

        if text_emb is not None:
            decide = torch.rand(1).item() < 0.1
            if not decide:
                text_input = text_emb
            else:
                text_input = None

        pred_r, pred_i = self.model(
            input_r=xt_r,
            input_i=xt_i,
            t=t,
            text_input=text_input
        )

        pred_i[:, 0, :] = 0
        if self.seq_length % 2 == 0:
            pred_i[:, -1, :] = 0

        L_f = x_real.shape[1]
        inv_w_r,inv_w_i = self._build_struct_weights(L_f, device)

        loss_r = (inv_w_r * (pred_r - noise_r) ** 2).mean()
        loss_i = (inv_w_i * (pred_i - noise_i) ** 2).mean()
        loss = loss_r + loss_i

        if text_emb is not None:
            loss = loss+loss_dc
        return loss

    @torch.no_grad()
    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        """unconditional generation"""
        L_full = self.seq_length // 2 + 1
        keep_idx = self._build_keep_indices(L_full, self.device)
        L_f = keep_idx.numel()

        xt_real = torch.randn(batch_size, L_f, self.feature_size, device=self.device)
        xt_imag = torch.randn(batch_size, L_f, self.feature_size, device=self.device)

        xt_imag[:, 0, :] = 0
        if self.seq_length % 2 == 0:
            xt_imag[:, -1, :] = 0

        for t in tqdm(range(self.sampling_timesteps), desc='Sampling'):
            t_val = self.sampling_timesteps - 1 - t
            t_batch = torch.full((batch_size,), t_val, dtype=torch.long, device=self.device)

            n_real, n_imag = self.model(input_r=xt_real, input_i=xt_imag, t=t_batch, text_input=None)

            xt_real = self.ddpm.p_sample(xt_real, n_real, t_batch)
            xt_imag = self.ddpm.p_sample(xt_imag, n_imag, t_batch)

            xt_imag[:, 0, :] = 0
            if self.seq_length % 2 == 0:
                xt_imag[:, -1, :] = 0

        x_freq_full = torch.complex(
            self._scatter_freq(xt_real, L_full, keep_idx),
            self._scatter_freq(xt_imag, L_full, keep_idx)
        )
        x_final = torch.fft.irfft(x_freq_full, n=self.seq_length, dim=1)
        return x_final

    # @torch.no_grad()
    # def generate_text(self, batch_size=16, x_raw=None,cond=None, sampling_steps=50, cfg_scale=7.5):
    #     """
    #     text_conditional generation
    #     """
    #     device = self.device
    #     L_f = self.seq_length // 2 + 1
    #     M = self.feature_size
    #     step = sampling_steps
    #
    #     if x_raw.ndim == 2:
    #         x_raw = x_raw.unsqueeze(-1)
    #     if x_raw.shape[1] > self.seq_length:
    #         x_raw = x_raw[:, :self.seq_length]
    #
    #     cond = self.text_projection(cond)
    #
    #     x_raw_freq = torch.fft.rfft(x_raw, dim=1)
    #
    #     x_real = x_raw_freq.real.float()
    #     x_imag = x_raw_freq.imag.float()
    #
    #     x_t_real = torch.randn_like(x_real).to(device)
    #     x_t_imag = torch.randn_like(x_imag).to(device)
    #
    #     for j in tqdm(range(step), desc="Generating TS via Text"):
    #
    #         t_real = torch.full((x_t_real.size(0),), math.floor(step - 1 - j), dtype=torch.long, device=device)
    #         t_imag = torch.full((x_t_imag.size(0),), math.floor(step - 1 - j), dtype=torch.long, device=device)
    #
    #         pred_uncond_real,pred_uncond_imag = self.model(input_r=x_t_real, input_i=x_t_imag, t=t_real, text_input=None)
    #         pred_cond_real,pred_cond_imag = self.model(input_r=x_t_real, input_i=x_t_imag, t=t_real, text_input=cond)
    #
    #         pred_real = pred_uncond_real + cfg_scale * (pred_cond_real - pred_uncond_real)
    #         pred_imag = pred_uncond_imag + cfg_scale * (pred_cond_imag - pred_uncond_imag)
    #
    #         x_t_real = self.ddpm.p_sample(x_t_real, pred_real, t_real)
    #         x_t_imag = self.ddpm.p_sample(x_t_imag, pred_imag, t_imag)
    #
    #     x_freq_final = torch.complex(x_t_real, x_t_imag)
    #     x_ac = torch.fft.irfft(x_freq_final, n=self.seq_length, dim=1)
    #
    #     pred_dc = self.dc_predictor(cond).unsqueeze(1)
    #     x_final = x_ac + pred_dc
    #
    #     return x_final,x_raw

    @torch.no_grad()
    def generate_text(self, batch_size=16, x_raw=None,cond=None, sampling_steps=50, cfg_scale=7.5):
        """
        text_conditional generation
        """
        device = self.device
        L_f = self.seq_length // 2 + 1
        M = self.feature_size
        step = sampling_steps

        if x_raw.ndim == 2:
            x_raw = x_raw.unsqueeze(-1)
        if x_raw.shape[1] > self.seq_length:
            x_raw = x_raw[:, :self.seq_length]

        cond = self.text_projection(cond)

        x_raw_freq = torch.fft.rfft(x_raw, dim=1)

        x_real = x_raw_freq.real.float()
        x_imag = x_raw_freq.imag.float()

        L_full = x_real.shape[1]
        keep_idx = self._build_keep_indices(L_full, device)
        x_real = self._gather_freq(x_real, keep_idx)
        x_imag = self._gather_freq(x_imag, keep_idx)

        L_f = x_real.shape[1]

        x_t_real = torch.randn_like(x_real).to(device)
        x_t_imag = torch.randn_like(x_imag).to(device)

        for j in tqdm(range(step), desc="Generating TS via Text"):
            t_real = torch.full((x_t_real.size(0),), math.floor(step - 1 - j), dtype=torch.long, device=device)
            t_imag = torch.full((x_t_imag.size(0),), math.floor(step - 1 - j), dtype=torch.long, device=device)

            pred_uncond_real,pred_uncond_imag = self.model(input_r=x_t_real, input_i=x_t_imag, t=t_real, text_input=None)
            pred_cond_real,pred_cond_imag = self.model(input_r=x_t_real, input_i=x_t_imag, t=t_real, text_input=cond)

            pred_real = pred_uncond_real + cfg_scale * (pred_cond_real - pred_uncond_real)
            pred_imag = pred_uncond_imag + cfg_scale * (pred_cond_imag - pred_uncond_imag)

            x_t_real = self.ddpm.p_sample(x_t_real, pred_real, t_real)
            x_t_imag = self.ddpm.p_sample(x_t_imag, pred_imag, t_imag)

        x_freq_full = torch.complex(
            self._scatter_freq(x_t_real, L_full, keep_idx),
            self._scatter_freq(x_t_imag, L_full, keep_idx)
        )
        x_ac = torch.fft.irfft(x_freq_full, n=self.seq_length, dim=1)

        pred_dc = self.dc_predictor(cond).unsqueeze(1)
        x_final = x_ac + pred_dc

        return x_final,x_raw
