import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_sinusoidal_positional_embeddings(num_positions, d_model):
    position = torch.arange(num_positions).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).unsqueeze(0)
    pos_embedding = torch.zeros(num_positions, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding.unsqueeze(0)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.unsqueeze(-1)
        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        return embedding.squeeze(1)


class DiTSolverV1Layer(nn.Module):
    def __init__(self, d_model=128, real_imag_interaction=True):
        super().__init__()
        self.real_imag_interaction = real_imag_interaction
        mlp_ratio = 2.0
        mlp_hidden_dim = int(d_model * mlp_ratio)
        mlp_input_dim = d_model * 2 if real_imag_interaction else d_model
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1_r = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2_r = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn_r = Attention(d_model, num_heads=4, qkv_bias=True)
        self.mlp_r = Mlp(
            in_features=mlp_input_dim,
            hidden_features=mlp_hidden_dim,
            out_features=d_model,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_r = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

        self.norm1_i = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2_i = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn_i = Attention(d_model, num_heads=4, qkv_bias=True)
        self.mlp_i = Mlp(
            in_features=mlp_input_dim,
            hidden_features=mlp_hidden_dim,
            out_features=d_model,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_i = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

    def forward(self, x_r, x_i, c):
        s_msa_r, sc_msa_r, g_msa_r, s_mlp_r, sc_mlp_r, g_mlp_r = self.adaLN_r(c).chunk(6, dim=1)
        x_r = x_r + g_msa_r.unsqueeze(1) * self.attn_r(modulate(self.norm1_r(x_r), s_msa_r, sc_msa_r))

        s_msa_i, sc_msa_i, g_msa_i, s_mlp_i, sc_mlp_i, g_mlp_i = self.adaLN_i(c).chunk(6, dim=1)
        x_i = x_i + g_msa_i.unsqueeze(1) * self.attn_i(modulate(self.norm1_i(x_i), s_msa_i, sc_msa_i))

        norm_x_r = modulate(self.norm2_r(x_r), s_mlp_r, sc_mlp_r)
        norm_x_i = modulate(self.norm2_i(x_i), s_mlp_i, sc_mlp_i)

        if self.real_imag_interaction:
            mlp_input_r = torch.cat([norm_x_r, norm_x_i], dim=-1)
            mlp_input_i = mlp_input_r
        else:
            mlp_input_r = norm_x_r
            mlp_input_i = norm_x_i

        x_r = x_r + g_mlp_r.unsqueeze(1) * self.mlp_r(mlp_input_r)
        x_i = x_i + g_mlp_i.unsqueeze(1) * self.mlp_i(mlp_input_i)
        return x_r, x_i


class DiTSolverV1(nn.Module):
    def __init__(self, channels=5, emb_size=128, patch_size=1, num_layers=4, real_imag_interaction=True):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.channels = channels
        self.real_imag_interaction = real_imag_interaction

        self.conv_r = nn.Conv1d(channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.conv_i = nn.Conv1d(channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = None

        self.layers = nn.ModuleList(
            [
                DiTSolverV1Layer(
                    emb_size,
                    real_imag_interaction=real_imag_interaction,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_r = nn.LayerNorm(emb_size)
        self.ln_i = nn.LayerNorm(emb_size)
        self.linear_emb_to_patch_r = nn.Linear(emb_size, patch_size * channels)
        self.linear_emb_to_patch_i = nn.Linear(emb_size, patch_size * channels)
        self.time_emb = TimeEmbedding(dim=emb_size)

        self.initialize_weights()

    def forward(self, input_r: torch.Tensor, input_i: torch.Tensor, t: torch.Tensor, condition=None):
        batch_size, seq_length, channels = input_r.shape
        pad_len = (self.patch_size - seq_length % self.patch_size) % self.patch_size
        if pad_len > 0:
            input_r = F.pad(input_r, (0, 0, 0, pad_len))
            input_i = F.pad(input_i, (0, 0, 0, pad_len))

        x_r = self.conv_r(input_r.permute(0, 2, 1))
        x_i = self.conv_i(input_i.permute(0, 2, 1))

        patch_count = x_r.size(2)
        x_r, x_i = x_r.permute(0, 2, 1), x_i.permute(0, 2, 1)

        if self.pos_embed is None or self.pos_embed.size(1) != patch_count:
            self.pos_embed = get_sinusoidal_positional_embeddings(patch_count, self.emb_size).to(x_r.device)
        x_r = x_r + self.pos_embed
        x_i = x_i + self.pos_embed

        t = self.time_emb(t)
        c = t + condition if condition is not None else t

        for layer in self.layers:
            x_r, x_i = layer(x_r, x_i, c)

        x_r, x_i = self.ln_r(x_r), self.ln_i(x_i)
        x_r, x_i = self.linear_emb_to_patch_r(x_r), self.linear_emb_to_patch_i(x_i)

        x_r = x_r.reshape(batch_size, patch_count * self.patch_size, channels)[:, :seq_length, :]
        x_i = x_i.reshape(batch_size, patch_count * self.patch_size, channels)[:, :seq_length, :]
        return x_r, x_i

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        for block in self.layers:
            nn.init.constant_(block.adaLN_r[-1].weight, 0)
            nn.init.constant_(block.adaLN_r[-1].bias, 0)
            nn.init.constant_(block.adaLN_i[-1].weight, 0)
            nn.init.constant_(block.adaLN_i[-1].bias, 0)


DiT_Solver_V1 = DiTSolverV1
