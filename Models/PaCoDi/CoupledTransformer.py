import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import math
import torch.nn.functional as F

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
        super(TimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        #t = t * 100.0
        t = t.unsqueeze(-1)
        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        embedding = embedding.squeeze(1)
        return embedding

class CoupledTransformerLayer(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        mlp_ratio = 2.0
        mlp_hidden_dim = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1_r = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2_r = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn_r = Attention(d_model, num_heads=4, qkv_bias=True)
        self.mlp_r = Mlp(in_features=d_model * 2, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=approx_gelu, drop=0)
        self.adaLN_r = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

        self.norm1_i = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2_i = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn_i = Attention(d_model, num_heads=4, qkv_bias=True)
        self.mlp_i = Mlp(in_features=d_model * 2, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=approx_gelu, drop=0)
        self.adaLN_i = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

        # Learnable fusion scaling factor
        self.fusion_r2i = nn.Parameter(torch.full((1,), 0.1))
        self.fusion_i2r = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x_r, x_i, c):
        s_msa_r, sc_msa_r, g_msa_r, s_mlp_r, sc_mlp_r, g_mlp_r = self.adaLN_r(c).chunk(6, dim=1)
        x_r = x_r + g_msa_r.unsqueeze(1) * self.attn_r(modulate(self.norm1_r(x_r), s_msa_r, sc_msa_r))

        s_msa_i, sc_msa_i, g_msa_i, s_mlp_i, sc_mlp_i, g_mlp_i = self.adaLN_i(c).chunk(6, dim=1)
        x_i = x_i + g_msa_i.unsqueeze(1) * self.attn_i(modulate(self.norm1_i(x_i), s_msa_i, sc_msa_i))

        norm_x_r = modulate(self.norm2_r(x_r), s_mlp_r, sc_mlp_r)
        norm_x_i = modulate(self.norm2_i(x_i), s_mlp_i, sc_mlp_i)

        x_combined = torch.cat([norm_x_r, norm_x_i], dim=-1)

        x_r = x_r + g_mlp_r.unsqueeze(1) * self.mlp_r(x_combined)
        x_i = x_i + g_mlp_i.unsqueeze(1) * self.mlp_i(x_combined)

        return x_r, x_i


class CoupledTransformer(nn.Module):
    def __init__(self, feature_size=5, emb_size=128, patch_size=2, num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.feature_size = feature_size
        self.conv_r = nn.Conv1d(feature_size, emb_size, kernel_size=patch_size, stride=patch_size)
        self.conv_i = nn.Conv1d(feature_size, emb_size, kernel_size=patch_size, stride=patch_size)
        self.text_projection = nn.Linear(128, emb_size)
        self.pos_embed = None

        self.layers = nn.ModuleList([CoupledTransformerLayer(emb_size) for _ in range(num_layers)])

        self.ln_r = nn.LayerNorm(emb_size)
        self.ln_i = nn.LayerNorm(emb_size)

        self.linear_emb_to_patch_r = nn.Linear(emb_size, patch_size * feature_size)
        self.linear_emb_to_patch_i = nn.Linear(emb_size, patch_size * feature_size)

        self.time_emb = TimeEmbedding(dim=emb_size)
        self.initialize_weights()

    def forward(self, input_r: torch.Tensor, input_i: torch.Tensor, t: torch.Tensor, text_input=None):
        """
        input_r/i: (B, L, M)
        """
        B, L, M = input_r.shape
        pad_len = (self.patch_size - L % self.patch_size) % self.patch_size
        if pad_len > 0:
            input_r = F.pad(input_r, (0, 0, 0, pad_len))
            input_i = F.pad(input_i, (0, 0, 0, pad_len))

        # Patchify
        x_r = self.conv_r(input_r.permute(0, 2, 1))
        x_i = self.conv_i(input_i.permute(0, 2, 1))

        L_patch = x_r.size(2)
        x_r, x_i = x_r.permute(0, 2, 1), x_i.permute(0, 2, 1)

        # Positional Embedding
        if self.pos_embed is None or self.pos_embed.size(1) != L_patch:
            self.pos_embed = get_sinusoidal_positional_embeddings(L_patch, self.emb_size).to(x_r.device)
        x_r = x_r + self.pos_embed
        x_i = x_i + self.pos_embed

        # Time/Condition Embedding
        t = self.time_emb(t)
        if text_input is not None:
            text_input = self.text_projection(text_input)
            c = t + text_input
        else:
            c = t

        # foward
        for layer in self.layers:
            x_r, x_i = layer(x_r, x_i, c)

        # Linear
        x_r, x_i = self.ln_r(x_r), self.ln_i(x_i)
        x_r, x_i = self.linear_emb_to_patch_r(x_r), self.linear_emb_to_patch_i(x_i)

        # Reshape
        x_r = x_r.reshape(B, L_patch * self.patch_size, M)[:, :L, :]
        x_i = x_i.reshape(B, L_patch * self.patch_size, M)[:, :L, :]

        return x_r, x_i

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                #torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        for block in self.layers:
            nn.init.constant_(block.adaLN_r[-1].weight, 0)
            nn.init.constant_(block.adaLN_r[-1].bias, 0)
            nn.init.constant_(block.adaLN_i[-1].weight, 0)
            nn.init.constant_(block.adaLN_i[-1].bias, 0)