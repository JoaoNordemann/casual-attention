import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        bias: bool = False,
        use_flash: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve ser divisível por n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_flash = use_flash

        # Projeção fundida: Q, K, V numa única matmul
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Máscara causal triangular (usada só no caminho manual)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)                      # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)               # 3 x (B, T, C)

        # (B, T, C) -> (B, H, T, Dh)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_flash:
            # Kernel fundido (FlashAttention / mem-efficient), máscara causal embutida
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Caminho explícito, útil para entender e debugar
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, H, T, T)
            scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v                                                     # (B, H, T, Dh)

        # (B, H, T, Dh) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))