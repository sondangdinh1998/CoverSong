import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightcover.layers.conformer import ConformerBlock
from lightcover.utils.common import length_to_mask, compute_statistics


class ConvolutionSubsampling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        factor: int,
        num_filters: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.stride = 2
        self.factor = factor

        in_channels = 1
        padding = (kernel_size - 1) // 2

        self.layers = nn.ModuleList()
        for _ in range(int(math.log(factor, 2))):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_filters,
                        kernel_size=kernel_size,
                        stride=self.stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_filters),
                    nn.SiLU(),
                )
            )
            in_channels = num_filters

        num_filters = 1 if len(self.layers) == 0 else num_filters
        self.proj = nn.Linear(
            num_filters * math.ceil(input_dim / self.factor),
            output_dim,
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = xs[:, None, :, :]
        masks = length_to_mask(x_lens, xs.size(2))
        masks = masks[:, None, :, None]

        for layer in self.layers:
            masks = masks[:, :, :: self.stride, :]
            xs = layer(xs) * masks

        b, c, t, f = xs.size()
        xs = xs.transpose(1, 2).contiguous().view(b, t, c * f)

        xs = self.proj(xs)
        xs = self.drop(xs)

        x_lens = torch.div(x_lens - 1, self.factor, rounding_mode="trunc")
        x_lens = (x_lens + 1).type(torch.long)

        return xs, x_lens


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, d_model: int, att_dim: int, emb_dim: int):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(d_model * 3, att_dim, 1),
            nn.ReLU(),
            nn.BatchNorm1d(att_dim),
        )
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(att_dim, d_model, 1)
        self.norm = nn.BatchNorm1d(d_model * 2)
        self.proj = nn.Conv1d(d_model * 2, emb_dim, 1)

    def forward(self, xs, x_lens):
        xs = xs.transpose(1, 2)
        L = xs.shape[-1]

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(x_lens, L)
        mask = mask.unsqueeze(1).float()

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        total = mask.sum(dim=2, keepdim=True).float()
        mean, std = compute_statistics(xs, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, L)
        std = std.unsqueeze(2).repeat(1, 1, L)
        attn = torch.cat([xs, mean, std], dim=1)

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0.0, -1e3)
        attn = F.softmax(attn, dim=2)
        mean, std = compute_statistics(xs, attn)

        stats = torch.cat((mean, std), dim=1)
        stats = stats.unsqueeze(2)

        outs = self.proj(self.norm(stats))
        outs = outs.transpose(1, 2).squeeze(1)

        return outs


class AttentiveTransformerPooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        emb_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token)

        self.attention = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        cls_token = self.cls_token.repeat_interleave(xs.size(0), dim=0)

        xs = torch.cat((cls_token, xs), dim=1)
        x_lens = x_lens + 1

        masks = length_to_mask(x_lens, xs.size(1), dtype=torch.bool)
        outs = self.attention(xs, src_key_padding_mask=~masks)

        embeds = self.projector(outs[:, 0, :])

        return embeds


class Conformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        subsampling_factor: int,
        subsampling_filters: int,
        subsampling_kernel: int,
        encoder_num_heads: int,
        encoder_ffn_dim: int,
        encoder_num_layers: int,
        encoder_kernel_size: int,
        pooling_n_heads: int,
        pooling_emb_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.subsampling = ConvolutionSubsampling(
            input_dim=input_dim,
            output_dim=d_model,
            factor=subsampling_factor,
            num_filters=subsampling_filters,
            kernel_size=subsampling_kernel,
            dropout=dropout,
        )

        self.encoder = ConformerBlock(
            input_dim=d_model,
            num_heads=encoder_num_heads,
            ffn_dim=encoder_ffn_dim,
            num_layers=encoder_num_layers,
            depthwise_conv_kernel_size=encoder_kernel_size,
            dropout=dropout,
        )

        self.pooling = AttentiveTransformerPooling(
            d_model=d_model,
            n_heads=pooling_n_heads,
            emb_dim=pooling_emb_dim,
            dropout=dropout,
        )

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, x_lens = self.subsampling(xs, x_lens)
        xs, x_lens = self.encoder(xs, x_lens)
        outs = self.pooling(xs, x_lens)
        return outs
