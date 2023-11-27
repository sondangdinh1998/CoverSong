import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginLoss(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scale: float,
        margin: float,
    ):
        super(ArcMarginLoss, self).__init__()

        self.scale = scale
        self.margin = margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeds, labels):
        x_norm = F.normalize(embeds, p=2.0, dim=1)
        w_norm = F.normalize(self.W, p=2.0, dim=1)

        cosine = torch.mm(x_norm, w_norm).clamp(-1.0, 1.0)
        sine = (1.0 - cosine.pow(2)).clamp(1e-12).sqrt()

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        mask = torch.zeros(cosine.size()).type_as(cosine)
        mask.scatter_(1, labels.unsqueeze(1).long(), 1)

        logits = (mask * phi) + ((1.0 - mask) * cosine)
        logits = self.scale * logits

        loss = F.cross_entropy(logits, labels)

        return loss
