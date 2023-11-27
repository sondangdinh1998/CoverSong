from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from lightcover.layers.campplus import (
    DenseLayer,
    StatsPool,
    TDNNLayer,
    CAMDenseTDNNBlock,
    TransitLayer,
    BasicResBlock,
    get_nonlinear,
)
from lightcover.utils.common import length_to_mask


class FCM(nn.Module):
    def __init__(
        self,
        block=BasicResBlock,
        num_blocks=[2, 2],
        m_channels=32,
        feat_dim=80,
    ):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(
            block,
            m_channels,
            num_blocks[0],
            stride=2,
        )
        self.layer2 = self._make_layer(
            block,
            m_channels,
            num_blocks[0],
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            m_channels,
            m_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])

        return out


class CAMPPlus(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        embedding_size=512,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
    ):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    )
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(
                    channels, channels // 2, bias=False, config_str=config_str
                ),
            )
            channels //= 2

        self.xvector.add_module(
            "out_nonlinear",
            get_nonlinear(config_str, channels),
        )

        self.pool = StatsPool()
        self.proj = DenseLayer(
            channels * 2,
            embedding_size,
            config_str="batchnorm_",
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, lens):
        x = x.permute(0, 2, 1)
        mask = length_to_mask(lens, x.size(2))

        x = self.head(x) * mask[:, None, :]
        x = self.xvector(x) * mask[:, None, ::2]

        m = mask[:, None, ::2]
        m = m / m.sum(dim=2, keepdim=True)

        x = self.pool(x, m)
        x = self.proj(x)

        return x
