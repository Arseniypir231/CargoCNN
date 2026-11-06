"""CNN implemented from scratch for cargo image classification (week 3)."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Two convolution layers followed by downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class CargoCNN(nn.Module):
    """Four-block CNN; accepts RGB images and returns three class logits.

    The defaults are the unchanged week-3 configuration. Week 4 can select a
    different width for each block and a dropout probability for the dense head.
    """

    def __init__(self, num_classes: int = 3,
                 filters: tuple[int, int, int, int] = (32, 64, 128, 256),
                 dropout: float = 0.0):
        super().__init__()
        if num_classes != 3:
            raise ValueError("CargoCNN is defined for exactly three classes.")
        filters = tuple(int(value) for value in filters)
        if len(filters) != 4 or any(value < 1 for value in filters):
            raise ValueError("filters должен содержать четыре положительных числа")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout должен быть в диапазоне [0, 1)")
        self.features = nn.Sequential(
            ConvBlock(3, filters[0]),
            ConvBlock(filters[0], filters[1]),
            ConvBlock(filters[1], filters[2]),
            ConvBlock(filters[2], filters[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[3], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        self.filters = filters
        self.dropout = float(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(images))
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.classifier[3](x)
