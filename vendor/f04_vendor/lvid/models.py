from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetHeatmap(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + identity)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        super().__init__()
        branches: list[nn.Module] = []
        for rate in rates:
            if rate == 1:
                branches.append(ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0))
            else:
                branches.append(ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=rate))
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.project = ConvBNReLU(out_channels * (len(rates) + 1), out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        features.append(pooled)
        return self.project(torch.cat(features, dim=1))


class DeepLabV3PlusHeatmap(nn.Module):
    """Lightweight DeepLabV3+ style heatmap head for LVID endpoint detection."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 32,
        aspp_channels: int = 128,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        aspp_c = int(aspp_channels)
        dec_c = int(decoder_channels)

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, c, kernel_size=3, stride=2),
            ConvBNReLU(c, c, kernel_size=3),
        )
        self.layer1 = ResidualBlock(c, c, stride=1)
        self.layer2 = nn.Sequential(
            ResidualBlock(c, c * 2, stride=2),
            ResidualBlock(c * 2, c * 2),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(c * 2, c * 4, stride=2),
            ResidualBlock(c * 4, c * 4),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(c * 4, c * 8, stride=2),
            ResidualBlock(c * 8, c * 8, dilation=2),
        )

        self.aspp = ASPP(c * 8, aspp_c)
        self.low_project = ConvBNReLU(c * 2, dec_c, kernel_size=1, padding=0)
        self.decoder = nn.Sequential(
            ConvBNReLU(aspp_c + dec_c, dec_c),
            ConvBNReLU(dec_c, dec_c),
            nn.Conv2d(dec_c, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.stem(x)
        x = self.layer1(x)
        low = self.layer2(x)
        x = self.layer3(low)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_project(low)
        x = self.decoder(torch.cat([x, low], dim=1))
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


class HRFuseBlock(nn.Module):
    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.refine1 = ResidualBlock(c1, c1)
        self.refine2 = ResidualBlock(c2, c2)
        self.refine3 = ResidualBlock(c3, c3)

        self.to1 = nn.ModuleList(
            [
                nn.Identity(),
                ConvBNReLU(c2, c1, kernel_size=1, padding=0),
                ConvBNReLU(c3, c1, kernel_size=1, padding=0),
            ]
        )
        self.to2 = nn.ModuleList(
            [
                ConvBNReLU(c1, c2, kernel_size=3, stride=2),
                nn.Identity(),
                ConvBNReLU(c3, c2, kernel_size=1, padding=0),
            ]
        )
        self.to3 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(c1, c1, kernel_size=3, stride=2),
                    ConvBNReLU(c1, c3, kernel_size=3, stride=2),
                ),
                ConvBNReLU(c2, c3, kernel_size=3, stride=2),
                nn.Identity(),
            ]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b1, b2, b3 = branches
        b1 = self.refine1(b1)
        b2 = self.refine2(b2)
        b3 = self.refine3(b3)

        size1 = b1.shape[-2:]
        size2 = b2.shape[-2:]
        size3 = b3.shape[-2:]

        f1 = self.to1[0](b1)
        f1 = f1 + F.interpolate(self.to1[1](b2), size=size1, mode="bilinear", align_corners=False)
        f1 = f1 + F.interpolate(self.to1[2](b3), size=size1, mode="bilinear", align_corners=False)

        f2 = self.to2[0](b1) + self.to2[1](b2)
        f2 = f2 + F.interpolate(self.to2[2](b3), size=size2, mode="bilinear", align_corners=False)

        f3 = self.to3[0](b1) + self.to3[1](b2) + self.to3[2](b3)
        if f3.shape[-2:] != size3:
            f3 = F.interpolate(f3, size=size3, mode="bilinear", align_corners=False)

        return self.relu(f1), self.relu(f2), self.relu(f3)


class HRNetHeatmap(nn.Module):
    """Lightweight HRNet-style multi-resolution heatmap model."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 32,
        fusion_blocks: int = 3,
        head_channels: int = 64,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, c, kernel_size=3, stride=2),
            ResidualBlock(c, c),
        )

        self.branch1 = ResidualBlock(c, c)
        self.make_branch2 = ConvBNReLU(c, c * 2, kernel_size=3, stride=2)
        self.make_branch3 = nn.Sequential(
            ConvBNReLU(c * 2, c * 4, kernel_size=3, stride=2),
            ResidualBlock(c * 4, c * 4),
        )

        self.fusions = nn.ModuleList([HRFuseBlock((c, c * 2, c * 4)) for _ in range(int(fusion_blocks))])
        head_c = int(head_channels)
        self.head = nn.Sequential(
            ConvBNReLU(c + c * 2 + c * 4, head_c),
            ConvBNReLU(head_c, head_c),
            nn.Conv2d(head_c, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.stem(x)
        b1 = self.branch1(x)
        b2 = self.make_branch2(b1)
        b3 = self.make_branch3(b2)

        for fuse in self.fusions:
            b1, b2, b3 = fuse((b1, b2, b3))

        size = b1.shape[-2:]
        features = [
            b1,
            F.interpolate(b2, size=size, mode="bilinear", align_corners=False),
            F.interpolate(b3, size=size, mode="bilinear", align_corners=False),
        ]
        logits = self.head(torch.cat(features, dim=1))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class ResNetFPNHeatmap(nn.Module):
    """ResNet backbone with an FPN-style heatmap decoder for LVID endpoint detection."""

    _CHANNELS = {
        "resnet18": (64, 64, 128, 256, 512),
        "resnet34": (64, 64, 128, 256, 512),
        "resnet50": (64, 256, 512, 1024, 2048),
    }

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        backbone: str = "resnet18",
        pretrained: bool = True,
        fpn_channels: int = 128,
        head_channels: int = 128,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        backbone = str(backbone).lower()
        if backbone not in self._CHANNELS:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")

        resnet = self._make_resnet(backbone, bool(pretrained))
        if int(in_channels) != 3:
            resnet.conv1 = self._adapt_first_conv(resnet.conv1, int(in_channels))

        self.normalize = bool(normalize)
        self.in_channels = int(in_channels)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        _, c2, c3, c4, c5 = self._CHANNELS[backbone]
        fpn_c = int(fpn_channels)
        self.lat2 = nn.Conv2d(c2, fpn_c, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, fpn_c, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, fpn_c, kernel_size=1)
        self.lat5 = nn.Conv2d(c5, fpn_c, kernel_size=1)
        self.smooth2 = ConvBNReLU(fpn_c, fpn_c)
        self.smooth3 = ConvBNReLU(fpn_c, fpn_c)
        self.smooth4 = ConvBNReLU(fpn_c, fpn_c)
        self.smooth5 = ConvBNReLU(fpn_c, fpn_c)

        head_c = int(head_channels)
        self.head = nn.Sequential(
            ConvBNReLU(fpn_c * 4, head_c),
            ConvBNReLU(head_c, head_c),
            nn.Conv2d(head_c, int(out_channels), kernel_size=1),
        )

        if self.in_channels == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        else:
            mean = torch.full((1, self.in_channels, 1, 1), 0.5, dtype=torch.float32)
            std = torch.full((1, self.in_channels, 1, 1), 0.25, dtype=torch.float32)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @staticmethod
    def _make_resnet(backbone: str, pretrained: bool):
        from torchvision import models as tv_models

        factory = getattr(tv_models, backbone)
        if not pretrained:
            try:
                return factory(weights=None)
            except TypeError:
                return factory(pretrained=False)

        depth = backbone.replace("resnet", "")
        weights_enum = getattr(tv_models, f"ResNet{depth}_Weights", None)
        try:
            if weights_enum is not None:
                return factory(weights=weights_enum.DEFAULT)
            return factory(pretrained=True)
        except Exception as exc:
            print(f"Warning: could not load pretrained {backbone} weights ({exc}); using random init.")
            try:
                return factory(weights=None)
            except TypeError:
                return factory(pretrained=False)

    @staticmethod
    def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            if in_channels == 1 and conv.weight.shape[1] == 3:
                new_conv.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
            else:
                base = conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(base.repeat(1, in_channels, 1, 1))
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        return new_conv

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return (x - self.mean) / self.std.clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self._norm(x)

        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        size = p2.shape[-2:]
        fused = torch.cat(
            [
                p2,
                F.interpolate(p3, size=size, mode="bilinear", align_corners=False),
                F.interpolate(p4, size=size, mode="bilinear", align_corners=False),
                F.interpolate(p5, size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        logits = self.head(fused)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class SwinFPNHeatmap(nn.Module):
    """Torchvision Swin Transformer backbone with an FPN heatmap decoder."""

    _CHANNELS = {
        "swin_t": (96, 192, 384, 768),
        "swin_s": (96, 192, 384, 768),
        "swin_b": (128, 256, 512, 1024),
    }

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        backbone: str = "swin_t",
        pretrained: bool = True,
        fpn_channels: int = 128,
        head_channels: int = 128,
        normalize: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        backbone = str(backbone).lower()
        if backbone not in self._CHANNELS:
            raise ValueError(f"Unsupported Swin backbone: {backbone}")

        swin = self._make_swin(backbone, bool(pretrained))
        self.in_channels = int(in_channels)
        self.normalize = bool(normalize)
        self.features = swin.features
        if self.in_channels != 3:
            self._adapt_patch_embed(self.in_channels)

        c2, c3, c4, c5 = self._CHANNELS[backbone]
        fpn_c = int(fpn_channels)
        self.lat2 = nn.Conv2d(c2, fpn_c, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, fpn_c, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, fpn_c, kernel_size=1)
        self.lat5 = nn.Conv2d(c5, fpn_c, kernel_size=1)
        self.smooth2 = ConvBNReLU(fpn_c, fpn_c)
        self.smooth3 = ConvBNReLU(fpn_c, fpn_c)
        self.smooth4 = ConvBNReLU(fpn_c, fpn_c)
        self.smooth5 = ConvBNReLU(fpn_c, fpn_c)

        head_c = int(head_channels)
        self.head = nn.Sequential(
            ConvBNReLU(fpn_c * 4, head_c),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            ConvBNReLU(head_c, head_c),
            nn.Conv2d(head_c, int(out_channels), kernel_size=1),
        )

        if self.in_channels == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        else:
            mean = torch.full((1, self.in_channels, 1, 1), 0.5, dtype=torch.float32)
            std = torch.full((1, self.in_channels, 1, 1), 0.25, dtype=torch.float32)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @staticmethod
    def _make_swin(backbone: str, pretrained: bool):
        from torchvision import models as tv_models

        factory = getattr(tv_models, backbone)
        if not pretrained:
            return factory(weights=None)
        weights_name = {
            "swin_t": "Swin_T_Weights",
            "swin_s": "Swin_S_Weights",
            "swin_b": "Swin_B_Weights",
        }[backbone]
        weights_enum = getattr(tv_models, weights_name, None)
        try:
            return factory(weights=weights_enum.DEFAULT if weights_enum is not None else None)
        except Exception as exc:
            print(f"Warning: could not load pretrained {backbone} weights ({exc}); using random init.")
            return factory(weights=None)

    def _adapt_patch_embed(self, in_channels: int) -> None:
        patch = self.features[0]
        conv = patch[0]
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("Unexpected Swin patch embedding layout; first module is not Conv2d.")
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            if in_channels == 1 and conv.weight.shape[1] == 3:
                new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            else:
                base = conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(base.repeat(1, in_channels, 1, 1))
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        patch[0] = new_conv

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return (x - self.mean) / self.std.clamp_min(1e-6)

    @staticmethod
    def _nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self._norm(x)

        outputs: list[torch.Tensor] = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in (1, 3, 5, 7):
                outputs.append(self._nchw(x))

        c2, c3, c4, c5 = outputs
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        size = p2.shape[-2:]
        fused = torch.cat(
            [
                p2,
                F.interpolate(p3, size=size, mode="bilinear", align_corners=False),
                F.interpolate(p4, size=size, mode="bilinear", align_corners=False),
                F.interpolate(p5, size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        logits = self.head(fused)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class DropPath(nn.Module):
    """Stochastic depth used by transformer blocks."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            int(in_channels),
            int(embed_dim),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
        )
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        return self.norm(x), height, width


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"SegFormer attention dim {dim} must be divisible by heads {num_heads}.")
        self.num_heads = num_heads
        self.sr_ratio = int(sr_ratio)
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None
        self.attn_drop = nn.Dropout(float(attention_dropout))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(float(projection_dropout))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        if self.sr is not None and self.norm is not None:
            pooled = x.transpose(1, 2).reshape(batch, channels, height, width)
            pooled = self.sr(pooled).reshape(batch, channels, -1).transpose(1, 2).contiguous()
            pooled = self.norm(pooled)
        else:
            pooled = x

        kv = self.kv(pooled).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(out))


class DWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(int(dim), int(dim), kernel_size=3, stride=1, padding=1, groups=int(dim))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class MixFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * float(mlp_ratio))
        self.fc1 = nn.Linear(int(dim), hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(hidden_dim, int(dim))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SegFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = EfficientSelfAttention(
            dim=int(dim),
            num_heads=int(num_heads),
            sr_ratio=int(sr_ratio),
            attention_dropout=float(attention_dropout),
            projection_dropout=float(dropout),
        )
        self.drop_path = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(int(dim))
        self.mlp = MixFFN(dim=int(dim), mlp_ratio=float(mlp_ratio), dropout=float(dropout))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class SegFormerHeatmap(nn.Module):
    """SegFormer-style MixVision Transformer with an MLP heatmap decoder."""

    _VARIANTS = {
        "b0": {
            "embed_dims": (32, 64, 160, 256),
            "depths": (2, 2, 2, 2),
            "num_heads": (1, 2, 5, 8),
            "sr_ratios": (8, 4, 2, 1),
        },
        "b1": {
            "embed_dims": (64, 128, 320, 512),
            "depths": (2, 2, 2, 2),
            "num_heads": (1, 2, 5, 8),
            "sr_ratios": (8, 4, 2, 1),
        },
        "b2": {
            "embed_dims": (64, 128, 320, 512),
            "depths": (3, 4, 6, 3),
            "num_heads": (1, 2, 5, 8),
            "sr_ratios": (8, 4, 2, 1),
        },
    }

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        variant: str = "b2",
        embed_dims: tuple[int, int, int, int] | None = None,
        depths: tuple[int, int, int, int] | None = None,
        num_heads: tuple[int, int, int, int] | None = None,
        sr_ratios: tuple[int, int, int, int] | None = None,
        mlp_ratio: float = 4.0,
        decoder_channels: int = 256,
        normalize: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        variant_cfg = self._VARIANTS.get(str(variant).lower())
        if variant_cfg is None and any(v is None for v in (embed_dims, depths, num_heads, sr_ratios)):
            raise ValueError(f"Unsupported SegFormer variant: {variant}")

        embed_dims = embed_dims or variant_cfg["embed_dims"]
        depths = depths or variant_cfg["depths"]
        num_heads = num_heads or variant_cfg["num_heads"]
        sr_ratios = sr_ratios or variant_cfg["sr_ratios"]
        if not (len(embed_dims) == len(depths) == len(num_heads) == len(sr_ratios) == 4):
            raise ValueError("SegFormer embed_dims/depths/num_heads/sr_ratios must all have 4 values.")

        self.in_channels = int(in_channels)
        self.normalize = bool(normalize)

        patch_args = [
            (self.in_channels, embed_dims[0], 7, 4, 3),
            (embed_dims[0], embed_dims[1], 3, 2, 1),
            (embed_dims[1], embed_dims[2], 3, 2, 1),
            (embed_dims[2], embed_dims[3], 3, 2, 1),
        ]
        self.patch_embeds = nn.ModuleList([OverlapPatchEmbed(*args) for args in patch_args])

        total_depth = int(sum(depths))
        drop_rates = torch.linspace(0, float(drop_path_rate), total_depth).tolist()
        cursor = 0
        stages: list[nn.ModuleList] = []
        norms: list[nn.LayerNorm] = []
        for stage_idx in range(4):
            blocks = nn.ModuleList(
                [
                    SegFormerBlock(
                        dim=int(embed_dims[stage_idx]),
                        num_heads=int(num_heads[stage_idx]),
                        sr_ratio=int(sr_ratios[stage_idx]),
                        mlp_ratio=float(mlp_ratio),
                        dropout=float(dropout),
                        attention_dropout=float(attention_dropout),
                        drop_path=float(drop_rates[cursor + block_idx]),
                    )
                    for block_idx in range(int(depths[stage_idx]))
                ]
            )
            cursor += int(depths[stage_idx])
            stages.append(blocks)
            norms.append(nn.LayerNorm(int(embed_dims[stage_idx])))
        self.stages = nn.ModuleList(stages)
        self.norms = nn.ModuleList(norms)

        dec_c = int(decoder_channels)
        self.decoder_projs = nn.ModuleList([nn.Conv2d(int(dim), dec_c, kernel_size=1) for dim in embed_dims])
        self.fuse = ConvBNReLU(dec_c * 4, dec_c, kernel_size=1, padding=0)
        self.decoder_dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.head = nn.Conv2d(dec_c, int(out_channels), kernel_size=1)

        if self.in_channels == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        else:
            mean = torch.full((1, self.in_channels, 1, 1), 0.5, dtype=torch.float32)
            std = torch.full((1, self.in_channels, 1, 1), 0.25, dtype=torch.float32)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return (x - self.mean) / self.std.clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self._norm(x)
        features: list[torch.Tensor] = []

        for patch_embed, blocks, norm in zip(self.patch_embeds, self.stages, self.norms):
            tokens, height, width = patch_embed(x)
            for block in blocks:
                tokens = block(tokens, height, width)
            tokens = norm(tokens)
            batch, _, channels = tokens.shape
            x = tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
            features.append(x)

        decode_size = features[0].shape[-2:]
        decoded = [
            F.interpolate(proj(feat), size=decode_size, mode="bilinear", align_corners=False)
            for proj, feat in zip(self.decoder_projs, features)
        ]
        fused = self.decoder_dropout(self.fuse(torch.cat(decoded, dim=1)))
        logits = self.head(fused)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def _optional_int_tuple(cfg: dict, key: str) -> tuple[int, int, int, int] | None:
    value = cfg.get(key)
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"{key} must have 4 values")
    return tuple(int(v) for v in value)


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("name", "unet_heatmap")
    if name == "unet_heatmap":
        return UNetHeatmap(
            in_channels=int(cfg.get("in_channels", 1)),
            out_channels=int(cfg.get("out_channels", 4)),
            base_channels=int(cfg.get("base_channels", 32)),
        )
    if name == "deeplabv3plus_heatmap":
        return DeepLabV3PlusHeatmap(
            in_channels=int(cfg.get("in_channels", 1)),
            out_channels=int(cfg.get("out_channels", 2)),
            base_channels=int(cfg.get("base_channels", 32)),
            aspp_channels=int(cfg.get("aspp_channels", 128)),
            decoder_channels=int(cfg.get("decoder_channels", 64)),
        )
    if name == "hrnet_heatmap":
        return HRNetHeatmap(
            in_channels=int(cfg.get("in_channels", 1)),
            out_channels=int(cfg.get("out_channels", 2)),
            base_channels=int(cfg.get("base_channels", 32)),
            fusion_blocks=int(cfg.get("fusion_blocks", 3)),
            head_channels=int(cfg.get("head_channels", 64)),
        )
    if name == "resnet_fpn_heatmap":
        return ResNetFPNHeatmap(
            in_channels=int(cfg.get("in_channels", 3)),
            out_channels=int(cfg.get("out_channels", 3)),
            backbone=str(cfg.get("backbone", "resnet18")),
            pretrained=bool(cfg.get("pretrained", True)),
            fpn_channels=int(cfg.get("fpn_channels", 128)),
            head_channels=int(cfg.get("head_channels", 128)),
            normalize=bool(cfg.get("normalize", True)),
        )
    if name == "swin_fpn_heatmap":
        return SwinFPNHeatmap(
            in_channels=int(cfg.get("in_channels", 3)),
            out_channels=int(cfg.get("out_channels", 3)),
            backbone=str(cfg.get("backbone", "swin_t")),
            pretrained=bool(cfg.get("pretrained", True)),
            fpn_channels=int(cfg.get("fpn_channels", 128)),
            head_channels=int(cfg.get("head_channels", 128)),
            normalize=bool(cfg.get("normalize", True)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if name == "segformer_heatmap":
        return SegFormerHeatmap(
            in_channels=int(cfg.get("in_channels", 3)),
            out_channels=int(cfg.get("out_channels", 3)),
            variant=str(cfg.get("variant", "b2")),
            embed_dims=_optional_int_tuple(cfg, "embed_dims"),
            depths=_optional_int_tuple(cfg, "depths"),
            num_heads=_optional_int_tuple(cfg, "num_heads"),
            sr_ratios=_optional_int_tuple(cfg, "sr_ratios"),
            mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
            decoder_channels=int(cfg.get("decoder_channels", 256)),
            normalize=bool(cfg.get("normalize", True)),
            dropout=float(cfg.get("dropout", 0.0)),
            attention_dropout=float(cfg.get("attention_dropout", 0.0)),
            drop_path_rate=float(cfg.get("drop_path_rate", 0.0)),
        )
    raise ValueError(f"Unsupported model: {name}")
