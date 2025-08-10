# models.py
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualBlock(nn.Module):
    """A small residual block with two conv layers and ReLU activation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class SmallEncoder(nn.Module):
    """
    Encoder from (Hs x Ws) small source (e.g. 16x16) to a bottleneck spatial map (4x4).
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        # conv downsampling layers
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),  # /4 -> expect 4x4 if start=16
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpsampleBlock(nn.Module):
    """
    Upsample by 2 using a conv to 4*C then pixelshuffle.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.ps(x)
        x = self.bn(x)
        return self.act(x)


class Generator(nn.Module):
    """
    Conditional generator that accepts source RGBA (4 channels), a label index, and an optional noise vector z.
    Outputs: rgb (-1..1 via tanh) and alpha (0..1 via sigmoid).

    Args:
        n_labels: number of categories
        z_dim: dimension of noise vector
        base_ch: base channels for encoder
        out_size: desired output spatial size (must be 16 * 2^k for integer k)
        in_channels: input channels for source (default 4 -> RGB + alpha)
    """

    def __init__(
        self,
        n_labels: int = 10,
        z_dim: int = 128,
        base_ch: int = 64,
        out_size: int = 512,
        in_channels: int = 4,
    ) -> None:
        super().__init__()
        assert out_size % 16 == 0 and ((out_size // 16) & ((out_size // 16) - 1)) == 0, "out_size must be 16 * power_of_two"
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(n_labels, 64)

        # encoder (source in 4 channels)
        self.encoder = SmallEncoder(in_channels=in_channels, base_channels=base_ch)
        bottleneck_ch = base_ch * 4  # e.g., 256

        # project z + label embedding to bottleneck spatial map
        self.z_proj = nn.Linear(z_dim + 64, bottleneck_ch * 4 * 4)

        # few residuals in bottleneck
        self.resblocks = nn.Sequential(ResidualBlock(bottleneck_ch), ResidualBlock(bottleneck_ch))

        # compute number of upsample steps from bottleneck (4x4) to out_size
        ups = int(math.log2(out_size // 4))
        self.ups = nn.ModuleList()
        in_ch = bottleneck_ch
        for i in range(ups):
            out_ch = max(in_ch // 2, 32)
            self.ups.append(UpsampleBlock(in_ch, out_ch))
            in_ch = out_ch

        # final conv to 4 channels (RGB + alpha)
        self.final = nn.Sequential(
            nn.Conv2d(in_ch, max(in_ch // 2, 32), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_ch // 2, 32), 4, 3, padding=1),  # 3 for rgb, 1 for alpha
        )

    def forward(
        self,
        source_rgba: torch.Tensor,
        labels: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_rgba: Tensor[B, 4, Hs, Ws] with RGB in -1..1 and alpha in {0,1}
            labels: Tensor[B] long
            z: Tensor[B, z_dim] noise. If None, sampled from N(0,1).
        Returns:
            rgb: Tensor[B,3,H,H] (-1..1)
            alpha: Tensor[B,1,H,H] (0..1)
        """
        b = source_rgba.shape[0]
        if z is None:
            z = source_rgba.new_empty((b, self.z_dim)).normal_(0, 1)
        lbl = self.label_emb(labels)  # B x 64
        zcat = torch.cat([z, lbl], dim=1)  # B x (z+lbl)
        # encode source
        enc = self.encoder(source_rgba)  # B x C x 4 x 4
        # project zcat
        zproj = self.z_proj(zcat).view(b, enc.shape[1], enc.shape[2], enc.shape[3])
        h = enc + zproj
        h = self.resblocks(h)
        for up in self.ups:
            h = up(h)
        out = self.final(h)
        rgb = torch.tanh(out[:, :3, :, :])
        alpha = torch.sigmoid(out[:, 3:4, :, :])
        return rgb, alpha


class PatchDiscriminator(nn.Module):
    """
    Patch discriminator that sees both (candidate) target RGBA and the upsampled source RGBA.
    It also uses a label embedding which is projected and fused with features.
    """

    def __init__(self, in_ch: int = 8, base: int = 64, n_labels: int = 10) -> None:
        """
        Args:
            in_ch: number of input channels (target_rgba 4 + source_rgba_up 4 = 8 typically)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final = nn.Conv2d(base * 8, 1, 3, 1, 1)
        self.label_emb = nn.Embedding(n_labels, base * 8)

    def forward(self, target_rgba: torch.Tensor, source_rgba_up: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_rgba: B x 4 x H x W (candidate/real target)
            source_rgba_up: B x 4 x H x W (source resized/upsampled to H x W)
            labels: B
        Returns:
            logits: B x 1 x h' x w' patch logits
        """
        x = torch.cat([target_rgba, source_rgba_up], dim=1)  # B x in_ch x H x W
        h = self.net(x)  # B x C x h' x w'
        out = self.final(h)  # B x 1 x h' x w'
        # label projection (global)
        pooled = F.adaptive_avg_pool2d(h, 1).view(h.shape[0], -1)  # B x C
        lbl = self.label_emb(labels)  # B x C
        proj = (pooled * lbl).sum(dim=1, keepdim=True)  # B x 1
        out = out + proj.view(-1, 1, 1, 1)
        return out


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG16 features. Accepts rgb tensors in -1..1.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        # layer indices to capture features
        self.layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3'}

    def forward(self, pred_rgb: torch.Tensor, tgt_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_rgb, tgt_rgb: tensors in -1..1 range
        Returns:
            perceptual L1 loss over selected features
        """
        # normalize to imagenet expected
        x = (pred_rgb + 1) / 2.0
        y = (tgt_rgb + 1) / 2.0
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        feats_x = {}
        feats_y = {}
        xi = x
        yi = y
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            xi = layer(xi); yi = layer(yi)
            key = str(i)
            if key in self.layers:
                loss = loss + F.l1_loss(xi, yi)
        return loss
