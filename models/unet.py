import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_skip = self.conv(x)  # for skip connection
        x_pool = self.pool(x_skip)
        return x_pool, x_skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # UpSampling
        self.conv = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, 32)

        # output
        self.out = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder part
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # bottleneck part
        x = self.bottleneck(x)

        # Decord part
        x = self.dec1(x, skip4)  # 上采样后与 skip4 拼接
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        # output
        x = self.out(x)

        x = self.sigmoid(x)  # only predict foreground and background

        return x
