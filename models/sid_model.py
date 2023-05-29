import torch
from torch import nn

from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SIDUNet(nn.Module):
    def __init__(self, block_size=2, channels=32) -> None:
        super().__init__()
        inchannels = block_size * block_size
        outchannels = block_size * block_size * 3

        self.conv1_1 = nn.Conv2d(inchannels, channels, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(channels * 8, channels * 16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(channels * 16, channels * 16, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(channels * 16, channels * 8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(channels * 16, channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(channels * 8, channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.conv9_1 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(channels, outchannels, kernel_size=1, stride=1)
        self.pixel_shuffle = nn.PixelShuffle(block_size)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        out = self.pixel_shuffle(conv10)
        return out

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt
