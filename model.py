import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlusEncoder(nn.Module):
    def __init__(self, skip_connections):
        super(UNetPlusPlusEncoder, self).__init__()
        self.skip_connections = skip_connections
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)

    def forward(self, x):
        x1_0 = self.enc1(x)
        x2_0 = self.enc2(nn.MaxPool2d(2)(x1_0))
        x3_0 = self.enc3(nn.MaxPool2d(2)(x2_0))
        x4_0 = self.enc4(nn.MaxPool2d(2)(x3_0))
        x5_0 = self.enc5(nn.MaxPool2d(2)(x4_0))

        self.skip_connections.extend([x1_0, x2_0, x3_0, x4_0])
        return x5_0

class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, skip_connections):
        super(UNetPlusPlusDecoder, self).__init__()
        self.skip_connections = skip_connections
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)

    def forward(self, x):
        x4_0 = self.skip_connections.pop()
        x3_0 = self.skip_connections.pop()
        x2_0 = self.skip_connections.pop()
        x1_0 = self.skip_connections.pop()

        x4_1 = self.decoder4(torch.cat([self.upconv4(x), x4_0], dim=1))
        x3_1 = self.decoder3(torch.cat([self.upconv3(x4_1), x3_0], dim=1))
        x2_1 = self.decoder2(torch.cat([self.upconv2(x3_1), x2_0], dim=1))
        x1_1 = self.decoder1(torch.cat([self.upconv1(x2_1), x1_0], dim=1))
        
        return x1_1

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.skip_connections = []
        self.encoder = UNetPlusPlusEncoder(self.skip_connections)
        self.decoder = UNetPlusPlusDecoder(self.skip_connections)
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        y = self.classifier(x)
        return self.sigmoid(y)
