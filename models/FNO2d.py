import torch
import torch.nn as nn
import torch.fft as fft

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) * (1 / (in_channels * out_channels))
        )

    def forward(self, x):
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')

        out_ft = torch.zeros(batch, self.out_c, h, w//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            'bixy, ioxy->boxy',
            x_ft[:, :, :self.modes1, :self.modes2], self.weights
        )

        x = torch.fft.irfft2(out_ft, s=(h, w), norm='ortho')
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(1, width)
        self.spec_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(4)
        ])
        self.local_convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(4)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        b, h, w, _ = x.shape
        x = x.view(b, h*w, 1)
        x = self.fc0(x)
        x = x.view(b, self.width, h, w)

        for spec, local in zip(self.spec_layers, self.local_convs):
            x1 = spec(x)
            x2 = local(x)
            x = x1 + x2
            x = torch.relu(x)

        x = x.permute(0, 2, 3, 1).view(b*h*w, self.width)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(b, h, w, 1)
