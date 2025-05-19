class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: apply learned complex-valued weights to the low-frequency modes
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        # complex-valued weights
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) * (1 / (in_channels * out_channels))
        )

    def forward(self, x):
        # x: (batch, in_c, height, width)
        batch, _, h, w = x.shape
        # forward FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')  # (batch, in_c, h, w//2+1)

        # allocate output in Fourier domain
        out_ft = torch.zeros(batch, self.out_c, h, w//2+1, dtype=torch.cfloat, device=x.device)
        # apply spectral weights to low modes
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            'bixy, ioxy->boxy',
            x_ft[:, :, :self.modes1, :self.modes2], self.weights
        )

        # inverse FFT back to physical space
        x = torch.fft.irfft2(out_ft, s=(h, w), norm='ortho')
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # initial lifting layer
        self.fc0 = nn.Linear(1, width)
        # four spectral convolution layers + 1x1 local conv to capture non-local & local interactions
        self.spec_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(4)
        ])
        self.local_convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(4)
        ])
        # final projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, h, w, 1)
        b, h, w, _ = x.shape
        x = x.view(b, h*w, 1)
        x = self.fc0(x)                           # (b, h*w, width)
        x = x.view(b, self.width, h, w)           # (b, width, h, w)

        # four Fourier + local conv blocks
        for spec, local in zip(self.spec_layers, self.local_convs):
            x1 = spec(x)
            x2 = local(x)
            x = x1 + x2
            x = torch.relu(x)

        # project back to scalar field
        x = x.permute(0, 2, 3, 1).view(b*h*w, self.width)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(b, h, w, 1)
