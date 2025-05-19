class SuperResolutionCNN(nn.Module):
    def __init__(self, in_channels=3, num_features=64):
        super(SuperResolutionCNN, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(num_features, num_features // 2, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Conv2d(num_features // 2, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x
