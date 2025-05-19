class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim,
                 branch_hidden, trunk_hidden, num_layers):
        super(DeepONet, self).__init__()
        b_layers = [nn.Linear(branch_input_dim, branch_hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            b_layers += [nn.Linear(branch_hidden, branch_hidden), nn.ReLU()]
        self.branch_net = nn.Sequential(*b_layers)

        t_layers = [nn.Linear(trunk_input_dim, trunk_hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            t_layers += [nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU()]
        self.trunk_net = nn.Sequential(*t_layers)

        assert branch_hidden == trunk_hidden, "Branch and trunk dims must match"

    def forward(self, branch_x, trunk_x):
        b = self.branch_net(branch_x)
        t = self.trunk_net(trunk_x)
        return torch.sum(b * t, dim=-1, keepdim=True)
