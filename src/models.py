import torch.nn as nn


class LinearProbe(nn.Module):
    """Linear probe that maps input features to class logits using a single linear layer.
    This is the simplest form of probing that can capture linear relationships in the features.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

        # Improved initialization using Kaiming initialization
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_out")
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """MLP probe that maps input features to class logits using multiple linear layers with non-linear activations.
    This can capture more complex non-linear relationships in the features compared to LinearProbe.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Add final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)
