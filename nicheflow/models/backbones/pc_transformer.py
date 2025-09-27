from jaxtyping import Float
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.LeakyReLU()

    def forward(
        self, x: Float[Tensor, "batch_size n_microenvs n_ponts {self.input_dim}"]
    ) -> Float[Tensor, "batch_size n_microenvs n_ponts {self.input_dim}"]:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual
