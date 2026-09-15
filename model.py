import math
import torch
from torch import nn

from geometry import center


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, values):
        half = self.dim // 2

        frequencies = torch.exp(
            torch.arange(
                half,
                device=values.device,
                dtype=values.dtype,
            )
            * (-math.log(10_000) / (half - 1))
        )

        angles = values[..., None] * frequencies

        return torch.cat(
            (angles.sin(), angles.cos()),
            dim=-1,
        )


class SingleScaleFlow(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 96

        self.coordinate_embedding = nn.Linear(3, dim)
        self.position_embedding = SinusoidalEmbedding(dim)
        self.time_embedding = SinusoidalEmbedding(dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 4,
            dropout=0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=3,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )

        self.velocity_head = nn.Linear(dim, 3)

        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(self, x_t, time):
        batch_size, length, _ = x_t.shape

        positions = torch.linspace(
            0,
            1,
            length,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        positions = positions.expand(batch_size, length)

        times = time.expand(batch_size, length)

        hidden = (
            self.coordinate_embedding(x_t)
            + self.position_embedding(positions)
            + self.time_embedding(times)
        )

        hidden = self.transformer(hidden)

        return self.velocity_head(hidden)

    

def flow_matching_loss(model, target):
    noise = center(torch.randn_like(target))

    time = torch.rand(
        target.shape[0],
        1,
        device=target.device,
        dtype=target.dtype,
    )

    x_t = time[..., None] * target + (1 - time[..., None]) * noise

    target_velocity = target - noise
    predicted_velocity = model(x_t, time)

    return (predicted_velocity - target_velocity).square().mean()