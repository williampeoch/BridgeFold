import torch
from torch import nn

from geometry import center


class SingleScaleFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, x_t, time):
        batch_size, length, _ = x_t.shape

        times = time[:, None, :].expand(batch_size, length, 1)

        positions = torch.linspace(
            0,
            1,
            length,
            device=x_t.device,
            dtype=x_t.dtype
        )
        positions = positions[None, :, None]
        positions = positions.expand(batch_size, length, 1)

        features = torch.cat((x_t, times, positions), dim=-1)

        return self.net(features)


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