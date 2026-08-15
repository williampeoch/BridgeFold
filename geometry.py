import torch
from torch.nn import functional as F

# tensor are like -> [B, L, 3]


x = torch.tensor([
    # 1. Tous les points au même endroit → Rg ≈ 1e-6
    [
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ],

    # 2. Points à distance 3 du centre → Rg = 3
    [
        [-3., 0., 0.],
        [ 3., 0., 0.],
        [-3., 0., 0.],
        [ 3., 0., 0.],
    ],

    # 3. Carré autour de l'origine → Rg = sqrt(2)
    [
        [ 1.,  1., 0.],
        [ 1., -1., 0.],
        [-1.,  1., 0.],
        [-1., -1., 0.],
    ],

    # 4. Même écart que le cas 2, mais translaté → Rg = 3
    [
        [ 8., -4., 7.],
        [14., -4., 7.],
        [ 8., -4., 7.],
        [14., -4., 7.],
    ],
])


def center(x):
    return x - x.mean(dim=-2, keepdim=True)


def radius_of_gyration(x):
    centered = center(x)
    return torch.sqrt(centered.square().sum(-1).mean(-1) + 1e-12)

