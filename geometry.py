import torch
from torch.nn import functional as F

# tensor are like -> [B, L, 3]

# x = torch.tensor([
#     [
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#     ],

#     [
#         [-3., 0., 0.],
#         [ 3., 0., 0.],
#         [-3., 0., 0.],
#         [ 3., 0., 0.],
#     ],

#     [
#         [ 1.,  1., 0.],
#         [ 1., -1., 0.],
#         [-1.,  1., 0.],
#         [-1., -1., 0.],
#     ],

#     [
#         [ 8., -4., 7.],
#         [14., -4., 7.],
#         [ 8., -4., 7.],
#         [14., -4., 7.],
#     ],
# ])

def center(x):
    return x - x.mean(dim=-2, keepdim=True)


def resize_backbone(x, length):
    if x.shape[-2] == length:
        return x
    x_t = x.transpose(1, 2)
    return F.interpolate(x_t, size=length, mode="linear", align_corners=False).transpose(1, 2)