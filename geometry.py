import torch
from torch.nn import functional as F

# center the protein
def center(x):
    return x - x.mean(dim=-2, keepdim=True)