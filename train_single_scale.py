import torch
from torch.utils.data import DataLoader
from data import load_datasets, coordinate_scale
from geometry import center
from model import SingleScaleFlow, flow_matching_loss


torch.manual_seed(13)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_data, _ = load_datasets("data/rcsb_ca64_cluster30.npz")

loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
)

scale = coordinate_scale(train_data)

# keep same batch to check if it learn
batch = next(iter(loader)).to(device)
target = center(batch) / scale

model = SingleScaleFlow().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []

for step in range(5001):
    optimizer.zero_grad()

    loss = flow_matching_loss(model, target)

    loss.backward()
    optimizer.step()

    losses.append(float(loss.detach()))

    if step % 50 == 0:
        recent_losses = losses[-50:]
        average = sum(recent_losses) / len(recent_losses)

        print(f"step={step:3d}  loss={float(loss.detach()):.4f}  average={average:.4f}")