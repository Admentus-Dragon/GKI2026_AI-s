import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from models_torch import SimpleCNN

DEVICE = "cuda"
BATCH = 512
EPOCHS = 30
LR = 1e-3

class HabitatDataset(Dataset):
    def __init__(self):
        self.patches = np.load("data/train/patches.npy", mmap_mode="r")
        df = pd.read_csv("data/train.csv")
        self.y = df["vistgerd_idx"].to_numpy().astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx].copy()).float()
        return x, int(self.y[idx])

ds = HabitatDataset()

# split (same idea as your two-stage script)
n = len(ds)
n_val = int(0.2 * n)
n_train = n - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

model = SimpleCNN(out_dim=71).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

def accuracy(pred, true):
    return (pred == true).mean()

@torch.no_grad()
def eval_epoch():
    model.eval()
    all_true = []
    all_pred = []
    for x, y in val_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    return accuracy(yp, yt)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x, y in train_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        opt.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item()

    val_acc = eval_epoch()
    print(f"Epoch {epoch:03d} | train_loss={total_loss:.4f} | val_fine_acc={val_acc:.4f}")

Path("models_baselines").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models_baselines/fine_only.pt")
print("Saved models_baselines/fine_only.pt")
