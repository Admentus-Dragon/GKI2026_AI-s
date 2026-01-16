import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import json
from torch.utils.data import WeightedRandomSampler
from models_torch import BetterCNN, SimpleCNN
from hierarchy_utils import load_fine_masks
from augmentations_torch import AugConfig, AspectConfig, apply_aspect_transform, apply_geometry_aug, apply_spectral_aug


from models_torch import SimpleCNN
from hierarchy_utils import load_fine_masks

DEVICE = "cuda"
BATCH = 64
EPOCHS = 100
VAL_FRAC = 0.2
SEED = 42


DEVICE = "cuda"
BATCH = 256          # consider lowering from 512 when using aug
EPOCHS = 100
VAL_FRAC = 0.1
SEED = 42

# ---- toggles ----
USE_BETTER_CNN = True

ASPECT_CFG = AspectConfig(mode="sincos")  # "raw" | "sincos" | "drop"

AUG_CFG = AugConfig(
    enable=True,
    random_rot90=True,
    random_flip=True,
    spectral_noise_std=0.01,        # try 0.005–0.02
    spectral_mult_jitter=0.05,      # try 0.02–0.08
    band_dropout_p=0.10,            # try 0.0–0.15
    band_dropout_max_bands=2
)

USE_ADAMW = True
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25


def infer_in_channels(aspect_cfg: AspectConfig) -> int:
    # original channels: 15
    if aspect_cfg.mode == "raw":
        return 15
    if aspect_cfg.mode == "sincos":
        return 16  # aspect 1ch -> 2ch
    if aspect_cfg.mode == "drop":
        return 14
    raise ValueError(aspect_cfg.mode)


class HabitatDataset(Dataset):
    def __init__(self, train: bool, aug_cfg: AugConfig, aspect_cfg: AspectConfig, seed: int = 42):
        self.train = train
        self.aug_cfg = aug_cfg
        self.aspect_cfg = aspect_cfg

        self.patches = np.load("data/train/patches.npy", mmap_mode="r")
        df = pd.read_csv("data/train.csv")
        self.fine = df["vistgerd_idx"].to_numpy()
        self.coarse = df["vistlendi_idx"].to_numpy()

        # generator for deterministic-ish augmentations per worker
        self.base_seed = seed

    def __len__(self):
        return len(self.fine)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx].copy()).float()  # (15,35,35)

        # aspect transform first (changes channel count)
        x = apply_aspect_transform(x, self.aspect_cfg, aspect_ch=14)

        if self.train and self.aug_cfg.enable:
            # Make per-sample randomness stable across workers:
            g = torch.Generator()
            g.manual_seed(self.base_seed + idx)

            # geometry
            x = apply_geometry_aug(x, self.aug_cfg, g=g)

            # spectral (ONLY first 12 channels remain spectral if aspect mode changes?)
            # Spectral is always channels 0..11 in original; aspect conversion only touches the last channel.
            x = apply_spectral_aug(x, self.aug_cfg, g=g, spectral_channels=12)

        return x, int(self.coarse[idx]), int(self.fine[idx])

def focal_loss(logits, targets, alpha=None, gamma=2.0):
    """
    logits: (B, C)
    targets: (B,)
    alpha: (C,) tensor of class weights or None
    """
    logp = torch.nn.functional.log_softmax(logits, dim=1)
    p = torch.exp(logp)
    tgt_logp = logp[torch.arange(logits.size(0), device=logits.device), targets]
    tgt_p = p[torch.arange(logits.size(0), device=logits.device), targets]

    loss = -(1 - tgt_p).pow(gamma) * tgt_logp
    if alpha is not None:
        loss = loss * alpha[targets]
    return loss.mean()


def load_fine_class_names():
    p = Path("data/class_names.json")
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Try a couple common shapes safely
    # If it's { "vistgerd": [...names...] } or similar, adjust here.
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # best guess: pick the longest list value
        lists = [v for v in obj.values() if isinstance(v, list)]
        if lists:
            return max(lists, key=len)
    return None


@torch.no_grad()
def evaluate(
    coarse_model,
    fine_model,
    dl,
    mask,
    fine_names=None,
    top_k_classes=10,
    top_k_confusions=5,
):
    coarse_model.eval()
    fine_model.eval()

    fine_true = []
    fine_pred = []
    coarse_true = []
    coarse_pred = []

    for x, coarse_y, fine_y in dl:
        x = x.to(DEVICE, non_blocking=True)

        # tensors -> numpy safely (no NumPy 2.x warning)
        coarse_y_np = torch.as_tensor(coarse_y).cpu().numpy().astype(np.int64)
        fine_y_np = torch.as_tensor(fine_y).cpu().numpy().astype(np.int64)

        # Stage 1
        coarse_logits = coarse_model(x)

        # top-2 coarse indices per sample: (B, 2)
        top2 = coarse_logits.topk(k=2, dim=1).indices  # torch int64 on GPU

        # top-1 coarse prediction (still useful for coarse accuracy)
        coarse_hat = top2[:, 0].detach().cpu().numpy().astype(np.int64)

        # Stage 2 (mask by union of top-2 coarse masks)
        fine_logits = fine_model(x)  # (B, 71)

        m1 = mask[top2[:, 0]]  # (B, 71) boolean
        m2 = mask[top2[:, 1]]  # (B, 71) boolean
        batch_mask = m1 | m2  # union

        fine_logits = fine_logits.masked_fill(~batch_mask, -1e9)
        fine_hat = fine_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

        fine_true.append(fine_y_np)
        fine_pred.append(fine_hat)
        coarse_true.append(coarse_y_np)
        coarse_pred.append(coarse_hat)

    y_f = np.concatenate(fine_true)
    p_f = np.concatenate(fine_pred)
    y_c = np.concatenate(coarse_true)
    p_c = np.concatenate(coarse_pred)

    fine_acc = accuracy_score(y_f, p_f)
    fine_f1w = f1_score(y_f, p_f, average="weighted")
    coarse_acc = accuracy_score(y_c, p_c)

    # ---- Confusion summary (fine) ----
    cm = confusion_matrix(y_f, p_f, labels=np.arange(71))
    support = cm.sum(axis=1)  # true counts per class
    correct = np.diag(cm)
    per_class_acc = np.where(support > 0, correct / support, 0.0)

    # choose "worst" by accuracy among classes that exist in val
    present = np.where(support > 0)[0]
    worst = present[np.argsort(per_class_acc[present])[:top_k_classes]]

    def name(i):
        if fine_names and i < len(fine_names):
            return f"{i}: {fine_names[i]}"
        return str(i)

    print("\n--- Confusion summary (fine) ---")
    for i in worst:
        row = cm[i].copy()
        row[i] = 0  # exclude correct
        conf_idx = row.argsort()[::-1]
        conf_idx = [j for j in conf_idx if row[j] > 0][:top_k_confusions]
        print(f"Class {name(i)} | support={support[i]} | acc={per_class_acc[i]:.3f}")
        for j in conf_idx:
            print(f" -> predicted as {name(j)} : {row[j]}")
    print("--- end confusion summary ---\n")

    return fine_acc, fine_f1w, coarse_acc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Build a base dataset for splitting indices
    base_ds = HabitatDataset(train=False, aug_cfg=AUG_CFG, aspect_cfg=ASPECT_CFG, seed=SEED)

    n_val = int(len(base_ds) * VAL_FRAC)
    n_train = len(base_ds) - n_val

    train_split, val_split = random_split(
        range(len(base_ds)),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Datasets that use the split indices
    train_ds = torch.utils.data.Subset(
        HabitatDataset(train=True, aug_cfg=AUG_CFG, aspect_cfg=ASPECT_CFG, seed=SEED),
        train_split.indices
    )
    val_ds = torch.utils.data.Subset(
        HabitatDataset(train=False, aug_cfg=AUG_CFG, aspect_cfg=ASPECT_CFG, seed=SEED),
        val_split.indices
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    in_ch = infer_in_channels(ASPECT_CFG)

    if USE_BETTER_CNN:
        coarse_model = BetterCNN(out_dim=13, in_ch=in_ch, dropout=DROPOUT).to(DEVICE)
        fine_model = BetterCNN(out_dim=71, in_ch=in_ch, dropout=DROPOUT).to(DEVICE)
    else:
        coarse_model = SimpleCNN(out_dim=13, in_ch=in_ch).to(DEVICE)
        fine_model = SimpleCNN(out_dim=71, in_ch=in_ch).to(DEVICE)

    if USE_ADAMW:
        opt = torch.optim.AdamW(
            list(coarse_model.parameters()) + list(fine_model.parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
    else:
        opt = torch.optim.Adam(
            list(coarse_model.parameters()) + list(fine_model.parameters()),
            lr=LR
        )

    mask = load_fine_masks(device=DEVICE)
    fine_names = load_fine_class_names()

    best_f1 = -1.0
    out_dir = Path("models_two_stage_torch")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        coarse_model.train()
        fine_model.train()

        total_loss = 0.0
        n_batches = 0

        for x, coarse_y, fine_y in train_dl:
            x = x.to(DEVICE, non_blocking=True)
            coarse_y = torch.as_tensor(coarse_y, device=DEVICE)
            fine_y = torch.as_tensor(fine_y, device=DEVICE)

            opt.zero_grad()

            # Stage 1
            coarse_logits = coarse_model(x)
            loss_coarse = torch.nn.functional.cross_entropy(coarse_logits, coarse_y)

            # Stage 2 (masked fine logits using TRUE coarse during training)
            fine_logits = fine_model(x)
            batch_mask = mask[coarse_y]  # (B,71)
            fine_logits = fine_logits.masked_fill(~batch_mask, -1e9)
            #loss_fine = torch.nn.functional.cross_entropy(fine_logits, fine_y)
            loss_fine = focal_loss(fine_logits, fine_y, alpha=None, gamma=2.0)



            loss = loss_coarse + loss_fine
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation metrics (full pipeline)
        val_acc, val_f1w, val_coarse_acc = evaluate(
            coarse_model, fine_model, val_dl, mask, fine_names=fine_names
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_fine_acc={val_acc:.4f} | "
            f"val_fine_f1w={val_f1w:.4f} | "
            f"val_coarse_acc={val_coarse_acc:.4f}"
        )

        # Save best by weighted F1
        if val_f1w > best_f1:
            best_f1 = val_f1w
            torch.save(coarse_model.state_dict(), out_dir / "coarse.pt")
            torch.save(fine_model.state_dict(), out_dir / "fine.pt")
            print(f" ✅ saved best weights (val_f1w={best_f1:.4f})")


if __name__ == "__main__":
    main()
