# habitat_classification/model.py
import torch
import numpy as np

from models_torch import SimpleCNN, BetterCNN
from hierarchy_utils import load_fine_masks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COARSE_PATH = "models_two_stage_torch/coarse.pt"
FINE_PATH   = "models_two_stage_torch/fine.pt"

def _infer_arch_and_in_ch(state_dict: dict) -> tuple[str, int]:
    """
    Detect whether checkpoint is BetterCNN or SimpleCNN by key names,
    and infer input channels from first conv weight shape.
    """
    keys = list(state_dict.keys())

    if any(k.startswith("stem.") for k in keys) and any(k.startswith("b1.") for k in keys):
        arch = "BetterCNN"
        # BetterCNN stem conv is stem.0.weight with shape (base, in_ch, 3, 3)
        in_ch = int(state_dict["stem.0.weight"].shape[1])
        return arch, in_ch

    if any(k.startswith("net.") for k in keys):
        arch = "SimpleCNN"
        # SimpleCNN first conv is net.0.weight with shape (32, in_ch, 3, 3)
        in_ch = int(state_dict["net.0.weight"].shape[1])
        return arch, in_ch

    raise RuntimeError("Unknown checkpoint format: cannot infer architecture.")

def _build_model(out_dim: int, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    arch, in_ch = _infer_arch_and_in_ch(sd)

    if arch == "BetterCNN":
        model = BetterCNN(out_dim=out_dim, in_ch=in_ch, dropout=0.0)
    else:
        model = SimpleCNN(out_dim=out_dim, in_ch=in_ch)

    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model, in_ch

# Load models (also learn expected in_ch)
coarse_model, IN_CH_COARSE = _build_model(out_dim=13, ckpt_path=COARSE_PATH)
fine_model,   IN_CH_FINE   = _build_model(out_dim=71, ckpt_path=FINE_PATH)

if IN_CH_COARSE != IN_CH_FINE:
    raise RuntimeError(f"Checkpoint mismatch: coarse in_ch={IN_CH_COARSE}, fine in_ch={IN_CH_FINE}")

IN_CH = IN_CH_COARSE
mask = load_fine_masks(device=DEVICE)

def _preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """
    patch: numpy (15,35,35) from the competition input.
    If the model expects 16 channels, we convert aspect -> sin/cos.
    If it expects 15, we pass through.
    """
    x = torch.from_numpy(patch).float()

    if IN_CH == 15:
        pass
    elif IN_CH == 16:
        # aspect -> sincos (assumes aspect is channel 14 in degrees)
        aspect_deg = x[14]
        aspect_rad = aspect_deg * (np.pi / 180.0)
        asp_sin = torch.sin(aspect_rad)
        asp_cos = torch.cos(aspect_rad)
        x = torch.cat([x[:14], asp_sin.unsqueeze(0), asp_cos.unsqueeze(0)], dim=0)
    elif IN_CH == 14:
        # aspect dropped
        x = torch.cat([x[:14]], dim=0)
    else:
        raise RuntimeError(f"Unsupported IN_CH={IN_CH}. Expected 14, 15, or 16.")

    return x.unsqueeze(0)

@torch.no_grad()
def predict(patch: np.ndarray) -> int:
    x = _preprocess_patch(patch).to(DEVICE)

    coarse_logits = coarse_model(x)
    top2 = coarse_logits.topk(k=2, dim=1).indices  # (1,2)

    fine_logits = fine_model(x)  # (1,71)
    m1 = mask[top2[:, 0]]
    m2 = mask[top2[:, 1]]
    batch_mask = m1 | m2

    fine_logits = fine_logits.masked_fill(~batch_mask, -1e9)
    return int(fine_logits.argmax(dim=1).item())
