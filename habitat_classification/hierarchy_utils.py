# habitat_classification/hierarchy_utils.py
import json
import torch
from pathlib import Path

def load_fine_masks(device="cuda"):
    """
    Returns:
      coarse_to_fine_mask: Tensor (13, 71) bool
    """
    with open(Path("data/hierarchy.json")) as f:
        h = {int(k): int(v) for k, v in json.load(f).items()}

    num_coarse = max(h.values()) + 1
    num_fine = max(h.keys()) + 1

    mask = torch.zeros(num_coarse, num_fine, dtype=torch.bool)
    for fine, coarse in h.items():
        mask[coarse, fine] = True

    return mask.to(device)
