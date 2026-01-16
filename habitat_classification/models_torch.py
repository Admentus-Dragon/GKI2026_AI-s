# habitat_classification/models_torch.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, out_dim, in_ch=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 35 → 17
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 17 → 8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class BetterCNN(nn.Module):
    """
    Conv-BN-ReLU blocks + GAP + Dropout.
    Works well for small patches and reduces overfit.
    """
    def __init__(self, out_dim, in_ch=15, base=32, dropout=0.2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.b1 = block(base, base)          # (base, 35, 35)
        self.p1 = nn.MaxPool2d(2)            # -> (base, 17, 17)

        self.b2 = block(base, base * 2)      # -> (2b, 17, 17)
        self.p2 = nn.MaxPool2d(2)            # -> (2b, 8, 8)

        self.b3 = block(base * 2, base * 4)  # -> (4b, 8, 8)

        self.gap = nn.AdaptiveAvgPool2d(1)   # -> (4b, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base * 4, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.p1(x)
        x = self.b2(x)
        x = self.p2(x)
        x = self.b3(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
