"""Eight encoder convolutions, two raw heads; CPU postprocessing is separate."""
from .runtime import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, widths):
        super().__init__()
        ci = 1
        for (stage, co) in enumerate(widths, 1):
            setattr(self, f'conv{stage}a', nn.Conv2d(ci, co, 3, padding=1))
            setattr(self, f'conv{stage}b', nn.Conv2d(co, co, 3, padding=1))
            ci = co
        self.convPa = nn.Conv2d(ci, 128, 3, padding=1)
        self.convPb = nn.Conv2d(128, 65, 1)
        self.convDa = nn.Conv2d(ci, 128, 3, padding=1)
        self.convDb = nn.Conv2d(128, 256, 1)

    def forward(self, x):
        for stage in range(1, 5):
            x = F.relu(getattr(self, f'conv{stage}a')(x))
            x = F.relu(getattr(self, f'conv{stage}b')(x))
            if stage < 4:
                x = F.max_pool2d(x, 2)
        return (self.convPb(F.relu(self.convPa(x))), self.convDb(F.relu(self.convDa(x))))
