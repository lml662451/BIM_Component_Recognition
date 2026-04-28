import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

class AIAModule(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        residual = torch.tanh(self.conv3(x)) * 0.05
        return torch.clamp(identity + residual, 0, 1)

class AIAYOLO(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', nc=None, verbose=True):
        self._aia_created = False
        super().__init__(cfg=cfg, nc=nc, verbose=verbose)
        self.aia = AIAModule(channels=3)
        self._aia_created = True
        
    def forward(self, x, *args, **kwargs):
        if self._aia_created:
            x = self.aia(x)
        return super().forward(x, *args, **kwargs)
        
    def _forward(self, x):
        if hasattr(self, 'aia') and self._aia_created:
            x = self.aia(x)
        return super()._forward(x)