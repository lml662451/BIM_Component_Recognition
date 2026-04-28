import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from dlfm import DLFM

class DLFMYOLO(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', nc=None, verbose=True):
        self._dlfm_created = False
        super().__init__(cfg=cfg, nc=nc, verbose=verbose)
        self.dlfm = DLFM(channels=3)
        self._dlfm_created = True
        
    def forward(self, x, *args, **kwargs):
        if self._dlfm_created:
            x = self.dlfm(x)
        return super().forward(x, *args, **kwargs)
        
    def _forward(self, x):
        if hasattr(self, 'dlfm') and self._dlfm_created:
            x = self.dlfm(x)
        return super()._forward(x)