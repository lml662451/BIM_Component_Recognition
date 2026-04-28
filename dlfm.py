import torch
import torch.nn as nn
import torch.nn.functional as F

class LEF(nn.Module):
    def __init__(self, kernel_size=15):
        super().__init__()
        self.kernel_size = kernel_size
        
    def forward(self, x):
        low_freq = F.avg_pool2d(x, self.kernel_size, stride=1, padding=self.kernel_size//2)
        high_freq = x - low_freq
        return x + high_freq * 0.3

class LightweightDiffusion(nn.Module):
    def __init__(self, channels=3, timesteps=4):
        super().__init__()
        self.timesteps = timesteps
        self.denoiser = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1)
        )
        
    def forward(self, x):
        noise = torch.randn_like(x) * 0.03
        x_noisy = x + noise
        for _ in range(self.timesteps):
            x_noisy = x_noisy - self.denoiser(x_noisy) * 0.03
        return torch.clamp(x_noisy, 0, 1)

class DLFM(nn.Module):
    def __init__(self, channels=3, freeze_diffusion=True, pretrained_path=None):
        super().__init__()
        self.lef = LEF()
        self.diffusion = LightweightDiffusion(channels)
        
        if pretrained_path:
            self.diffusion.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print(f"Loaded pretrained diffusion weights from {pretrained_path}")
        
        if freeze_diffusion:
            for param in self.diffusion.parameters():
                param.requires_grad = False
        
        self.fem = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.lef(x)
        x = self.diffusion(x)
        x = x + self.fem(x) * 0.1
        return torch.clamp(x, 0, 1)