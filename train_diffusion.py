import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from dlfm import LightweightDiffusion

class PairDataset(Dataset):
    def __init__(self, low_img_dir, normal_img_dir, img_size=320):
        self.low_paths = []
        self.normal_paths = []
        
        low_files = os.listdir(low_img_dir)
        print(f"Low light images found: {len(low_files)}")
        
        for f in low_files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            base = f.split('_bf')[0]
            
            for ext in ['.png', '.jpg', '.jpeg']:
                normal_path = os.path.join(normal_img_dir, base + ext)
                if os.path.exists(normal_path):
                    self.low_paths.append(os.path.join(low_img_dir, f))
                    self.normal_paths.append(normal_path)
                    break
        
        self.img_size = img_size
        print(f"Found {len(self.low_paths)} paired images")
        
        if len(self.low_paths) > 0:
            print(f"Example low: {self.low_paths[0]}")
            print(f"Example normal: {self.normal_paths[0]}")
        
    def __len__(self):
        return len(self.low_paths)
    
    def __getitem__(self, idx):
        low = cv2.imread(self.low_paths[idx])
        normal = cv2.imread(self.normal_paths[idx])
        
        if low is None or normal is None:
            return self.__getitem__((idx + 1) % len(self.low_paths))
        
        low = cv2.resize(low, (self.img_size, self.img_size))
        normal = cv2.resize(normal, (self.img_size, self.img_size))
        low = torch.from_numpy(low).float().permute(2,0,1) / 255.0
        normal = torch.from_numpy(normal).float().permute(2,0,1) / 255.0
        return low, normal

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    dataset = PairDataset(
        low_img_dir='datasets_lowlight/BIM/images/train',
        normal_img_dir='datasets/BIM/images/train',
        img_size=320
    )
    
    if len(dataset) == 0:
        print("No paired images found. Exiting.")
        exit()
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = LightweightDiffusion(channels=3, timesteps=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.MSELoss()

    print("Starting diffusion pre-training...")
    for epoch in range(50):
        total_loss = 0
        for low, normal in dataloader:
            low, normal = low.to(device), normal.to(device)
            optimizer.zero_grad()
            enhanced = model(low)
            loss = criterion(enhanced, normal)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/50, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'diffusion_pretrained.pt')
    print("Diffusion model saved as diffusion_pretrained.pt")