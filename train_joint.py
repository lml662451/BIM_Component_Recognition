import torch
from dlfm_yolo import DLFMYOLO
from dlfm import DLFM
from ultralytics import YOLO
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

num_classes = 13

print("Building DLFM-YOLO model with pretrained diffusion...")
model = DLFMYOLO(cfg='yolov8n.yaml', nc=num_classes).to(device)

model.dlfm = DLFM(channels=3, freeze_diffusion=True, pretrained_path='diffusion_pretrained.pt')

pretrained_path = 'runs/detect/train2/weights/best.pt'
print(f"Loading YOLO pretrained weights: {pretrained_path}")
pretrained_dict = torch.load(pretrained_path, map_location=device)
model.load_state_dict(pretrained_dict, strict=False)
print("YOLO weights loaded")

for param in model.dlfm.diffusion.parameters():
    param.requires_grad = False

custom_model = YOLO('yolov8n.pt')
custom_model.model = model
custom_model.save('dlfm_yolov8_joint_init.pt')
print("Joint model saved as dlfm_yolov8_joint_init.pt")

print("Starting joint training...")
results = custom_model.train(
    data='components_lowlight.yaml',
    epochs=60,
    imgsz=640,
    batch=16,
    device=device,
    workers=4,
    lr0=0.002,
    cls=0.2,
    box=0.05,
    project='D:/BIM_Component_Recognition/runs/detect',
    name='train_joint',
    patience=20,
    exist_ok=True
)

print("\nTraining completed.")
print(f"New model saved at: D:/BIM_Component_Recognition/runs/detect/train_joint/weights/best.pt")