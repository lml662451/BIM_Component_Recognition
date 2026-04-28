import torch
from aia_yolo import AIAYOLO
from ultralytics import YOLO
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

num_classes = 13

print("Building AIA-YOLO model...")
model = AIAYOLO(cfg='yolov8n.yaml', nc=num_classes).to(device)

pretrained_path = 'runs/detect/train2/weights/best.pt'
print(f"Loading pretrained weights: {pretrained_path}")
pretrained_dict = torch.load(pretrained_path, map_location=device)
model.load_state_dict(pretrained_dict, strict=False)
print("Pretrained weights loaded, AIA module initialized randomly")

trainable_params = 0
for param in model.parameters():
    param.requires_grad = True
    trainable_params += param.numel()
print(f"All parameters trainable: {trainable_params/1e6:.2f}M")

custom_model = YOLO('yolov8n.pt')
custom_model.model = model
custom_model.save('aia_yolov8_init.pt')
print("Initialized model saved as aia_yolov8_init.pt")

print("Starting training...")
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
    name='train6',
    patience=20,
    exist_ok=True
)

print("\nTraining completed.")
print(f"New model saved at: D:/BIM_Component_Recognition/runs/detect/train6/weights/best.pt")
print(f"Original model saved at: runs/detect/train2/weights/best.pt")