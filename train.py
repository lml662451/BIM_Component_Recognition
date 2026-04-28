from ultralytics import YOLO

model = YOLO('./yolov8n.pt')

model.train(data='components.yaml', workers=0, epochs=30, batch=8)