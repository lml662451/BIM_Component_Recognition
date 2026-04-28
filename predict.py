from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train2/weights/best.pt')

def predict_image(image_path):
    results = model(image_path)
    for r in results:
        if len(r.boxes) == 0:
            print("未识别到任何构件")
        else:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"识别到: {cls_name}, 置信度: {conf:.2f}")
        annotated_img = r.plot()
        cv2.imshow('识别结果', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

predict_image('test_images/1139.png')