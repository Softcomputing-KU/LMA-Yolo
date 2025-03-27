

from ultralytics import YOLO

model = YOLO(r"/root/autodl-tmp/ultralytics-main/ultralytics/cfg/models/v5/yolov5.yaml")
# model.load("yolov8n.pt")

results = model.train(data=r"/root/autodl-tmp/dataset_remote_sensing/ROSD/rosd.yaml", imgsz=640, epochs=200, batch=16, device=0, optimizer="SGD", workers=12,amp=True)
metrics=model.val()


