from ultralytics import YOLOv10

model = YOLOv10('./best.pt')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.predict(source="C:\\Users\yao19\Downloads\yolov10\datasets\images\\train\\0a0b4e09cbd0d703fcb65143a3ed18cd.png", save=True, imgsz=640, conf=0.3)
