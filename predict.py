from ultralytics10 import YOLOv10

model = YOLOv10('./best.pt')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.predict(source=".\datasets\images\\train\\1a8cbf306b1867e4b3f144439e231135.png", save=True, imgsz=640, conf=0.3)
