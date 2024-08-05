import os.path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from ultralytics10 import YOLOv10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
model = YOLOv10('./best.pt')
model.export(format="onnx", opset=13, max_det=100, conf=0.25, iou=0.65, nms=True, simplify=True)
