import os.path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from ultralytics import YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# torch.cuda.set_device(0)
# 加载预训练模型（例如YOLOv8n）
# model = YOLOv10('./best.pt')
# model.export(format="engine")
tensorrt_model = YOLO("best.engine",task="detect")
results = tensorrt_model("E:\code\\apex-yolo10\datasets\images\\val\\2d7cf30e47743c7a8bfc7720533704fa.png")
print(results)