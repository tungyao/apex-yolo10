import os.path

import torch
from ultralytics import YOLOv10 as YOLO

# 加载预训练模型（例如YOLOv8n）
model = YOLO('./yolov10s.pt')

# 使用 'yolo train' 命令训练模型

results = model.train(
    data=os.path.abspath('./datasets/data.yaml'),  # 数据集配置文件路径
    epochs=10,  # 训练轮数
    imgsz=640,  # 图像大小
    batch=16,  # 批次大小
    name='apex_yolov10_experiment'  # 实验名称
)

# 保存训练好的模型
model.save('apex_yolov10_model.pt')
