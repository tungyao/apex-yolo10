import os.path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from ultralytics10 import YOLOv10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# torch.cuda.set_device(0)
# 加载预训练模型（例如YOLOv8n）
model = YOLOv10('./yolov10s.pt')

# 使用 'yolo train' 命令训练模型

results = model.train(
    data=os.path.abspath('./datasets/dataset.yaml'),  # 数据集配置文件路径
    epochs=20,  # 训练轮数
    imgsz=640,  # 图像大小
    batch=16,  # 批次大小
    name='apex_yolov10_experiment2',  # 实验名称
    device=0,
    workers=0
)

# 保存训练好的模型
model.save('apex_yolov10_model.pt')
