基于YOLOv10的APEX的敌我识别辅助瞄准

---
基本快完成了 重新整理下 参考一下几位的仓库代码（copy + 修改） 到 yolov10
1. [罗技鼠标驱动移动代码和加载模型detect](https://github.com/EthanH3514/AL_Yolo)
2. [PID瞄准](https://github.com/Tang895/yolov5-apex-tang/blob/main/apex-yolov5/mouse_lock.py)
3. [dxshot截图终于不卡了](https://github.com/AI-M-BOT/DXcam/releases)
---
  
模型的训练和加载与YOLOv8如出一辙,但是有些地方值得注意
1. 全程应该跟随官方教程来做 https://github.com/THU-MIG/yolov10
2. 在train的时候，使用`train.py`,在detect代码的时候使用官方包`ultralytics` (我是因为会报错才出此下策)
3. 导出模型两种方式都可使用 运行`export.py` 如果报错将v10官方仓库的`ultralytics`复制过去覆盖官方包,或者 `yolo export model=.\best.pt format=onnx opset=13 simplify max_det=100 conf=0.25 iou=0.65 nms`
4. 导出tensorRT模型 `trtexec --onnx=best.onnx --saveEngine=best.engine --fp16`

---
环境配置 **最好使用conda**
1. python=3.9
2. conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
3. pip install pycuda
4. cuda 11.8 对应的三件套 cuDNN tensorRT
5. 其他的缺什么安什么