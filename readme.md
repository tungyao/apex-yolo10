c++ runtime from https://github.com/abbodi1406/vcredist/releases

Install nuitka by using command pip install nuitka
Go to your working path, run python -m nuitka --mingw64 --module --show-progress --no-pyi-file --remove-output --follow-import-to=dxcam dxshot.py

https://github.com/AI-M-BOT/DXcam/releases

run cut.py

yolo export model=.\best.pt format=onnx opset=13 simplify max_det=100 conf=0.25 iou=0.65 nms

trtexec --onnx=best.onnx --saveEngine=best.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16
