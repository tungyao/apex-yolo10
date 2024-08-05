import cv2
import pynput
import torch
from ultralytics.utils.ops import non_max_suppression

from capture import LoadScreen, LEFT, TOP, SIZE
from ultralytics.nn.autobackend import AutoBackend
import time

from general import xyxy2xywh, scale_boxes
from mouse_lock import MouseLock

#要处理2k屏
screen_width, screen_height = (1920, 1080)
offet_Shot_Screen = 0  # 屏幕截图偏移量,
left_top_x = LEFT
left_top_y = TOP
right_bottom_x = LEFT + SIZE
right_bottom_y = TOP + SIZE
shot_Width = 640   # 截屏区域的实际大小需要乘以2，因为是计算的中心点
shot_Height = 640
lock_mode  = 0



def on_click(x, y, button, pressed):
    global lock_mode, isX2Down
    if button == button.right:  # 使用鼠标上面一个侧键切换锁定模式，需要在apex设置中调整按键避免冲突
        isX2Down = pressed
        lock_mode = pressed
        print(f'isX2Down: {isX2Down}')


listener = pynput.mouse.Listener(on_click=on_click)
listener.start()
mouselock = MouseLock(shot_Width, shot_Height)


class YOLOv10Detect:
    def __init__(self):
        self.showFPS = False
        self.save_path = "datasets/img"
        self.imgsz = (640, 640)

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bs = 1
        dataset = LoadScreen()
        model = AutoBackend("./model.engine", device=device, dnn=False, data="./datasets/dataset.yaml", fp16=True)
        stride, names, pt = model.stride, model.names, model.pt
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *self.imgsz))  # warmup
        frame_cnt = 0
        that_time = 0
        # cv2.namedWindow('YOLOv10 Detection', cv2.WINDOW_NORMAL)
        for path, im, img0, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            pred = model(im, augment=False, visualize=False)
            # TODO 调节这里的参数 pt和engine模型输出pred不一样
            pred = non_max_suppression(pred, 0.5, 0.1, 0, False, max_det=10)
            aims = []
            for i, det in enumerate(pred):
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh)  # label format
                        aim = ('%g ' * len(line)).rstrip() % line
                        aim = aim.split(' ')
                        # print("aim:",aim)
                        aims.append(aim)
            if len(aims):
                mouselock.set_lock_state(lock_mode)
                mouselock.lock(aims)
                # 屏幕显示拉框
                # for i, det in enumerate(aims):
                #     tag, x_center, y_center, width, height = det
                #     x_center, width = shot_Width * float(x_center), shot_Width * float(width)
                #     y_center, height = shot_Height * float(y_center), shot_Height * float(height)
                #     top_left = (int(x_center - width / 2.0), int(y_center - height / 2.0))
                #     bottom_right = (int(x_center + width / 2.0), int(y_center + height / 2.0))
                #     color = (0, 0, 255)  # BGR
                #     cv2.rectangle(img0, top_left, bottom_right, color, thickness=2)
            else:
                mouselock.set_lock_state(False)  # no target, unlock
            # cv2.imshow("aaa", img0)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            if self.showFPS:
                now_time = time.time()
                frame_cnt += 1
                duration_time = now_time - that_time
                fps = frame_cnt / duration_time
                if frame_cnt >= 100:
                    that_time = now_time
                    frame_cnt = 0
                print("Fps is ", fps)
