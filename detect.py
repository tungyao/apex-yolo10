import torch
from ultralytics.utils.ops import non_max_suppression

from capture import LoadScreen
from ultralytics.nn.autobackend import AutoBackend
import time
import cv2
from math import atan2
import os
from mouse_driver.MouseMove import ghub_mouse_move
# from mouse_driver.MouseMove import ghub_mouse_move
from win32mouse.MouseMove import mouse_move
from pynput.mouse import Listener, Button

class YOLOv10Detect:
    def __init__(self):
        self.showFPS = 1
        self.size = 640
        self.offset = torch.tensor([self.size / 2, self.size / 2], device=0)
        self.mul = 0.4
        self.smooth = 0.42
        self.mouse_on_click = True
        self.showFPS = True
        self.enemy_label = 0
        self.should_stop = False  # flag to stop
        self.enable_mouse_lock = True
    def is_click(self, x, y, button, pressed):
        if self.enable_mouse_lock:
            if button in [Button.left, Button.right]:
                if pressed:
                    self.mouse_on_click = True
                    print("鼠标锁定已开启")
                else:
                    self.mouse_on_click = False
                    print("鼠标锁定已关闭")
    def get_dis(self, vec):  # must not null
        return (((vec[0] + vec[2] - self.size) / 2) ** 2 + ((vec[1] + vec[3] - self.size) / 2) ** 2) ** (1 / 2)

    def lock_target(self, target):
        rel_target = [item * self.smooth for item in
                      [(target[0] + target[2] - self.size) / 2, (target[1] + target[3] - self.size) / 2]]
        move_rel_x, move_rel_y = [atan2(item, self.size) * self.size for item in rel_target]
        ghub_mouse_move(move_rel_x, move_rel_y)

    def run(self):
        device = torch.device("cuda")
        imgsz = (640, 640)
        bs = 1
        dataset = LoadScreen()
        save_path = "datasets/img"
        model = AutoBackend("./best.engine", device=device, dnn=False, data="./datasets/dataset.yaml", fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        frame_cnt = 0
        that_time = 0
        cv2.namedWindow('YOLOv5 Detection', cv2.WINDOW_NORMAL)
        for im, im0 in dataset:  # main loop

            im = torch.from_numpy(im).to(model.device)

            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32

            im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference

            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, 0.3, 0.45, None, False, max_det=10)

            bound = pred[0].cpu().numpy()
            # 显示图像
            if self.enable_mouse_lock and len(bound) > 0:
                # chose target which is closest to center
                cv2.imwrite(os.path.join(save_path, str(time.time()) + ".jpg"), im0)
                target = bound[0]
                min_dis = self.get_dis(target)
                for vec in bound:
                    now_dis = self.get_dis(vec)
                    class_name = model.names[vec[5]]

        # 在图像上绘制类别名称和置信度
                    label = f'{class_name} {vec[4]}'
                    cv2.putText(im0, label, (int(vec[0]), int(vec[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(im0, (int(vec[0]), int(vec[1])), (int(vec[2]), int(vec[3])), (0, 255, 0), 2)
                    if now_dis < min_dis and vec[5] == self.enemy_label:
                        # only update target when it is enemy
                        target = vec
                        min_dis = now_dis
                if self.enable_mouse_lock and self.mouse_on_click and target[5] == self.enemy_label:
                    # only lock target when label is enemy and mouse is clicked
                    # self.lock_target(target)
                    pass
            cv2.imshow("aaa",im0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 如果按下 q 键，则退出循环
                break
            # FPS calculate
            if self.showFPS:
                now_time = time.time()
                frame_cnt += 1
                duration_time = now_time - that_time
                fps = frame_cnt / duration_time
                if frame_cnt >= 100:
                    that_time = now_time
                    frame_cnt = 0

                print("Fps is ", fps)


    def stop(self):
        self.should_stop = True

    def start_mouse(self):
        self.enable_mouse_lock = True

    def stop_mouse(self):
        self.enable_mouse_lock = False
