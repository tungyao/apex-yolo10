import numpy as np
import pynput
import torch
from ultralytics.utils.ops import non_max_suppression

from capture import LoadScreen, LEFT, TOP, SIZE
from ultralytics.nn.autobackend import AutoBackend
import time
import cv2

from general import xyxy2xywh, scale_boxes
from mouse_lock import MouseLock

screen_width, screen_height = (1920, 1080)
offet_Shot_Screen = 0  # 屏幕截图偏移量,
left_top_x = LEFT
left_top_y = TOP
right_bottom_x = LEFT + SIZE
right_bottom_y = TOP + SIZE
shot_Width = 640 *2  # 截屏区域的实际大小需要乘以2，因为是计算的中心点
shot_Height = 640*2
lock_mode = False  # don's edit this

window_Name = "apex-tang"
auto = True

lock_mode = False  # don's edit this
lock_button = "left"  # 无用，apex为按住鼠标左或者右其中一个为就为lock模式，建议在游戏设置按住开镜
isShowDebugWindow = True  # 可修改为True，会出现调试窗口
isRightKeyDown = False
isLeftKeyDown = False
isX2KeyDown = False
mouseFlag = 0  # 0, 1 2 3



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
        self.aim = False
        self.showFPS = 1
        self.size = 640
        self.offset = torch.tensor([self.size / 2, self.size / 2], device=0)
        self.mul = 0.4
        self.smooth = 0.42
        self.mouse_on_click = True
        self.showFPS = False
        self.enemy_label = 0
        self.should_stop = False  # flag to stop
        self.enable_mouse_lock = True
        self.last_time = 0

    def run(self):
        print(1)
        device = torch.device("cuda")
        imgsz = (640, 640)
        bs = 1
        dataset = LoadScreen()
        save_path = "datasets/img"
        model = AutoBackend("./model.engine", device=device, dnn=False, data="./datasets/dataset.yaml", fp16=True)
        stride, names, pt = model.stride, model.names, model.pt
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        frame_cnt = 0
        that_time = 0
        cv2.namedWindow('YOLOv10 Detection', cv2.WINDOW_NORMAL)
        cnt = 0
        imgsz = [640, 640],  # inference size (height, width)
        conf_thres = 0.25,  # confidence threshold
        iou_thres = 0.45,  # NMS IOU threshold
        max_det = 100,  # maximum detections per image
        classes = 0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False,  # class-agnostic NMS

        for path, im, img0, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(model.device)

            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32

            im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, 0.4, 0.1, classes, False, max_det=10)
            aims = []
            for i, det in enumerate(pred):
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        # bbox:(tag, x_center, y_center, x_width, y_width)
                        """
                        0 ct_head  1 ct_body  2 t_head  3 t_body
                        """
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh)  # label format
                        aim = ('%g ' * len(line)).rstrip() % line
                        aim = aim.split(' ')
                        # print("aim:",aim)
                        aims.append(aim)
            if len(aims):
                mouselock.set_lock_state(lock_mode)
                mouselock.lock(aims)
                # print(f"set mouse lock state to {lock_mode}")
                # print(f"mouse lock state: {lock_state}")
                for i, det in enumerate(aims):
                    tag, x_center, y_center, width, height = det
                    x_center, width = shot_Width * float(x_center), shot_Width * float(width)
                    y_center, height = shot_Height * float(y_center), shot_Height * float(height)
                    top_left = (int(x_center - width / 2.0), int(y_center - height / 2.0))
                    bottom_right = (int(x_center + width / 2.0), int(y_center + height / 2.0))
                    color = (0, 0, 255)  # BGR
                    cv2.rectangle(img0, top_left, bottom_right, color, thickness=2)
            else:
                mouselock.set_lock_state(False)  # no target, unlock
            cv2.imshow("aaa", img0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            # bound = pred[0].cpu().numpy()
            # # 显示图像
            # if self.enable_mouse_lock and len(bound) > 0:
            #     # chose target which is closest to center
            #     # cv2.imwrite(os.path.join(save_path, str(time.time()) + ".jpg"), im0)
            #     target = bound[0]
            #     min_dis = self.get_dis(target)
            #     if time.time() - self.last_time > 2 and target[4] > 0.6:
            #         pass
            #         # print("capture")
            #         # cv2.imwrite(os.path.join(save_path, str(time.time()) + ".jpg"), im0)
            #
            #     for vec in bound:
            #         now_dis = self.get_dis(vec)
            #         class_name = model.names[vec[5]]
            #
            #         # 在图像上绘制类别名称和置信度
            #         label = f'{class_name} {vec[4]}'
            #         cv2.putText(im0, label, (int(vec[0]), int(vec[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
            #                     2)
            #         cv2.rectangle(im0, (int(vec[0]), int(vec[1])), (int(vec[2]), int(vec[3])), (0, 255, 0), 2)
            #         if now_dis < min_dis and vec[5] == self.enemy_label:
            #             # only update target when it is enemy
            #             target = vec
            #             min_dis = now_dis
            #     # print(target[5], self.enemy_label)
            #     if self.aim and target[5] == self.enemy_label:
            #         # only lock target when label is enemy and mouse is clicked
            #         self.lock_target(target)
            #         # pass
            # cv2.imshow("aaa", im0)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):  # 如果按下 q 键，则退出循环
            #     break
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
