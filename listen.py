import numpy as np
import win32api
import win32gui
import win32print
import win32con
from pynput import keyboard, mouse
import winsound
import time

detecting = False
listening = True
shift_pressed = False
mouse1_pressed = False
mouse2_pressed = False
left_lock = False  # lock on target when the left mouse button is pressed  # 左键锁, Left, 按鼠标左键时锁
right_lock = False  # lock when pressing the right mouse button (scoping)  # 右键锁, Right, 按鼠标右键(开镜)时锁
auto_fire = False
time_fire = time.time()
backforce = 0
screen_size = np.array([win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)])
screen_center = screen_size / 2
destination = screen_center
last = destination
width = 0
hDC = win32gui.GetDC(0)
scale = win32print.GetDeviceCaps(hDC, win32con.LOGPIXELSX) / 96
pre_error = integral = np.array([0., 0.])


def listen_init(args):
    global caps_lock
    caps_lock = args.caps_lock


def get_D_L():
    global detecting, listening, left_lock, caps_lock
    if caps_lock:
        if win32api.GetKeyState(0x14):
            if not left_lock:
                detecting = False
                left_lock = True
                winsound.Beep(800, 100)
        else:
            if left_lock:
                detecting = False
                left_lock = False
                winsound.Beep(400, 100)
    return detecting, listening


def listen_k_press(key):
    global detecting, listening, shift_pressed, left_lock, right_lock, auto_fire, caps_lock
    if key == keyboard.Key.home:
        detecting = False
        listening = False
        print("listeners stop")
        winsound.Beep(700, 100)
        winsound.Beep(600, 100)
        return False
    if key == keyboard.Key.shift:
        shift_pressed = True
        if not detecting:
            detecting = True
            print("Start detection: ", detecting)
    # if key == keyboard.Key.left:
    #     detecting = False
    #     left_lock = not left_lock
    #     winsound.Beep(800 if left_lock else 400, 200)
    if key == keyboard.Key.right:
        detecting = False
        right_lock = not right_lock
        winsound.Beep(900 if right_lock else 500, 200)
    if key == keyboard.Key.up:  # AUTO_FIRE is detected by EAC
        detecting = False
        auto_fire = not auto_fire
        winsound.Beep(850 if auto_fire else 450, 200)
    if not caps_lock:
        if key == keyboard.KeyCode.from_char('1') or key == keyboard.KeyCode.from_char('2'):
            if not left_lock:
                detecting = False
                left_lock = True
                winsound.Beep(800, 100)
        if key == keyboard.KeyCode.from_char('g'):
            if left_lock:
                detecting = False
                left_lock = False
                winsound.Beep(400, 100)


def listen_k_release(key):
    global detecting, shift_pressed, mouse1_pressed, mouse2_pressed, left_lock, right_lock
    if key == keyboard.Key.shift:
        shift_pressed = False
        if not (left_lock and mouse1_pressed) and not (right_lock and mouse2_pressed):
            detecting = False
            print("Start detection: ", detecting)


def listen_m_click(x, y, button, pressed):
    global detecting, shift_pressed, mouse1_pressed, mouse2_pressed, left_lock, right_lock, backforce
    # if button == mouse.Button.left and left_lock:
    #     if pressed:
    #         backforce = 3
    #         mouse1_pressed = True
    #         if not detecting:
    #             detecting = True
    #             print("Start detection: ", detecting)
    #     else:
    #         backforce = 0
    #         mouse1_pressed = False
    #         if not shift_pressed and (not right_lock or (right_lock and not mouse2_pressed)):
    #             detecting = False
    #             print("Start detection: ", detecting)
    if button == mouse.Button.left:
        if pressed:
            mouse1_pressed = True
            if left_lock:
                backforce = 6
                if not detecting:
                    detecting = True
                    print("Start detection: ", detecting)
        else:
            mouse1_pressed = False
            if left_lock:
                backforce = 0
                if not shift_pressed and (not right_lock or (right_lock and not mouse2_pressed)):
                    detecting = False
                    print("Start detection: ", detecting)
    # if button == mouse.Button.right and right_lock:
    #     if pressed:
    #         mouse2_pressed = True
    #         if not detecting:
    #             detecting = True
    #             print("Start detection: ", detecting)
    #     else:
    #         mouse2_pressed = False
    #         if not shift_pressed and (not left_lock or (left_lock and not mouse1_pressed)):
    #             detecting = False
    #             print("Start detection: ", detecting)
    if button == mouse.Button.right:
        if pressed:
            mouse2_pressed = True
            if right_lock:
                if not detecting:
                    detecting = True
                    print("Start detection: ", detecting)
        else:
            mouse2_pressed = False
            if right_lock:
                if not shift_pressed and (not left_lock or (left_lock and not mouse1_pressed)):
                    detecting = False
                    print("Start detection: ", detecting)


def PID(args, error):
    global integral, pre_error, backforce
    integral += error
    derivative = error - pre_error
    pre_error = error
    output = args.Kp*error + args.Ki*integral + args.Kd*derivative
    output[1] += backforce
    return output.astype(int)


def move_mouse(args):
    global detecting, screen_center, destination, last, width, scale, pre_error, integral, pos
    global auto_fire, time_fire, shift_pressed, right_lock, mouse2_pressed, mouse1_pressed
    if detecting:
        if destination[0] == -1:
            if last[0] == -1:
                pre_error = integral = np.array([0., 0.])
                return
            else:
                mouse_vector = np.array([0, 0])
        else:
            mouse_vector = (destination - pos) / scale
        norm = np.linalg.norm(mouse_vector)
        # if destination not in region
        if norm <= args.aim_deadzone or (destination[0] == screen_center[0] and destination[1] == screen_center[1]): return
        if norm > width*args.aim_fov: return
        if args.pid:
            move = PID(args, mouse_vector)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(move[0]), int(move[1] / 2))
            last_mv = last - destination + mouse_vector
            if not auto_fire or time.time()-time_fire <= 0.125: return  # 125ms
            # norm <= width/2  # higher divisor increases precision but limits fire rate
            # move[0]*last_mv[0] >= 0  # ensures tracking
            if ( shift_pressed and not right_lock and mouse2_pressed and not mouse1_pressed  # scope fire
            and norm <= width*2/3 and move[0]*last_mv[0] >= 0 ):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                time_fire = time.time()
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            elif ( ((shift_pressed and not mouse2_pressed) or (right_lock and mouse2_pressed and not mouse1_pressed))  # hip fire
            and norm <= width*3/4 and move[0]*last_mv[0] >= 0 ):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                time_fire = time.time()
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            return
        if norm <= width*4/3:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouse_vector[0] / 3), int(mouse_vector[1] / 3))
            return
    else:
        pre_error = integral = np.array([0., 0.])


# redirect the mouse closer to the nearest box center
def mouse_redirection(args, boxes):
    global screen_size, screen_center, destination, last, width, pos
    if boxes.shape[0] == 0:
        width = -1
        last = destination
        destination = np.array([-1, -1])
        return
    # pos = np.array(win32api.GetCursorPos())  # GetCursorPos is monitored by BattlEye
    pos = np.array(mouse.Controller().position)

    # get the center of the boxes
    boxes_center = (
        (boxes[:, :2] + boxes[:, 2:]) / 2
    )
    boxes_center[:, 1] = (
        # boxes[:, 1] * 0.6 + boxes[:, 3] * 0.4  # torso
        boxes[:, 1] * 0.7 + boxes[:, 3] * 0.3  # chest
    )

    # map the box from the image coordinate to the screen coordinate
    start_point = screen_center - screen_size[1] * args.crop_size / 2
    start_point = list(map(int, start_point))
    boxes_center[:, 0] = boxes_center[:, 0] + start_point[0]
    boxes_center[:, 1] = boxes_center[:, 1] + start_point[1]

    # find the nearest box center
    dis = np.linalg.norm(boxes_center - pos, axis=-1)
    min_index = np.argmin(dis)
    width = boxes[min_index, 2] - boxes[min_index, 0]
    last = destination
    destination = boxes_center[np.argmin(dis)].astype(int)
    # print(destination)
