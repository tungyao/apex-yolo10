from math import *

from mouse_driver.MouseMove import ghub_mouse_move

lock_choice = [0]
lock_smooth = 1.9
lock_sen = 3.0
def lock(aims, mouse, top_x, top_y, len_x, len_y, args, pidx, pidy):
    mouse_pos_x, mouse_pos_y = mouse.position
    aims_copy = aims.copy()
    detect_arange =15000    #範圍可自行調整
    aims_copy = [x for x in aims_copy if x[0] in lock_choice and (len_x * float(x[1]) + top_x - mouse_pos_x) ** 2 + (len_y * float(x[2]) + top_y - mouse_pos_y) ** 2 < detect_arange]
    k = 4.07 * (1 / lock_smooth)
    print(len(aims_copy))

    if len(aims_copy):
        dist_list = []
        for det in aims_copy:
            _, x_c, y_c, _, _ = det
            dist = (len_x * float(x_c) + top_x - mouse_pos_x) ** 2 + (len_y * float(y_c) + top_y - mouse_pos_y) ** 2
            dist_list.append(dist)
        if dist_list:
            det = aims_copy[dist_list.index(min(dist_list))]
            tag, x_center, y_center, width, height = det
            x_center, width = len_x * float(x_center) + top_x, len_x * float(width)
            y_center, height = len_y * float(y_center) + top_y, len_y * float(height)
            rel_x = int(k / lock_sen * atan((mouse_pos_x - x_center) / 640) * 640)
            rel_y = int(k / lock_sen * atan((mouse_pos_y - y_center + 1 / 4 * height) / 640) * 640)#瞄準高度可自行調整(建議為1/4)
            pid_movex = pidx(rel_x)
            pid_movey = pidy(rel_y)
            print(2222)
            ghub_mouse_move(round(pid_movex), round(pid_movey))