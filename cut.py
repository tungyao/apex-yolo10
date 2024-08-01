import d3dshot
import cv2
import time
import os
from PIL import ImageGrab

d = d3dshot.create()
res = d.display.resolution
w = res[0]
v = res[1]
x = int((w - 640) / 2)
y = int((v - 640) / 2)
x1 = x + 640
y1 = y + 640
print(x, y, x1, y1)

save_path = "./dataset/img"


def cut_display():
    while True:
        img = d.screenshot(region=(x, y, x1, y1))
        img.save(os.path.join(save_path, str(time.time()) + ".jpg"))
        print(str(time.time()) + ":  " + str(time.time()) + ".jpg" + "已保存")
        time.sleep(1)  # 一秒截一张图


if __name__ == "__main__":
    cut_display()
