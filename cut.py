import time
import dxshot
import cv2
import time
import os

d = dxshot.create()
res = [2560, 1440]
w = res[0]
v = res[1]
x = int((w - 640) / 2)
y = int((v - 640) / 2)
x1 = x + 640
y1 = y + 640
save_path = "datasets/img"


def benchmark():
    shape = 640
    left, top = (1920 - shape) // 2, (1080 - shape) // 2
    right, bottom = left + shape, top + shape
    region = (left, top, right, bottom)
    title = "[DXcam] FPS benchmark"
    cam = dxshot.create()
    start_time = time.perf_counter()
    fps = 0
    while fps < 1000:
        start = time.perf_counter()
        frame = cam.grab(region=region)
        if frame is not None:
            print(time.perf_counter() - start)
            # start = now_time
            fps += 1
            cv2.imshow('', frame)
            cv2.waitKey(1)

    end_time = time.perf_counter() - start_time

    print(f"{title}: {fps / end_time}")
    del cam


def capture():
    target_fps = 60
    camera = dxshot.create(output_idx=0, output_color="RGB")
    camera.start(target_fps=target_fps, video_mode=True)
    writer = cv2.VideoWriter(
        "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (1920, 1080)
    )
    for i in range(600):
        writer.write(camera.get_latest_frame())
    camera.stop()
    writer.release()


def cut_display():
    print(x, y, x1, y1)
    cam = dxshot.create(output_color="RGB")
    region = (x, y, x1, y1)

    while True:
        frame = cam.grab(region=region)
        cv2.imwrite(os.path.join(save_path, str(time.time()) + ".jpg"), frame)
        print(str(time.time()) + ":  " + str(time.time()) + ".jpg" + "已保存")
        time.sleep(1)  # 一秒截一张图


if __name__ == "__main__":
    cut_display()
