import time
import dxshot
import cv2


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
    camera = dxshot.create(output_idx=0, output_color="BGR")
    camera.start(target_fps=target_fps, video_mode=True)
    writer = cv2.VideoWriter(
        "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (1920, 1080)
    )
    for i in range(600):
        writer.write(camera.get_latest_frame())
    camera.stop()
    writer.release()


if __name__ == "__main__":
    capture()
