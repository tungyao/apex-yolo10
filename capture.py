import dxshot
import numpy as np

WIDTH, HEIGHT = (2560,1440)
CENTER = [WIDTH / 2, HEIGHT / 2]
SIZE = 640
LEFT = int(CENTER[0] - SIZE / 2)
TOP = int(CENTER[1] - SIZE / 2)
REGION = (LEFT, TOP, LEFT + SIZE, TOP + SIZE)


class LoadScreen:
    def __init__(self, region: tuple[int, int, int, int] = REGION):
        self.region = region
        self.camera = dxshot.create(region=self.region, output_color="RGB")
        self.camera.start(target_fps=100, video_mode=False)
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        self.top = TOP
        self.left = LEFT
        self.width = WIDTH
        self.height = WIDTH
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        return self

    def __next__(self):
        # now_time = time.time()
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        im0 = self.camera.get_latest_frame()
        while im0 is None:
            im0 = self.camera.get_latest_frame()

        im = im0.copy()
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s
