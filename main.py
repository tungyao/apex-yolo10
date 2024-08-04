import pynput

from args_ import arg_init
from detect import YOLOv10Detect
import threading

detector = YOLOv10Detect()
import argparse

if __name__ == "__main__":

    detector_thread = threading.Thread(target=detector.run)
    detector_thread.start()