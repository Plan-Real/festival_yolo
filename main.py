import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.predict_center import Central_point

if __name__ == '__main__':
    model = YOLO("./checkpoints/yolov8n-face.pt")
    Launcher = Central_point(model)
    Launcher.stream_webcam()
