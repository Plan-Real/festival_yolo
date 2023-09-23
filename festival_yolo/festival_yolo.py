# -*- coding: utf-8 -*-
import pyrealsense2.pyrealsense2 as rs
import torch
import torch.backends.cudnn as cudnn
from models import experimental
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size,
    non_max_suppression,
    # apply_classifier,
    scale_coords,
    # xyxy2xywh,
    # strip_optimizer,
    set_logging,
)
from utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
)
import random
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from rclpy.node import Node
import rclpy
import cv2
import numpy as np
import time
import argparse
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# ROS2 Humble


# yolo v5

# Realsense


class YoloV5:
    def __init__(self, yolov5_yaml_path):
        with open(yolov5_yaml_path, "r", encoding="utf-8") as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        self.colors = [
            [np.random.randint(0, 255) for _ in range(3)]
            for class_id in range(self.yolov5["class_num"])
        ]
        self.init_model()

    @torch.no_grad()
    def init_model(self):
        set_logging()
        device = select_device(self.yolov5["device"])
        is_half = device.type != "cpu"
        model = experimental.attempt_load(
            str(ROOT) + self.yolov5["weight"], map_location=device
        )
        input_size = check_img_size(
            self.yolov5["input_size"], s=model.stride.max()
        )
        if is_half:
            model.half()

        cudnn.benchmark = True  # set True to speed up constant image size inference

        img_torch = torch.zeros(
            (1, 3, self.yolov5["input_size"], self.yolov5["input_size"]), device=device
        )
        print(is_half)
        _ = model(img_torch.half() if is_half else img_torch)
        # _ = model(img_torch.half() if is_half else img) if device.type != 'cpu' else None
        self.is_half = is_half
        self.device = device
        self.model = model
        self.img_torch = img_torch

    def preprocessing(self, img):
        img_resize = letterbox(
            img,
            new_shape=(self.yolov5["input_size"], self.yolov5["input_size"]),
            auto=False,
        )[0]
        img_arr = np.stack([img_resize], 0)
        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img_arr = np.ascontiguousarray(img_arr)
        return img_arr

    @torch.no_grad()
    def detect(self, img, canvas=None, view_img=True):
        img_resize = self.preprocessing(img)
        self.img_torch = torch.from_numpy(img_resize).to(self.device)
        self.img_torch = (
            self.img_torch.half() if self.is_half else self.img_torch.float()
        )
        self.img_torch /= 255.0
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)
        t1 = time_synchronized()

        pred = self.model(self.img_torch, augment=False)[0]
        # pred = self.model_trt(self.img_torch, augment=False)[0]
        pred = non_max_suppression(
            pred,
            self.yolov5["threshold"]["confidence"],
            self.yolov5["threshold"]["iou"],
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
        )
        t2 = time_synchronized()
        det = pred[0]
        gain_whwh = torch.tensor(img.shape)[[1, 0, 1, 0]]  # [w, h, w, h]

        if view_img and canvas is None:
            canvas = np.copy(img)
        xyxy_list = []
        conf_list = []
        class_id_list = []
        if det is not None and len(det):
            det[:, :4] = scale_coords(
                img_resize.shape[2:], det[:, :4], img.shape
            ).round()
            for *xyxy, conf, class_id in reversed(det):
                class_id = int(class_id)
                # if class_id != 0:
                #     continue
                # int(class_id)
                xyxy_list.append(xyxy)
                conf_list.append(conf)
                class_id_list.append(class_id)
                if view_img:
                    label = "%s %.2f" % (self.yolov5["class_name"][-1], conf)
                    # self.plot_one_box(
                    #     xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)
                print(class_id_list)
        return canvas, class_id_list, xyxy_list, conf_list

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        tl = (
            line_thickness or round(
                0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # if label:
        #     tf = max(tl - 1, 1)  # font thickness
        #     t_size = cv2.getTextSize(
        #         label, 0, fontScale=tl / 3, thickness=tf)[0]
        #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
        #                 [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class Yolo_people(Node):
    def __init__(self):
        super().__init__("yolo_people")
        self.bridge = CvBridge()
        self.realsense_publisher = self.create_publisher(
            Image, "/image_raw", 10)

        self.human = TransformStamped()
        self.end = TransformStamped()
        # Realsense pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.image_checker = True

        self.model = YoloV5(str(ROOT) + str("/config/yolov5.yaml"))

        self.peopletf_broadcaster = TransformBroadcaster(self)
        self.end_tf = TransformBroadcaster(self)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.peopletf)

        # self.peopletf()
        # # print("TFuvk")

    def peopletf(self):
        self.peopletf_broadcaster.sendTransform(self.human)
        self.peopletf_broadcaster.sendTransform(self.end)

        people_pose = self.yolopeople_callback()

        self.human.header.stamp = self.get_clock().now().to_msg()
        self.human.header.frame_id = "front_camera_link"
        self.human.child_frame_id = "human"

        self.end.header.stamp = self.get_clock().now().to_msg()
        self.end.header.frame_id = "front_camera_link"
        self.end.child_frame_id = "end"
        howmanypeoplearethere = len(people_pose)
        # print(people_pose)
        if howmanypeoplearethere > 0:
            x = people_pose[0][2]
            y = -people_pose[0][0]
            z = -people_pose[0][1]

            self.human.transform.translation.x = people_pose[0][2]
            self.human.transform.translation.y = -people_pose[0][0]
            self.human.transform.translation.z = -people_pose[0][1]
            # Original
            # self.human.transform.translation.x = people_pose[0][2] * 1
            # self.human.transform.translation.y = -people_pose[0][0] * 1
            # self.human.transform.translation.z = -people_pose[0][1] * 1

            # Photo Shot
            self.end.transform.translation.x = 0.9848 * x + 0.1736 * z - 0.8
            self.end.transform.translation.y = 0.0
            self.end.transform.translation.z = -0.1736 * x + 0.9848 * z + 0.25

        # if howmanypeoplearethere == 2:
        #     # print("It's two people")
        #     human.transform.translation.z = (people_pose[0][0] + people_pose[1][0]) / 2
        #     human.transform.translation.x = (people_pose[0][1] + people_pose[1][1]) / 2
        #     human.transform.translation.y = (people_pose[0][2] + people_pose[1][2]) / 2
        # elif howmanypeoplearethere > 0:
        #     human.transform.translation.z = people_pose[0][0] * 1
        #     human.transform.translation.x = people_pose[0][1] * 1
        #     human.transform.translation.y = people_pose[0][2] * 1
        # else:
        #     human.transform.translation.x = 0.0
        #     human.transform.translation.y = 0.0
        #     human.transform.translation.z = 0.0

    def yolopeople_callback(self):
        # Get image to realsense
        (
            intr,
            depth_intrin,
            color_image,
            depth_image,
            aligned_depth_frame,
        ) = self.get_aligned_images()
        while not depth_image.any() or not color_image.any():
            if self.image_checker:
                self.image_checker = False
                print("[Wait] Image wait")
        if self.image_checker:
            self.image_checker = False
            print("[Done] Success get Image")

        # Image Publish
        image_raw = self.bridge.cv2_to_imgmsg(color_image)
        self.realsense_publisher.publish(image_raw)

        # Load image to realsense
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image, depth_colormap))

        # load yolo v5 model
        t_start = time.time()
        canvas, class_id_list, xyxy_list, conf_list = self.model.detect(
            color_image)
        t_end = time.time()

        camera_xyz_list = []
        if xyxy_list:
            for i in range(len(xyxy_list)):
                ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)
                uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)
                dis = aligned_depth_frame.get_distance(ux, uy)
                camera_xyz = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, (ux, uy), dis
                )

                camera_xyz = np.round(np.array(camera_xyz), 3)
                camera_xyz = camera_xyz.tolist()
                # print(camera_xyz)
                # if camera_xyz[0] == 0 or camera_xyz[1] == 0 or camera_xyz[2] == 0:
                #     print("====================0====================")
                cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                cv2.putText(
                    canvas,
                    str(camera_xyz),
                    (ux + 20, uy + 10),
                    0,
                    1,
                    [225, 255, 255],
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                camera_xyz_list.append(camera_xyz)

        fps = int(1.0 / (t_end - t_start))
        cv2.putText(
            canvas,
            text="FPS: {}".format(fps),
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=2,
            lineType=cv2.LINE_AA,
            color=(0, 0, 0),
        )
        cv2.namedWindow(
            "detection",
            flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
        )
        cv2.imshow("detection", canvas)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            # break

        # print(len(camera_xyz_list))

        return camera_xyz_list

    def get_aligned_images(self):
        # class self realsense config
        # self.pipeline, self.config, self.profile, self.align, self.align

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        """camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                            'ppx': intr.ppx, 'ppy': intr.ppy,
                            'height': intr.height, 'width': intr.width,
                            'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                            }"""

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_image_3d = np.dstack(
            (depth_image_8bit, depth_image_8bit, depth_image_8bit)
        )
        color_image = np.asanyarray(color_frame.get_data())

        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


def main(args=None):
    rclpy.init(args=args)
    yolo_people = Yolo_people()
    rclpy.spin(yolo_people)
    yolo_people.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
