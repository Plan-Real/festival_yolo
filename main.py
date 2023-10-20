import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.predict_center import FaceDetector
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")
        # ros2 parameter
        self.declare_parameter("model_path", "./checkpoints/yolov8n-face.pt")
        self.declare_parameter("camera_frame", "camera_color_frame")
        self.declare_parameter("face_frame", "face")
        self.declare_parameter("FOV_H", 69)
        self.declare_parameter("FOV_V", 42)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.model_path = self.get_parameter("model_path").value
        self.camera_link = self.get_parameter("camera_frame").value
        self.face_frame = self.get_parameter("face_frame").value
        self.FOV_H = self.get_parameter("FOV_H").value
        self.FOV_V = self.get_parameter("FOV_V").value
        self.frame_width = self.get_parameter("frame_width").value
        self.frame_height = self.get_parameter("frame_height").value

        # ros2 publisher
        self.image_pub = self.create_publisher(Image, "yolo/image", 10)
        self.compress_pub = self.create_publisher(
            CompressedImage, "yolo/compressed", 10)

        # ros2 tf broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()

        # ros2 timer
        self.create_timer(0.033, self.timer_callback)

        self.model = YOLO(self.model_path)
        self.face_detector = FaceDetector(self.model)

    def search_center(self, frame):
        return self.face_detector.search_center(frame)

    def stream_realsense(self, only_point=False):
        return self.face_detector.stream_realsense(only_point)

    def publish_tf(self, x, y, depth):
        # 69° × 42°
        x = x-self.frame_width/2
        y = y-self.frame_height/2

        # degree
        x_degree = x/self.frame_width*self.FOV_H
        y_degree = y/self.frame_height*self.FOV_V
        x_degree = math.radians(x_degree)
        y_degree = math.radians(y_degree)

        # relative distance
        x = depth*math.cos(x_degree)*math.cos(y_degree)
        z = -x*math.tan(y_degree)
        y = -x*math.tan(x_degree)

        # publish tf
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.camera_link
        transform.child_frame_id = self.face_frame
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        self.tf_broadcaster.sendTransform(transform)

    def pub_image(self, frame):
        image_msg = Image()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = self.camera_link
        image_msg.height = frame.shape[0]
        image_msg.width = frame.shape[1]
        image_msg.encoding = "bgr8"
        image_msg.data = frame.tobytes()
        self.image_pub.publish(image_msg)

    def pub_compressed(self, frame):
        compress_msg = CompressedImage()
        compress_msg.header.stamp = self.get_clock().now().to_msg()
        compress_msg.header.frame_id = self.camera_link
        compress_msg.format = "jpeg"
        compress_msg.data = np.array(cv2.imencode(".jpg", frame)[1]).tostring()
        self.compress_pub.publish(compress_msg)

    def timer_callback(self):
        frame = self.face_detector.stream()
        x, y, depth = self.face_detector.get_face_info(frame)
        if x != -1:
            self.publish_tf(x, y, depth)
        self.pub_compressed(frame)
        self.pub_image(frame)

    def __del__(self):
        self.face_detector.release()


if __name__ == '__main__':
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()
