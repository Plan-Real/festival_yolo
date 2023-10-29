from typing import Iterator
import cv2
import math
from ultralytics import YOLO
from utils.predict_center import FaceDetector

import numpy as np
from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.service import Service
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)

class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")
        # ros2 parameter
        self.declare_parameter("model", "yolov8n-face.pt")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("face_frame", "face")
        self.declare_parameter("FOV_H", 69)
        self.declare_parameter("FOV_V", 42)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.model = self.get_parameter("model").value
        self.model_path = os.path.join(
            parent_directory, "checkpoints", self.model)
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

        # ros2 service
        self.pic_start_client = self.create_client(Trigger, "pic_start_srv")
        self.pic_stop_client = self.create_client(Trigger, "pic_stop_srv")

        # ros2 tf broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        
        stream_group = None
        yolo_group = ReentrantCallbackGroup()

        # ros2 timer
        self.create_timer(0.03, self.timer_callback_yolo, callback_group=yolo_group)
        self.create_timer(0.03, self.timer_callback_stream, callback_group=stream_group)

        self.yolo = YOLO(self.model_path)
        self.face_detector = FaceDetector(self.yolo)

        # binding
        self.face_detector.set_start(self.start)
        self.face_detector.set_stop(self.stop)

        self.frame = None
        self.thread = Thread(
            target=self.face_detector.streaming_server_setting)
        self.thread.start()

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
        if 0.0 < depth <= 1.5:
            x = depth*math.cos(x_degree)*math.cos(y_degree)
            z = -x*math.tan(y_degree)
            y = -x*math.tan(x_degree)
            self.depth = depth

        else:
            x = self.depth*math.cos(x_degree)*math.cos(y_degree)
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
        transform.transform.rotation.x = -0.5
        transform.transform.rotation.y = 0.5
        transform.transform.rotation.z = -0.5
        transform.transform.rotation.w = 0.5
        if x ==0 :
            pass
        else:
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

    def timer_callback_stream(self):
        self.frame = self.face_detector.stream()
        
    def timer_callback_yolo(self):
        if self.frame is not None:
            x, y, depth = self.face_detector.get_face_info()
            if depth != -1:
                self.get_logger().info("Human detected")
                self.publish_tf(x, y, depth)
            else :
                # pass
                self.get_logger().info("Human not detected")
            self.pub_image(self.frame)

    def start(self):
        request = Trigger.Request()

        # while not self.pic_start_client.wait_for_service(timeout_sec=1.0):
        self.get_logger().info("start")
        future = self.pic_start_client.call(request)
        # rclpy.spin_until_future_complete(self, future)
        # try:
        #     response = future.result()
        #     self.get_logger().info("Result of start : %r" % (response.success))
        # except Exception as e:
        #     self.get_logger().info("Service call failed %r" % (e,))
        # return response.success
    
    def stop(self):
        request = Trigger.Request()
        # while not self.pic_stop_client.wait_for_service(timeout_sec=1.0):
        self.get_logger().info("stop")
        future = self.pic_stop_client.call(request)
        # rclpy.spin_until_future_complete(self, future)
        # try:
        #     response = future.result()
        #     self.get_logger().info("Result of stop : %r" % (response.success))
        # except Exception as e:
        #     self.get_logger().info("Service call failed %r" % (e,))
        # return response.success

    def __del__(self):
        self.thread.join()
        self.face_detector.release()


if __name__ == '__main__':
    rclpy.init()
    node = YoloNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info("Beginning client, shut down with CTRL-C")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.\n")
    node.destroy_node()
    rclpy.shutdown()
