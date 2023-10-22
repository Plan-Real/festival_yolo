import pyrealsense2 as rs
import numpy as np
import torch
import cv2


class FaceDetector(object):
    def __init__(self, model) -> None:
        self.model = model
        self.prev_p1 = 0
        self.prev_p2 = 0
        self.current_frame = None
        
        if not self._realsense_open():
            print("connection failed")
            exit()
    
    def streaming_server_setting(self):
        """
            Building Server for Streaming Socket
        """
        from flask import Flask
        from flask_socketio import SocketIO, emit
        import base64

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins='*')
        
        @self.socketio.on("get_video")
        def push_video():
            if self.current_frame is not None:
                _, buffer = cv2.imencode(".jpg", self.current_frame)
                frame = base64.b64encode(buffer).decode('utf-8')
                b64_src = 'data:image/jpeg;base64,'
                stringData = b64_src + frame
                emit("video_frame", stringData)
                
        self.socketio.run(self.app)
    
    def stream(self):
        """
            Looping Process -> update frame (: FPS)
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        self.aligned_depth_info = aligned_depth_frame.as_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        self.current_frame = color_frame
        return True
        
    def get_face_info(self):
        """
            Looping Process -> return coordinate by Using Yolo to find face 
        """
        _, center = self._search_center(self.current_frame)
        center_depth = -1
        if center != -1:
            center_depth = self.aligned_depth_info.get_distance(center[0], center[1])
            self.prev_p1 = center[0]
            self.prev_p2 = center[1]
            return center[0], center[1], center_depth
        return self.prev_p1, self.prev_p2, center_depth
    
    def release(self):
        """
            Camera Stream Release
        """
        self.pipeline.stop()

    def _realsense_open(self):
        """
            RealSense Camera connecting
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == "L500":
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        clipping_distance_in_meters = 1
        clipping_distance = clipping_distance_in_meters / depth_scale

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        return True

    def _search_center(self, frame):
        """
            Find Face center coordinate By Using Yolo_v8-Face
        """
        results = self.model.predict(
            source=frame,
            verbose=False,
            stream=False,
            conf=0.25,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            half=False,
            classes=0,
            max_det=1
        )

        box = results[0].boxes.xyxy
        center = -1

        if len(box):
            p1, p2 = (int(box[0][0]), int(box[0][1])
                      ), (int(box[0][2]), int(box[0][3]))
            center = (int((p1[0] + p2[0]) // 2), int((p1[1] + p2[1]) // 2))

        return frame, center

    