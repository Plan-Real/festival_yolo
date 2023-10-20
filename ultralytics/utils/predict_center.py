import numpy as np
import torch
import cv2

class Central_point(object):
    def __init__(self, model) -> None:
        self.model = model
    
    def stream_realsense(self):
        try:
            import pyrealsense2 as rs
        except:
            raise ImportError
        pipeline = rs.pipline()

        config = rs.config()
        pipeline_wrapper = rs.pipline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
            if not found_rgb:
                print("The demo requires Depth Camera with Color Sensor")
                exit(0)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == "L500":
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # Streaming loop
        try:
            while True:
                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)

                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    continue
                
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                grey_color = 153
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

                # Render images:
                #   depth align to color on left
                #   depth on right
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((bg_removed, depth_colormap))

                cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                cv2.imshow('Align Example', images)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            pipeline.stop()


    def stream_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print('video open failed')
            exit()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            results = self.model.predict(
                source=frame,
                verbose=False,
                stream=False,
                conf=0.25,
                device="cpu",
                half=False,
                classes=0,
                max_det=1
            )

            box = results[0].boxes.xyxy

            if len(box):
                p1, p2 = (int(box[0][0]), int(box[0][1])), (int(box[0][2]), int(box[0][3]))
                c = ((p1[0] + p2[0]) // 2 , (p1[1] + p2[1]) // 2)
                frame =  cv2.rectangle(frame, c, c, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)

            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()