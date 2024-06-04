#!/usr/bin/env python3

import json
import math
import statistics
import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import cv_viewer.tracking_viewer as cv_viewer

lock = Lock()
run_signal = False
exit_signal = False


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output

def distance_calc(depth_np, det, type=sl.MEASURE.XYZRGBA):
    """Calculate distance of object from camera
    Coordinate reference: https://www.stereolabs.com/docs/api/group__Core__group.html#ga1114207cbac18c1c0e4f77a6b36a8cb2
    Coordinate system: RIGHT_HANDED_Y_UP
    """
    obj_dist_list = []
    im_width = 2208
    im_height = 1242
    for i, o in enumerate(det.object_list):
        box = o.bounding_box_2d
        xmin = box[0][0]
        ymin = box[0][1]
        xmax = box[1][0]
        ymax = box[2][1]
        color = o.raw_label

        x_vect = []
        y_vect = []
        z_vect = []

        # For XYZRGBA mode
        if type == sl.MEASURE.XYZRGBA:
            for x_pix in range(int(xmin), int(xmax)):
                for y_pix in range(int(ymin), int(ymax)):
                    if x_pix >= im_width:
                        x_pix = im_width - 1
                    if y_pix >= im_height:
                        y_pix = im_height - 1
                    z_ = depth_np[y_pix, x_pix, 2]  # depth_np is in XYZRGBA mode, (X,Y,Z, color) for each pixel where color in 4 channel
                    if not np.isnan(z_) and not np.isinf(z_):
                        x_vect.append(depth_np[y_pix, x_pix, 0])
                        y_vect.append(depth_np[y_pix, x_pix, 1])
                        z_vect.append(z_)

        if len(z_vect) > 0:
            x = statistics.median(x_vect)
            y = statistics.median(y_vect)
            z = statistics.median(z_vect)

            # print("X: ", x)
            # print("Y: ", y)
            # print("Z: ", z)

            distance = math.sqrt(x * x + y * y + z * z)

            obj_dist_list.append((i, x, y, z, distance, color))

    return obj_dist_list

def draw_obj_dist(image, obj, dist, img_scale):
    """Draw distance of objects onto image.
    dist: (index, x, y, z, distance), where index is the object number of detection list
    """
    for d in dist:
        o = obj.object_list[d[0]]
        xmin = o.bounding_box_2d[0][0]* img_scale[0]
        ymin = o.bounding_box_2d[0][1]* img_scale[1]
        item = model_labels.get(o.raw_label) #TODO
        label = "".join([item, " ", str(d[4])])
        image = cv2.putText(image, label, (int(xmin), int(ymin)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 2,
                            cv2.LINE_AA)
    return image

def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, model_labels

    print("Intializing Network...")

    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)

            # Print image info from numpy nd.array
            # shape = img.shape
            # dtype = img.dtype
            # print(f"Image shape: {shape}, dtype: {dtype}")
            # min_val = np.min(img)
            # max_val = np.max(img)
            # print(f"Min: {min_val}, Max: {max_val}")

            # print(type(img))
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes
            model_labels = model.names
            # print(model_labels)
            # print(type(det))
            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    zed.set_svo_position(12300)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    # print(camera_infos.camera_configuration.resolution.width, camera_infos.camera_configuration.resolution.height)
    print(f"Camera Resolution: width = {camera_infos.camera_configuration.resolution.width}, height = {camera_infos.camera_configuration.resolution.height}")
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    # point_cloud_render = sl.Mat()
    # viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    # point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    point_cloud = sl.Mat()
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    # camera_config = camera_infos.camera_configuration
    # tracks_resolution = sl.Resolution(400, display_resolution.height)
    # track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    # track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    # image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()

    # while viewer.is_available() and not exit_signal:
    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            # point_cloud.copy_to(point_cloud_render) 
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            obj_dist = distance_calc(point_cloud.get_data(), objects)
            # print("OBJ: ", obj_dist)
            # print(type(obj_dist))

            # 3D rendering
            # viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            
            # print(type(image_scale))
            # global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            # track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            # We measure the distance camera - object using Euclidean distance
            # print("width: ", image_left.get_width(), ", height: ", image_left.get_height())
            # print(type(point_cloud))
            # print(point_cloud.get_width(), point_cloud.get_height())
            # print(point_cloud)
            # x = round(image_left.get_width() / 2)
            # y = round(image_left.get_height() / 2)
            # err, point_cloud_value = point_cloud.get_value(x, y)

            # if math.isfinite(point_cloud_value[2]):
            #     distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
            #                         point_cloud_value[1] * point_cloud_value[1] +
            #                         point_cloud_value[2] * point_cloud_value[2])
            #     print(f"Distance to Camera at {{{x};{y}}}: {distance}")
            # else : 
            #     print(f"The distance can not be computed at {{{x};{y}}}")

            buoys = {}
            # create a json string of all the elements 
            for obj in obj_dist:
                buoy = {"x" : str("{}".format(obj[1])), \
                        "y" : str("{}".format(obj[2])), \
                        "z" : str("{}".format(obj[3])), \
                        "range" : str("{}".format(obj[4])), \
                        "colour" : str("{}".format(obj[5]))}
                buoys.update({"{}".format(obj[0]):buoy})
            json_string = json.dumps(buoys)
            print("{}".format(json_string))

            cv2.imshow("ZED Detect Buoys", image_left_ocv)
            key = cv2.waitKey(10) # Use esc to exit
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

    # viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()

# python3 detector.py --svo /workspaces/Object_Detection/FirstGo.svo --weights /workspaces/Object_Detection/best.pt   