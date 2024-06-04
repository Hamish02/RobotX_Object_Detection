#!/usr/bin/env python3
import torch
import numpy as np
#import ros_numpy as rnp
import cv2 as cv
# from cv_bridge import CvBridge
import statistics
import math 
import sys
from threading import Lock, Thread
from time import sleep
from ultralytics import YOLO
# import rospy
# from std_msgs.msg import String
# from std_msgs.msg import Float64
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import PoseStamped
import json

# --------------------------- ZED imports
sys.path.insert(0, '/workspaces/Object_Detection/')
import pyzed.sl as sl
# import cv_viewer.tracking_viewer as cv_viewer
# import ogl_viewer.viewer as gl

# ---------------------------- ROS message imports
# from sensor_msgs.msg import Image
# from std_msgs.msg import String



# Functions
class ObjectDetection:

    def __init__(self):
        # Global variables
        # self.bridge = CvBridge()
        # Image variables
        self.im_scale = 2  # Scaling of image 736*416
        self.im_width = 736 * self.im_scale  # Image width of base 736 but originally from ZED is 2208
        self.im_height = 416 * self.im_scale  # Image height of base 416 but originally from ZED is 1248
        self.confidence = 0.5  # Confidence threshold of object detection
        self.image_np_global = np.zeros([self.im_width, self.im_height, 3], dtype=np.uint8)  # Numpy array containing ZED colour image
        self.depth_np_global = np.zeros([self.im_width, self.im_height, 4], dtype=float)  # Numpy array containing ZED XYZRGBA image


        # Thread variables
        self.lock = Lock()
        self.exit_sig = False  # boolean for quitting the object detection loop
        self.run_sig = False  # True if an image is taken from
        self.detection_thread = Thread(target=self.detect_thread,
                                       kwargs={'conf_thres': self.confidence})

        # Model labels
        self.model_labels = {}
        self.detections = []
        
        # self.detectionPub = rospy.Publisher('zedr_objects', String)#, queue_size=10)

        # Data to send with the object detection - current poses etc 
        self.current_pose = False
        self.yeehaw = False 


    def detect_thread(self, conf_thres=0.2):
        weights = '/workspaces/Object_Detection/best.pt'
        img_size = 416,
        iou_thres=0.45
        # --------------------------Setup Pytorch
        print("Starting the detection")

        print(torch.cuda.is_available())
        print(torch.version.cuda)
        model = YOLO(weights)
        self.lock.acquire()
        # stride, self.model_labels, pt = model.stride, model.names, model.pt
        self.model_labels = model.names
        self.lock.release()
        # print("Stride: ", stride)
        # print("Name: ", self.model_labels)
        # print("Pt: ", pt)

        print("Initialized detection")

        

        while not self.exit_sig:
            if self.run_sig:
                self.lock.acquire()

                img = cv.cvtColor(self.image_np_global, cv.COLOR_BGRA2RGB)
                # https://docs.ultralytics.com/modes/predict/#video-suffixes
                det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

                # ZED CustomBox format (with inverse letterboxing tf applied)
                self.detections = det.pandas().xyxy[0]
                self.lock.release()
                self.run_sig = False
            sleep(0.01)

    def create_bounding_box(self, xmin, ymin, xmax, ymax):
        """Create 2D bounding box box for object"""
        # Bounding box array reference from ZED SDK
        # A ------ B
        # | Object |
        # D ------ C
        output = np.zeros((4, 2))

        # A
        output[0][0] = xmin
        output[0][1] = ymin

        # B
        output[1][0] = xmax
        output[1][1] = ymin

        # D
        output[2][0] = xmin
        output[2][1] = ymax

        # C
        output[3][0] = xmax
        output[3][1] = ymax

        return output

    def image_grabbing(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.lock.acquire()
        self.image_np_global = np.asarray(cv_image)
        self.run_sig = True
        self.lock.release()

    # def grab_xyz(self, msg):
    #     self.depth_np_global = rnp.point_cloud2.get_xyz_points_toarray(msg, remove_nans=False, dtype=np.float)
    #     for

    def load_image_into_numpy_array(self, image):
        ar = image.get_data()
        ar = ar[:, :, 0:3]
        (im_height, im_width, channels) = image.get_data().shape
        return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)

    def load_depth_into_numpy_array(self, depth):
        ar = depth.get_data()
        ar = ar[:, :, 0:4]
        (im_height, im_width, channels) = depth.get_data().shape
        return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)

    def detection_to_obj(self, det):
        """Convert detections from model to custom objects for ZED SDK"""
        objects_in = []
        for i, obj in det.iterrows():
            tmp = sl.CustomBoxObjectData()
            tmp.unique_object_id = sl.generate_unique_id()
            tmp.probability = obj['confidence']
            tmp.label = obj['class']
            box = self.create_bounding_box(obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'])
            tmp.bounding_box_2d = box
            tmp.is_grounded = True  # objects are moving on the floor plane and tracked in 2D only
            objects_in.append(tmp)
        return objects_in

    def draw_obj_dist(self, image, obj, dist):
        """Draw distance of objects onto image.
        dist: (index, x, y, z, distance), where index is the object number of detection list
        """
        for d in dist:
            o = obj.object_list[d[0]]
            xmin = o.bounding_box_2d[0][0]
            ymin = o.bounding_box_2d[0][1]
            item = self.model_labels.get(o.raw_label)
            label = "".join([item, " ", str(d[4])])
            image = cv.putText(image, label, (int(xmin), int(ymin)),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 2,
                                cv.LINE_AA)
        return image

    def distance_calc(self, depth_np, det, type=sl.MEASURE.XYZRGBA):
        """Calculate distance of object from camera
        Coordinate reference: https://www.stereolabs.com/docs/api/group__Core__group.html#ga1114207cbac18c1c0e4f77a6b36a8cb2
        Coordinate system: RIGHT_HANDED_Y_UP
        """
        obj_dist_list = []

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
                        if x_pix >= self.im_width:
                            x_pix = self.im_width - 1
                        if y_pix >= self.im_height:
                            y_pix = self.im_height - 1
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

    def main_processing(self, filepath):
        self.detection_thread.start()
        print('Starting the ZED')

        zed = sl.Camera()

        svo_filepath = filepath
        input_type = sl.InputType()
        if svo_filepath is not None:
            input_type.set_from_svo_file(svo_filepath)

        init_params = sl.InitParameters(input_t=input_type)
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 15
        init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.svo_real_time_mode = False
        init_params.depth_maximum_distance = 40

        err = zed.open(init_params)
        print(err)
        while err != sl.ERROR_CODE.SUCCESS:
            err = zed.open(init_params)
            print(err)
            sleep(1)

        print("Initialized Camera")

        positional_tracking_parameters = sl.PositionalTrackingParameters()
        zed.enable_positional_tracking(positional_tracking_parameters)

        # obj_param = sl.ObjectDetectionParameters()
        # obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        # obj_param.enable_tracking = True
        # zed.enable_object_detection(obj_param)
        # Enable object detection
        detection_param = sl.ObjectDetectionParameters()
        detection_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        detection_param.enable_tracking = True
        err = zed.enable_object_detection(detection_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print("enable_object_detection", err, "\nExit program.")
            zed.close()
            exit(-1)

        camera_infos = zed.get_camera_information()
        camera_res = camera_infos.camera_configuration.resolution
        image_mat = sl.Mat(camera_res.width,
                           camera_res.height, sl.MAT_TYPE.U8_C3)
        depth_mat = sl.Mat(camera_res.width,
                           camera_res.height, sl.MAT_TYPE.F32_C4)
        runtime_parameters = sl.RuntimeParameters()  # measure3D_reference_frame=sl.REFERENCE_FRAME.WORLD
        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

        # Display
        # 3D Mapping
        point_cloud_res = sl.Resolution(min(camera_res.width, 720),
                                        min(camera_res.height, 404))
        point_cloud_render = sl.Mat()

        # 2D Object Detection
        display_resolution = sl.Resolution(self.im_width, self.im_height)
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

        # zed.set_svo_position(12300)
        try:
            while not self.exit_sig:
                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image_mat, sl.VIEW.RIGHT, resolution=display_resolution)
                    zed.retrieve_measure(depth_mat, sl.MEASURE.XYZRGBA, resolution=display_resolution)
                    self.lock.acquire()
                    self.image_np_global = self.load_image_into_numpy_array(image_mat)
                    self.depth_np_global = self.load_depth_into_numpy_array(depth_mat)
                    self.run_sig = True
                    self.lock.release()

                    while self.run_sig:
                        sleep(0.001)

                    obj = self.detection_to_obj(self.detections)
                    # load object into zed object tracking
                    self.lock.acquire()
                    zed.ingest_custom_box_objects(obj)
                    self.lock.release()

                    zed.retrieve_objects(objects, obj_runtime_param)

                    obj_dist = self.distance_calc(depth_mat.get_data(), objects)

                    np.copyto(image_left_ocv, image_mat.get_data())

                    if obj_dist:
                        image_left_ocv = self.draw_obj_dist(image_left_ocv, objects, obj_dist)
                        buoys = {}
                        # create a json string of all the elements 
                        for obj in obj_dist:
                            buoy = {"x" : str("{}".format(obj[1])), \
                                    "y" : str("{}".format(obj[2])), \
                                    "z" : str("{}".format(obj[3])), \
                                    "range" : str("{}".format(obj[4])), \
                                    "colour" : str("{}".format(obj[5]))}
                            buoys.update({"{}".format(obj[0]):buoy})

                        # Calculate the orientation from the current pose 
                        if self.current_pose is not False and self.yeehaw is not False: 
                            #self.current_pose.orientation.x = 0
                            #self.current_pose.orientation.y = 0
                            #mag = math.sqrt(self.current_pose.orientation.w * self.current_pose.orientation.w \
                            #            + self.current_pose.orientation.z * self.current_pose.orientation.z)
                            #self.current_pose.orientation.w /= mag
                            #self.current_pose.orientation.z /= mag
                            #ang = 2 * math.acos(self.current_pose.orientation.w);        
                            
                            #yaw = math.atan2(2.0 * (self.current_pose.orientation.z * self.current_pose.orientation.w\
                            #        + self.current_pose.orientation.x * self.current_pose.orientation.y),\
                            #         -1.0 + 2.0 * (self.current_pose.orientation.w * self.current_pose.orientation.w\
                            #        + self.current_pose.orientation.x * self.current_pose.orientation.x)\
                            #        - self.current_pose.orientation.y * self.current_pose.orientation.y\
                            #        - self.current_pose.orientation.z * self.current_pose.orientation.z)

                            #atan2(2.0f * (w * z + x * y), w * w + x * x - y * y - z * z);

                            #yaw   = math.atan2(2.0 * (q.q3 * q.q0 + q.q1 * q.q2) , - 1.0 + 2.0 * (q.q0 * q.q0 + q.q1 * q.q1))
                            # Add to the dictionary of dictionaries
                            buoys.update({"where" : {"x" : self.current_pose.position.x,\
                                                    "y" : self.current_pose.position.y,\
                                                    "z" : self.current_pose.position.z,\
                                                    "yeehaw" : str(self.yeehaw)}})

                        # Create a json serialised representation of the objects and the current position
                        json_string = json.dumps(buoys)
                        print("{}".format(json_string))
                        
                        # pub.publish(json_string)
                        
                    # cv_viewer.render_2D(image_left_ocv, [1, 1], objects, detection_param.enable_tracking)
                    print("check")
                    cv.imshow("ZED", image_left_ocv)  # cv2.hconcat([image_left_ocv, image_track_ocv])

                    if cv.waitKey(10) & 0xFF == ord('q'):
                        cv.destroyAllWindows()
                        self.exit_sig = True

                elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    self.exit_sig = True
                    break
        
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            self.exit_sig = True
            zed.close()

    def where_callback(self, data):
        self.current_pose = data.pose
        
    def yeehaw_callback(self, msg):
        self.yeehaw = msg.data

if __name__ == "__main__":

    #rate = rospy.Rate(4) 

    od = ObjectDetection()

    ## Create the publishing topic for the object detection 
    # pub = rospy.Publisher('object_publisher', String, queue_size=10)
    # rospy.Subscriber('/mavros/local_position/pose', PoseStamped, od.where_callback)
    # rospy.Subscriber('/mavros/global_position/compass_hdg', Float64, od.yeehaw_callback)

    # # Register this node to the rosmaster
    # rospy.init_node('object_starboard_node', anonymous=False)

    #od.main_processing(None)

    od.main_processing("/workspaces/Object_Detection/demo.svo")