#include <sl/Camera.hpp>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: ./object_detection_svo <path_to_svo_file>" << std::endl;
        return 1;
    }

    // Initialize the ZED camera
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.input.setFromSVOFile(argv[1]);
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_params.coordinate_units = sl::UNIT::METER;

    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Error opening the ZED camera: " << sl::toString(err) << std::endl;
        return 1;
    }

    // Enable object detection
    sl::ObjectDetectionParameters obj_det_params;
    obj_det_params.detection_model = sl::DETECTION_MODEL::MULTI_CLASS_BOX;
    obj_det_params.enable_tracking = true;
    obj_det_params.image_sync = true;
    obj_det_params.enable_mask_output = false;

    err = zed.enableObjectDetection(obj_det_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Error enabling object detection: " << sl::toString(err) << std::endl;
        return 1;
    }

    // Define runtime parameters
    sl::ObjectDetectionRuntimeParameters obj_runtime_params;
    obj_runtime_params.detection_confidence_threshold = 50;

    // Runtime loop
    while (zed.grab() == sl::ERROR_CODE::SUCCESS) {
        sl::Objects objects;
        err = zed.retrieveObjects(objects, obj_runtime_params);
        if (err == sl::ERROR_CODE::SUCCESS) {
            for (auto &object : objects.object_list) {
                std::cout << "Object ID: " << object.id << std::endl;
                std::cout << "Label: " << object.label << std::endl;
                std::cout << "Confidence: " << object.confidence << std::endl;
                std::cout << "Position: [" << object.position.x << ", " << object.position.y << ", " << object.position.z << "]" << std::endl;
                std::cout << "Bounding Box 2D: [";
                for (auto &point : object.bounding_box_2d) {
                    std::cout << "(" << point.x << ", " << point.y << "), ";
                }
                std::cout << "]" << std::endl;
            }
        }
    }

    // Disable object detection and close the camera
    zed.disableObjectDetection();
    zed.close();
    return 0;
}
