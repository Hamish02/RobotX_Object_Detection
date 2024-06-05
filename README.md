# ZED SDK - Object Detection

This container allows for the use of custom trained YOLOv8 models to a ZED camera and extract 3D informations and tracking for each objects.

## Getting Started

 - Install Docker desktop
 - Open directory in VS code
 - Ensure devcontainer extension is installed
 - Control + Shift + P and open in devcontainer

## Run the program

*NOTE: The ZED v1 is not compatible with this module*

```
python3 detector.py --svo /workspaces/Object_Detection/wvh_4.svo --weights /workspaces/Object_Detection/best_small_e50.pt --img_size 2208 
```

### Features

 - 3D bounding boxes with class and distance around detected objects are drawn
 - Objects classes and confidences can be changed

## Support

If you need assistance please email me at hamish.lithgow@uq.net