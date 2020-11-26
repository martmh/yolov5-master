# Setup

## Boat.yaml

Update the boat.yaml file with correct path to image train/valid in dir 'boatData/'
Example of file structure:

train: H:/Develop/AI/yolov5-master/Dataset/boatData/images/train
valid: H:/Develop/AI/yolov5-master/Dataset/boatData/images/valid

nc: 8
names: ['cranefront','craneleft','craneright','greenboxleft','lifeboat','spotlightright','watchtower','winchright']
## Get ready for training Copy the boat.yaml to yolov5-master/
or run the script copy.py

## Train

To train, run yolov5-master/train.py.

## To detect Run from terminal: (current path should be yolov5-master/)
python detect.py --source "can be camera or folder with images example for camera on link : http://10.0.0.10:8080/"
--weights "path to the weights Example: H:/Develop/AI/yolov5-master/runs/exp5_boat/weights/best.pt" --save-dir Hvor det skal lagres, default er inference/output.
