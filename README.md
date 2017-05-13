# Video Processing on Vehicle Front-View Camera

This is the home repository of EE368: Digital Image Processing's Final Project.

The demo videos illustrate road sign detection, traffic light recognition, and speed 
prediction on three KITTI benchmark videos. Due to the computation time of running 
feature mapping and training the model, road sign detection and parameters of the 
learning model for speed prediction are pre-computed, while traffic light detection 
is computed on the fly. The sign detection results are stored as csv file under the 
video source directory and model parameters are stored in `parameters.txt` under 
`scratch\` directory. For speed prediction, a 11x12 segmentation is picked based on 
optimum result for velocity. 

The project was written in python using 
+ `OpenCV` for image processing
+ `scrikit-learn` (`sklearn`) for the linear regression model in velocity prediction, which also depends on `scipy`
+ `matplotlib` for visualization
+ `numpy` for matrix operation

All required libraries are installed on Stanford corn machines. There is a `setup.sh` 
script to install/upgrade all required python libraries for user only (does not require
`sudo`). **NOTICE** the `scripy` library required by `sklearn` is outdated on corn. So 
importing `linear_module` from `sklearn` directly will fail. So **PLEASE** run the `setup.sh` 
script to upgrade the library before running the demo. Although developed in OpenCV 3.1.0 
on Mac, I also modified the script to support OpenCV 2.4 and tested on Linux. Due to 
depended module by OpenCV is missing in latest macOS Sierra, only master branch of OpenCV
works on macOS Sierra, which is of version 3.1.0. Nonetheless, the demo should also work 
on OpenCV 2.4 for other Mac operating system but it is not tested. 

The dataset is also uploaded in the repository. However, given the repository size, cloning
is mostly likely to fail due to connection time out. So please download the
zip file instead from [https://github.com/blackwings-01/DrivingPerception/archive/master.zip].

Demo Videos:
2011_09_26-1
2011_09_26-2
2011_09_26-3


There are set of demos that combines road sign, traffic light and speed together and demos for each individual component. 
A subset of the demos are recorded and uploaded in case installation failed or playing on corn
is too slow. You can download them from
[https://drive.google.com/drive/folders/0BzFuAterjDpFV3p6d3VObHVheGc?usp=sharing]

To get start, first
```
wget https://github.com/blackwings-01/DrivingPerception/archive/master.zip
unzip master.zip
cd DrivingPerception-master/src
./setup.sh
```
Then choose any of the following demos to run:

1. Visualization of roadsign, trafficlight and speed. # can be one of [1,2,3] for three videos:

  ```
  python play.py --mode all --demo <#>
  ```
2. Visualization of trafficlight with color mask

  ```
  python play.py --mode detlight --demo <#>
  ```
3. Visualization of optical flow and averaged flow (Unlike mode all, this is computed on the fly)

  ```
  python play.py --mode flow --demo <#>
  ```
4. Visualization of roadsign of one specific sign with matching explicitly drew (Unlike mode all, this is computed on the fly)

  ```
  python play.py --mode roadsign --demo <#> --sign <signname>
  ```
  `signname` can choose from any of the following: 
    `children_crossing`
    `give_way`
    `keep_right`
    `no_entry`
    `no_parking_end`
    `no_parking_start`
    `no_stopping`
    `no_stopping_end`
    `no_stopping_start`
    `parking`
    `parking_area_end`
    `parking_area_start`
    `pedestrian_crossing_left`
    `pedestrian_crossing_right`
    `speed_limit_30`
    `stop_sign`
5. Picked interesting frames from 3 videos
  
  ```
  ./demo.sh
  ```

Additional options in play.py:
`--path <string>` directory to the frame path. If set `--demo` is ignored
`--delay <float>` amount of delay to add between two frames to adjust video speed. Do not set to 0
`--start-frame <int>` starting frame for playing. Default 0
`--end-frame <int>` ending frame for playing. Default -1 for last frame
`--num-frame <init>` number of frame for playing. Default -1 for all frames. If both `--end-frame` and `--num-frame` are set, whichever ends first will stop the playing
`--no-sign` Disable sign detection.

