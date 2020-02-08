# SDC Final Competition - Group 5
## Member
* 0751081 陳靖雯 Ching-Wen Chen
* 0751903 林春榮 Chun-Jong Lin
* 0751906 黃偉嘉

## Procedure
* open 1st terminal 
    * `cd [workspace]`
    * `source devel/setup.bash or devel/setup.zsh`
    * `roslaunch final_tracking final_track.launch`
* open 2nd terminal
    * `rosbag play tracking_1.bag --clock -r 0.05`
