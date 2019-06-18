// data_association.cpp



/* Header */
//====================================================================================================

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <time.h>
#include <math.h>

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>

#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/fpfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/kld_adaptive_particle_filter.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/coherence.h>
#include <pcl/registration/icp.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

#include <Eigen/Dense>
#include <Eigen/Dense>

#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/BoundingBox.h>

#include <image_transport/image_transport.h>

#include <cv_bridge/cv_bridge.h>

using namespace boost::filesystem;

/* Private variable */
//====================================================================================================

const int WIDTH = 1280, HEIGHT = 720;

float velocity[3] = {0, 0, 0};

Eigen::Matrix4f l_t_c;

cv::Mat INTRINSIC_MAT(3, 3, cv::DataType<double>::type); // Intrinsics
cv::Mat DIST_COEFFS(5, 1, cv::DataType<double>::type); // Distortion vector
cv::Mat rVec(3, 1, cv::DataType<float>::type); // Rotation vector
cv::Mat tVec(3, 1, cv::DataType<float>::type); // Translation vector

cv::Mat match_frame;

ros::Publisher imu_pub;
ros::Publisher plane_filtered_pub;
ros::Publisher cluster_pub, predict_pub;
ros::Publisher markerArrayPub;

image_transport::Publisher image_pub;

Eigen::Matrix4f guess;

cv_bridge::CvImagePtr cv_ptr;

darknet_ros_msgs::BoundingBoxes result_boxes;

visualization_msgs::Marker line_strip_odom, line_strip_ekf_odom;

pcl::PointCloud<pcl::PointXYZI>::Ptr prev_scan (new pcl::PointCloud<pcl::PointXYZI> );

std::vector<std::vector<float> > centroid_series;
std::vector<std::vector<float> > current_centroid;

std::vector<pcl::PointCloud<pcl::PointXYZI> > pre_track_targets;
std::vector<pcl::PointCloud<pcl::PointXYZI> > track_targets;