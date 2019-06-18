#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>
#include <tf/transform_broadcaster.h>
// pcl include
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

typedef pcl::PointXYZI PointType;

ros::Publisher pub_points;
pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

Eigen::Matrix3d euler2rotation(Eigen::Vector3d rpy){
  Eigen::Matrix3d rx, ry, rz;
  rz <<  cos(rpy(2)), sin(rpy(2)), 0,
        -sin(rpy(2)), cos(rpy(2)), 0,
                   0,           0, 1;

  rx << 1,            0,           0,
        0,  cos(rpy(0)), sin(rpy(0)),
        0, -sin(rpy(0)), cos(rpy(0));

  ry << cos(rpy(1)), 0, -sin(rpy(1)),
                  0, 1,            0,
        sin(rpy(1)), 0,  cos(rpy(1));

  return rz * ry * rx;
}

void settingInitialTF(Eigen::Matrix4d &trans){

  trans << 1, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 1;
  double roll = 0 * M_PI / 180;
  double pitch = 0 * M_PI / 180;
  double yaw = 125 * M_PI / 180;
  trans.topLeftCorner<3, 3>() = euler2rotation(Eigen::Vector3d(roll, pitch, yaw));
}

void initTransform(pcl::PointCloud<PointType>::Ptr cloudIn,
                   pcl::PointCloud<PointType>::Ptr cloudOut){
  Eigen::Matrix4d transfrom;
  settingInitialTF(transfrom);
  cloudOut = cloudIn;
  for (int i = 0; i < cloudIn->points.size(); i++){
    Eigen::Vector4d point(cloudIn->points[i].x,
                          cloudIn->points[i].y,
                          cloudIn->points[i].z,
                                            1);
    Eigen::Vector4d point_tr = transfrom * point;
    cloudOut->points[i].x = point_tr(0);
    cloudOut->points[i].y = point_tr(1);
    cloudOut->points[i].z = point_tr(2);
  }
}

void cb_points(const sensor_msgs::PointCloud2::ConstPtr &msg){
  pcl::fromROSMsg(*msg, *cloud);
  initTransform(cloud, cloud);
  sensor_msgs::PointCloud2 point_transform;
  pcl::toROSMsg(*cloud, point_transform);
  pub_points.publish(point_transform);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_raw_transform");
  ros::NodeHandle nh;

  ros::Subscriber lidar_sub = nh.subscribe<sensor_msgs::PointCloud2>("/points_raw", 10, cb_points);
  pub_points = nh.advertise<sensor_msgs::PointCloud2>("/points_transform", 10);
  ros::spin();
}
