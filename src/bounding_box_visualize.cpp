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
#include <pcl/common/transforms.h>
#include <string>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <visualization_msgs/Marker.h>

typedef pcl::PointXYZI PointType;
typedef pcl::Normal NormalType;

ros::Publisher pub_marker;

void visual_setting(visualization_msgs::Marker &line_list, std_msgs::Header header, Eigen::Vector3f p1, Eigen::Vector3f p2){
  line_list.header = header;
  line_list.ns = "points_and_lines";
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.pose.orientation.w = 1.0;
  line_list.id = 2;
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.scale.x = 0.1;
  // Line list is red
  line_list.color.r = 1.0;
  line_list.color.a = 1.0;
  geometry_msgs::Point p_1, p_2;
  p_1.x = p1(0);
  p_1.y = p1(1);
  p_1.z = p1(2);
  p_2.x = p2(0);
  p_2.y = p2(1);
  p_2.z = p2(2);
  line_list.points.push_back(p_1);
  line_list.points.push_back(p_2);
}

pcl::PointCloud<PointType>::Ptr Cluster_All(new pcl::PointCloud<PointType>);
void cb_cluster(const sensor_msgs::PointCloud2::ConstPtr& msg){
  std::vector<pcl::PointCloud<PointType>::Ptr> cluster_vector;
  pcl::fromROSMsg(*msg, *Cluster_All);
  // separate cluster
  int intensity_ = -1;
  pcl::PointCloud<PointType>::Ptr aCluster(new pcl::PointCloud<PointType>);
  aCluster->clear();
  for (int i = 0; i < Cluster_All->points.size(); i++){
    PointType pointSel = Cluster_All->points[i];
    if(intensity_ != Cluster_All->points[i].intensity){
      intensity_ = Cluster_All->points[i].intensity;
      if (aCluster->points.size() != 0){
        cluster_vector.push_back(aCluster);
//        aCluster->clear();
        pcl::PointCloud<PointType>::Ptr aCluster_(new pcl::PointCloud<PointType>);
        aCluster = aCluster_;
      }
      aCluster->push_back(pointSel);
    }
    else{
      aCluster->push_back(pointSel);
      if(i == Cluster_All->points.size() - 1){
        cluster_vector.push_back(aCluster);
      }
    }
  }

  // compute bounding box for each cluster
  visualization_msgs::Marker line_list;
  for (int i = 0; i < cluster_vector.size(); i++){
    pcl::PointCloud<PointType>::Ptr cluster = cluster_vector[i];
    Eigen::Vector4f pointCentroid;
    pcl::compute3DCentroid(*cluster, pointCentroid);

    // extract eigen vector and eigen values
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cluster, pointCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
    eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));

    // transform cluster with eigen vector
    Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f tm_inv = Eigen::Matrix4f::Identity();
    tm.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();   //R. as dot project
    tm.block<3, 1>(0, 3) = -1.0f * (eigenVectorsPCA.transpose()) *(pointCentroid.head<3>());//  -R*t
    tm_inv = tm.inverse();
    pcl::PointCloud<PointType>::Ptr transformedCloud(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*cluster, *transformedCloud, tm);

    PointType min_p1, max_p1;
    Eigen::Vector3f c1, c, c2;
    pcl::getMinMax3D(*transformedCloud, min_p1, max_p1);
    c1 = 0.5f * (min_p1.getVector3fMap() + max_p1.getVector3fMap());
    c2 = 0.5f * (max_p1.getVector3fMap() - min_p1.getVector3fMap());
//    std::cout << "centroid(3x1):\n" << c1 << std::endl;

    Eigen::Affine3f tm_inv_aff(tm_inv);
    pcl::transformPoint(c1, c, tm_inv_aff); // c1 transform to c with tm_inv_aff

    Eigen::Vector3f corner_vector[8];
    Eigen::Vector3f corner_vector_transform[8];
    corner_vector[0] = Eigen::Vector3f((c1(0) + c2(0)), (c1(1) + c2(1)), (c1(2) + c2(2)));
    corner_vector[1] = Eigen::Vector3f((c1(0) + c2(0)), (c1(1) + c2(1)), (c1(2) - c2(2)));
    corner_vector[2] = Eigen::Vector3f((c1(0) + c2(0)), (c1(1) - c2(1)), (c1(2) + c2(2)));
    corner_vector[3] = Eigen::Vector3f((c1(0) + c2(0)), (c1(1) - c2(1)), (c1(2) - c2(2)));
    corner_vector[4] = Eigen::Vector3f((c1(0) - c2(0)), (c1(1) + c2(1)), (c1(2) + c2(2)));
    corner_vector[5] = Eigen::Vector3f((c1(0) - c2(0)), (c1(1) + c2(1)), (c1(2) - c2(2)));
    corner_vector[6] = Eigen::Vector3f((c1(0) - c2(0)), (c1(1) - c2(1)), (c1(2) + c2(2)));
    corner_vector[7] = Eigen::Vector3f((c1(0) - c2(0)), (c1(1) - c2(1)), (c1(2) - c2(2)));

    for (int i = 0; i < 8; i++){
      pcl::transformPoint(corner_vector[i], corner_vector_transform[i], tm_inv_aff);
    }


    visual_setting(line_list, msg->header, corner_vector_transform[0], corner_vector_transform[2]);
    visual_setting(line_list, msg->header, corner_vector_transform[0], corner_vector_transform[4]);
    visual_setting(line_list, msg->header, corner_vector_transform[2], corner_vector_transform[6]);
    visual_setting(line_list, msg->header, corner_vector_transform[4], corner_vector_transform[6]);
    visual_setting(line_list, msg->header, corner_vector_transform[0], corner_vector_transform[1]);
    visual_setting(line_list, msg->header, corner_vector_transform[2], corner_vector_transform[3]);
    visual_setting(line_list, msg->header, corner_vector_transform[4], corner_vector_transform[5]);
    visual_setting(line_list, msg->header, corner_vector_transform[6], corner_vector_transform[7]);
    visual_setting(line_list, msg->header, corner_vector_transform[1], corner_vector_transform[3]);
    visual_setting(line_list, msg->header, corner_vector_transform[1], corner_vector_transform[5]);
    visual_setting(line_list, msg->header, corner_vector_transform[3], corner_vector_transform[7]);
    visual_setting(line_list, msg->header, corner_vector_transform[5], corner_vector_transform[7]);
    pub_marker.publish(line_list);
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "bounding_box_visualize");
  ros::NodeHandle nh;

  ros::Subscriber sub_cluster = nh.subscribe<sensor_msgs::PointCloud2>("/cluster_cloud", 10, cb_cluster);
  pub_marker = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 10);


  ros::spin();
}
