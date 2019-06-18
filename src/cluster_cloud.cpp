// final_tracking.cpp



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
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
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

typedef pcl::PointXYZI PointType;
//====================================================================================================


/* Private variable */
//====================================================================================================

ros::Publisher plane_filtered_pub, plane_filtered_remove_pub;
ros::Publisher cluster_pub, predict_pub;
ros::Publisher markerArrayPub;

darknet_ros_msgs::BoundingBoxes result_boxes;

std::vector<pcl::PointCloud<PointType> > pre_track_targets;
std::vector<pcl::PointCloud<PointType> > track_targets;

//===================================================================================================
double planeFilter_thr1;
double planeFilter_thr2;

/* Private function */
//====================================================================================================

void ground_removal(const sensor_msgs::PointCloud2::ConstPtr &msg);

void clustering(const pcl::PointCloud<PointType>::Ptr& cloud_filtered);

void updateTarget(const geometry_msgs::Twist::ConstPtr& msg);

//===================================================================================================

void DownsizeFilter(pcl::PointCloud<PointType>::Ptr cloud_in, pcl::PointCloud<PointType>::Ptr &cloud_out, float size){
  pcl::VoxelGrid<PointType> vox;
  vox.setInputCloud(cloud_in);
  vox.setLeafSize(size, size, size);
  vox.filter(*cloud_out);
}

void planeFilter(pcl::PointCloud<PointType>::Ptr point_in,
                 pcl::PointCloud<PointType>::Ptr point_out,
                 double thresh, bool neg){
  Eigen::Vector3f axis = Eigen::Vector3f(0.0,0.0,1.0);
  pcl::SACSegmentation<PointType> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  seg.setAxis(axis);
  seg.setEpsAngle(30.0f * (M_PI/180.0f) );
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (10000);
  seg.setDistanceThreshold (thresh);
  seg.setInputCloud (point_in);
  seg.segment (*inliers, *coefficients);
  if (inliers->indices.size () == 0)
  {
    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }
  // Extract the plane inliers from the input cloud
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud (point_in);
  extract.setIndices (inliers);
  extract.setNegative (neg);
  extract.filter (*point_out);
}

std::vector<pcl::PointIndices> L2Cluster(pcl::PointCloud<PointType>::Ptr cloud_in,
                                         double dth, int minSize, int maxSize){
  pcl::EuclideanClusterExtraction<PointType> euclidean_cluster;
  pcl::search::KdTree<PointType>::Ptr tree       (new pcl::search::KdTree<pcl::PointXYZI>);
  std::vector<pcl::PointIndices> cluster_indices;
  tree->setInputCloud (cloud_in);
  euclidean_cluster.setClusterTolerance (dth); // 50cm
  euclidean_cluster.setMinClusterSize (minSize);
  euclidean_cluster.setMaxClusterSize (maxSize);
  euclidean_cluster.setSearchMethod (tree);
  euclidean_cluster.setInputCloud (cloud_in);
  euclidean_cluster.extract (cluster_indices);
  return cluster_indices;
}

void ground_removal(const sensor_msgs::PointCloud2::ConstPtr &msg)
{

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*msg, *cloud);

    /// Filter points
    pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>);
    DownsizeFilter(cloud, cloud_filtered, 0.1);

    pcl::PointCloud<PointType>::Ptr cloud_plane (new pcl::PointCloud<PointType> ());
    planeFilter(cloud_filtered, cloud_plane, 0.5, true);
    pcl::PointCloud<PointType>::Ptr cloud_f2(new pcl::PointCloud<PointType>);
    planeFilter(cloud_plane, cloud_f2, 0.9, false);


    sensor_msgs::PointCloud2 filtered_cloud;
    pcl::toROSMsg(*cloud_f2, filtered_cloud);
    filtered_cloud.header.frame_id = "velodyne";
    plane_filtered_pub.publish(filtered_cloud);

    sensor_msgs::PointCloud2 filtered_cloud_remove;
    pcl::toROSMsg(*cloud_plane, filtered_cloud_remove);
    filtered_cloud_remove.header.frame_id = "velodyne";
    plane_filtered_remove_pub.publish(filtered_cloud_remove);

    clustering(cloud_f2);
}

//===================================================================================================

void clustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered)
{
  int j = 50;

  int counter = 0;

  static bool first_add = true;

  bool over_height = false;

  std::ostringstream os;

  Eigen::Matrix3f covariance_matrix;

  Eigen::Vector4f centroid;

  sensor_msgs::PointCloud2 clustered_cloud;

  visualization_msgs::MarkerArray centroid_array;

  pcl::PointCloud<PointType>::Ptr cloud_clusters (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<PointType>::Ptr cloud_cluster  (new pcl::PointCloud<pcl::PointXYZI>);

  std::vector<pcl::PointIndices> cluster_indices = L2Cluster(cloud_filtered, 0.4, 30, 1000);

  track_targets.clear();

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      cloud_filtered->points[*pit].intensity = j;

      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*

      if (cloud_filtered->points[*pit].z > 0.3f)
      {
        over_height = true;

        break;
      }
    }

    if (!over_height)
    {

      cloud_cluster->width = cloud_cluster->points.size ();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;

      pcl::compute3DCentroid(*cloud_cluster, centroid);

      j+=2;

      pcl::computeCovarianceMatrix (*cloud_cluster, centroid, covariance_matrix);

      if (abs(covariance_matrix(0,1)) < 500)
      {

        visualization_msgs::Marker marker;

          marker.header.frame_id = "velodyne";
          marker.header.stamp = ros::Time::now();
          marker.ns = "basic_shapes";
          marker.action = visualization_msgs::Marker::ADD;
          marker.pose.orientation.w = 1.0;
          marker.id = counter;
          marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
          marker.scale.z = 1;
          marker.color.b = 1.0;
          marker.color.g = 1.0;
          marker.color.r = 1.0;
          marker.color.a = 1.0;
          marker.lifetime = ros::Duration(0.5);

          os.str("");
          os.clear();

          os << counter;

        marker.text = os.str();

            marker.pose.position.x = centroid[0];
            marker.pose.position.y = centroid[1];
            marker.pose.position.z = centroid[2];

            centroid_array.markers.push_back(marker);

        *cloud_clusters += *cloud_cluster;

        if (first_add)
        {
          pre_track_targets.push_back(*cloud_cluster);
          track_targets.push_back(*cloud_cluster);
        }
        else
        {
          track_targets.push_back(*cloud_cluster);
        }

        counter++;
    }
  }

  over_height = false;

  cloud_cluster->points.clear();
  }

    pcl::toROSMsg(*cloud_clusters, clustered_cloud);

    clustered_cloud.header.frame_id = "velodyne";

    cluster_pub.publish(clustered_cloud);

    markerArrayPub.publish(centroid_array);

    first_add = false;
}

//===================================================================================================

void updateTarget(const geometry_msgs::Twist::ConstPtr& msg)
{
  int i, k, num_pts, number_targets;

  float velocity[3];

  sensor_msgs::PointCloud2 predict_cloud;

  pcl::PointCloud<pcl::PointXYZI>::Ptr predict_points (new pcl::PointCloud<pcl::PointXYZI>);

  number_targets = pre_track_targets.size();

  velocity[0] = msg->linear.x;
  velocity[1] = msg->linear.y;
  velocity[2] = msg->linear.z;

  if (number_targets > 0)
  {
    for (i = 0; i < number_targets; i++)
    {
      num_pts = pre_track_targets[i].points.size();

      for (k = 0; k < num_pts; k++)
      {
        pre_track_targets[i].points[k].x -= 100*velocity[0];
        pre_track_targets[i].points[k].y -= 100*velocity[1];
        pre_track_targets[i].points[k].z -= velocity[2];
      }


      *predict_points += pre_track_targets[i];
    }

      pcl::toROSMsg(*predict_points, predict_cloud);

      predict_cloud.header.frame_id = "velodyne";

      predict_pub.publish(predict_cloud);

  }

}

//===================================================================================================

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cluster_cloud");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    ros::Subscriber lidar_sub   = nh.subscribe("/points_transform", 10, ground_removal);
    ros::Subscriber imu_vel_sub = nh.subscribe("/imu_velocity", 10, updateTarget);

    plane_filtered_pub = nh.advertise<sensor_msgs::PointCloud2 >("plane_filtered_pub_points", 1000);
    plane_filtered_remove_pub = nh.advertise<sensor_msgs::PointCloud2> ("plane_filtered_remove", 1000);
    cluster_pub        = nh.advertise<sensor_msgs::PointCloud2 >("cluster_cloud", 1000);

    predict_pub        = nh.advertise<sensor_msgs::PointCloud2 >("predict_cloud", 1000);

    markerArrayPub     = nh.advertise<visualization_msgs::MarkerArray>("visualization_msgs/MarkerArray", 10);

    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        loop_rate.sleep();
        ros::spinOnce();
    }
}

