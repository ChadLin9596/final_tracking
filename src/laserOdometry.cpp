#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>

#include "final_tracking//common.h"
#include "final_tracking/tic_toc.h"

bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

pcl::IterativeClosestPoint<PointType, PointType> icp_corner;
pcl::IterativeClosestPoint<PointType, PointType> icp_surface;

// Transformation from current frame to world frame
Eigen::Quaternionf q_w_curr(1, 0, 0, 0);
Eigen::Vector3f t_w_curr(0, 0, 0);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>
                                           ("/scanRegistration/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>
                                               ("/scanRegistration/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>
                                        ("/scanRegistration/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>
                                            ("/scanRegistration/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>
                                           ("/scanRegistration/velodyne_cloud_2", 100, laserCloudFullResHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>
                                             ("/laserOdometry/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laserOdometry/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
                                          ("/laserOdometry/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>
                                      ("/laserOdometry/laser_odom_to_init", 100);
    ros::Publisher pubLaserFrameOdometry = nh.advertise<nav_msgs::Odometry>
                                           ("/laserOdometry/laser_last_to_odom", 100);

    int frameCount = 0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }

            // extract the data
            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            // initializing
            if (!systemInited)
            {
                cornerPointsLessSharp->clear();
                pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
                cornerLessSharpBuf.pop();

                surfPointsLessFlat->clear();
                pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
                surfLessFlatBuf.pop();

                icp_corner.setInputTarget(cornerPointsLessSharp);
                icp_surface.setInputTarget(surfPointsLessFlat);

                systemInited = true;
                std::cout << "Initialization finished \n";
            } else {
                Eigen::Matrix4f transform;
                transform << 1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1;

                for (int iter = 0; iter < 1; iter++){
                  pcl::PointCloud<PointType> Final;
                  icp_corner.setInputSource(cornerPointsSharp);
                  icp_corner.align(Final, transform);
                  transform = icp_corner.getFinalTransformation();

                  icp_surface.setInputSource(surfPointsFlat);
                  icp_surface.align(Final, transform);
                  transform = icp_surface.getFinalTransformation();
                }

                Eigen::Quaternionf q_last_curr{transform.topLeftCorner<3, 3>()};
                Eigen::Vector3f t_last_curr = transform.topRightCorner<3, 1>();

                // accumulate transform
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;

                // finish icp, set current less feature point into icp target
                cornerPointsLessSharp->clear();
                pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
                cornerLessSharpBuf.pop();

                surfPointsLessFlat->clear();
                pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
                surfLessFlatBuf.pop();

                icp_corner.setInputTarget(cornerPointsLessSharp);
                icp_surface.setInputTarget(surfPointsLessFlat);

                nav_msgs::Odometry laserFrameOdometry;
                laserFrameOdometry.header.frame_id = "/velodyne_init";
                laserFrameOdometry.child_frame_id = "/velodyne";
                laserFrameOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserFrameOdometry.pose.pose.orientation.x = q_last_curr.x();
                laserFrameOdometry.pose.pose.orientation.y = q_last_curr.y();
                laserFrameOdometry.pose.pose.orientation.z = q_last_curr.z();
                laserFrameOdometry.pose.pose.orientation.w = q_last_curr.w();
                laserFrameOdometry.pose.pose.position.x = t_last_curr.x();
                laserFrameOdometry.pose.pose.position.y = t_last_curr.y();
                laserFrameOdometry.pose.pose.position.z = t_last_curr.z();
                pubLaserFrameOdometry.publish(laserFrameOdometry);
            }

            TicToc t_pub;

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/velodyne_init";
            laserOdometry.child_frame_id = "/velodyne";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserCloudCornerLast2.header.frame_id = "/velodyne";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserCloudSurfLast2.header.frame_id = "/velodyne";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

            sensor_msgs::PointCloud2 laserCloudFullRes3;
            pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
            laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserCloudFullRes3.header.frame_id = "/velodyne";
            pubLaserCloudFullRes.publish(laserCloudFullRes3);

            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}
