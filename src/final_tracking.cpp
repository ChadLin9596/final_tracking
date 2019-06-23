// final_tracking.cpp



/* Header */
//====================================================================================================

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <time.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <queue>
#include <mutex>

#include <ros/ros.h>
#include <ros/package.h>

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
#include <nav_msgs/Odometry.h>
using namespace boost::filesystem;
/* Parameter setting*/
//===================================================================================================
double cluster_tolerance;
int MinClusterSize;
int MaxClusterSize;
double matrix_00;
double matrix_00_;
double matrix_01;
double matrix_10;
double matrix_11;
double matrix_11_;
double matrix_20;
double matrix_21;
double std_low;
double std_high;
//====================================================================================================
//====================================================================================================

/* Private definition */
//====================================================================================================

// typedef pcl::PointXYZI RefPointType;
// typedef ParticleXYZRPY ParticleT;
// typedef pcl::PointCloud<pcl::PointXYZI> Cloud;
// typedef Cloud::Ptr CloudPtr;
// typedef Cloud::ConstPtr CloudConstPtr;
// typedef ParticleFilterTracker<RefPointType, ParticleT> ParticleFilter;

// ParticleFilter::Ptr tracker_;

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

cv_bridge::CvImagePtr cv_ptr;

darknet_ros_msgs::BoundingBoxes result_boxes;

visualization_msgs::Marker line_strip_odom, line_strip_ekf_odom;

std::vector<Eigen::Vector4f> target_cen;
std::vector<Eigen::Vector4f> obser_cen;

std::vector<int> new_track_list;
std::vector<int> candidate_list;

std::vector<int> miss_track_count;

std::vector<pcl::PointCloud<pcl::PointXYZI> > track_targets;
std::vector<pcl::PointCloud<pcl::PointXYZI> > obser_targets;

std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;

std::mutex mutex;

std::ofstream file;

//===================================================================================================

typedef struct _label_{

    double score;

    float distance;

}_label;


/* Private function */
//====================================================================================================

void initializeGlobalParams();

void matching_method(const sensor_msgs::ImageConstPtr& msg);

void ground_removal(const sensor_msgs::PointCloud2::ConstPtr &msg);

void clustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered,
                ros::Time&                                  time_stamp);

void updateTarget();

void formNewTrack();

bool compareTarget(const int& target_idx);

void Odom_callback(const nav_msgs::Odometry::ConstPtr& msg);

double ICP_process(pcl::PointCloud<pcl::PointXYZI>& input,
                 pcl::PointCloud<pcl::PointXYZI>& target);



//===================================================================================================

void initializeGlobalParams()
{
    l_t_c << 0.84592974185943604, 0.53328412771224976, -0.0033089336939156055, 0.092240132391452789,
           0.045996580272912979, -0.079141519963741302, -0.99580162763595581, -0.35709697008132935,
           -0.53130710124969482, 0.84222602844238281, -0.091477409005165100, -0.16055910289287567,
           0, 0, 0, 1;

    INTRINSIC_MAT.at<double>(0, 0) = 698.939;
    INTRINSIC_MAT.at<double>(1, 0) = 0;
    INTRINSIC_MAT.at<double>(2, 0) = 0;

    INTRINSIC_MAT.at<double>(0, 1) = 0;
    INTRINSIC_MAT.at<double>(1, 1) = 698.939;
    INTRINSIC_MAT.at<double>(2, 1) = 0;

    INTRINSIC_MAT.at<double>(0, 2) = 641.868;
    INTRINSIC_MAT.at<double>(1, 2) = 385.788;
    INTRINSIC_MAT.at<double>(2, 2) = 1.0;

    DIST_COEFFS.at<double>(0) = -0.171466;
    DIST_COEFFS.at<double>(1) = 0.0246144;
    DIST_COEFFS.at<double>(2) = 0;
    DIST_COEFFS.at<double>(3) = 0;
    DIST_COEFFS.at<double>(4) = 0;

    rVec.at<float>(0) = 0.0f;
    rVec.at<float>(1) = 0.0f;
    rVec.at<float>(2) = 0.0f;

    tVec.at<float>(0) = 0.0f;
    tVec.at<float>(1) = 0.0f;
    tVec.at<float>(2) = 0.0f;
}

//===================================================================================================

void matching_method(const sensor_msgs::ImageConstPtr& msg)
{
        static bool initiailize = false;

        int rows, cols;

        double minVal;

    cv::Point minLoc;

        cv::Mat current_frame, half_match_img, half_compare_img, temp, result;
        cv::Mat gray_frame;

        cv_bridge::CvImagePtr cv_ptr;

        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        current_frame = cv_ptr->image.clone();

        cv::cvtColor(current_frame, gray_frame, CV_BGR2GRAY);

        rows = current_frame.rows>>2;
        cols = current_frame.cols>>2;

        std::cout << rows << " " << cols <<std::endl;

        if (initiailize)
        {

                cv::cvtColor(match_frame(cv::Rect(cols>>1, rows>>1, cols, rows)), temp, CV_BGR2GRAY);

                cv::cvtColor(current_frame(cv::Rect(0, 0, match_frame.cols>>1, match_frame.rows>>1)), half_compare_img, CV_BGR2GRAY);

                cv::matchTemplate(half_compare_img, temp, result, CV_TM_SQDIFF);

                cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

                cv::rectangle(gray_frame, minLoc, cv::Point(minLoc.x+temp.cols , minLoc.y+temp.rows), cv::Scalar::all(0), 3);

        }

        match_frame = cv_ptr->image.clone();

        initiailize = true;

        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", gray_frame).toImageMsg();

        image_pub.publish(img_msg);
}

//===================================================================================================

void ground_removal(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    ros::Time time_stamp = msg->header.stamp;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    fromROSMsg(*msg, *cloud);


    /// Set weights of segmenting ground points
    Eigen::Vector3f axis = Eigen::Vector3f(0.0,1.0,0.0);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZI> ());
    pcl::PCDWriter writer;

    seg.setAxis(axis);
    seg.setEpsAngle(30.0f * (M_PI/180.0f) );
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (10000);
    seg.setDistanceThreshold (0.5);

    // Segment ground plane
    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0) 
    {
        std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    // Extract the plane inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZI>);

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud = *cloud_f;

    sensor_msgs::PointCloud2 filtered_cloud;
    pcl::toROSMsg(*cloud, filtered_cloud);
    filtered_cloud.header.frame_id = "velodyne";
    plane_filtered_pub.publish(filtered_cloud);

    
    clustering(cloud, time_stamp);
}

//===================================================================================================

void clustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered,
                ros::Time&                                  time_stamp)
{
    int j = 50;
    int k = 0;
    int i, idx, num, m;
    int obser_size;

    int counter = 0;

    static bool first_add = true;

    bool over_height = false;
    bool ret;

    std::vector<pcl::PointIndices> cluster_indices;

    std::ostringstream os;

    Eigen::Matrix3f covariance_matrix;

    Eigen::Vector4f centroid;

    sensor_msgs::PointCloud2 clustered_cloud;

    visualization_msgs::MarkerArray centroid_array;

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> euclidean_cluster;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clusters (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster  (new pcl::PointCloud<pcl::PointXYZI>);

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree       (new pcl::search::KdTree<pcl::PointXYZI>);

    std::vector <pcl::PointIndices> clusters;

    tree->setInputCloud (cloud_filtered);

    euclidean_cluster.setClusterTolerance (cluster_tolerance);
    euclidean_cluster.setMinClusterSize (MinClusterSize);
    euclidean_cluster.setMaxClusterSize (MaxClusterSize);
    euclidean_cluster.setSearchMethod (tree);
    euclidean_cluster.setInputCloud (cloud_filtered);
    euclidean_cluster.extract (cluster_indices);

    obser_targets.clear();
    obser_cen.clear();

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {

            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            {
                cloud_filtered->points[*pit].intensity = j;

                cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*

                if (cloud_filtered->points[*pit].z > 0.2f)
                {
                        over_height = true;

                        break;
                }
            }

            if (!over_height)
            {

                cloud_cluster->width = cloud_cluster->points.size();
                cloud_cluster->height = 1;
                cloud_cluster->is_dense = true;
                j+=2;

                //PCA
                pcl::compute3DCentroid(*cloud_cluster, centroid);
                pcl::computeCovarianceMatrix (*cloud_cluster, centroid, covariance_matrix);
                // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
                // Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
                // Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();


                // Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
                // transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
                // transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3,3>(0,0)) * (centroid.head<3>());

                // pcl::PointCloud<pcl::PointXYZI>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZI>);
                // pcl::transformPointCloud(*cloud_cluster, *transformedCloud, transform);


                // //standard deviation of the cluster in eigen vector direction
                // std::vector<float> feature_points;
                // float sum,mean,variance,stdDeviation;
                // for (size_t i=0; i<cloud_cluster->points.size();++i)
                // {

                //         float feature_point = cloud_cluster->points[i].x*eigenVectorsPCA(2,0)+cloud_cluster->points[i].y*eigenVectorsPCA(2,1)+cloud_cluster->points[i].z*eigenVectorsPCA(2,2);
                //         feature_points.push_back(feature_point);
                // }

                // for(size_t i=0; i< feature_points.size();++i)
                // {
                //     sum += feature_points[i];
                // }
                // mean = sum/feature_points.size();

                // for(size_t i=0; i< feature_points.size();++i)
                // {
                // variance += pow(feature_points[i] - mean, 2);
                // }
                // variance=variance/feature_points.size();
                // stdDeviation = sqrt(variance);
                // std::cout<<stdDeviation<<std::endl;



                if ((abs(covariance_matrix(1,1)) < matrix_11_) && (abs(covariance_matrix(0,0)) < matrix_00_))//(abs(covariance_matrix(0,0)) > matrix_00) && (abs(covariance_matrix(0,0)) < matrix_00_) && (abs(covariance_matrix(1,1)) > matrix_11) && (abs(covariance_matrix(1,1)) < matrix_11_) && (abs(covariance_matrix(0,1)) > matrix_01) && (abs(covariance_matrix(1,0)) > matrix_10) && (abs(covariance_matrix(2,0)) > matrix_20) && (abs(covariance_matrix(2,1)) > matrix_21))
                {
                    if(true)//stdDeviation < std_high)&&(stdDeviation > std_low))
                    {
                        

                        if (first_add)
                        {
                            *cloud_clusters += *cloud_cluster;

                            track_targets.push_back(*cloud_cluster);
                        
                            target_cen.push_back(centroid);

                            visualization_msgs::Marker marker;

                            marker.header.frame_id = "velodyne"; 
                            marker.header.stamp = time_stamp; 
                            marker.ns = "basic_shapes"; 
                            marker.action = visualization_msgs::Marker::ADD; 
                            marker.pose.orientation.w = 1.0; 
                            marker.id = counter; 
                            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING; 
                            marker.scale.z = 3; 
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

                            file << marker.header.stamp << ", " << counter << ", " << marker.pose.position.x << ", " << marker.pose.position.y << ", " << marker.pose.position.z << "\n";
                            
                            counter++;
                    
                        }
                        else
                        {
                            *cloud_clusters += *cloud_cluster;

                            obser_targets.push_back(*cloud_cluster);

                            obser_cen.push_back(centroid);
                        }

                        counter++;

                        }
                }

                //feature_points.clear();

            }

            over_height = false;

            cloud_cluster->points.clear();
    }

    if (!first_add)
    {
        //ICP_process(translate, cloud_clusters);

        candidate_list.clear();
        new_track_list.clear();

        obser_size = obser_cen.size();

        new_track_list.resize(obser_size);

        for (m = 0; m < obser_size; m++)
        {
            new_track_list[m] = 0;
        }

        num = target_cen.size();

        for (i = 0; i < num; i++)
        {
            ret = compareTarget(i);

            visualization_msgs::Marker marker;

            marker.header.frame_id = "velodyne"; 
            marker.header.stamp = time_stamp; 
            marker.ns = "basic_shapes"; 
            marker.action = visualization_msgs::Marker::ADD; 
            marker.pose.orientation.w = 1.0; 
            marker.id = i; 
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING; 
            marker.scale.z = 3; 
            marker.color.b = 1.0; 
            marker.color.g = 1.0; 
            marker.color.r = 1.0; 
            marker.color.a = 1.0;
            marker.lifetime = ros::Duration(0.5);

            os.str("");
            os.clear();

            os << i;

            marker.text = os.str();

            marker.pose.position.x = target_cen[i][0];
            marker.pose.position.y = target_cen[i][1];
            marker.pose.position.z = target_cen[i][2];

            if (ret)
            {
                centroid_array.markers.push_back(marker);
                file << marker.header.stamp << ", " << i << ", " << marker.pose.position.x << ", " << marker.pose.position.y << ", " << marker.pose.position.z << "\n";

            }
        }

    }

    formNewTrack();

    updateTarget();

    pcl::toROSMsg(*cloud_clusters, clustered_cloud);

    clustered_cloud.header.frame_id = "velodyne";

    cluster_pub.publish(clustered_cloud);

    markerArrayPub.publish(centroid_array);


    first_add = false;
}

//===================================================================================================

bool compareTarget(const int& target_idx)
{
    bool tracked = false;
    bool break_flag = false;
    bool ret = false;

    static int counter = 0;

    const float score_thres = 0.1f;

    int i, j, closet_idx, remove_num;
    int obser_num, hypothesis_num = 0;
    int K = 1;

    float abs_dis;

    double score;
    double min_score = DBL_MAX;

    double min_distance = DBL_MAX;

    Eigen::Matrix2f distance;
    Eigen::MatrixXf substraction(1, 3);
    Eigen::Matrix3f covariance_matrix;

    pcl::PointXYZ searchPoint;

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    _label label_buffer;

    std::vector<_label> candidate;

    pcl::PointCloud<pcl::PointXYZ>::Ptr tree (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    obser_num = obser_targets.size();

    remove_num = candidate_list.size();

    for (i = 0; i < obser_num; i++)
    {
        break_flag = false;

        for (j = 0; j < remove_num; j++)
        {
            if (candidate_list[j] == i)
            {
                break_flag = true;

                new_track_list[i] = 0;
            }
        }

        if (!break_flag)
        {
            substraction(0,0) = (obser_cen[i][0] - (target_cen[target_idx][0]));
            substraction(0,1) = (obser_cen[i][1] - (target_cen[target_idx][1]));
            substraction(0,2) = (obser_cen[i][2] - (target_cen[target_idx][2]));

            pcl::computeCovarianceMatrix (track_targets[target_idx], target_cen[target_idx], covariance_matrix);

            //distance = substraction * covariance_matrix.inverse() * substraction.transpose();

            distance(0,0) = fabs(substraction(0,0) + substraction(0,1));

            abs_dis = (distance(0, 0) > 0.0f) ? distance(0,0) : -distance(0,0);

            // if (abs_dis > 1.5f && abs_dis < 2.0f)
            // {
            //     new_track_list[i]++;
            // }

            if (abs_dis < 3.0f)
            {
                score = ICP_process(track_targets[target_idx], obser_targets[i]);

                new_track_list[i]++;

                if (min_score > score && abs_dis < 2.0f)
                {
                    min_distance = abs_dis;

                    min_score = score;

                    closet_idx = i;

                    label_buffer.score = min_score;
                    label_buffer.distance = min_distance;

                    candidate.push_back(label_buffer);

                    hypothesis_num++;
                }

                if (score < 0.5f)
                {
                    new_track_list[i]++;
                }
            }
        }
    }

    //std::cout << counter << " : " << min_distance << "\t" << min_score <<std::endl;


    if (min_score < 5.0f)
    {
        target_cen[target_idx] = obser_cen[closet_idx];

        pcl::copyPointCloud(obser_targets[closet_idx], track_targets[target_idx]);

        candidate_list.push_back(closet_idx);

        //obser_targets.erase(obser_targets.begin() + closet_idx);

        //obser_cen.erase(obser_cen.begin() + closet_idx);

        ret = true;
    }

    counter++;


    return ret;
}

//===================================================================================================

void formNewTrack()
{
    int i, obser_num;
    int threshold;

    obser_num = (int)new_track_list.size();

    threshold = int(0.1 * (float)track_targets.size());

    for (i = 0; i < obser_num; i++)
    {
        if (new_track_list[i] > threshold)
        {
            track_targets.push_back(obser_targets[i]);

            target_cen.push_back(obser_cen[i]);
        }

    }

}

//===================================================================================================

void Odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
    mutex.lock();

    odom_buffer.push(msg);

    mutex.unlock();
}
//===================================================================================================


double ICP_process(pcl::PointCloud<pcl::PointXYZI>& input,
                   pcl::PointCloud<pcl::PointXYZI>& target)
{
    int i, j;
    int pre_size, cur_size;

    double max_score = DBL_MAX;

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    pcl::PointCloud<pcl::PointXYZI> icp_result;

    icp.setMaximumIterations (100);

    icp.setInputSource(input.makeShared());
    icp.setInputTarget(target.makeShared());

    icp.align(icp_result);

    if (icp.hasConverged())
    {
        max_score = icp.getFitnessScore();
    }

    return max_score;
}

//===================================================================================================

void updateTarget()
{
    int i, k, num_pts, number_targets;

    sensor_msgs::PointCloud2 predict_cloud;

    pcl::PointCloud<pcl::PointXYZI>::Ptr predict_points (new pcl::PointCloud<pcl::PointXYZI>);

    if (!odom_buffer.empty())

    {

        //std::cout << "receive predict" << std::endl;

        number_targets = track_targets.size();

        mutex.lock();

        velocity[0] = odom_buffer.front()->pose.pose.position.x;
        velocity[1] = odom_buffer.front()->pose.pose.position.y;
        velocity[2] = odom_buffer.front()->pose.pose.position.z;

        odom_buffer.pop();

        mutex.unlock();;

        if (number_targets > 0)
        {
            for (i = 0; i < number_targets; i++)
            {
                num_pts = track_targets[i].points.size();

                for (k = 0; k < num_pts; k++)
                {
                    track_targets[i].points[k].x -= velocity[0];
                    track_targets[i].points[k].y -= velocity[1];
                    track_targets[i].points[k].z -= velocity[2];
                }

                *predict_points += track_targets[i];
            }

            pcl::toROSMsg(*predict_points, predict_cloud);

            predict_cloud.header.frame_id = "velodyne";

            predict_pub.publish(predict_cloud);

        }

        number_targets = target_cen.size();

        for (i = 0; i < number_targets; i++)
        {
            target_cen[i][0] -= velocity[0];
            target_cen[i][1] -= velocity[1];
            target_cen[i][2] -= velocity[2];
        }


    }

}

//===================================================================================================

int main(int argc, char **argv)
{
    std::string folder_path;

    ros::init(argc, argv, "final_tracking");
    ros::NodeHandle nh;
    nh.param<double>("cluster_tolerance",cluster_tolerance,0.4);
    nh.param<int>("MinClusterSize",MinClusterSize,50);
    nh.param<int>("MaxClusterSize",MaxClusterSize,2000);
    nh.param<double>("matrix_00",matrix_00,500);
    nh.param<double>("matrix_00_",matrix_00_,500);
    nh.param<double>("matrix_01",matrix_01,500);
    nh.param<double>("matrix_10",matrix_10,500);
    nh.param<double>("matrix_11",matrix_11,500);
    nh.param<double>("matrix_11_",matrix_11_,500);
    nh.param<double>("matrix_20",matrix_20,500);
    nh.param<double>("matrix_21",matrix_21,500);
    nh.param<double>("std_low",std_low,0);
    nh.param<double>("std_high",std_high,500);
    image_transport::ImageTransport it(nh);

    line_strip_odom.header.frame_id = "velodyne";
    line_strip_odom.id = 2;
    line_strip_odom.type = visualization_msgs::Marker::LINE_STRIP;
    line_strip_odom.scale.x = 0.03;
    line_strip_odom.color.r = 1.0;
    line_strip_odom.color.a = 1.0;

    plane_filtered_pub = nh.advertise<sensor_msgs::PointCloud2 >("plane_filtered_pub_points", 1000);
    cluster_pub        = nh.advertise<sensor_msgs::PointCloud2 >("cluster_cloud", 1000);

    predict_pub        = nh.advertise<sensor_msgs::PointCloud2 >("predict_cloud", 1000);

    markerArrayPub     = nh.advertise<visualization_msgs::MarkerArray>("visualization_msgs/MarkerArray", 10);

    imu_pub            = nh.advertise<sensor_msgs::Imu> ("/imu_data", 2);

    image_pub          = it.advertise("/camera/output", 1);

    ros::Subscriber lidar_sub   = nh.subscribe("/points_transform", 100, ground_removal);
    ros::Subscriber imu_vel_sub = nh.subscribe("/predict/pos2", 100, Odom_callback);

    folder_path = ros::package::getPath("final_tracking");
    folder_path += "/output.csv";

    file.open(folder_path);

    //image_transport::Subscriber img_sub = it.subscribe("/camera/rgb/image_raw", 1, matching_method);

    initializeGlobalParams();
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        loop_rate.sleep();
        ros::spinOnce();
    }


}

