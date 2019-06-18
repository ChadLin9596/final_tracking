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


typedef struct _track_pattern_{

    std::vector<float> centroid;

    pcl::PointCloud<pcl::PointXYZI> point_cloud;

}_track_pattern;

typedef struct _label_{

    int index;

    float distance;

    double score;

}_label;

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

std::vector<Eigen::Vector4f> centroid_series;
std::vector<Eigen::Vector4f> current_centroid;

std::vector<std::vector<int> > track_label;

std::vector<int > label_list;

std::vector<pcl::PointCloud<pcl::PointXYZI> > pre_track_targets;
std::vector<pcl::PointCloud<pcl::PointXYZI> > track_targets;

std::vector<_track_pattern> track_patterns;

//===================================================================================================



/* Private function */
//====================================================================================================

void initializeGlobalParams();

void matching_method(const sensor_msgs::ImageConstPtr& msg);

void ground_removal(const sensor_msgs::PointCloud2::ConstPtr &msg);

void clustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered);

void updateTarget(const nav_msgs::Odometry::ConstPtr& msg);

int compareTarget(Eigen::Vector4f centroid, 
                  float* translate,
                  pcl::PointCloud<pcl::PointXYZI> target);

void gateAssociation();

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

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    fromROSMsg(*msg, *cloud);

    /// Filter points
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.1f, 0.1f, 0.1f);
    vg.filter (*cloud_filtered);

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
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0) 
    {
    	std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    // Extract the plane inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZI>);

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;

    sensor_msgs::PointCloud2 filtered_cloud;
    pcl::toROSMsg(*cloud_filtered, filtered_cloud);
    filtered_cloud.header.frame_id = "velodyne";
    plane_filtered_pub.publish(filtered_cloud);

    
    clustering(cloud_filtered);
}

//===================================================================================================

void clustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered)
{
	int j = 50;
    int k = 0;
    int i, idx, num;

    int counter = 0;

    float translate[3] = {0, 0, 0};

	static bool first_add = true;

	bool over_height = false;

	std::vector<float> centroid_tmp(3);

    std::vector<pcl::PointIndices> cluster_indices;

    std::vector<int> labeled;

	std::ostringstream os;

	Eigen::Matrix3f covariance_matrix;

    Eigen::Vector4f centroid;

    sensor_msgs::PointCloud2 clustered_cloud;

    visualization_msgs::MarkerArray centroid_array;

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> euclidean_cluster;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clusters (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster  (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr sec_cloud_clusters (new pcl::PointCloud<pcl::PointXYZI>);

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree       (new pcl::search::KdTree<pcl::PointXYZI>);

    pcl::MinCutSegmentation<pcl::PointXYZI> clustering;
    pcl::PointCloud<pcl::PointXYZI>::Ptr foregroundPoints(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointXYZI point;
    point.x = 100.0;
    point.y = 100.0;
    point.z = 100.0;
    foregroundPoints->points.push_back(point);
    clustering.setForegroundPoints(foregroundPoints);
    clustering.setSigma(0.02);
    clustering.setRadius(0.01);
    clustering.setNumberOfNeighbours(20);
    clustering.setSourceWeight(0.6);

    std::vector <pcl::PointIndices> clusters;

    tree->setInputCloud (cloud_filtered);

    euclidean_cluster.setClusterTolerance (0.4); // 50cm
    euclidean_cluster.setMinClusterSize (50);
    euclidean_cluster.setMaxClusterSize (5000);
    euclidean_cluster.setSearchMethod (tree);
    euclidean_cluster.setInputCloud (cloud_filtered);
    euclidean_cluster.extract (cluster_indices);

    track_targets.clear();
    current_centroid.clear();

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

		    pcl::compute3DCentroid(*cloud_cluster, centroid);

		    j+=2;

	    	pcl::computeCovarianceMatrix (*cloud_cluster, centroid, covariance_matrix);

	    	if (abs(covariance_matrix(0,1)) < 500)
	    	{
		    	*cloud_clusters += *cloud_cluster;

	    		centroid_tmp[0] = centroid[0];
	    		centroid_tmp[1] = centroid[1];
	    		centroid_tmp[2] = centroid[2];

		    	if (first_add)
		    	{
		    		pre_track_targets.push_back(*cloud_cluster);
		    		track_targets.push_back(*cloud_cluster);
		    		
		    		centroid_series.push_back(centroid);

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
		    	}
		    	else
		    	{
		    		track_targets.push_back(*cloud_cluster);

		    		current_centroid.push_back(centroid);

		    	}

	    		counter++;
			}
		}

		over_height = false;

		cloud_cluster->points.clear();
    }

    // clustering.setInputCloud(cloud_clusters);
    // clustering.extract(clusters);

 //    if (!first_add)
 //    {

	//     for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
	//     {
	//         pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);

	//         for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
	//             cluster->points.push_back(cloud_clusters->points[*point]);

	//         cluster->width = cluster->points.size();
	//         cluster->height = 1;
	//         cluster->is_dense = true;

	//         if (cluster->points.size() <= 0)
	//             break;

	//   //       track_targets.push_back(*cluster);

	//   //       pcl::compute3DCentroid(*cluster, centroid);

	//   //       centroid_tmp[0] = centroid[0];
	// 		// centroid_tmp[1] = centroid[1];
	// 		// centroid_tmp[2] = centroid[2];

	// 		// current_centroid.push_back(centroid_tmp);

	//         *sec_cloud_clusters += *cluster;
	//     }
	// }

    if (!first_add)
    {
    	//ICP_process(translate, cloud_clusters);

    	num = current_centroid.size();

        label_list.clear();

    	for (i = 0; i < num; i++)
    	{
    		idx = compareTarget(current_centroid[i], translate, track_targets[i]);

            labeled.push_back(idx);

    		visualization_msgs::Marker marker;

	        marker.header.frame_id = "velodyne"; 
	        marker.header.stamp = ros::Time::now(); 
	        marker.ns = "basic_shapes"; 
	        marker.action = visualization_msgs::Marker::ADD; 
	        marker.pose.orientation.w = 1.0; 
	        marker.id = i; 
	        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING; 
	        marker.scale.z = 1; 
	        marker.color.b = 1.0; 
	        marker.color.g = 1.0; 
	        marker.color.r = 1.0; 
	        marker.color.a = 1.0;
	        marker.lifetime = ros::Duration(0.5);

	        os.str("");
	        os.clear();

	        os << idx;

    		marker.text = os.str();

            marker.pose.position.x = current_centroid[i][0];
            marker.pose.position.y = current_centroid[i][1];
            marker.pose.position.z = current_centroid[i][2];

            centroid_array.markers.push_back(marker);
    	}
    }

    pcl::toROSMsg(*cloud_clusters, clustered_cloud);

    clustered_cloud.header.frame_id = "velodyne";

    cluster_pub.publish(clustered_cloud);

    markerArrayPub.publish(centroid_array);

    pcl::copyPointCloud(*cloud_clusters, *prev_scan);

    if (first_add)
    {
        track_label.resize(track_targets.size());
    }

    first_add = false;
}

//===================================================================================================

int compareTarget(Eigen::Vector4f centroid, 
				  float* translate,
                  pcl::PointCloud<pcl::PointXYZI> target)
{
    static bool initialize = true;

    bool break_flag = false;

	int i, j, idx, num_list;
    int track_num = 0;
	int traget_num = centroid_series.size();
    int K = 1;

    double score;
    double min_score = DBL_MAX;

    float min_distance = DBL_MAX;

	Eigen::MatrixXf distance(1,1);
    Eigen::MatrixXf substraction(1, 3);
    Eigen::Matrix3f covariance_matrix;

    _label tmp_label;

	std::vector<float> centroid_tmp(3);

    std::vector<_label> target_label;

    pcl::PointCloud<pcl::PointXYZ>::Ptr tree (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    pcl::PointXYZ searchPoint;

    num_list = label_list.size();

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

	for (i = 0; i < traget_num; i++)
	{

        substraction(0,0) = (centroid[0] - (centroid_series[i][0]));
        substraction(0,1) = (centroid[1] - (centroid_series[i][1]));
        substraction(0,2) = (centroid[2] - (centroid_series[i][2]));

        pcl::computeCovarianceMatrix (pre_track_targets[i], centroid_series[i], covariance_matrix);

        distance = substraction * covariance_matrix.inverse() * substraction.transpose();

		if (distance(0,0) < 6.0f && distance(0,0) > 0.0f)
		{
            score = ICP_process(pre_track_targets[i], target);

            if (min_score > score)
            {
                min_score = score;
                idx = i;

                tmp_label.index = i;
                tmp_label.score = score;
                tmp_label.distance = abs(distance(0,0));

                target_label.push_back(tmp_label);

                track_num++;

                // track_label[i].push_back(idx);
            }

		}
	}


//    if (track_num > 0)
//    {

//        tree->width = track_num;
//        tree->height = 1;

//        tree->points.resize(track_num);

//        for (i = 0; i < track_num; i++)
//        {
//            tree->points[i].x = target_label[i].score;
//            tree->points[i].y = target_label[i].score;
//            tree->points[i].z = target_label[i].score;
//        }

//        searchPoint.x = 0.0f;
//        searchPoint.y = 0.0f;
//        searchPoint.z = 0.0f;

//        kdtree.setInputCloud (tree);

//        if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
//        {
//            idx = pointIdxNKNSearch[0];
//        }
//    }

	if (min_score > 0.2f)
	{
		idx = traget_num;

		centroid_series.push_back(centroid);

        pre_track_targets.push_back(target);
	}

    initialize = false;

	return idx;

}

//===================================================================================================

void gateAssociation()
{




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

void updateTarget(const nav_msgs::Odometry::ConstPtr& msg)
{
	int i, k, num_pts, number_targets;

	sensor_msgs::PointCloud2 predict_cloud;

	pcl::PointCloud<pcl::PointXYZI>::Ptr predict_points (new pcl::PointCloud<pcl::PointXYZI>);

  std::cout << "receive predict" << std::endl;
	number_targets = pre_track_targets.size();

  velocity[0] = msg->pose.pose.position.x;
  velocity[1] = msg->pose.pose.position.y;
  velocity[2] = msg->pose.pose.position.z;




	if (number_targets > 0)
	{
		for (i = 0; i < number_targets; i++)
		{
			num_pts = pre_track_targets[i].points.size();

			for (k = 0; k < num_pts; k++)
			{
        pre_track_targets[i].points[k].x -= velocity[0];
        pre_track_targets[i].points[k].y -= velocity[1];
				pre_track_targets[i].points[k].z -= velocity[2];
			}

			*predict_points += pre_track_targets[i];
		}

	    pcl::toROSMsg(*predict_points, predict_cloud);

	    predict_cloud.header.frame_id = "velodyne";

	    predict_pub.publish(predict_cloud);

	}

	number_targets = centroid_series.size();

	for (i = 0; i < number_targets; i++)
	{
		centroid_series[i][0] -= 100*velocity[0];
		centroid_series[i][1] -= 100*velocity[1];
		centroid_series[i][2] -= velocity[2];
	}

}

//===================================================================================================

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "final_tracking");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    line_strip_odom.header.frame_id = "velodyne";
    line_strip_odom.id = 2;
    line_strip_odom.type = visualization_msgs::Marker::LINE_STRIP;
    line_strip_odom.scale.x = 0.03;
    line_strip_odom.color.r = 1.0;
    line_strip_odom.color.a = 1.0;

    guess << 1, 0, 0, 0,
	         0, 1, 0, 0,
	         0, 0, 1, 0,
	         0, 0, 0, 1;

    plane_filtered_pub = nh.advertise<sensor_msgs::PointCloud2 >("plane_filtered_pub_points", 1000);
    cluster_pub        = nh.advertise<sensor_msgs::PointCloud2 >("cluster_cloud", 1000);

    predict_pub        = nh.advertise<sensor_msgs::PointCloud2 >("predict_cloud", 1000);

    markerArrayPub     = nh.advertise<visualization_msgs::MarkerArray>("visualization_msgs/MarkerArray", 10);

    imu_pub            = nh.advertise<sensor_msgs::Imu> ("/imu_data", 2);

    image_pub          = it.advertise("/camera/output", 1);

    ros::Subscriber lidar_sub   = nh.subscribe("/points_transform", 100, ground_removal);
    ros::Subscriber imu_vel_sub = nh.subscribe("/predict/pos2", 100, updateTarget);

    //image_transport::Subscriber img_sub = it.subscribe("/camera/rgb/image_raw", 1, matching_method);

    initializeGlobalParams();
    ros::Rate loop_rate(10);
    while (ros::ok()) 
    {
        loop_rate.sleep();
        ros::spinOnce();
    }
}

