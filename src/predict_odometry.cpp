#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <string>
#include <queue>
#define GRAVITY 9.81

typedef std::pair<std::vector<sensor_msgs::Imu>, std_msgs::Header> combinedData;
typedef std::vector<combinedData> combinedDatas;

nav_msgs::Odometry f2fodom1, f2fodom2;

std::queue<sensor_msgs::Imu> imu_buf;
std::queue<std_msgs::Header> header_buf;

bool systemInit = false;

void getTimeDur(double &last, double &cur, double &dur){
  if (last == 0){
    last = cur;
    dur = cur - last;
  }	else {
    dur = cur - last;
    last = cur;
  }
}

// caculate rotation matrix from t to delta t
void DCM_rot(Eigen::Vector3d w, Eigen::Matrix3d &DCM, double dt){
  Eigen::Vector3d r_X, r_Y, r_Z;
  Eigen::Vector3d r_x, r_y, r_z;
  r_X << 1, 0, 0;
  r_Y << 0, 1, 0;
  r_Z << 0, 0, 1;
  r_x = r_X + dt * w.cross(r_X);
  r_y = r_Y + dt * w.cross(r_Y);
  r_z = r_Z + dt * w.cross(r_Z);

  DCM << r_X.dot(r_x), r_X.dot(r_y), r_X.dot(r_z),
         r_Y.dot(r_x), r_Y.dot(r_y), r_Y.dot(r_z),
         r_Z.dot(r_x), r_Z.dot(r_y), r_Z.dot(r_z);
}

void DCM_rot1(Eigen::Vector3d w, Eigen::Matrix3d &DCM, double dt){
  DCM <<        1, -w(2)*dt,  w(1)*dt,
          w(2)*dt,        1, -w(0)*dt,
         -w(1)*dt,  w(0)*dt,        1;
}

void visualize(visualization_msgs::Marker& line_strip, Eigen::Vector3d &pos){
  line_strip.header.frame_id = "/map";
  line_strip.header.stamp = ros::Time::now();
  line_strip.ns = "points_and_lines";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.id = 2;
  line_strip.type = visualization_msgs::Marker::LINE_STRIP;
  line_strip.scale.x = 0.1;
  line_strip.color.b = 1.0;
  line_strip.color.a = 1.0;
  geometry_msgs::Point p;
  p.x = pos(0);
  p.y = pos(1);
  p.z = pos(2);
  line_strip.points.push_back(p);
}

void cb_imu(const sensor_msgs::Imu::ConstPtr& msg){
  sensor_msgs::Imu imu = *msg;
//  std::cout << "imu time " << imu.header.stamp.sec << "." << imu.header.stamp.nsec<< std::endl;
  imu_buf.push(imu);
//  t_now = imu_in.header.stamp.sec + imu_in.header.stamp.nsec*1e-9;
//  getTimeDur(t_last, t_now, t_dur);

//  Eigen::Vector3d w;
//  w << imu_in.angular_velocity.x,
//       imu_in.angular_velocity.y,
//       imu_in.angular_velocity.z;


//  DCM_rot1(w, DCM, t_dur);

//  // update rotation matrix
//  //DCM_rot(w, DCM, t_dur);

//  acc_b << imu_in.linear_acceleration.x,
//           imu_in.linear_acceleration.y,
//           imu_in.linear_acceleration.z;
//  acc_g = DCM_l * acc_b;
//  acc_g(2) -= GRAVITY;

//  vel_g += acc_g * t_dur;
//  pos += vel_g * t_dur;
//  DCM_l = DCM_l * DCM;

//  std::cout << pos << std::endl << std::endl;
//  pos_g.x = pos(0);
//  pos_g.y = pos(1);
//  pos_g.z = pos(2);
//  visualize(line_strip, pos);
//  pub_pos.publish(line_strip);
}

void cb_points(const sensor_msgs::PointCloud2::ConstPtr &msg){
  std_msgs::Header header = msg->header;
//  std::cout << "points time : " << header.stamp.sec << "." << header.stamp.nsec << std::endl;
  header_buf.push(header);
}


combinedDatas getMeasurement(){
  combinedDatas measurements;
  while(true){
      // check that feature point is empty or not
      if (imu_buf.empty() || header_buf.empty())
          return measurements;

      double td = 0.0;
      if (!(imu_buf.back().header.stamp.toSec() > header_buf.front().stamp.toSec() + td))
      {
//          std::cout << "imu time > header time" << std::endl;
          return measurements;
      }

      // only time of imu front < feature time + td
      if (!(imu_buf.front().header.stamp.toSec() < header_buf.front().stamp.toSec() + td))
      {
//          std::cout << "imu time < header time" << std::endl;
          imu_buf.pop();
          header_buf.pop();
          continue;
      }

      // extract header msg
      combinedData measurement;
      std_msgs::Header header = header_buf.front();
      header_buf.pop();
      while((imu_buf.front().header.stamp.toSec() < header.stamp.toSec() + td))
      {
          measurement.first.push_back(imu_buf.front());
          imu_buf.pop();
      }
      measurement.first.push_back(imu_buf.front());
      measurement.second = header;
      measurements.push_back(measurement);
      if(imu_buf.empty())
          ROS_WARN("no imu between two image");
  }
  return measurements;
}

void initState(Eigen::Vector3d &pos, Eigen::Vector3d &vel, Eigen::Matrix3d &rot){
  pos.setZero();
  vel.setZero();
  rot << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;
}

void imuPropagate(Eigen::Vector3d &pos, Eigen::Vector3d &vel, Eigen::Matrix3d &rot,
                  Eigen::Vector3d acc, Eigen::Vector3d gyro, double dt){
  Eigen::Matrix3d rot_;
  // create rotation matrix
  DCM_rot1(gyro, rot_, dt);
  rot_.normalize();
  rot = rot * rot_;
  pos = pos + vel * dt + 0.5 * rot * acc * dt * dt;
  vel = vel + rot * acc * dt;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "predict_odometry");
  ros::NodeHandle nh;

  ros::Subscriber sub_imu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 100, cb_imu);
  ros::Subscriber sub_point = nh.subscribe<sensor_msgs::PointCloud2> ("/points_raw", 100, cb_points);
  ros::Publisher pub_pos1 = nh.advertise<nav_msgs::Odometry> ("/predict/pos1", 2);
  ros::Publisher pub_pos2 = nh.advertise<nav_msgs::Odometry> ("/predict/pos2", 2);

  std_msgs::Header header;
  sensor_msgs::Imu imu_last;
  ros::Rate rate(10);
  while(ros::ok()){
    if (!imu_buf.empty() && !header_buf.empty()){
//      std::cout << "buffer before " << imu_buf.size() << " " << header_buf.size() << std::endl;
      combinedDatas measurement = getMeasurement();
//      std::cout << "buffer after " << imu_buf.size() << " " << header_buf.size() << std::endl;

      if (measurement.size() !=  0){
//        std::cout << "size : " << measurement[0].first.size() << "\n" << std::endl;
        Eigen::Vector3d pos, vel;
        Eigen::Matrix3d rot;
        initState(pos, vel, rot);

        // Mainly do imu propagate
        if(!systemInit){
          systemInit = true;
          header = measurement[0].second;
          // initial vel update
          for(int i = 0; i < measurement[0].first.size(); i++){
            sensor_msgs::Imu imu = measurement[0].first[i];
            if (i == 0)
              imu_last = imu;
            Eigen::Vector3d acc(0.5 * (imu.linear_acceleration.x + imu_last.linear_acceleration.x),
                                0.5 * (imu.linear_acceleration.y + imu_last.linear_acceleration.y),
                                0.5 * (imu.linear_acceleration.z + imu_last.linear_acceleration.z));
            Eigen::Vector3d gyr(0.5 * (imu.angular_velocity.x + imu_last.angular_velocity.x),
                                0.5 * (imu.angular_velocity.y + imu_last.angular_velocity.y),
                                0.5 * (imu.angular_velocity.z + imu_last.angular_velocity.z));
            double dt = imu.header.stamp.toSec() - imu_last.header.stamp.toSec();
            imuPropagate(pos, vel, rot, acc, gyr, dt);
            imu_last = imu;
          }
        } else {
          // propagate
          for(int i = 0; i < measurement[0].first.size(); i++){
            sensor_msgs::Imu imu = measurement[0].first[i];
            Eigen::Vector3d acc(0.5 * (imu.linear_acceleration.x + imu_last.linear_acceleration.x),
                                0.5 * (imu.linear_acceleration.y + imu_last.linear_acceleration.y),
                                0.5 * (imu.linear_acceleration.z + imu_last.linear_acceleration.z));
            Eigen::Vector3d gyr(0.5 * (imu.angular_velocity.x + imu_last.angular_velocity.x),
                                0.5 * (imu.angular_velocity.y + imu_last.angular_velocity.y),
                                0.5 * (imu.angular_velocity.z + imu_last.angular_velocity.z));
            double dt = imu.header.stamp.toSec() - imu_last.header.stamp.toSec();
            imuPropagate(pos, vel, rot, acc, gyr, dt);
            imu_last = imu;
          }
          Eigen::Quaterniond q(rot);
          f2fodom1.header = header;
          f2fodom1.child_frame_id = "velodyne_current";
          f2fodom1.pose.pose.position.x = pos(0);
          f2fodom1.pose.pose.position.y = pos(1);
          f2fodom1.pose.pose.position.z = pos(2);
          f2fodom1.pose.pose.orientation.x = q.x();
          f2fodom1.pose.pose.orientation.y = q.y();
          f2fodom1.pose.pose.orientation.z = q.z();
          f2fodom1.pose.pose.orientation.w = q.w();

          // publish data
          pub_pos1.publish(f2fodom1);
          pub_pos2.publish(f2fodom2);
        }
      }
    }
    ros::spinOnce();
    rate.sleep();
  }
}
