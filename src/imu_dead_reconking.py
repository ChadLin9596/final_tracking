#!/usr/bin/env python

import rospy, math
import numpy as np

from sensor_msgs.msg import Imu

from geometry_msgs.msg import Quaternion, Pose, Point, Vector3, Twist

from visualization_msgs.msg import Marker

from std_msgs.msg import Header, ColorRGBA

GRAVITY = 9.8

class ROS_NODE(object):

	def __init__(self):

		self.subscriber = rospy.Subscriber("/imu/data", Imu, self.callback)

		self.ang_vel    = [0, 0, 0]
		self.lin_acc    = [0, 0, 0]

		self.publisher = rospy.Publisher('/vis/marker', Marker, queue_size = 1)
		self.imu_vel_pub = rospy.Publisher('/imu_velocity', Twist, queue_size = 1)

		self.imu_velocity = Twist()

		self.markers = Marker()

		self.markers.header.frame_id = "velodyne"
		self.markers.type 			 = Marker.LINE_STRIP
		self.markers.action 		 = Marker.ADD
		self.markers.id  			 = 2

		self.markers.scale.x = 0.03
		self.markers.scale.y = 0.03
		self.markers.scale.z = 0.03

		self.markers.color.a = 1.0
		self.markers.color.r = 0.0
		self.markers.color.g = 0.0
		self.markers.color.b = 1.0 

		self.markers.pose.orientation.x = 0.0
		self.markers.pose.orientation.y = 0.0
		self.markers.pose.orientation.z = 0.0
		self.markers.pose.orientation.w = 1.0

		self.markers.pose.position.x = 0.0
		self.markers.pose.position.y = 0.0
		self.markers.pose.position.z = 0.0

		self.markers.points = []

		self.position = np.zeros((3,1))
		self.velocity = np.zeros((3,1))

		self.I = np.identity(3)
		self.C = np.identity(3)

		self.counter = 0

		self.new_point = Point()

		self.new_point.x = 0.0
		self.new_point.y = 0.0
		self.new_point.z = 0.0

		self.markers.points.append(self.new_point)

	def callback(self, data):

		self.ang_vel[0] = data.angular_velocity.x
		self.ang_vel[1] = data.angular_velocity.y
		self.ang_vel[2] = data.angular_velocity.z

		ang_vel = np.transpose(np.expand_dims(np.array(self.ang_vel), axis=0))

		self.lin_acc[0] = data.linear_acceleration.x
		self.lin_acc[1] = data.linear_acceleration.y
		self.lin_acc[2] = data.linear_acceleration.z

		body_lin_acc = np.transpose(np.expand_dims(np.array(self.lin_acc), axis=0))

		if (self.counter == 0):

			self.pre_time_stamp = data.header.stamp.secs + (float(data.header.stamp.nsecs) / 1000000000)

		else:

			self.time_stamp = data.header.stamp.secs + (float(data.header.stamp.nsecs) / 1000000000)

			duration = self.time_stamp - self.pre_time_stamp

			sigma = ang_vel * duration

			B = np.array([ [0,                       -self.ang_vel[2]*duration, self.ang_vel[1]*duration],\
						   [self.ang_vel[2]*duration, 0,                        -self.ang_vel[0]*duration],\
						   [-self.ang_vel[1]*duration,self.ang_vel[0]*duration, 0]])


			self.cal_C_matrix(sigma, B)

			glo_lin_acc = np.dot(self.C, body_lin_acc)

			glo_lin_acc[2] = glo_lin_acc[2] - GRAVITY

			self.velocity = self.velocity + (duration * glo_lin_acc)

			print(self.velocity)

			self.position = self.position + (self.velocity * duration)

			self.new_point = Point()
			self.new_point.x = self.position[0]
			self.new_point.y = self.position[1]
			self.new_point.z = self.position[2]

			self.markers.points.append(self.new_point)

			self.pre_time_stamp = self.time_stamp

			self.imu_velocity.linear.x = self.velocity[0] * duration
			self.imu_velocity.linear.y = self.velocity[1] * duration
			self.imu_velocity.linear.z = self.velocity[2] * duration

			self.imu_vel_pub.publish(self.imu_velocity)

			self.publisher.publish(self.markers)

		self.counter+=1

	def cal_C_matrix(self, sigma, B):

		sig_norm = np.linalg.norm(sigma)
		sin_sig_norm = math.sin(sig_norm)
		cos_sig_norm = math.cos(sig_norm)

		weight = self.I + ((sin_sig_norm/sig_norm)*B) + (((1-cos_sig_norm)/(sig_norm*sig_norm))*B*B)

		self.C = np.dot(self.C, weight)


if __name__ == "__main__":

	rospy.init_node("dead_reckoning", anonymous = True)

	ROS_NODE()

	try:
		rospy.spin()

	except KeyboardInterrupt:

		print("shutting down")