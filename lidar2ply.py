#! /usr/bin/env python3

import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os
import shutil
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import pcl_ros
from tqdm import tqdm



def pointcloud2numpy(msg):
	points_list = []
	for data in pc2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True):
		points_list.append([data[0], data[1], data[2]])
	return np.array(points_list)

def pointcloud2_to_array(cloud_msg):
	# Use the point_cloud2 module to convert to a list of tuples
	point_list = list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")))
	return np.array(point_list)

def save_ply(points, filename):
	# Prepare the PLY file header
	header = f'''ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
'''
	# Write header and points to file
	with open(filename, 'w') as f:
		f.write(header)
		np.savetxt(f, points, fmt='%f %f %f')




if __name__ == '__main__':

	bag_msgs = []
	num_messages_to_process = 10

	bag_file = "2024-05-15-19-02-16_2.bag"
	lidar2ply_dir = "lidar2ply"
	
	for path in [lidar2ply_dir]:
		try:
			shutil.rmtree(path)
			print(f"Directory '{path}' has been removed successfully.")
		except OSError as e:
			print(f"Error: {e.strerror}")
	
	for path in [lidar2ply_dir]:
		os.makedirs( path, exist_ok=True)

	with rosbag.Bag(bag_file, "r") as bag:
		total_messages = bag.get_message_count()
		print(f"The bag file contains {total_messages} messages!")

		for topic, msg, t in tqdm(bag.read_messages(), total=total_messages, unit='msgs'):
			# print(f"Parsing {idx}th message!")
			if topic == "/ouster/points":
				bag_msgs.append((topic, msg, t))
			# print(f"idx: {idx}")
			# print(f"topic: {topic}")
			# print(f"type(msg): {type(msg)}")

			# points = pointcloud2_to_array(msg)
			# save_ply(points, f"{rosbag2ply_dir}/pcl_{idx}.ply")

			#write_as_ply(msg, f"{rosbag2ply_dir}/pcl_{idx}.ply")            
			#print(f"t: {t}")
			#print(f"msg: {msg}")  # Uncomment this line to print the message data
			#break  # Exit the loop after processing the first message
			
			#np_arr = pointcloud2numpy(msg)
			#print(f"np_arr.shape: {np_arr.shape}")
			#np.save(f"{rosbag2pcl_dir}/pcl_{idx}.npy", np_arr)s        
		print("Finished saving all the bag_msgs!")
		print("Reversing the bag messages!")
	
	reversed_bag_msgs = list(reversed(bag_msgs))[:num_messages_to_process]
	for idx, (topic, msg, t) in enumerate(tqdm(reversed_bag_msgs, desc="Processing messages", unit="msg")):
	#for idx, (topic, msg, t) in enumerate(reversed(bag_msgs)):
		if idx > 10:
			break
		points = pointcloud2_to_array(msg)
		save_ply(points, f"{lidar2ply_dir}/lidar_{idx}.ply")

	
