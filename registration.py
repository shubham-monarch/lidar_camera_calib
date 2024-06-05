#! /usr/bin/env python3
import open3d as o3d
# from open3d.pipelines import registration as treg
import numpy as np
import copy
import time
import math

if o3d.__DEVICE_API__ == 'cuda':
	import open3d.cuda.pybind.t.pipelines.registration as treg
else:
	import open3d.cpu.pybind.t.pipelines.registration as treg

# TO-DO: 
# - multilevel icp , robust icp
# - Try point to plane registration method
# - remove outliers

def rotation_matrix_y(theta):
    """
    Returns the 4x4 transformation matrix for rotating around the y-axis
    by the given angle (in degrees).
    """
    theta = math.radians(theta)  # Convert to radians
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_theta, 0, -sin_theta, 0],
        [0, 1, 0, 0],
        [sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ])
    
    return rotation_matrix

def rotation_matrix_x(theta):
    """
    Returns the 4x4 transformation matrix for rotating around the x-axis
    by the given angle (in degrees).
    """
    theta = math.radians(theta)  # Convert to radians
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, cos_theta, sin_theta, 0],
        [0, -sin_theta, cos_theta, 0],
        [0, 0, 0, 1]
    ])
    
    return rotation_matrix

def rotation_matrix_z(theta):
	"""
	Returns the 4x4 transformation matrix for rotating around the z-axis
	by the given angle (in degrees).
	"""
	theta = math.radians(theta)  # Convert to radians
	
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	
	rotation_matrix = np.array([
		[cos_theta, sin_theta, 0, 0],
		[-sin_theta, cos_theta, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	])
	
	return rotation_matrix


def draw_registration_result(source, target, transformation):
	source_a = source.clone()
	source_b = source.clone()

	target_temp = target.clone()

	source_a.paint_uniform_color([255, 0, 0])  # Red color
	source_a.transform(transformation)

	source_b.paint_uniform_color([0, 0, 255])  # Blue color
	
	# source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])
	# target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])

	frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
	frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[60, 0, 0])
    
	# tf1 = np.asarray([[1.0, 0.0 , 0,  0],
	# 				[0 , 1.0, 0, 0],
	# 				[0 , 0, 1.0, 0],
	# 				[0.0, 0.0, 0.0, 1.0]])

	# tf2 = np.asarray([[1.0, 0.0 , 0,  0],
	# 				[0 , 1.0, 0, 60],
	# 				[0 , 0, 1.0, 0],
	# 				[0.0, 0.0, 0.0, 1.0]])

	# frame1.transform(tf1)
	# frame2.transform(tf2)	

	# This is patched version for tutorial rendering.
	# Use `draw` function for you application.
	o3d.visualization.draw_geometries(
		[source_a.to_legacy(),
		 source_b.to_legacy(),
		 # target_temp.to_legacy(),
		 frame1,
		 frame2],
		zoom=0.4459,
		front=[0.9288, -0.2951, -0.2242],
		lookat=[1.6784, 2.0612, 1.4451],
		up=[-0.3402, -0.9189, -0.1996])


def get_pcl_aabb(pcd):
	aabb = pcd.get_axis_aligned_bounding_box()

	# Calculate spans
	horizontal_span = aabb.get_extent()[0]  # span along the x-axis
	vertical_span = aabb.get_extent()[2]    # span along the z-axis (assuming z is vertical)

	# print(f"Horizontal span: {horizontal_span}")
	# print(f"Vertical span: {vertical_span}")
	return [horizontal_span, vertical_span]

def run_vanilla_icp(lidar_path  , svo_path):
	
	source = o3d.t.io.read_point_cloud(lidar_path)
	target = o3d.t.io.read_point_cloud(svo_path)
	
	# Example callback_after_iteration lambda function:
	callback_after_iteration = lambda updated_result_dict : print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
	updated_result_dict["iteration_index"].item(),
	updated_result_dict["fitness"].item(), 
	updated_result_dict["inlier_rmse"].item()))


	# Search distance for Nearest Neighbour Search [Hybrid-Search is used].
	max_correspondence_distance = 7

	# Initial alignment or source to target transform.
	init_source_to_target = np.asarray([[0.862, 0.011, -0.507, 0.5],
										[-0.139, 0.967, -0.215, 0.7],
										[0.487, 0.255, 0.835, -1.4],
										[0.0, 0.0, 0.0, 1.0]])

	# Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
	# estimation = treg.TransformationEstimationPointToPlane()
	estimation = treg.TransformationEstimationPointToPoint()

	# Convergence-Criteria for Vanilla ICP
	criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
										relative_rmse=0.000001,
										max_iteration=50)
	# Down-sampling voxel-size.
	voxel_size = 0.025

	# Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
	save_loss_log = True
	
	s = time.time()

	
	registration_icp = treg.icp(source, target, max_correspondence_distance,
								init_source_to_target, estimation, criteria,
								voxel_size, callback_after_iteration)

	icp_time = time.time() - s
	print("Time taken by ICP: ", icp_time)
	print("Inlier Fitness: ", registration_icp.fitness)
	print("Inlier RMSE: ", registration_icp.inlier_rmse)

	draw_registration_result(source, target, registration_icp.transformation)


if __name__ == "__main__":
	
	lidar_path = "lidar2ply/lidar_0.ply"
	svo_path = "svo2ply/svo_0.ply"
	
	# lidar_pcd = o3d.io.read_point_cloud(lidar_ply, remove_nan_points=True, remove_infinite_points=True)
	# svo_pcd = o3d.io.read_point_cloud(svo_ply, remove_nan_points=True, remove_infinite_points=True) 
	
	# lidar_pcd = o3d.io.read_point_cloud(lidar_ply)
	# svo_pcd = o3d.io.read_point_cloud(svo_ply) 

	# source = copy.copy(lidar_pcd)
	# target = copy.copy(svo_pcd)

	print(f"{o3d.__DEVICE_API__}")

	# run_vanilla_icp(lidar_ply, svo_ply)
	print(f"lidar_pcl_span: {get_pcl_aabb(o3d.io.read_point_cloud(lidar_path))}")
	print(f"svo_pcl_span: {get_pcl_aabb(o3d.io.read_point_cloud(svo_path))}")	

	# transformation = np.array()
	final_transformation = np.eye(4)
	print(f"Initial Transformation: {final_transformation}")

	
	# rotate 90 dgree around x-axis
	# rot_1 = np.array([[1, 0, 0, 0],
	# 				  [0, 0, -1, 0],
	# 				  [0, 1, 0, 0],
	# 				  [0, 0, 0, 1]])

	final_transformation = final_transformation @ rotation_matrix_x(90)
	# final_transformation = final_transformation @ rotation_matrix_y(90)
	# final_transformation = final_transformation @ rotation_matrix_z(90)
	np.set_printoptions(precision=3, suppress=True)

	print(f"Final Transformation: {final_transformation}")

	draw_registration_result(o3d.t.io.read_point_cloud(lidar_path), o3d.t.io.read_point_cloud(svo_path), final_transformation)
	# run_vanilla_icp(lidar_path, svo_path)