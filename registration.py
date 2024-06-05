#! /usr/bin/env python3
import open3d as o3d
# from open3d.pipelines import registration as treg
import numpy as np
import copy
import time

if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg

# TO-DO: 
# - multilevel icp , robust icp
# - Try point to plane registration method
# - remove outliers

def draw_registration_result(source, target, transformation):
	print("Inside draw_registration_result")
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)

	source_temp.transform(transformation)

	# This is patched version for tutorial rendering.
	# Use `draw` function for you application.
	o3d.visualization.draw_geometries(
		[source_temp,target_temp],
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

def run_vanilla_icp(lidar_ply , svo_ply):
	
	source = o3d.t.io.read_point_cloud(lidar_ply)
	target = o3d.t.io.read_point_cloud(svo_ply)
	
	# Example callback_after_iteration lambda function:
	callback_after_iteration = lambda updated_result_dict : print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    updated_result_dict["iteration_index"].item(),
    updated_result_dict["fitness"].item(), 
    updated_result_dict["inlier_rmse"].item()))


	# Search distance for Nearest Neighbour Search [Hybrid-Search is used].
	max_correspondence_distance = 0.07

	# Initial alignment or source to target transform.
	init_source_to_target = np.asarray([[0.862, 0.011, -0.507, 0.5],
										[-0.139, 0.967, -0.215, 0.7],
										[0.487, 0.255, 0.835, -1.4],
										[0.0, 0.0, 0.0, 1.0]])

	# Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
	estimation = treg.TransformationEstimationPointToPlane()

	# Convergence-Criteria for Vanilla ICP
	criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
										relative_rmse=0.000001,
										max_iteration=50)
	# Down-sampling voxel-size.
	voxel_size = 0.00025

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
	
	lidar_ply = "lidar2ply/lidar_0.ply"
	svo_ply = "svo2ply/svo_0.ply"
	
	lidar_pcd = o3d.io.read_point_cloud(lidar_ply, remove_nan_points=True, remove_infinite_points=True)
	svo_pcd = o3d.io.read_point_cloud(svo_ply, remove_nan_points=True, remove_infinite_points=True) 
	
	# lidar_pcd = o3d.io.read_point_cloud(lidar_ply)
	# svo_pcd = o3d.io.read_point_cloud(svo_ply) 

	# source = copy.copy(lidar_pcd)
	# target = copy.copy(svo_pcd)

	print(f"{o3d.__DEVICE_API__}")

	# run_vanilla_icp(lidar_ply, svo_ply)
	print(f"lidar_pcl_span: {get_pcl_aabb(lidar_pcd)}")
	print(f"svo_pcl_span: {get_pcl_aabb(svo_pcd)}")