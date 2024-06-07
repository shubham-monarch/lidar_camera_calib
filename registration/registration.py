#! /usr/bin/env python3
import open3d as o3d
# from open3d.pipelines import registration as treg
import numpy as np
import copy
import time
import math
import fast_global_registration 
import utils

if o3d.__DEVICE_API__ == 'cuda':
	import open3d.cuda.pybind.t.pipelines.registration as treg
else:
	import open3d.cpu.pybind.t.pipelines.registration as treg

# TO-DO: 
# try plane fitting to tractor hood pcd
# play with tractor hood pcd + one crop row on each side


# robust icp -> outlier elimination
# downsample target pcl
# trying pre-processing for zed pointcloud
# use pcd.segment_plane

# try manual initial alignment / global registration methods



def draw_registration_result(source, target, transformation):

	source_temp = source.clone()
	target_temp = target.clone()

	source_temp.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
	source_temp.transform(transformation) 

	target_temp.paint_uniform_color([0.5, 1.0, 0.5])  # Green color

	# source_b.paint_uniform_color([0.0, 0.0,1.0])  # Blue color
	
	# source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])
	# target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])

	frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])	
	# frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[60, 0, 0])
	
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
		[
		source_temp.to_legacy(),
		#  source_b.to_legacy(),
		target_temp.to_legacy(),
		 frame1
		#  frame2
		],
		zoom=0.4459,
		front=[0.9288, -0.2951, -0.2242],
		lookat=[1.6784, 2.0612, 1.4451],
		# lookat=[0, 0, 0],
		up=[-0.3402, -0.9189, -0.1996])

def prepare_pcd(pcd, min_bound, max_bound):
	pcd_temp = pcd.clone()

	print(f"[BEFORE CROPPING] Number of points in pcd: {np.asarray(pcd_temp.to_legacy().points).shape[0]}")
	
	bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
	pcd_temp = pcd_temp.crop(bbox) 
	
	print(f"[AFTER CROPPING] Number of points in pcd: {np.asarray(pcd_temp.to_legacy().points).shape[0]}")
	
	return pcd_temp		

def run_multiview_icp(source, target):
	s = time.time()

	print("[AFTER DOWNSAMPLING] Number of points in source: ", np.asarray(source.to_legacy().points).shape[0])
	print("[AFTER DOWNSAMPLING] Number of points in target: ", np.asarray(target.to_legacy().points).shape[0])

	source_cuda = source.cuda(0)
	target_cuda = target.cuda(0)
	
	target_cuda.estimate_normals(max_nn=30, radius=0.1)
	
	# voxel_sizes = o3d.utility.DoubleVector([1, 0.09, 0.089])
	voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])

	criteria_list = [
    treg.ICPConvergenceCriteria(relative_fitness=0.0001,
                                relative_rmse=0.0001,
                                max_iteration=20),
    treg.ICPConvergenceCriteria(0.00001, 0.00001, 15),
    treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
]

	# `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
	max_correspondence_distances = o3d.utility.DoubleVector([20, 20, 20])

	init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

	# estimation = treg.TransformationEstimationPointToPlane()
	# estimation = treg.TransformationEstimationPointToPoint()

	mu, sigma = 0, 1.0 # mean and standard deviation
	estimation = treg.TransformationEstimationPointToPlane(
	treg.robust_kernel.RobustKernel(
		treg.robust_kernel.RobustKernelMethod.GMLoss))

	# estimation = treg.TransformationEstimationPointToPoint()

	# Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
	callback_after_iteration = lambda loss_log_map : print("Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
		loss_log_map["iteration_index"].item(),
		loss_log_map["scale_index"].item(),
		loss_log_map["scale_iteration_index"].item(),
		loss_log_map["fitness"].item(),
		loss_log_map["inlier_rmse"].item()))

	registration_ms_icp = treg.multi_scale_icp(source_cuda, target_cuda,
										   voxel_sizes, criteria_list,
										   max_correspondence_distances,
										   init_source_to_target, estimation,
										   callback_after_iteration)

	ms_icp_time = time.time() - s
	print("Time taken by Multi-Scale ICP: ", ms_icp_time)
	print("Inlier Fitness: ", registration_ms_icp.fitness)
	print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)

	draw_registration_result(source, target, registration_ms_icp.transformation) 	


	
def numpy_to_pointcloud(points):
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	return pcd

if __name__ == "__main__":
	
	lidar_path = "../lidar2ply/lidar_0.ply"
	svo_path = "../svo2ply/svo_0.ply"
	
	print(f"{o3d.__DEVICE_API__}")

	frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])	

	s = time.time()

	source = o3d.t.io.read_point_cloud(lidar_path)	
	
	target = o3d.t.io.read_point_cloud(svo_path)
	
	# target = target.voxel_down_sample(voxel_size=0.08)
	
	# tractor hood only
	# source = prepare_pcd(source, [0.0, -0.5, -1.0], [1.2 , 0.6, 1])
	# target = prepare_pcd(target, [-0.6, 0.0, 0.0], [0.6, 1.0, 2.0])
	
	# hood + vine row
	# source = prepare_pcd(source, [0.0, -5.0, -1.0], [10.0 , 5.0, 4.0])
	# target = prepare_pcd(target, [-4.0, -10.0, -10.0], [4.0, 10.0, 10.0])
	
	
	# target = target.voxel_down_sample(voxel_size=0.02)
	# target.paint_uniform_color([0.0, 1.0, 0.0])	

	
	# o3d.visualization.draw_geometries([target.to_legacy(), frame1],
	# 	zoom=0.4459,
	# 	front=[0.9288, -0.2951, -0.2242],
	# 	lookat=[1.6784, 2.0612, 1.4451],
	# 	up=[-0.3402, -0.9189, -0.1996])
	
	# o3d.visualization.draw_geometries([ target.to_legacy(), frame1])	
	# source = source.voxel_down_sample(voxel_size=0.02)
	# o3d.visualization.draw_geometries([ source.to_legacy(), frame1])	
	# o3d.visualization.draw_geometries([ source.to_legacy(), target.to_legacy(), frame1])	

	# target = target.voxel_down_sample(voxel_size=0.03)
	run_multiview_icp(source, target)

	# source plane segmentation
	# source_plane, source_inliers = source.segment_plane(distance_threshold=0.1, ransac_n=300, num_iterations=1000)
	# source_filtered = source.select_by_index(source_inliers)	

	# source.paint_uniform_color([0.5, 1.0, 0.5])
	# source_filtered.paint_uniform_color([1.0, 1.0, 0.0])
	# o3d.visualization.draw_geometries([source.to_legacy(), source_filtered.to_legacy()], window_name="Point Cloud Filtering")	

	# target plane segmentation
	# target_plane, target_inliers = target.segment_plane(distance_threshold=0.01, ransac_n=30, num_iterations=1000)
	# target_filtered = target.select_by_index(target_inliers)

	# target.paint_uniform_color([0.5, 1.0, 0.5])
	# target_filtered.paint_uniform_color([1.0, 1.0, 0.0])
	# o3d.visualization.draw_geometries([target.to_legacy(), target_filtered.to_legacy()], window_name="Point Cloud Filtering")

	
	
	
	