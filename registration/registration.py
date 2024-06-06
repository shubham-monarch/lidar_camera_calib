#! /usr/bin/env python3
import open3d as o3d
# from open3d.pipelines import registration as treg
import numpy as np
import copy
import time
import math
import fast_global_registration 


if o3d.__DEVICE_API__ == 'cuda':
	import open3d.cuda.pybind.t.pipelines.registration as treg
else:
	import open3d.cpu.pybind.t.pipelines.registration as treg

# TO-DO: 
# robust icp -> outlier elimination
# downsample target pcl
# try croppping
# try arun's method
# try manual initial alignment / global registration methods

def draw_registration_result(source, target, transformation):

	source_temp = source.clone()
	target_temp = target.clone()

	source_temp.paint_uniform_color([0.0, 0.0, 0.0])  # Red color
	source_temp.transform(transformation) 

	# target_temp.paint_uniform_color([0.0, 1.0, 0.0])  # Green color

	# source_b.paint_uniform_color([0.0, 0.0,1.0])  # Blue color
	
	# source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])
	# target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])

	frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
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
		up=[-0.3402, -0.9189, -0.1996])

def prepare_pcd(pcd, min_bound, max_bound):
	pcd_temp = pcd.clone()

	print(f"[BEFORE CROPPING] Number of points in pcd: {np.asarray(pcd_temp.to_legacy().points).shape[0]}")
	
	bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
	pcd_temp = pcd_temp.crop(bbox) 
	
	print(f"[AFTER CROPPING] Number of points in pcd: {np.asarray(pcd_temp.to_legacy().points).shape[0]}")
	
	return pcd_temp		


if __name__ == "__main__":
	
	lidar_path = "../lidar2ply/lidar_0.ply"
	svo_path = "../svo2ply/svo_0.ply"
	
	print(f"{o3d.__DEVICE_API__}")

	# run_vanilla_icp(lidar_ply, svo_ply)
	# print(f"lidar_pcl_span: {get_pcl_aabb(o3d.io.read_point_cloud(lidar_path))}")
	# print(f"svo_pcl_span: {get_pcl_aabb(o3d.io.read_point_cloud(svo_path))}")	

	frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])	

	s = time.time()

	source = o3d.t.io.read_point_cloud(lidar_path)	
	target = o3d.t.io.read_point_cloud(svo_path)	

	source = prepare_pcd(source, [-5.0, -0.6, -50.0], [2.0 , 0.6, 10.0])
	target = prepare_pcd(target, [-0.7, -10, -50.0], [0.8, 0.5, 2.0])

	# source_num_points = np.asarray(source.to_legacy().points).shape[0]
	# # target_num_points = np.asarray(target.to_legacy().points).shape[0]
	# print(f"[BEFORE CROPPING] Number of points in source: {source_num_points}")
	# # print(f"[BEFORE CROPPING] Number of points in target: {target_num_points}")
		
	# source_min_bound = [-5.0, -0.6, -50.0]
	# source_max_bound = [2.0 , 0.6, 10.0]
	# source_bbox = o3d.t.geometry.AxisAlignedBoundingBox(source_min_bound, source_max_bound)

	# # target_min_bound = [-0.7, -10, -50.0]
	# # target_max_bound = [0.8, 0.5, 2.0]
	# # target_bbox = o3d.t.geometry.AxisAlignedBoundingBox(target_min_bound, target_max_bound)

	# source = source.crop(source_bbox) 
	# # target = target.crop(target_bbox) 
	

	# source_num_points = np.asarray(source.to_legacy().points).shape[0]
	# # target_num_points = np.asarray(target.to_legacy().points).shape[0]
	# print(f"[AFTER CROPPING] Number of points in source: {source_num_points}")
	# # print(f"[AFTER CROPPING] Number of points in source: {target_num_points}")

	o3d.visualization.draw_geometries([source.to_legacy(),target.to_legacy(), frame1])	
	
	
	
	exit(0)





	source_cuda = source.cuda(0)
	target_cuda = target.cuda(0)

	


	
	
	# Downsampling the svo pcl
	voxel_size = 0.1  # Adjust the voxel size as needed
	downsampled_target_cuda = target_cuda.voxel_down_sample(voxel_size)
	o3d.visualization.draw_geometries([downsampled_target_cuda.to_legacy()])




	exit(0)
	target_cuda.estimate_normals(max_nn=30, radius=0.1)
	downsampled_target_cuda.estimate_normals(max_nn=30, radius=0.1)

	voxel_sizes = o3d.utility.DoubleVector([0.01, 0.0009, 0.00089])

	criteria_list = [
		treg.ICPConvergenceCriteria(relative_fitness=0.001,
									relative_rmse=0.0001,
									max_iteration=100),
		treg.ICPConvergenceCriteria(0.0001, 0.0001, 20),
		treg.ICPConvergenceCriteria(0.001, 0.001, 5)
	]

	# `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
	max_correspondence_distances = o3d.utility.DoubleVector([100, 100, 100])

	init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

	estimation = treg.TransformationEstimationPointToPlane()

	# mu, sigma = 0, 1.0
	# estimation = treg.TransformationEstimationPointToPlane(
    # treg.robust_kernel.RobustKernel(
    #     treg.robust_kernel.RobustKernelMethod.CauchyLoss, sigma))

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