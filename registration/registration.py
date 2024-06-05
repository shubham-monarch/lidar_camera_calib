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





def draw_registration_result(source, target, transformation):

	source_a = source.clone()
	source_b = source.clone()

	target_temp = target.clone()

	source_a.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
	source_a.transform(transformation) 

	# source_b.paint_uniform_color([0.0, 0.0,1.0])  # Blue color
	
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
		[
		source_a.to_legacy(),
		#  source_b.to_legacy(),
		 target_temp.to_legacy()
		#  frame1,
		#  frame2
		],
		zoom=0.4459,
		front=[0.9288, -0.2951, -0.2242],
		lookat=[1.6784, 2.0612, 1.4451],
		up=[-0.3402, -0.9189, -0.1996])

def pick_point_callback(vis, event):
    # Get the picked point and its coordinates
    point = vis.get_picked_point(event)
    if point is not None:
        print(f"Picked point: {point}")


def get_pcl_aabb(pcd):
	aabb = pcd.get_axis_aligned_bounding_box()

	# Calculate spans
	horizontal_span = aabb.get_extent()[0]  # span along the x-axis
	vertical_span = aabb.get_extent()[2]    # span along the z-axis (assuming z is vertical)

	# print(f"Horizontal span: {horizontal_span}")
	# print(f"Vertical span: {vertical_span}")
	return [horizontal_span, vertical_span]


if __name__ == "__main__":
	
	lidar_path = "../lidar2ply/lidar_0.ply"
	svo_path = "../svo2ply/svo_0.ply"
	
	print(f"{o3d.__DEVICE_API__}")

	# run_vanilla_icp(lidar_ply, svo_ply)
	print(f"lidar_pcl_span: {get_pcl_aabb(o3d.io.read_point_cloud(lidar_path))}")
	print(f"svo_pcl_span: {get_pcl_aabb(o3d.io.read_point_cloud(svo_path))}")	

		
	s = time.time()

	source = o3d.t.io.read_point_cloud(lidar_path)	
	target = o3d.t.io.read_point_cloud(svo_path)
	
	source_cuda = source.cuda(0)
	target_cuda = target.cuda(0)

	target_cuda.estimate_normals(max_nn=30, radius=0.1)

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

	# registration_ms_icp = treg.multi_scale_icp(source, target, voxel_sizes,
	# 										criteria_list,
	# 										max_correspondence_distances,
	# 										init_source_to_target, estimation,
	# 										callback_after_iteration)


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