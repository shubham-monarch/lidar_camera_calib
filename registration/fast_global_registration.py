#! /usr/bin/env python3
import open3d as o3d
import copy
import time

def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	source_temp.paint_uniform_color([1, 0.706, 0])
	target_temp.paint_uniform_color([0, 0.651, 0.929])
	source_temp.transform(transformation)
	o3d.visualization.draw_geometries([source_temp, target_temp],
									  zoom=0.4559,
									  front=[0.6452, -0.3036, -0.7011],
									  lookat=[1.9892, 2.0208, 1.8945],
									  up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
	print(":: Downsample with a voxel size %.3f." % voxel_size)
	pcd_down = pcd.voxel_down_sample(voxel_size)

	radius_normal = voxel_size * 2
	print(":: Estimate normal with search radius %.3f." % radius_normal)
	pcd_down.estimate_normals(
		o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

	radius_feature = voxel_size * 5
	print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
	pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
		pcd_down,
		o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
	return pcd_down, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
									 target_fpfh, voxel_size):
	distance_threshold = voxel_size * 0.5
	print(":: Apply fast global registration with distance threshold %.3f" \
			% distance_threshold)
	result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
		source_down, target_down, source_fpfh, target_fpfh,
		o3d.pipelines.registration.FastGlobalRegistrationOption(
			maximum_correspondence_distance=distance_threshold))
	return result


def main(lidar_path, svo_path):
	
	voxel_size = 1  # means 5cm for the dataset
	source_down, source_fpfh = preprocess_point_cloud(o3d.io.read_point_cloud(lidar_path), voxel_size)
	target_down, target_fpfh = preprocess_point_cloud(o3d.io.read_point_cloud(svo_path), voxel_size)

	print(f"type(source_down): {type(source_down)}")	

	start = time.time()
	result_fast = execute_fast_global_registration(source_down, target_down,
												source_fpfh, target_fpfh,
												voxel_size)
	print("Fast global registration took %.3f sec.\n" % (time.time() - start))
	print(result_fast)
	# draw_registration_result(source_down, target_down, result_fast.transformation)
	draw_registration_result(source_down, target_down, result_fast.transformation)

	print(f"type(result_fast.transformation): {type(result_fast.transformation)}")
	print(f"result_fast.transformation: \n{result_fast.transformation}")	

	return result_fast.transformation


