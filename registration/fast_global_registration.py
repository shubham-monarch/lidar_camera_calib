#! /usr/bin/env python3
import open3d as o3d

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
