#! /usr/bin/env python3

import sys
import pyzed.sl as sl
import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm	
import open3d as o3d



if __name__ == '__main__':

	svo_file = "front_2024-05-15-19-04-18.svo"
	svo2ply_dir = "svo2ply"
	num_frames_to_process = 10
	
	for path in [svo2ply_dir]:
		try:
			shutil.rmtree(path)
			print(f"Directory '{path}' has been removed successfully.")
		except OSError as e:
			print(f"Error: {e.strerror}")

	os.makedirs( svo2ply_dir, exist_ok=True)
	
	input_type = sl.InputType()
	input_type.set_from_svo_file(svo_file)
	
	init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
	init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Use ULTRA depth mode
	init_params.coordinate_units = sl.UNIT.METER 

	zed = sl.Camera()
	status = zed.open(init_params)
			
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.enable_fill_mode	= True

	pointcloud = sl.Mat()
	frame_counter = 0
	total_frame_cnt = zed.get_svo_number_of_frames()
	
	print(f"Writing last {num_frames_to_process} frames to {svo2ply_dir}!")

	for frame_number in tqdm(range(total_frame_cnt - 1, total_frame_cnt - 1 - num_frames_to_process, -1), desc=f"Processing last {num_frames_to_process} frames", unit="frame"):
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :				
			
			zed.retrieve_measure(pointcloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
			pointcloud.write( os.path.join(svo2ply_dir, f'svo_{frame_counter}.ply'))
			
			# point_cloud_np = pointcloud.get_data()

			#  # Convert numpy array to lists
			# points_list = point_cloud_np[:, :, :3].reshape(-1, 3).tolist()  # Extract XYZ coordinates
			# colors_list = point_cloud_np[:, :, 3:].reshape(-1, 3).tolist()   # Extract RGB colors

			# # Create Open3D point cloud
			# pcd = o3d.geometry.PointCloud()
			# pcd.points = o3d.utility.Vector3dVector(points_list)
			# pcd.colors = o3d.utility.Vector3dVector(colors_list)

			# # Write the point cloud to a PLY file
			# o3d.io.write_point_cloud(f"{svo2ply_dir}/svo_{frame_counter}.ply", pcd)
	
			frame_counter += 1

	zed.close()