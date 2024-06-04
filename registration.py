#! /usr/bin/env python3
import open3d as o3d
import numpy as np

# TO-DO: 
# - Try colored registration
# - Try point to plane registration method
# - remove outliers

def draw_registration_result(source, target, transformation):
    print("Inside draw_registration_result")
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp.to_legacy(),
         target_temp.to_legacy()],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])

if __name__ == "__main__":
    
    lidar_ply = "lidar2ply/lidar_0.ply"
    svo_ply = "svo2ply/svo_0.ply"
    
    lidar_pcd = o3d.io.read_point_cloud(lidar_ply, remove_nan_points=True, remove_infinite_points=True)
    svo_pcd = o3d.io.read_point_cloud(svo_ply, remove_nan_points=True, remove_infinite_points=True) 
    # o3d.visualization.draw_geometries([lidar_pcd],
    #                               zoom=0.3412,
    #                               front=[0.4257, -0.2125, -0.8795],
    #                               lookat=[2.6172, 2.0475, 1.532],
    #                               up=[-0.0694, -0.9768, 0.2024])

    o3d.visualization.draw_geometries([svo_pcd])