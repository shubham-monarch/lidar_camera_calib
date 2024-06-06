#! /usr/bin/env python3

import numpy as np
import math
import time
import open3d as o3d

if o3d.__DEVICE_API__ == 'cuda':
	import open3d.cuda.pybind.t.pipelines.registration as treg
else:
	import open3d.cpu.pybind.t.pipelines.registration as treg


def create_transformation_matrix(R, t):
    """
    Create a 4x4 transformation matrix from a 3x3 rotation matrix and 3x1 translation vector.
    """
    T = np.eye(4)  # Create a 4x4 identity matrix
    T[:3, :3] = R  # Set the rotation matrix part
    T[:3, 3] = t.squeeze()  # Set the translation vector part

    return T


def arun(A, B):
    """Solve 3D registration using Arun's method: B = RA + t"""
    N = A.shape[1]  # Number of points
    assert B.shape[1] == N

    # Calculate centroids
    A_centroid = np.mean(A, axis=0).reshape(1, 3)
    B_centroid = np.mean(B, axis=0).reshape(1, 3)

    # Calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid

    # Rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[i, :]  # Extract the ith row (shape (3,))
        bi = B_prime[i, :]  # Extract the ith row (shape (3,))
        H += np.outer(ai, bi)

    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

    # Translation estimation
    t = B_centroid.T - R @ A_centroid.T

    return R, t



# def arun(A, B):
# 	"""Solve 3D registration using Arun's method: B = RA + t
# 	"""
# 	print(f"Inside arun!")
	
# 	N = A.shape[1]
# 	assert B.shape[1] == N

# 	print(f"A.shape: {A.shape} B.shape: {B.shape}")

# 	# calculate centroids
# 	# A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
# 	# B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))

# 	A_centroid = np.reshape(1/N * (np.sum(A, axis=0)), (1,3))
# 	B_centroid = np.reshape(1/N * (np.sum(B, axis=0)), (1,3))


# 	# calculate the vectors from centroids
# 	A_prime = A - A_centroid
# 	B_prime = B - B_centroid

# 	print(f"A_prime and B_prime have been calculated!")

# 	# rotation estimation
# 	H = np.zeros([3, 3])
# 	for i in range(N):
# 		ai = A_prime[i, :]
# 		bi = B_prime[i, :]
# 		H = H + np.outer(ai, bi)
# 	U, S, V_transpose = np.linalg.svd(H)
# 	V = np.transpose(V_transpose)
# 	U_transpose = np.transpose(U)
# 	R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

# 	# translation estimation
# 	t = B_centroid - R @ A_centroid

# 	return R, t


def get_pcl_aabb(pcd):
	aabb = pcd.get_axis_aligned_bounding_box()

	# Calculate spans
	horizontal_span = aabb.get_extent()[0]  # span along the x-axis
	vertical_span = aabb.get_extent()[2]    # span along the z-axis (assuming z is vertical)

	# print(f"Horizontal span: {horizontal_span}")
	# print(f"Vertical span: {vertical_span}")
	return [horizontal_span, vertical_span]



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