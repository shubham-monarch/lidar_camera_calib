#! /usr/bin/env python3

import numpy as np
import math

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