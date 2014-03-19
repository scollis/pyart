#! /usr/bin/env python
"""Simple functions for calculating texture and using it to 
threshold reflectivity and other data"""
from pyart.correct.phase_proc import smooth_and_trim
from numpy import abs

def texture(field, new_standard_name, new_long_name):
	textr = np.zeros_like(field)
	for i in range(textr.shape[0]):
		this_ray_of_data = field['data'][i,:]
		signal = smooth_and_trim(this_ray_of_data)
		noise = smooth_and_trim(np.sqrt((this_ray_of_data - signal) ** 2))
		this_ray_texture = abs(signal) / noise
		textr[i,:] = 

	field_dict = {'data' : SNR_VR,
				  'units' : 'Unitless',
				  'long_name' : new_long_name,
				  'standard_name' : new_standard_name}
	return field_dict
