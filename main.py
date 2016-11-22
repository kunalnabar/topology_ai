import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from topology_ai import solve

# constants
MTYPES = ['inside','closure','connectedness','holes']
KSIZE = 1
IDEAL_H = 250
IDEAL_W = 300

def determine_shrink(h,w):
	if h < IDEAL_H or w < IDEAL_W:
		return 1
	sw = w // IDEAL_W
	sh = h // IDEAL_H
	return (sw + sh) // 2


def init(method,sno):
	"""
	Initialize the test set to the specific method and test set
	
	Found by reading the files found at the path './$(method)/$(sno)/'

	Note: this method is not part of the topological ai

	:param method: the type of topological analysis method
	:param sno: the set name of the method in question

	:returns: the set of images specificed by the method and the set number
	"""
	# check assertions
	assert method in MTYPES

	# walk through all the files in the test set
	image_set = []
	path = os.path.join(*(os.curdir, method, sno))
	_,_,files = next(os.walk(path))
	test_set = [os.path.join(path,f) for f in files]
	for f in test_set:
		im = cv2.imread(f,cv2.IMREAD_GRAYSCALE)

		# shrink the image into a smaller one for faster processing
		shrink = determine_shrink(*im.shape)

		nx,ny = (im.shape[0]//shrink, im.shape[1]//shrink)
		im_shrink = cv2.resize(im, dsize=(nx,ny))

		# blur the image to make thresholding easier
		#im_blur = cv2.medianBlur(im_shrink, KSIZE)

		# Use Otsu's method to find the threshold for the binary image
		# https://en.wikipedia.org/wiki/Otsu's_method
		_, im_bw = cv2.threshold(im_shrink, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		
		image_set.append(im_bw)
	return image_set


image_set = init('closure','set1')

s_index = solve(image_set)

plt.imshow(image_set[s_index])
plt.show()
