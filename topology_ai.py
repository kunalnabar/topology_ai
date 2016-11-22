import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

BLACK = 0
WHITE = 255
RESET = 128

def seed_images(image_set):
	"""
	Select a seed pixel for each image
	One black pixel will be selected and one white pixel will be selected for each image

	:param image_set: the set of images
	:returns: a set of tuples, such that each tuple t_i corresponds to the image I_i. And each tuple has the form (black seed, white seed) where the black and white seeds are pixel values
	"""
	seed_set = []
	random_seed = lambda image,target: random.choice(list(zip(*np.where(image == target))))
	for image in image_set:
		bseed = random_seed(image, BLACK)
		wseed = random_seed(image, WHITE)
		seed_set.append((bseed, wseed))
	return seed_set


def flood(image, seed):
	(x,y) = seed
	# copy thresholded image
	im_floodfill = image.copy()
	h,w = image.shape
	# mask must be slightly larger than the original image
	mask = np.zeros((h+2,w+2), np.uint8)

	# have to flip the seed since this takes in (col,row)
	# flood fill the region from the seed point, set it to a new value (not black or white)
	cv2.floodFill(im_floodfill, mask, (y,x), RESET)
	#im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	im_out = (im_floodfill == RESET).astype(np.uint8)
	return im_out

def grow(image, seeds):
	bseed, wseed = seeds
	white_mask = flood(image, wseed)
	black_mask = flood(image, bseed)
	return (black_mask, white_mask)

def fillContours(image_set):
	filled_image_set = [None] * len(image_set)
	for i,image in enumerate(image_set):
				imcopy = image.copy()
				ret,thresh = cv2.threshold(imcopy,20,255,0)
				_, contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				cv2.drawContours(imcopy, contours, 1, 0, -1)
				filled_image_set[i] = imcopy
	return filled_image_set

def solve(image_set):
	n = len(image_set)
	seed_set = seed_images(image_set)
	sim = np.array([0] * n, np.float)
	for i in range(n):
		(bc,wc) = grow(image_set[i], seed_set[i])
		o = (bc|wc).astype(np.uint8).flatten()
		sim[i] = float(sum(o))/len(o)

	unions = np.where(sim == 1)[0]
	s_index = -1

	if len(unions) == 0:
		filled_image_set = fillContours(image_set)
		s_index = solve(filled_image_set)
	elif len(unions) > 1:
		s_index = np.argmin(sim)
	else:
		s_index = np.argmax(sim)
	return s_index