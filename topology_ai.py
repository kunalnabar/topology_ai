# Kunal Nabar

import cv2
import numpy as np
import random

BLACK = 0
WHITE = 255
RESET = 128
IDEAL_H = 250
IDEAL_W = 300

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
	"""
	Grow the image from the seed location. Return a mask of the grown image.

	:param image: the image that you are performing the flood on
	:param seed: the seed from the image that you are flooding
	:returns: an image of 0/1's that corresponds to the location that was flooded
	"""
	(x,y) = seed
	# copy thresholded image
	im_floodfill = image.copy()
	h,w = image.shape
	# mask must be slightly larger than the original image
	mask = np.zeros((h+2,w+2), np.uint8)

	# have to flip the seed since this takes in (col,row)
	# flood fill the region from the seed point, set it to a new value (not black or white)
	cv2.floodFill(im_floodfill, mask, (y,x), RESET)
	
	# find only the spots that were flood filled	
	im_out = (im_floodfill == RESET).astype(np.uint8)
	return im_out

def grow(image, seeds):
	"""
	Flood both the black and white regions of the image

	:param image: the image that is to be seeded
	:param seeds: the black and white seed, respectively
	"""
	bseed, wseed = seeds
	# get the flooded black and white regions, return them as image masks
	white_mask = flood(image, wseed)
	black_mask = flood(image, bseed)
	return (black_mask, white_mask)

def fillContours(image_set):
	"""
	Fill the contours of the image with black
	"""
	filled_image_set = [None] * len(image_set)
	for i,image in enumerate(image_set):
				imcopy = image.copy()
				ret,thresh = cv2.threshold(imcopy,20,255,0)
				_, contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				cv2.drawContours(imcopy, contours, 1, 0, -1)
				filled_image_set[i] = imcopy
	return filled_image_set

def determine_shrink(h,w):
	"""
	Determine how much to shrink the image in order to run the AI faster.

	:param h: the height of the image
	:param w: the width of the image
	:returns: the shrink factor of the image
	"""
	if h < IDEAL_H or w < IDEAL_W:
		return 1
	sw = w // IDEAL_W
	sh = h // IDEAL_H
	return (sw + sh) // 2

def clean(IS):
	"""
	Initialize the test set to the specific method and test set
	
	Found by reading the files found at the path './$(method)/$(sno)/'

	Note: this method is not part of the topological ai

	:param method: the type of topological analysis method
	:param sno: the set name of the method in question

	:returns: the set of images specificed by the method and the set number
	"""
	image_set = []
	for im in IS:
		# shrink the image into a smaller one for faster processing
		shrink = determine_shrink(*im.shape)
		nx,ny = (im.shape[0]//shrink, im.shape[1]//shrink)
		im_shrink = cv2.resize(im, dsize=(nx,ny))

		# Use Otsu's method to find the threshold for the binary image
		# https://en.wikipedia.org/wiki/Otsu's_method
		_, im_bw = cv2.threshold(im_shrink, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		
		image_set.append(im_bw)
	return image_set

def solve(IS):
	# clean the image set
	image_set = clean(IS)
	n = len(image_set)

	# find the seed of the image
	seed_set = seed_images(image_set)
	sim = np.array([0] * n, np.float)

	# grow each region and find the percentage of the image covered
	for i in range(n):
		(bc,wc) = grow(image_set[i], seed_set[i])
		o = (bc|wc).astype(np.uint8).flatten()
		sim[i] = float(sum(o))/len(o)

	unions = np.where(sim == 1)[0]
	s_index = -1

	# this will occur in the case of inside topology.
	# refill the contours and try again
	if len(unions) == 0:
		filled_image_set = fillContours(image_set)
		s_index = solve(filled_image_set)
	elif len(unions) > 1:
		# if more than one index has a 1, then you want the one that didn't get filled. 
		s_index = np.argmin(sim)
	else:
		# otherwise, take the max argument
		s_index = np.argmax(sim)
	return s_index