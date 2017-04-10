# Kunal Nabar

import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from topology_ai import solve

# constants
def init(p):
	"""
	Initialize the test set to the specific method and test set
	
	Found by reading the files found at the path './$(method)/$(sno)/'

	Note: this method is not part of the topological ai

	:param method: the type of topological analysis method
	:param sno: the set name of the method in question

	:returns: the set of images specificed by the method and the set number
	"""
	# check assertions

	# walk through all the files in the test set
	image_set = []
	path = os.path.join(*(os.curdir, p))
	_,_,files = next(os.walk(path))
	test_set = [os.path.join(path,f) for f in files]
	IS = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in test_set]
	return files,IS

# Read in the two parameters from the image set

if not sys.argv[1:]:
	print("""Running the system:
python main.py "path"
	path: the path to the folder containing all image files
The folder must only contain image files, even hidden files will cause a crash.
The the path is relative to this main.py file.""")
	sys.exit()

files,IS = init(*sys.argv[1:2])
show_image = False
save_image = False
save_path = './'
if len(sys.argv) == 3 and sys.argv[2] == '-sh':
		show_image = True
if len(sys.argv) == 4 and sys.argv[2] == '-sa':
		save_image = True
		save_path = sys.argv[3]

start = time.time()
s_index = solve(IS)
dur = time.time()-start

print("The different image is %s" % (files[s_index]))
print("It took %.3f seconds." % (dur))

if show_image:
	imout = cv2.cvtColor(IS[s_index],cv2.COLOR_GRAY2BGRA)
	plt.imshow(imout)
	plt.show()
if save_image:
	imout = cv2.cvtColor(IS[s_index], cv2.COLOR_GRAY2BGRA)
	cv2.imwrite(os.path.join(save_path, 'answer.png'), imout)
	print('Answer in: %s' % (os.path.join(save_path, 'answer.png')))

