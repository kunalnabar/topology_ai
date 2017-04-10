# Topology Recognition AI

## System Description
This system is a visual based artificial intelligence system to determine the "odd one out" in a specific set of images (images that outline specific types of topology for closure, connectedness, and an element being inside another element). The AI system can be given any of these inputs and will determine the image that does not adhere to the rules of the other images. This project was inspired by the paper *Core Knowledge of Geometry in an Amazonian Indigene Group* by Dehaene et al. A more in depth explanation of the project can be found in *Topology_AI.pdf*. 

This project was developed for CS8395 at Vanderbilt University. All work represented here is my own. 

## Technologies Used
All the code is written in Python.
* **OpenCV** Used for image I/O, performing basic transformations (resizing images proportionally, binary threshold calculation, contour filling) and the flood fill operation.
* **Numpy** Used to store the image and pixel sets over the life cycle of the program and some basic indexing operations.
* **Matplotlib** Used to output the image at the end of the execution if the “show” option is selected
* **Random, sys, os, timer** Built-in Python Libraries that are used for miscellaneous simple operations.



## Running the system:
`python main.py "path"`
    path: the path to the folder containing all image files

The the path is relative to this main.py file.
*Note: The folder must only contain image files, even hidden files will cause a crash.*

To show the image:
`python main.py "path" -sh`

To save the image:
`python main.py "path" -sa "path-to-save"`

The image will output as *answer.png* in the path to save the image.