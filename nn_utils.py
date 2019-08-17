import os, random
import numpy as np
from skimage.draw import line


def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.path.dirname(__file__), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """
    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png', 'bmp']
    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def thermal_images():
	"""Return a list of image paths to thermal spectrum images"""
	return list_images(relpath('dataset/thermal'), True)


def visible_images():
	"""Return a list of image paths to visible spectrum images"""
	return list_images(relpath('dataset/color'), True)


def v_to_t(visual_path):
	"""returns path of thermal image corresponding to the given visual image path"""
	return visual_path.replace('color', 'thermal').replace('V', 'L')


def t_to_v(thermal_path):
	"""returns path of visual image corresponding to the given thermal image path"""
	return visual_path.replace('thermal', 'color').replace('L', 'V')

