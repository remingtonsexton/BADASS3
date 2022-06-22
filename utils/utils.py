import numpy as np

from utils.constants import *

def find_nearest(array, value):
	"""
	This function finds the nearest value in an array and returns the 
	closest value and the corresponding index.
	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx], idx

# TODO: implement generating numbered output diretories
def get_default_outdir(infile):
	return infile.parent.joinpath(DEFAULT_OUTDIR)
