#!/usr/bin/env python

import numpy as np
import pandas as pd
# Plotting Libraries
import matplotlib.pyplot as plt
plt.style.use('dark_background') # For cool tron-style dark plots
plt.rcParams['text.usetex'] = True
# Basic Libraries
import sys
import os
import natsort
import glob
import copy
# Astropy
from astropy.io import fits
import scipy.optimize as op
from scipy.ndimage import generic_filter


