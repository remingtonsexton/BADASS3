#!/usr/bin/env python
# coding: utf-8

# ## Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS)
# ### Single Spectrum with Options File
# ####  Remington O. Sexton$^{1,2}$, Sara M. Doan$^{1}$, Michael A. Reefe$^{1}$, William Matzko$^{1}$ 
# $^{1}$George Mason University, $^{2}$United States Naval Observatory
# 

# In[1]:
import glob
import time
import natsort
#from IPython.display import clear_output
# import multiprocess as mp
import os
import psutil
import pathlib
import natsort
# Import BADASS here
import badass as badass
import badass_utils as badass_utils

#from IPython.display import display, HTML
#display(HTML("<style>.container { width:90% !important; }</style>"))


# ### BADASS Options

# In[2]:


# A .py file containing options
options_file = "BADASS_options.py"


# ### Run BADASS on a single spectrum
# 
# The following is shows how to fit single SDSS spectra.

# #### Directory Structure

# In[3]:


########################## Directory Structure #################################
spec_dir = 'examples/'#"G:\\Research\MUSE\\J104457\\sdss\\"#'examples/' # folder with spectra in it
#spec_dir = 'G:\\Research\\Reefe_DR8_CLs\\Reefe_DR8_CLs\\f_6\\'
#spec_dir = 'G:\\Research\\Reefe_DR8_CLs\\Old\\Reefe_DR8_BPT_K01_AGN_Controls_Short_argo_in\\f_39\\'
# Get full list of spectrum folders; these will be the working directories
spec_loc = natsort.natsorted( glob.glob(spec_dir+'*') )
################################################################################
print(len(spec_loc))
print(spec_loc)

# #### Choose Spectrum

# In[4]:


nobj = -2# Object in the spec_loc list
file = glob.glob(spec_loc[nobj]+'/*.fits')[0] # Get name of FITS spectra file

print(f"Fitting {file = } with {options_file = }\n")

# #### Run IRSA Dust Query
# To correct for Galactic extinction.  This only needs to be done once so that the data is stored locally.

# In[5]:

#sys
badass_utils.fetch_IRSA_dust(spec_loc[nobj])

 
# #### Run 

# In[6]:


# Call the main function in BADASS
badass.run_BADASS(pathlib.Path(file),
                  options_file = options_file,
                 )
    #


# ###### 
