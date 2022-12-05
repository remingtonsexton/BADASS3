#!/usr/bin/env python
# coding: utf-8

# ## Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS)
# ### MUSE IFU Cube
# 
# ####  Remington O. Sexton$^{1,2}$, Sara M. Doan$^{1}$, Michael A. Reefe$^{1}$, William Matzko$^{1}$
# $^{1}$George Mason University, $^{2}$United States Naval Observatory
# 

# In[1]:


import glob
import time
import natsort
#from IPython.display import clear_output
import os
import psutil
import pathlib
import shutil
from itertools import repeat
# To see full list of imported packages and modules, see 
import badass as badass # <<<---       Import BADASS here
import badass_tools.badass_ifu as ifu  # <<---  Import the IFU submodule here

#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:80% !important; }</style>"))


# ### BADASS Options

# In[2]:


################################## Fit Options #################################
# Fitting Parameters
fit_options={
"fit_reg"    : (4400,5350),                           # Fitting region; Note: Indo-US Library=(3460,9464)
"good_thresh":  0.0,                                  # percentage of "good" pixels required in fig_reg for fit.
"mask_bad_pix": False,                                # mask pixels SDSS flagged as 'bad' (careful!)
"mask_emline" : False,                                # mask emission lines for continuum fitting.
"interp_metal": False,                                # interpolate over metal absorption lines for high-z spectra
"n_basinhop": 15,                                     # Number of consecutive basinhopping thresholds before solution achieved
"test_line": {"bool": True,                           # boolean of whether or not to test lines
              "line":["OUT_OIII_5007","BR_H_BETA"],   # list of lines to test
              "cont_fit":True,                        # If True, continues with initial SciPy fit to obtain a fit with all detected lines in the test, with detection criteria specified below. If false, just outputs line test results
              "conf":0.9,                             # Bayesian A/B test confidence that line is there. Percentage (e.g. 0.9) to be detected, or None to not do this test. At least one test must not be None. Default is None.
              "f_conf":0.95,                          # F-test confidence percentage (e.g. 0.9) or None. Default is 0.9
              "chi2_ratio":None,                      # minimum chi square ratio for line detection (e.g. 3), int, float, or None. Default is None.
              "ssr_ratio":None,                       # minimum sum square ratio for line detection (e.g. 3), int, float or None. Default is None
              "linetest_mask":"or"},                  # If using multiple tests, can choose a detection to pass ALL criteria ("and") or just one criterion ("or"). Case INsensitive. Works as expected if only use one test. Default is or.  
"mask_metal": False,
"max_like_niter": 10,                                 # number of maximum likelihood iterations
"output_pars": False,                                 # only output free parameters of fit and stop code (diagnostic)
"fit_stat": "ML"
}
################################################################################

########################### MCMC algorithm parameters ##########################
mcmc_options={
"mcmc_fit"    : False,       # Perform robust fitting using emcee
"nwalkers"    : 100,         # Number of emcee walkers; min = 2 x N_parameters
"auto_stop"   : False,       # Automatic stop using autocorrelation analysis
"conv_type"   : "all",       # "median", "mean", "all", or (tuple) of parameters
"min_samp"    : 1000,        # min number of iterations for sampling post-convergence
"ncor_times"  : 1.0,         # number of autocorrelation times for convergence
"autocorr_tol": 10.0,        # percent tolerance between checking autocorr. times
"write_iter"  : 100,         # write/check autocorrelation times interval
"write_thresh": 100,         # iteration to start writing/checking parameters
"burn_in"     : 1500,        # burn-in if max_iter is reached
"min_iter"    : 2500,        # min number of iterations before stopping
"max_iter"    : 2500,        # max number of MCMC iterations
}
################################################################################

############################ Fit component op dtions #############################
comp_options={
"fit_opt_feii"     : True,        # optical FeII
"fit_uv_iron"      : False,       # UV Iron
"fit_balmer"       : False,       # Balmer continuum (<4000 A)
"fit_losvd"        : True,        # stellar LOSVD
"fit_host"         : False,       # host template
"fit_power"        : True,        # AGN power-law
"fit_narrow"       : True,        # narrow lines
"fit_broad"        : True,        # broad lines
"fit_outflow"      : True,        # outflow lines
"fit_absorp"       : False,       # absorption lines
"tie_line_disp"    : False,       # tie line widths
"tie_line_voff"    : False,       # tie line velocity offsets
"na_line_profile"  : "gaussian",  # narrow line profile
"br_line_profile"  : "voigt",     # broad line profile
"out_line_profile" : "gaussian",  # outflow line profile
"abs_line_profile" : "gaussian",  # absorption line profile
"n_moments"        : 4,           # number of Gauss-Hermite moments for Gauss-Hermite line profiles
                                    # must be >2 and <10 for higher-order moments (default = 4)
}
################################################################################

########################### Emission Lines & Options ###########################
# If not specified, defaults to SDSS-QSO Emission Lines (http://classic.sdss.org/dr6/algorithms/linestable.html)
################################################################################
user_lines = {
    # NArrow lines:
    "NA_HeI_4471"  :{"center":4471.479, "amp":"free", "disp":"free", "voff":"free", "line_type":"na","label":r"He I"},
    "NA_HeII_4687" :{"center":4687.021, "amp":"free", "disp":"free", "voff":"free", "line_type":"na","label":r"He II"},
    "NA_H_BETA"    :{"center":4862.691, "amp":"free", "disp":"NA_OIII_5007_DISP", "voff":"NA_OIII_5007_VOFF", "h3":"NA_OIII_5007_h3", "h4":"NA_OIII_5007_h4", "shape":"NA_OIII_5007_shape", "line_type":"na","label":r"H$\beta$"},
    #"NA_FeVII_4893":{"center":4893.370, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VII]"},
    "NA_OIII_4960" :{"center":4960.295, "amp":"(NA_OIII_5007_AMP/2.98)", "disp":"NA_OIII_5007_DISP", "voff":"NA_OIII_5007_VOFF", "h3":"NA_OIII_5007_h3", "h4":"NA_OIII_5007_h4", "shape":"NA_OIII_5007_shape", "line_type":"na","label":r"[O III]"},
    "NA_OIII_5007" :{"center":5008.240, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[O III]"},
    # broad lines:
    "BR_H_BETA"   :{"center":4862.691, "amp":"free", "disp":"free", "voff":"free", "line_type":"br"},
    # outflow lines:
#    "OUT_H_BETA"   :{"center":4862.691, "amp":"OUT_OIII_5007_AMP/NA_OIII_5007_AMP*NA_H_BETA_AMP", "disp":"free", "voff":"free", "h3":"out_h3", "h4":"out_h4", "shape":"out_shape", "line_type":"out"},
    "OUT_OIII_4960":{"center":4960.295, "amp":"OUT_OIII_5007_AMP/NA_OIII_5007_AMP*NA_OIII_5007_AMP/2.98", "disp":"OUT_OIII_5007_DISP", "voff":"OUT_OIII_5007_VOFF", "h3":"OUT_h3", "h4":"OUT_h4", "shape":"out_shape", "line_type":"out"},
    "OUT_OIII_5007":{"center":5008.240, "amp":"free", "disp":"free", "voff":"free", "h3":"OUT_h3", "h4":"OUT_h4", "shape":"out_shape", "line_type":"out"},
    #"NA_FeVI_5146" :{"center":5145.750, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VI]"},
    #"NA_FeVII_5159":{"center":5158.890, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VII]"},
    #"NA_FeVI_5176" :{"center":5176.040, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VI]"},
    #"NA_FeVII_5276":{"center":5276.380, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VII]"},
    #"NA_FeXIV_5303":{"center":5302.860, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe XIV]"},
    #"NA_CaV_5309"  :{"center":5309.110, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Ca V]"},
    #"NA_FeVI_5335" :{"center":5335.180, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VI]"},
}

user_constraints = [
("OUT_OIII_5007_DISP","NA_OIII_5007_DISP"),
("NA_OIII_5007_AMP", "OUT_OIII_5007_AMP"),
#("NA_OIII_5007_VOFF","OUT_OIII_5007_VOFF"), # careful, voff is negative!
("BR_H_BETA_DISP","OUT_H_BETA_DISP"),
("BR_H_BETA_DISP","NA_OIII_5007_DISP")

]
# User defined masked regions (list of tuples)
user_mask = [
# (6250,6525),
]

combined_lines = {
    "OIII_5007_COMP":["NA_OIII_5007", "OUT_OIII_5007"],
    "OIII_4960_COMP":["NA_OIII_4960", "OUT_OIII_4960"],
#    "H_BETA_COMP"   :["NA_H_BETA","OUT_H_BETA"]

}
########################## LOSVD Fitting & Options ##################
# For direct fitting of the stellar kinematics (stellar LOSVD), one can 
# specify a stellar template library (Indo-US, Vazdekis 2010, or eMILES).
# One can also hold velocity or dispersion constant to avoid template
# convolution during the fitting process.
################################################################################

losvd_options = {
"library"   : "IndoUS", # Options: IndoUS, Vazdekis2010, eMILES
"vel_const" : {"bool":False, "val":0.0},
"disp_const": {"bool":False, "val":250.0},
"losvd_apoly": {"bool":True , "order":3},
}


########################## SSP Host Galaxy Template & Options ##################
# The default is zero velocity, 100 km/s dispersion 10 Gyr template from 
# the eMILES stellar library. 
################################################################################

host_options = {
"age"       : [1.0,5.0,10.0], # Gyr; [0.09 Gyr - 14 Gyr] 
"vel_const" : {"bool":False, "val":0.0},
"disp_const": {"bool":False, "val":250.0}
}

########################### AGN power-law continuum & Options ##################
# The default is a simple power law.
################################################################################

power_options = {
"type" : "simple" # alternatively, "broken" for smoothly-broken power-law
}

########################### Polynomial Continuum Options #######################
# Disabled by default.  Options for a power series polynomial continuum, 
# additive legendre polynomial, or multiplicative polynomial to be included in 
# the fit.
################################################################################

poly_options = {
"ppoly" : {"bool": False, "order": 3}, # positive definite additive polynomial 
"apoly" : {"bool": False , "order": 3}, # Legendre additive polynomial 
"mpoly" : {"bool": False, "order": 3}, # Legendre multiplicative polynomial 
}

############################### Optical FeII options ###############################
# Below are options for fitting FeII.  For most objects, you don't need to 
# perform detailed fitting on FeII (only fit for amplitudes) use the 
# Veron-Cetty 2004 template ('VC04') (2-6 free parameters)
# However in NLS1 objects, FeII is much stronger, and sometimes more detailed 
# fitting is necessary, use the Kovacevic 2010 template 
# ('K10'; 7 free parameters).

# The options are:
# template   : VC04 (Veron-Cetty 2004) or K10 (Kovacevic 2010)
# amp_const  : constant amplitude (default False)
# disp_const : constant disp (default True)
# voff_const : constant velocity offset (default True)
# temp_const : constant temp ('K10' only)

opt_feii_options={
"opt_template"  :{"type":"VC04"}, 
"opt_amp_const" :{"bool":False, "br_opt_feii_val":1.0   , "na_opt_feii_val":1.0},
"opt_disp_const":{"bool":False, "br_opt_feii_val":3000.0, "na_opt_feii_val":500.0},
"opt_voff_const":{"bool":False, "br_opt_feii_val":0.0   , "na_opt_feii_val":0.0},
}
# or
# opt_feii_options={
# "opt_template"  :{"type":"K10"},
# "opt_amp_const" :{"bool":False,"f_feii_val":1.0,"s_feii_val":1.0,"g_feii_val":1.0,"z_feii_val":1.0},
# "opt_disp_const":{"bool":False,"opt_feii_val":1500.0},
# "opt_voff_const":{"bool":False,"opt_feii_val":0.0},
# "opt_temp_const":{"bool":True,"opt_feii_val":10000.0},
# }
################################################################################

############################### UV Iron options ################################
uv_iron_options={
"uv_amp_const"  :{"bool":False, "uv_iron_val":1.0},
"uv_disp_const" :{"bool":False, "uv_iron_val":3000.0},
"uv_voff_const" :{"bool":False, "uv_iron_val":0.0},
"uv_legendre_p" :{"bool":False , "uv_iron_val":3},
}
################################################################################

########################### Balmer Continuum options ###########################
# For most purposes, only the ratio R, and the overall amplitude are free paramters
# but if you want to go crazy, you can fit everything.
balmer_options = {
"R_const"          :{"bool":False, "R_val":1.0},                 # ratio between balmer continuum and higher-order balmer lines
"balmer_amp_const" :{"bool":False, "balmer_amp_val":1.0},        # amplitude of overall balmer model (continuum + higher-order lines)
"balmer_disp_const":{"bool":False,  "balmer_disp_val":5000.0},   # broadening of higher-order Balmer lines
"balmer_voff_const":{"bool":False,  "balmer_voff_val":0.0},      # velocity offset of higher-order Balmer lines
"Teff_const"       :{"bool":True,  "Teff_val":15000.0},          # effective temperature
"tau_const"        :{"bool":True,  "tau_val":1.0},               # optical depth
}

################################################################################

############################### Plotting options ###############################
plot_options={
"plot_param_hist"    : False,    # Plot MCMC histograms and chains for each parameter
"plot_flux_hist"     : False,    # Plot MCMC hist. and chains for component fluxes
"plot_lum_hist"      : False,    # Plot MCMC hist. and chains for component luminosities
"plot_eqwidth_hist"  : False,    # Plot MCMC hist. and chains for equivalent widths 
"plot_HTML"          : False,    # make interactive plotly HTML best-fit plot
}
################################################################################

################################ Output options ################################
output_options={
"write_chain" : False, # Write MCMC chains for all paramters, fluxes, and
                         # luminosities to a FITS table We set this to false 
                         # because MCMC_chains.FITS file can become very large, 
                         # especially  if you are running multiple objects.  
                         # You only need this if you want to reconstruct chains 
                         # and histograms. 
"verbose"     : True,  # prints steps of fitting process in Notebook
}
################################################################################


# ### Run BADASS using `multiprocessing.pool` to fit $N$ spectra simultaenously
# 
# The following is shows how to fit multiple SDSS spectra simultaneously using `multiprocessing.pool()`.  The number of spectra $N$ you can fit simultaneously ultimately depends on the number of CPU cores and RAM available on your system.

# #### Directory Structure: This is where IFU data will differ from fitting normal 1D spectra

# In[3]:


########################## Directory Structure #################################

########################## MaNGA Example Fit ###################################
spec_dir = r'examples/MUSE/'      # folder with spectra in it
redshifts = [0.00379]             # redshifts for each object must be entered manually (for now) since they are not included
                                  # in MUSE data
apertures = [[8, 9, 3, 4]]        # Square apertures for each cube -- only fit the spectra within the aperture.
                                  # number is half the side length of the cube, in pixels
    
# # Get full list of spectrum files - will make sub-directories when decomposing the IFU Cube(s), so it is assumed
# # that the cube FITS files are within spec_dir directly.
spec_loc = natsort.natsorted( glob.glob(spec_dir+'*.fits') )
spec_loc = spec_loc[:]

################################################################################
print(len(spec_loc))
print(spec_loc)


# #### Run

# In[5]:


# Iterate linearly over each MANGA cube to be fit, and fit each 1D spectrum within the cube in parallel
for cube, z, ap in zip(spec_loc, redshifts, apertures):
# for cube, z in zip(spec_loc, redshifts):

    # Unpack the spectra into 1D FITS files
    print(f'Unpacking {cube} into subfolders...')
    # The formats currently supported are 'muse' and 'manga'
    wave,flux,ivar,mask,fwhm_res,binnum,npixels,xpixbin,ypixbin,z,dataid,objname = ifu.prepare_ifu(cube, z, format='muse', 
                                                                                  aperture=ap, 
                                                                                  targetsn = 25.0 ,
                                                                                  snr_threshold = 0.,
                                                                                  voronoi_binning=False,
#                                                                                   maxbins=500,
                                                                                  use_and_mask = False,
                                                                                  )
#     # Plot the cube data
    ifu.plot_ifu(cube, wave, flux, ivar, mask, binnum, npixels, xpixbin, ypixbin, z, dataid, ap, objname)
#     ifu.plot_ifu(cube, wave, flux, ivar, mask, binnum, npixels, xpixbin, ypixbin, z, dataid, object_name=objname)
    
    cube_subdir = os.path.join(os.path.dirname(cube), cube.split(os.sep)[-1].replace('.fits','')) + os.sep

#     sys.exit()
    if __name__ == "__main__":
        badass.run_BADASS(cube_subdir,
                         nprocesses       = 4,
#                          nobj             = (0,1),
                         fit_options      = fit_options,
                         mcmc_options     = mcmc_options,
                         comp_options     = comp_options,
                         user_lines       = user_lines,
                         user_constraints = user_constraints,
                         losvd_options    = losvd_options,
                         host_options     = host_options,
                         power_options    = power_options,
                         poly_options     = poly_options,
                         opt_feii_options = opt_feii_options,
                         uv_iron_options  = uv_iron_options,
                         balmer_options   = balmer_options,
                         plot_options     = plot_options,
                         output_options   = output_options,
                         combined_lines   = combined_lines,  
                         sdss_spec        = False,
                         ifu_spec         = True
                         )


# ## Now we reconstruct the cube data and make some plots!

# In[6]:


for cube in spec_loc:
    
    _, _, n = ifu.reconstruct_ifu(cube)
    
    # If you want the fit results to be saved as an animated .mp4 file, change animated=True.
    # This option requires that you have both the python package ffmpeg and the ffmpeg software itself 
    # installed on your system.  See https://www.ffmpeg.org/download.html for details.
    ifu.plot_reconstructed_cube(os.path.join(os.path.dirname(cube), f'MCMC_output_{n}'), animated=False)

