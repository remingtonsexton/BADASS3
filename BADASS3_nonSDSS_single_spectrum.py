#!/usr/bin/env python
# coding: utf-8

# ## Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS)
# ### Non-SDSS Single Spectrum
# 
# ####  Remington O. Sexton$^{1,2}$, Sara M. Doan$^{1}$, Michael A. Reefe$^{1}$, William Matzko$^{1}$
# $^{1}$George Mason University, $^{2}$United States Naval Observatory,

# In[ ]:


import glob
import time
import natsort
#from IPython.display import clear_output
# import multiprocess as mp
import os
import psutil
import pathlib
# Import BADASS here
import badass as badass
import badass_utils as badass_utils
from astroquery.irsa_dust import IrsaDust
from astropy import coordinates
import astropy.units as u
import numpy as np
from scipy.ndimage import gaussian_filter as gf
#from IPython.display import display, HTML
#display(HTML("<style>.container { width:90% !important; }</style>"))


# ### BADASS Options

# In[ ]:


################################## Fit Options #################################
# Fitting Parameters
fit_options={
"fit_reg"    : (6400,6800),                           # Fitting region; Note: Indo-US Library=(3460,9464)
"good_thresh":  0.0,                                  # percentage of "good" pixels required in fig_reg for fit.
"mask_bad_pix": False,                                # mask pixels SDSS flagged as 'bad' (careful!)
"mask_emline" : False,                                # mask emission lines for continuum fitting.
"interp_metal": False,                                # interpolate over metal absorption lines for high-z spectra
"n_basinhop": 15,                                      # Number of consecutive basinhopping thresholds before solution achieved
"test_line": {"bool": False,                           # boolean of whether or not to test lines
              "test_indv": True,                      # boolean of whether or not to test lines individually. If True, tests lines one at a time (more control over components). If False, tests all lines at once (all or nothing). Default is True.
              "line":["OUT_OIII_5007","BR_H_BETA"],   # list of lines to test
              "cont_fit":True,                        # If True, continues fitting after line test. Will do a (initial) fit with all lines that passed the line test. If false, just outputs line test results
              "conf":0.9,                             # Bayesian A/B test confidence threshold that line is present. Minimum percentage (e.g. 0.9) for detection, or None to not do this test. At least one test must not be None. Default is None.
              "f_conf":0.95,                          # F-test confidence percentage (e.g. 0.9) or None. Default is 0.9
              "chi2_ratio":None,                      # minimum chi square ratio for line detection (e.g. 3), int, float, or None. Default is None.
              "ssr_ratio":None,                       # minimum sum square ratio for line detection (e.g. 3), int, float or None. Default is None
              "linetest_mask":"or"},                  # If using multiple tests, can choose a detection to pass ALL criteria ("and") or just one criterion ("or"). Case INsensitive. Works as expected if only use one test. Default is or.  
"mask_metal": False,
"max_like_niter": 10,                                  # number of maximum likelihood iterations
"output_pars": False,                                 # only output free parameters of fit and stop code (diagnostic)
"fit_stat": "RCHI2"
}
################################################################################

########################### MCMC algorithm parameters ##########################
mcmc_options={
"mcmc_fit"    : False, # Perform robust fitting using emcee
"nwalkers"    : 100,  # Number of emcee walkers; min = 2 x N_parameters
"auto_stop"   : False, # Automatic stop using autocorrelation analysis
"conv_type"   : "all", # "median", "mean", "all", or (tuple) of parameters
"min_samp"    : 1000,  # min number of iterations for sampling post-convergence
"ncor_times"  : 1.0,  # number of autocorrelation times for convergence
"autocorr_tol": 10.0,  # percent tolerance between checking autocorr. times
"write_iter"  : 100,   # write/check autocorrelation times interval
"write_thresh": 100,   # iteration to start writing/checking parameters
"burn_in"     : 1000, # burn-in if max_iter is reached
"min_iter"    : 1000, # min number of iterations before stopping
"max_iter"    : 2500, # max number of MCMC iterations
}
################################################################################

############################ Fit component op dtions #############################
comp_options={
"fit_opt_feii"     : False, # optical FeII
"fit_uv_iron"      : False, # UV Iron 
"fit_balmer"       : False, # Balmer continuum (<4000 A)
"fit_losvd"        : False, # stellar LOSVD
"fit_host"         : True, # host template
"fit_power"        : True, # AGN power-law
"fit_poly"         : False, # Add polynomial continuum component
"fit_narrow"       : True, # narrow lines
"fit_broad"        : True, # broad lines
"fit_outflow"      : False, # outflow lines
"fit_absorp"       : False, # absorption lines
"tie_line_disp"    : False, # tie line widths
"tie_line_voff"    : False, # tie line velocity offsets
"na_line_profile"  : "gaussian",     # narrow line profile
"br_line_profile"  : "voigt",     # broad line profile
"out_line_profile" : "gaussian",     # outflow line profile
"abs_line_profile" : "gaussian",     # absorption line profile
"n_moments"        : 4, # number of Gauss-Hermite moments for Gauss-Hermite line profiles
                        # must be >2 and <10 for higher-order moments (default = 4)
}
################################################################################

########################### Emission Lines & Options ###########################
# If not specified, defaults to SDSS-QSO Emission Lines (http://classic.sdss.org/dr6/algorithms/linestable.html)

user_lines = {
"NA_NII_6549"  :{"center":6549.859, "amp":"NA_NII_6585_AMP/3", "disp":"45.1852", "voff":"-56.178932", "line_type":"na","label":r"[N II]"},
"NA_H_ALPHA"   :{"center":6564.632, "amp":"free","amp_prior":{"type":"gaussian","loc":3850,"scale":0.1}, "disp":"45.1852", "voff":"-56.1789", "line_type":"na","label":r"H$\alpha$"},
"NA_NII_6585"  :{"center":6585.278, "amp":"12.4045", "disp":"45.1852", "voff":"-56.178932", "line_type":"na","label":r"[N II]"},
"NA_SII_6718"  :{"center":6718.294, "amp":"35.015934", "disp":"48.366700", "voff":"-55.783392", "h3":"NA_NII_6585_h3", "h4":"NA_NII_6585_h4", "shape":"NA_NII_6585_shape", "line_type":"na","label":r"[S II]"},
"NA_SII_6732"  :{"center":6732.668, "amp":"28.117003", "disp":"48.366700P", "voff":"-55.783392", "h3":"NA_NII_6585_h3", "h4":"NA_NII_6585_h4", "shape":"NA_NII_6585_shape", "line_type":"na","label":r"[S II]"},
"NA_UNK_6677"  :{"center":6677.000, "amp":"33.406151", "disp":"66.514431", "voff":"69.186165", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"UNK"},
#"BR_H_ALPHA"  :{"center":6564.278, "amp":"free", "disp":"free", "voff":"free", "line_type":"br","line_profile":"lorentzian"},
#"BR2_H_ALPHA" :{"center":6564.278, "amp":"free", "disp":"free", "voff":"free", "line_type":"br", "line_profile":"gauss-hermite"},
#"BR3_H_ALPHA" :{"center":6564.278, "amp":"free", "disp":"free", "voff":"free", "line_type":"br","line_profile":"gaussian"},
#"BR3_H_ALPHA" :{"center":6564.278, "amp":"free", "disp":"free", "voff":"free", "line_type":"br","line_profile":"gaussian"},
"BR_H_ALPHA" :{"center":6564.278, "amp":"free", "disp":"free", "voff":"free", "line_type":"br", "line_profile":"gauss-hermite"},
"BR2_H_ALPHA"  :{"center":6564.278, "amp":"free", "disp":"free", "voff":"free", "line_type":"br","line_profile":"lorentzian"},
}

################################################################################
# user_lines = {
#     # narrow lines:
#     #"NA_OI_6302"   :{"center":6302.046, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[O I]"},
#     #"NA_SIII_63012":{"center":6312.060, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[S III]"},
#     #"NA_OI_6365"   :{"center":6365.535, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[O I]"},
#     #"NA_FeX_6374"  :{"center":6374.510, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe X]"}, # Coronal Line
#     "NA_NII_6549"  :{"center":6549.859, "amp":"NA_NII_6585_AMP/3", "disp":"NA_NII_6585_DISP", "voff":"NA_NII_6585_VOFF", "h3":"NA_NII_6585_h3", "h4":"NA_NII_6585_h4", "shape":"NA_H_alpha_shape", "line_type":"na","label":r"[N II]"},
#     #"NA_H_ALPHA"   :{"center":6564.632, "amp":"free", "disp":"NA_NII_6585_DISP", "voff":"free", "h3":"NA_NII_6585_h3", "h4":"NA_NII_6585_h4", "shape":"NA_NII_6585_shape", "line_type":"na","label":r"H$\alpha$"},
#     #"NA2_H_ALPHA"  :{"center":6564.278, "amp":"138.88", "disp":"74.347", "voff":"-58.8739", "line_type":"na"},
#     "NA_NII_6585"  :{"center":6585.278, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[N II]"},
#     "NA_SII_6718"  :{"center":6718.294, "amp":"free", "disp":"NA_NII_6585_DISP", "voff":"NA_NII_6585_VOFF", "h3":"NA_NII_6585_h3", "h4":"NA_NII_6585_h4", "shape":"NA_NII_6585_shape", "line_type":"na","label":r"[S II]"},
#     "NA_SII_6732"  :{"center":6732.668, "amp":"free", "disp":"NA_NII_6585_DISP", "voff":"NA_NII_6585_VOFF", "h3":"NA_NII_6585_h3", "h4":"NA_NII_6585_h4", "shape":"NA_NII_6585_shape", "line_type":"na","label":r"[S II]"},
#     "NA_UNK_6677"  :{"center":6677.000, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"UNK"},
#     #broad lines:
#     "BR_H_ALPHA"  :{"center":6564.278, "amp":"free", "disp":"free", "voff":"free", "line_type":"br", "line_profile":"lorentzian"},
#     "BR2_H_ALPHA" :{"center":6564.278, "amp":"free", "disp":"free", "voff":"free", "line_type":"br","line_profile":"gauss-hermite"},
#     #outflow lines:
#     #"out_NII_6549" :{"center":6549.859, "amp":"out_NII_6585_amp/na_NII_6585_amp*na_NII_6585_amp/2.93", "disp":"out_disp", "voff":"out_voff", "h3":"out_h3", "h4":"out_h4", "shape":"out_shape", "line_type":"out"},
#     #"out_H_ALPHA"  :{"center":6564.632, "amp":"out_NII_6585_amp/na_NII_6585_amp*na_H_alpha_amp", "disp":"out_disp", "voff":"out_voff", "h3":"out_h3", "h4":"out_h4", "shape":"out_shape", "line_type":"out"},
#     #"out_NII_6585" :{"center":6585.278, "amp":"free", "disp":"out_disp", "voff":"out_voff", "h3":"out_h3", "h4":"out_h4", "shape":"out_shape", "line_type":"out"},
#     #"out_SII_6718" :{"center":6718.294, "amp":"out_NII_6585_amp/na_NII_6585_amp*na_SII_6718_amp", "disp":"out_disp", "voff":"out_voff", "h3":"out_h3", "h4":"out_h4", "shape":"out_shape", "line_type":"out"},
#     #"out_SII_6732" :{"center":6732.668, "amp":"out_NII_6585_amp/na_NII_6585_amp*na_SII_6732_amp", "disp":"out_disp", "voff":"out_voff", "h3":"out_h3", "h4":"out_h4", "shape":"out_shape", "line_type":"out"},
# }
    

# user_lines = {
#     # NArrow lines:
#     "NA_HeI_4471"  :{"center":4471.479, "amp":"free", "disp":"free", "voff":"free", "line_type":"na","label":r"He I"},
#     "NA_HeII_4687" :{"center":4687.021, "amp":"free", "disp":"free", "voff":"free", "line_type":"na","label":r"He II"},
#     #"OUT_HeII_4687"  :{"center":4686.479, "amp":"free", "disp":"NA_HeII_4687_DISP", "voff":"free", "shape":"out_shape", "line_type":"out",},
#     "NA_H_BETA"    :{"center":4862.691, "amp":"free", "disp":"NA_OIII_5007_DISP", "voff":"free", "h3":"NA_OIII_5007_h3", "h4":"NA_OIII_5007_h4", "shape":"NA_OIII_5007_shape", "line_type":"na","label":r"H$\beta$"},
#     #"NA_FeVII_4893":{"center":4893.370, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VII]"},
#     "NA_OIII_4960" :{"center":4960.295, "amp":"(NA_OIII_5007_AMP/2.98)", "disp":"NA_OIII_5007_DISP", "voff":"NA_OIII_5007_VOFF", "h3":"NA_OIII_5007_h3", "h4":"NA_OIII_5007_h4", "shape":"NA_OIII_5007_shape", "line_type":"na","label":r"[O III]"},
#     "NA_OIII_5007" :{"center":5008.240, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[O III]"},
#     # broad lines:
#     #"BR_H_BETA"   :{"center":4862.691, "amp":"free", "disp":"free", "voff":"free", "line_type":"br"},
#     # outflow lines:
# #    "OUT_H_BETA"   :{"center":4862.691, "amp":"OUT_OIII_5007_AMP/NA_OIII_5007_AMP*NA_H_BETA_AMP", "disp":"free", "voff":"free", "h3":"out_h3", "h4":"out_h4", "shape":"out_shape", "line_type":"out"},
#     "OUT_OIII_4960":{"center":4960.295, "amp":"OUT_OIII_5007_AMP/NA_OIII_5007_AMP*NA_OIII_5007_AMP/2.98", "disp":"OUT_OIII_5007_DISP", "voff":"OUT_OIII_5007_VOFF", "h3":"OUT_h3", "h4":"OUT_h4", "shape":"out_shape", "line_type":"out"},
#     "OUT_OIII_5007":{"center":5008.240, "amp":"free", "disp":"free", "voff":"free", "h3":"OUT_h3", "h4":"OUT_h4", "shape":"out_shape", "line_type":"out"},
#     #"NA_FeVI_5146" :{"center":5145.750, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VI]"},
#     #"NA_FeVII_5159":{"center":5158.890, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VII]"},
#     #"NA_FeVI_5176" :{"center":5176.040, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VI]"},
#     #"NA_FeVII_5276":{"center":5276.380, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VII]"},
#     #"NA_FeXIV_5303":{"center":5302.860, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe XIV]"},
#     #"NA_CaV_5309"  :{"center":5309.110, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Ca V]"},
#     #"NA_FeVI_5335" :{"center":5335.180, "amp":"free", "disp":"free", "voff":"free", "h3":"free", "h4":"free", "shape":"free", "line_type":"na","label":r"[Fe VI]"},
# }

user_constraints = [
    #("BR_H_ALPHA_DISP","NA_NII_6585_DISP"), # Broad H-alpha must be broader than the narrow lines
    #('NA_NII_6585_AMP','BR_H_ALPHA_AMP'),
    #('BR_H_ALPHA_DISP','NA_H_ALPHA_DISP'),
    #('NA2_H_ALPHA_AMP','BR_H_ALPHA_AMP'),
    #('NA2_H_ALPHA_AMP','BR2_H_ALPHA_AMP')
    #("br_H_alpha_fwhm","out_fwhm"), # Broad H-alpha must be broader than the outflow lines
    #("out_fwhm","na_NII_6585_fwhm"), # Outflow must be broader than narrow lines
    #('NA_OIII_5007_VOFF','OUT_OIII_5007_VOFF'),
    #('OUT_OIII_5007_VOFF','2500')
]
# User defined masked regions (list of tuples)
user_mask = [
#(6500,6545),
#(6553,6567),
#(6567,6580),
#(6588,6650)
#(6562,6563)
]

combined_lines = {
#"H_ALPHA_COMP":["NA_H_ALPHA","BR_H_ALPHA"],
#"NA_HeII_COMP":["NA_HeII_4687","OUT_HeII_4687"]
#'OIII_5007_COMP':['NA_OIII_5007', 'OUT_OIII_5007'],
#'OIII_4960_COMP':['NA_OIII_4960', 'OUT_OIII_4960'],
'H_ALPHA_COMP':['BR_H_ALPHA','BR2_H_ALPHA','NA_H_ALPHA']
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
"losvd_apoly": {"bool":False , "order":3},
}

########################## SSP Host Galaxy Template & Options ##################
# The default is zero velocity, 100 km/s dispersion 10 Gyr template from 
# the eMILES stellar library. 
################################################################################

host_options = {
"age"       : [0.1,1.0,5.0,10.0], # Gyr; [0.09 Gyr - 14 Gyr] 
"vel_const" : {"bool":False, "val":0.0},
"disp_const": {"bool":False, "val":150.0}
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
"ppoly" : {"bool": False, "order": 1}, # positive definite additive polynomial 
"apoly" : {"bool": True , "order": 1}, # Legendre additive polynomial 
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
"uv_voff_const" :{"bool":True,  "uv_iron_val":0.0},
"uv_legendre_p" :{"bool":False, "uv_iron_val":3},
}
################################################################################

########################### Balmer Continuum options ###########################
# For most purposes, only the ratio R, and the overall amplitude are free paramters
# but if you want to go crazy, you can fit everything.
balmer_options = {
"R_const"          :{"bool":True,  "R_val":1.0}, # ratio between balmer continuum and higher-order balmer lines
"balmer_amp_const" :{"bool":False, "balmer_amp_val":1.0}, # amplitude of overall balmer model (continuum + higher-order lines)
"balmer_disp_const":{"bool":True,  "balmer_disp_val":5000.0}, # broadening of higher-order Balmer lines
"balmer_voff_const":{"bool":True,  "balmer_voff_val":0.0}, # velocity offset of higher-order Balmer lines
"Teff_const"       :{"bool":True,  "Teff_val":15000.0}, # effective temperature
"tau_const"        :{"bool":True,  "tau_val":1.0}, # optical depth
}

################################################################################

############################### Plotting options ###############################
plot_options={
"plot_param_hist"    : False, # Plot MCMC histograms and chains for each parameter
"plot_flux_hist"     : False, # Plot MCMC hist. and chains for component fluxes
"plot_lum_hist"      : False, # Plot MCMC hist. and chains for component luminosities
"plot_eqwidth_hist"  : False, # Plot MCMC hist. and chains for equivalent widths 
"plot_HTML"          : False, # make interactive plotly HTML best-fit plot
"plot_pca"           : False, # Plot PCA reconstructed spectrum. If doing PCA, you probably want this as True
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


# ### Run BADASS on a single spectrum
# 
# The following is shows how to fit single SDSS spectra.

# #### Directory Structure

# In[ ]:


########################## Directory Structure #################################
spec_dir = 'G:\\Research\\MUSE\\J104457\\ADP_2021_05_17T13_24_16_108\\'#'examples/' # folder with spectra in it
# Get full list of spectrum folders; these will be the working directories
spec_loc = natsort.natsorted( glob.glob(spec_dir+'*') )
################################################################################
#print(spec_loc)


# #### Choose Spectrum

# In[ ]:


nobj = -7 # Object in the spec_loc list
work_dir = spec_loc[nobj]+'\\' # working directory
print(work_dir)
# Set up run ('MCMC_output_#') directory
file = glob.glob(work_dir+'*.FITS')[0] # Get name of FITS spectra file
#
print(file)

#sys.exit()
# #### Load the Spectrum 

# In[ ]:

# For non-SDSS spectra, you must explicitly pass vectors for the spectrum (spec), 
# linearly-binned wavelength (wave), error spectrum (err), FWHM resolution in Å (fwhm_res),
# redshift (z), and Galactic reddening E(B-V) (ebv).
from astropy.io import fits
import matplotlib.pyplot as plt
hdulist = fits.open(file)

with fits.open(file) as hdu:
    wave = hdu[1].data['wave']
    spec = hdu[1].data['flux'] * 10**(-20)
    #err = np.sqrt(hdu[1].data['ivar']) * 10**(-20) 
    err = spec * 0.05 * gf(spec*0.05, sigma = 3)
    fwhm_res = hdu[1].data['specres']
    
    ra = hdu[2].data['ra'][0]
    dec = hdu[2].data['dec'][0]
    z = hdu[2].data['z'][0]
    
co = coordinates.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
try: 
    table = IrsaDust.get_query_table(co,section='ebv')
    ebv = table['ext SandF mean'][0]
except: 
    ebv = 0.04 # average Galactic E(B-V)
        # If E(B-V) is large, it can significantly affect normalization of the 
        # spectrum, in addition to changing its shape.  Re-normalizing the spectrum
        # throws off the maximum likelihood fitting, so instead of re-normalizing, 
        # we set an upper limit on the allowed ebv value for Galactic de-reddening.
if (ebv>=1.0):
    ebv = 0.04 # average Galactic E(B-V)
    

#header = hdulist[0].header
#z = header['z']
#fwhm_res = header['fwhm']
#ebv = header['ebv']
#spec = hdulist[0].data
#wave = hdulist[1].data
#err  = hdulist[2].data
# Plot
#fig = plt.figure(figsize=(22,6))
#ax1 = fig.add_subplot(1,1,1)
#ax1.plot(wave,spec,linewidth=0.5,label=r'Spectrum')
#ax1.plot(wave,err,linewidth=0.5,label=r'$1\sigma$ Error')
#fontsize=14
#ax1.set_title("Keck/LRIS Un-normalized, Observed Spectrum Input, $z=%0.4f$ (Sexton et al. 2019)" % z,fontsize=fontsize)
#ax1.set_xlabel(r"$\lambda_{\rm{observed}}$ ($\rm{\AA}$)",fontsize=fontsize)
#ax1.set_ylabel(r"$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)",fontsize=fontsize)
#ax1.legend(fontsize=fontsize)
#plt.tight_layout()

# ### Run 

# In[ ]:

# Note: we set sdss_spec=False for non-SDSS spectrum.  This tells BADASS
# to use the spec, wave, err, fwhm_res, z, and ebv keywords for the data input.

# Call the main function in BADASS
badass.run_BADASS(pathlib.Path(file),
                  fit_options          = fit_options,
                  mcmc_options         = mcmc_options,
                  comp_options         = comp_options,
                  user_lines           = user_lines, # User-lines
                  user_constraints     = user_constraints, # User-constraints
                  user_mask            = user_mask, # User-mask
                  combined_lines       = combined_lines,
                  losvd_options        = losvd_options,
                  host_options         = host_options,
                  power_options        = power_options,
                  poly_options         = poly_options,
                  opt_feii_options     = opt_feii_options,
                  uv_iron_options      = uv_iron_options,
                  balmer_options       = balmer_options,
                  plot_options         = plot_options,
                  output_options       = output_options,
                  # Here is where we specify that we are fitting a non-SDSS user-input spectrum:
                  sdss_spec            = False,
                  spec                 = spec,
                  wave                 = wave, # observed wavelength
                  err                  = err, # 1-sigma uncertainty
                  fwhm_res             = fwhm_res, # linear FWHM resolution in Angstroms
                  z                    = z, # BADASS assumes spectrum is NOT corrected for redshift.
                  ebv                  = ebv # for Galactic extinction correction.
                 )
    #


# In[ ]:




