################################## Fit Options #################################
# Fitting Parameters
fit_options={
"fit_reg"    : (4400,5500),# Fitting region; Note: Indo-US Library=(3460,9464)
# "fit_reg"    : (2000,3000),# Fitting region; Note: Indo-US Library=(3460,9464)

"good_thresh": 0.0, # percentage of "good" pixels required in fig_reg for fit.
"mask_bad_pix": False, # mask pixels SDSS flagged as 'bad' (careful!)
"mask_emline" : False, # automatically mask lines for continuum fitting.
"mask_metal": False, # interpolate over metal absorption lines for high-z spectra
"fit_stat": "RCHI2", # fit statistic; ML = Max. Like. , OLS = Ordinary Least Squares, RCHI2 = reduced chi2
"n_basinhop": 25, # Number of consecutive basinhopping thresholds before solution achieved
"test_lines": True,
"max_like_niter": 25, # number of maximum likelihood iterations
"output_pars": False, # only output free parameters of fit and stop code (diagnostic)
"cosmology": {"H0":70.0, "Om0": 0.30}, # Flat Lam-CDM Cosmology
}

################################################################################

########################### MCMC algorithm parameters ##########################
mcmc_options={
"mcmc_fit"    : True, # Perform robust fitting using emcee
"nwalkers"    : 100,  # Number of emcee walkers; min = 2 x N_parameters
"auto_stop"   : False, # Automatic stop using autocorrelation analysis
"conv_type"   : "all", # "median", "mean", "all", or (tuple) of parameters
"min_samp"    : 1000,  # min number of iterations for sampling post-convergence
"ncor_times"  : 10.0,  # number of autocorrelation times for convergence
"autocorr_tol": 10.0,  # percent tolerance between checking autocorr. times
"write_iter"  : 100,   # write/check autocorrelation times interval
"write_thresh": 100,   # iteration to start writing/checking parameters
"burn_in"     : 1500, # burn-in if max_iter is reached
"min_iter"    : 1000, # min number of iterations before stopping
"max_iter"    : 3000, # max number of MCMC iterations
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
"fit_poly"         : True, # Add polynomial continuum component
"fit_narrow"       : True, # narrow lines
"fit_broad"        : False, # broad lines
"fit_absorp"       : False, # absorption lines
"tie_line_disp"    : False, # tie line widths (dispersions)
"tie_line_voff"    : False, # tie line velocity offsets
}

# Line options for each narrow, broad, and absorption.
narrow_options = {
#     "amp_plim": (0,1), # line amplitude parameter limits; default (0,)
    "disp_plim": (0,1000), # line dispersion parameter limits; default (0,)
    "voff_plim": (-1000,1000), # line velocity offset parameter limits; default (0,)
    "line_profile": "gaussian", # line profile shape*
    "n_moments": 4, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)
}

broad_options ={
#     "amp_plim": (0,40), # line amplitude parameter limits; default (0,)
#     "disp_plim": (300,3000), # line dispersion parameter limits; default (0,)
#     "voff_plim": (-1000,1000), # line velocity offset parameter limits; default (0,)
    "line_profile": "gauss-hermite", # line profile shape*
    "n_moments": 6, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)
}

absorp_options = {
#     "amp_plim": (-1,0), # line amplitude parameter limits; default (0,)
#     "disp_plim": (0,10), # line dispersion parameter limits; default (0,)
#     "voff_plim": (-2500,2500), # line velocity offset parameter limits; default (0,)
    "line_profile": "gaussian", # line profile shape*
    "n_moments": 4, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)        
}

# Choices for line profile shape include 'gaussian', 'lorentzian', 'voigt',
# 'gauss-hermite', 'laplace', and 'uniform'
################################################################################

########################### Emission Lines & Options ###########################
# If not specified, defaults to SDSS-QSO Emission Lines (http://classic.sdss.org/dr6/algorithms/linestable.html)
################################################################################
# User lines overrides the default line list with a user-input line list!
user_lines = {
    "NA_H_BETA"      :{"center":4862.691,"amp":"free","disp":"NA_OIII_5007_DISP","voff":"free","h3":"NA_OIII_5007_H3","h4":"NA_OIII_5007_H4","line_type":"na","label":r"H$\beta$","ncomp":1,},
    "NA_H_BETA_2"    :{"center":4862.691,"amp":"NA_H_BETA_AMP*NA_OIII_5007_2_AMP/NA_OIII_5007_AMP","disp":"NA_OIII_5007_2_DISP","voff":"NA_OIII_5007_2_VOFF","h3":"NA_OIII_5007_2_H3","h4":"NA_OIII_5007_2_H4","line_type":"na","ncomp":2,"parent":"NA_H_BETA"},
    "NA_H_BETA_3"    :{"center":4862.691,"amp":"NA_H_BETA_AMP*NA_OIII_5007_3_AMP/NA_OIII_5007_AMP","disp":"NA_OIII_5007_3_DISP","voff":"NA_OIII_5007_3_VOFF","h3":"NA_OIII_5007_3_H3","h4":"NA_OIII_5007_3_H4","line_type":"na","ncomp":3,"parent":"NA_H_BETA"}, 
    "NA_H_BETA_4"    :{"center":4862.691,"amp":"NA_H_BETA_AMP*NA_OIII_5007_4_AMP/NA_OIII_5007_AMP","disp":"NA_OIII_5007_4_DISP","voff":"NA_OIII_5007_4_VOFF","h3":"NA_OIII_5007_4_H3","h4":"NA_OIII_5007_4_H4","line_type":"na","ncomp":4,"parent":"NA_H_BETA"}, 
    "NA_H_BETA_5"    :{"center":4862.691,"amp":"NA_H_BETA_AMP*NA_OIII_5007_5_AMP/NA_OIII_5007_AMP","disp":"NA_OIII_5007_5_DISP","voff":"NA_OIII_5007_5_VOFF","h3":"NA_OIII_5007_5_H3","h4":"NA_OIII_5007_5_H4","line_type":"na","ncomp":5,"parent":"NA_H_BETA"}, 

    "NA_OIII_4960"   :{"center":4960.295,"amp":"(NA_OIII_5007_AMP/2.98)","disp":"NA_OIII_5007_DISP","voff":"NA_OIII_5007_VOFF","h3":"NA_OIII_5007_H3","h4":"NA_OIII_5007_H4","line_type":"na","label":r"[O III]","ncomp":1,},
    "NA_OIII_4960_2" :{"center":4960.295,"amp":"(NA_OIII_5007_2_AMP/2.98)","disp":"NA_OIII_5007_2_DISP","voff":"NA_OIII_5007_2_VOFF","h3":"NA_OIII_5007_2_H3","h4":"NA_OIII_5007_2_H4","line_type":"na","ncomp":2,"parent":"NA_OIII_4960"},
    "NA_OIII_4960_3" :{"center":4960.295,"amp":"(NA_OIII_5007_3_AMP/2.98)","disp":"NA_OIII_5007_3_DISP","voff":"NA_OIII_5007_3_VOFF","h3":"NA_OIII_5007_3_H3","h4":"NA_OIII_5007_3_H4","line_type":"na","ncomp":3,"parent":"NA_OIII_4960"},
    "NA_OIII_4960_4" :{"center":4960.295,"amp":"(NA_OIII_5007_4_AMP/2.98)","disp":"NA_OIII_5007_4_DISP","voff":"NA_OIII_5007_4_VOFF","h3":"NA_OIII_5007_4_H3","h4":"NA_OIII_5007_4_H4","line_type":"na","ncomp":4,"parent":"NA_OIII_4960"},
    "NA_OIII_4960_5" :{"center":4960.295,"amp":"(NA_OIII_5007_5_AMP/2.98)","disp":"NA_OIII_5007_5_DISP","voff":"NA_OIII_5007_5_VOFF","h3":"NA_OIII_5007_5_H3","h4":"NA_OIII_5007_5_H4","line_type":"na","ncomp":5,"parent":"NA_OIII_4960"},

    "NA_OIII_5007"   :{"center":5008.240,"amp":"free","disp":"free","voff":"free","h3":"free","h4":"free","line_type":"na","label":r"[O III]","ncomp":1,},
    "NA_OIII_5007_2" :{"center":5008.240,"amp":"free","disp":"free","voff":"free","h3":"free","h4":"free","line_type":"na","ncomp":2,"parent":"NA_OIII_5007"},
    "NA_OIII_5007_3" :{"center":5008.240,"amp":"free","disp":"free","voff":"free","h3":"free","h4":"free","line_type":"na","ncomp":3,"parent":"NA_OIII_5007"},
    "NA_OIII_5007_4" :{"center":5008.240,"amp":"free","disp":"free","voff":"free","h3":"free","h4":"free","line_type":"na","ncomp":4,"parent":"NA_OIII_5007"},
    "NA_OIII_5007_5" :{"center":5008.240,"amp":"free","disp":"free","voff":"free","h3":"free","h4":"free","line_type":"na","ncomp":5,"parent":"NA_OIII_5007"},
    
    "BR_H_BETA"      :{"center":4862.691,"amp":"free","disp":"free","voff":"free","line_type":"br","ncomp":1,},
    "BR_H_BETA_2"    :{"center":4862.691,"amp":"free","disp":"free","voff":"free","line_type":"br","ncomp":2,"parent":"BR_H_BETA"},
    "BR_H_BETA_3"    :{"center":4862.691,"amp":"free","disp":"free","voff":"free","line_type":"br","ncomp":3,"parent":"BR_H_BETA"},

    "NA_UNK_1"       :{"center":5200,"line_type":"na"},

}


test_options = {
"test_mode":"line",
"lines": [["NA_OIII_5007","NA_OIII_4960","NA_H_BETA"]], # The lines to test
"ranges":[(4900,5100)], # The range over which the test is performed must include the tested line
# "groups": [["NA_OIII_5007","NA_OIII_4960","NA_H_BETA"],["BR_H_BETA"]], # groups of line associated lines including the lines being tested
"metrics": ["BADASS", "ANOVA", "CHI2_RATIO", "AON"],# Fitting metrics to use when determining the best model
"thresholds": [0.95, 0.95, 0.10, 3.0],
"conv_mode": "any", # "any" single threshold satisfies the solution, or "all" must satisfy thresholds
"auto_stop":False, # automatically stop testing once threshold is reached; False test all no matter what
"full_verbose":False, # prints out all test fitting to screen
"plot_tests":True, # plot the fit of each model comparison
"force_best":True, # this forces the more-complex model to have a fit better than the previous.
"continue_fit":True, # continue the fit with the best chosen model
}

# test_options = {
# "test_mode":"line",
# "lines": "BR_H_BETA", # The lines to test
# "ranges":(4700,4940), # The range over which the test is performed must include the tested line
# "metrics": ["BADASS", "ANOVA", "CHI2_RATIO","AON"],# Fitting metrics to use when determining the best model
# "thresholds": [0.95, 0.95, 0.25, 3.0],
# "auto_stop":True, # automatically stop testing once threshold is reached; False test all no matter what
# "plot_tests":True,
# "force_best":True, # this forces the more-complex model to have a fit better than the previous.
# "continue_fit":True, # continue the fit with the best chosen model
# }

# test_options = {
# "test_mode":"line",
# "lines": [["NA_OIII_5007","NA_OIII_4960","NA_H_BETA"],"BR_H_BETA","NA_UNK_1"], # The lines to test
# "ranges":[(4900,5050),(4700,4940),(5100,5200)], # The range over which the test is performed must include the tested line
# "metrics": ["BADASS", "ANOVA", "CHI2_RATIO","AON"],# Fitting metrics to use when determining the best model
# "thresholds": [0.95, 0.95, 0.10, 3.0],
# "auto_stop":True, # automatically stop testing once threshold is reached; False test all no matter what
# "plot_tests":True,
# "force_best":True, # this forces the more-complex model to have a fit better than the previous.
# "continue_fit":True, # continue the fit with the best chosen model
# }

# test_options = {
# "test_mode":"line",
# "lines":["BR_MGII_2799","NA_MGII_2799"], # The lines to test
# "ranges":[(2600,3000),(2600,3000)], # The range over which the test is performed must include the tested line
# "metrics": ["BADASS", "ANOVA", "CHI2_RATIO","AON"],# Fitting metrics to use when determining the best model
# "thresholds": [0.95, 0.95, 0.25, 5.0],
# "auto_stop":True, # automatically stop testing once threshold is reached; False test all no matter what
# "plot_tests":True,
# "force_best":True, # this forces the more-complex model to have a fit better than the previous.
# "continue_fit":True, # continue the fit with the best chosen model
# }


user_constraints = [
    ("NA_OIII_5007_2_DISP","NA_OIII_5007_DISP"),
    ("NA_OIII_5007_3_DISP","NA_OIII_5007_2_DISP"),
    ("NA_OIII_5007_4_DISP","NA_OIII_5007_3_DISP"),
    ("NA_OIII_5007_5_DISP","NA_OIII_5007_4_DISP"),

]
# User defined masked regions (list of tuples)
user_mask = [
#     (4840,5015),
]

combined_lines = {
    # "H_BETA_COMP"   :["NA_H_BETA","NA_H_BETA_2","BR_H_BETA"],
}
########################## LOSVD Fitting & Options ##################
# For direct fitting of the stellar kinematics (stellar LOSVD), one can 
# specify a stellar template library (Indo-US, Vazdekis 2010, or eMILES).
# One can also hold velocity or dispersion constant to avoid template
# convolution during the fitting process.
################################################################################

losvd_options = {
"library"   : "IndoUS", # Options: IndoUS, Vazdekis2010, eMILES
"vel_const" :  {"bool":False, "val":0.0},
"disp_const":  {"bool":False, "val":250.0},
}

########################## SSP Host Galaxy Template & Options ##################
# The default is zero velocity, 100 km/s dispersion 10 Gyr template from 
# the eMILES stellar library. 
################################################################################

host_options = {
"age"       : [1.0,5.0,10.0], # Gyr; [0.09 Gyr - 14 Gyr] 
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
"apoly" : {"bool": True , "order": 7}, # Legendre additive polynomial 
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
# disp_const : constant dispersion (default True)
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
"plot_param_hist"    : False,# Plot MCMC histograms and chains for each parameter
"plot_HTML"          : True,# make interactive plotly HTML best-fit plot
}
################################################################################

################################ Output options ################################
output_options={
"write_chain"  : False, # Write MCMC chains for all paramters, fluxes, and
                         # luminosities to a FITS table We set this to false 
                         # because MCMC_chains.FITS file can become very large, 
                         # especially  if you are running multiple objects.  
                         # You only need this if you want to reconstruct chains 
                         # and histograms. 
"write_options": False,  # output restart file
"verbose"      : True,  # prints steps of fitting process in Notebook
}
################################################################################