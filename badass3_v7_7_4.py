#!/usr/bin/env python

"""Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS3), Version 7.7.4, Python 3.6 Version

BADASS is an open-source spectral analysis tool designed for detailed decomposition
of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the 
fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical. 
The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte 
Carlo sampler emcee for robust parameter and uncertainty estimation, as well 
as autocorrelation analysis to access parameter chain convergence.
"""

import numpy as np
from numpy.polynomial import legendre, hermite
from numpy import linspace, meshgrid
import scipy.optimize as op
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm
import matplotlib.gridspec as gridspec
from scipy import optimize, linalg, special, fftpack
from scipy.interpolate import griddata, interp1d
from scipy.stats import kde, norm, f_oneway, kruskal, bartlett, levene, f
import scipy
from scipy.integrate import simps
from astropy.io import fits
import glob
import time
import datetime
from os import path
import os
import shutil
import sys
import emcee
from astropy.stats import mad_std
from astroquery.irsa_dust import IrsaDust
import astropy.units as u
from astropy import coordinates
from astropy.cosmology import FlatLambdaCDM
import re
import natsort
import corner
# import StringIO
import psutil
# Import BADASS tools modules
cwd = os.getcwd() # get current working directory
sys.path.insert(1,cwd+'/badass_tools/')
import badass_utils as badass_utils

plt.style.use('dark_background') # For cool tron-style dark plots
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 100000
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
import gc # Garbage collector

__author__ = "Remington O. Sexton (UCR), William Matzko (GMU), Nicholas Darden (UCR)"
__copyright__ = "Copyright (c) 2020 Remington Oliver Sexton"
__credits__ = ["Remington O. Sexton (UCR)", "William Matzko (GMU)", "Nicholas Darden (UCR)"]
__license__ = "MIT"
__version__ = "7.7.4"
__maintainer__ = "Remington O. Sexton"
__email__ = "remington.sexton@email.ucr.edu"
__status__ = "Release"

##########################################################################################################

# Note: Minor tweaks needed to port Python 2 version of BADASS into Python 3 (thanks to W. Matzko).
# - First, I had to set dtype = [("fluxes",dict)]. Without the brackets [], you get the error 
#   TypeError: data type "fluxes" not understood. Adding the additional brackets causes subsequent 
#   results to be returned in brackets. To that end, I needed to make the following changes in the 
#   flux_plots() function:

#   ~line 5180: for key in flux_blob[0][0]: --> for key in flux_blob[0][0][0]:
#   ~line 5196: same change as above
#   ~line 5200: flux_dict[key]['chain'][j,i] = flux_blob[i][j[key] --> flux_dict[key]['chain'][j,i] = flux_blob[i][j][0][key]

# 	Comment out "import StringIO" (not needed)
#	TypeError, " " -> TypeError(" ")
##########################################################################################################

# Revision History

# Versions 1-5
# - Very unstable, lots of bugs, kinda messy, not many options or features.  We've made a lot of front- and back-end changes and improvements.
# - Versions 1-4 were not very flexible, and were originally optimized for Keck LRIS spectra (See 
#	[Sexton et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)) and then optimized for large samples of SDSS spectra.
# - In Version 5 we performed a complete overhaul with more options, features.  The most improved-upon feature was the addition of autocorrelation
#   analysis for parameter chain convergence, which now produces the most robust estimates.

# Version 6
# - Improved autocorrelation analysis and options.  One can now choose the number of autocorrelation times and tolerance for convergence.  
#	Posterior sampling now restarts if solution jumps prematurely out of convergence.
# - Simplified the Jupyter Notebook control panel and layout.  Most of the BADASS machinery is now contained in the badass_v6_0.py file.
# - Output of black hole mass based on availability of broad line (based on Woo et al. (2015) (https://ui.adsabs.harvard.edu/abs/2015ApJ...801...38W/abstract)
#   H-alpha BH mass estimate, and Sexton et al. (2019) (https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract) H-beta BH mass estimate.
# - Output of systemic stellar velocity (redshift) and it's uncertainty.
# - Output of BPT diagnostic ratios and plot if both H$\alpha$ and H$\beta$ regions are fit simultaneously.
# - Minor memory leak improvements by optimizing plotting functions and deleting large arrays from memory via garbage collection.
# - Fixed issues with the outflow test function
# - Added minimum S/N option for fitting the LOSVD
# - MCMC fitting with emcee is now optional with `mcmc_fit`; one can fit using only Monte Carlo bootstrapping with any number of `max_like_niter` iterations
#   to estimate uncertainties if one does not require a fit of the LOSVD.  If you need LOSVD measurements, you still must (and *should*) use emcee.
# - One can now perform more than a single maximum likelihood fit for intial parameter values for emcee by changing `max_like_niter`, be advised this will 
#	take longer for large regions of spectra, but generally produces better initial parameter values.
# - BPT diagnostic classification includes the classic Kewley+01 & Kauffmann+03 diagram to separate starforming from AGN dominated objects, but also the [SII]
#   diagnostic to distinguish Seyferts from LINERs.  The BPT classification is now written to the log file. 
# - Store autocorrelation times and tolerances for each parameter in a dictionary and save to a `.npy` file
# - Cleaned up Notebook
# - Major changes and improvements in how monte carlo bootstrapping is performed for maximum likelihood and outflow testing functions.

# Version 7.0.0
# - Added minimum width for emission lines which improves outflow testing; this is based on the dispersion element of a single noise spike.  
# - Emission lines widths are now measured as Gaussian dispersion (disp) instead of Gaussian FWHM (fwhm).
# - Added warning flags to best fit parameter files and logfile if parameters are consistent with lower or upper limits to within 1-sigma.
# - While is is *not recommended*, one can now test for outflows in the H-alpha/[NII]/[SII] region independently of the H-beta/[OIII] region, as well as
#   fit for outflows in this region.  However, if the region includes H-beta/[OIII], then the default constraint is to still use [OIII]5007 to constrain 
#	outflow amplitude, dispersion, and velocity offset.
# - Plotting options, as well as corner plot added (defualt is *not* to output this file because there is lots of overhead)
# - More stable outflow testing and maximum likelihood estimation

# Version 7.1.0
# - Fixed a critical bug in resolution correction for emission lines
# - misc. bug fixes

# Version 7.2.0
# - Feature additions; one can suppress print output completely for use when
# running multiprocessing pool

# Version 7.3.0
# - Feature additions; Jupyter Notebook now supports multiprocessing in place
# of for loops which do not release memory.  
# - Outflow test options; outflow fitting no longer constrains velocity offset
# to be less than core (blueshifted), and now only tests for blueshifts if 
# option is selected.  Only amplitude and FHWM are constrained. 
# - Better outflow testing; test now compare outflow to no-outflow models
# to check if there is significant improvement in residuals, as well as flags
# models in which the bounds are reached and good fits cannot be determined.

# Version 7.3.1-7.3.3
# - bug fixes.

# Version 7.4.0
# - changes to how outflow tests are performed; different residual improvement metric. 
# - new default host galaxy template for non-LOSVD fitting; using MILES 10.0 Gyr SSP
# with a dispersion of 100 km/s that better matches absorption features.

# Version 7.4.1-7.4.3
# - writing outflow test metrics to log file for post-fit analysis
# - Improved outflow/max-likelihood fitting using scipy.optimize.basinhopping.
#   While basinhopping algorithm requires more runtime, it produces a significantly
#   better fit, namely for the power-law slope parameter which never varies with 
#   the SLSQP algorithm due to the fact that it is stuck in a local minima.
# - Added F-statistic (ratio of variances between no outflow and outflow model)
# - Changed default outflow statistic settings
# - Bug fixes; fixed problems with parameters in 'list' option conv_type getting
#   removed.  Now if a user-defined parameter in conv_type is wrong or removed,
#   it uses the remaining valid parameters for convergence, or defaults to 'median'.

# Version 7.5.0 - 7.5.3
# - test outflow residual statistic replaced with f-statistic (ratio-of-variances)
#   to compare model residuals.
# - added interpolation of bad pixels based on SDSS flagged pixels.
# - bug fixes

# Version 7.6.0 - 7.6.8
# - Writing no-outflow parameters from test_outflows run to log file
# - bug fixes

# Version 7.7.0
# - NLS1 support; more detailed option for FeII template fitting (fwhm and voff
#   fitting options); Lorentzian emission line profile option.
# - Kovacevic et al. 2010 FeII template added, which includes a paramter for 
# - temperature.
# - Relaxed wavelength requirement for outflow tests for higher-redshift targets

# Version 7.7.1 (MNRAS Publication Version)
# - Added statistical F-test for ascertaining confidence between single-Gaussian
#   and double-Gaussian models for the outflow test. Removed the ratio-of-variance
#   test and replaced it with a sum-of-squares of residuals ratio. 
# - Added "n_basinhop" to fit_options, which allows user to choose how many initial
#	basinhopping success iterations before a solution is achieved.  This can 
#	drastically reduce the basinhopping fit time, at the expense of fit quality.
# - Bug fixes.

# Version 7.7.2  - 7.7.4
# - Fixed problem with FeII emission lines at the edge of the fitting region
#   This is done by setting the variable edge_pad=0.
# - Fixed F-test NaN confidence bug
# - Bug fixes

##########################################################################################################

#### Run BADASS ##################################################################

def run_BADASS(file,run_dir,temp_dir,
			   fit_options,
               mcmc_options,
               comp_options,
               feii_options,
               outflow_test_options,
               plot_options,
               output_options,
               mp_options,
               ):
               
	"""
	This is the main function calls all other sub-functions in order. 
	"""

	# Check argument dictionaries
	fit_options          = badass_utils.check_fit_options(fit_options,comp_options)
	mcmc_options         = badass_utils.check_mcmc_options(mcmc_options)
	comp_options         = badass_utils.check_comp_options(comp_options)
	feii_options         = badass_utils.check_feii_options(feii_options)
	outflow_test_options = badass_utils.check_outflow_test_options(outflow_test_options)
	plot_options         = badass_utils.check_plot_options(plot_options)
	output_options       = badass_utils.check_output_options(output_options)
	mp_options           = badass_utils.check_mp_options(mp_options)

	# Unpack dictionaries
	# fit_options
	fit_reg            = fit_options['fit_reg']
	good_thresh        = fit_options['good_thresh']
	interp_bad         = fit_options['interp_bad']
	n_basinhop		   = fit_options['n_basinhop']
	test_outflows      = fit_options['test_outflows']
	outflow_test_niter = fit_options['outflow_test_niter']
	max_like_niter     = fit_options['max_like_niter']
	min_sn_losvd       = fit_options['min_sn_losvd']
	line_profile       = fit_options['line_profile']
	# mcmc_options
	mcmc_fit           = mcmc_options['mcmc_fit']
	nwalkers           = mcmc_options['nwalkers']
	auto_stop          = mcmc_options['auto_stop']
	conv_type          = mcmc_options['conv_type']
	min_samp           = mcmc_options['min_samp']
	ncor_times         = mcmc_options['ncor_times']
	autocorr_tol       = mcmc_options['autocorr_tol']
	write_iter         = mcmc_options['write_iter']
	write_thresh       = mcmc_options['write_thresh']
	burn_in            = mcmc_options['burn_in']
	min_iter           = mcmc_options['min_iter']
	max_iter           = mcmc_options['max_iter']
	# comp_options
	fit_feii           = comp_options['fit_feii']
	fit_losvd          = comp_options['fit_losvd']
	fit_host           = comp_options['fit_host']
	fit_power          = comp_options['fit_power']
	fit_broad          = comp_options['fit_broad']
	fit_narrow         = comp_options['fit_narrow']
	fit_outflows       = comp_options['fit_outflows']
	tie_narrow         = comp_options['tie_narrow']
	# plot_options
	plot_param_hist    = plot_options['plot_param_hist']
	plot_flux_hist     = plot_options['plot_flux_hist']
	plot_lum_hist      = plot_options['plot_lum_hist']
	plot_mbh_hist      = plot_options['plot_mbh_hist']
	plot_corner        = plot_options['plot_corner']
	plot_bpt           = plot_options['plot_bpt']
	# output_options   
	write_chain        = output_options['write_chain']
	print_output       = output_options['print_output']
	# mp_options
	threads 		   = mp_options['threads']

	# Start fitting process
	print('\n > Starting fit for %s' % file.split('/')[-1][:-5])
	# Start Timer for total runtime
	start_time = time.time()

	# Determine fitting region
	fit_reg,good_frac = determine_fit_reg(file,good_thresh,run_dir,fit_reg)
	if (fit_reg is None) or ((fit_reg[1]-fit_reg[0])<100.):
		print('\n Fit region too small! Moving to next object...')
		cleanup(run_dir)
		return None
	elif (good_frac < good_thresh) and (fit_reg is not None): # if fraction of good pixels is less than good_threshold, then move to next object
		print('\n Not enough good channels above threshold! Moving onto next object...')
		cleanup(run_dir)
		return None
	elif (good_frac >= good_thresh) and (fit_reg is not None):
		pass

	# Prepare SDSS spectrum + stellar templates for fitting
	lam_gal,galaxy,noise,velscale,vsyst,temp_list,z,ebv,npix,ntemp,temp_fft,npad,fwhm_gal = sdss_prepare(file,fit_reg,interp_bad,temp_dir,run_dir,plot=True)
	if print_output:

		print('\n')
		print('-----------------------------------------------------------')
		print('{0:<25}{1:<30}'.format(' file:'		   , file.split('/')[-1]				  ))
		print('{0:<25}{1:<30}'.format(' SDSS redshift:'  , '%0.5f' % z						  ))
		print('{0:<25}{1:<30}'.format(' fitting region:' , '(%d,%d)' % (fit_reg[0],fit_reg[1])  ))
		print('{0:<25}{1:<30}'.format(' velocity scale:' , '%0.2f (km/s/pixel)' % velscale	  ))
		print('{0:<25}{1:<30}'.format(' galactic E(B-V):', '%0.3f' % ebv						))
		print('-----------------------------------------------------------')
	###########################################################################################################

	# Testing for outflows
	# Issue a warning to the user about independently fitting outflows in the Ha/[NII]/[SII] region
	if ( (fit_reg[0]>=5008.) and (fit_reg[1] >= 6750.) and (fit_outflows==True) and (print_output==True)):
		print('\n *WARNING* Not including Hb/[OIII] region for fitting outflows will lead to poorly constrained outflows in Ha/[NII]/[SII] region! \
				  \n		   For best results, include the Hb/[OIII] region to constrain outflows in Ha/[NII]/[SII].\n')

	# We store the user-defined fit component options in an array that we pass to the outflow testing functions.  This allows the user to fit for outflows
	# with or without broad lines, FeII, or power-law components.  All other components are included in the fit by default.
	fit_comp_options = {'fit_feii':fit_feii, 'fit_power':fit_power, 'fit_broad':fit_broad}

	if (test_outflows==True) & (fit_outflows==True) & (outflow_test_niter>0) & (fit_reg[1]>4600.): # 
		if print_output:
			print('\n Running outflow tests.')
			print('----------------------------------------------------------------------------------------------------')
		if ( (fit_reg[0]<=4400.) & (fit_reg[1] >=5015.) ): # if Hb/[OIII] region available
			if print_output:
				print(' Testing the Hb/[OIII] region...')
			fit_outflows,outflow_res,res_sigma = outflow_test_oiii(lam_gal,galaxy,noise,run_dir,line_profile,fwhm_gal,feii_options,
																   velscale,n_basinhop,outflow_test_niter,outflow_test_options,fit_comp_options,print_output)
			if fit_outflows==True:
				if print_output:
					print('\n Outflows detected: including outflow components in fit.')
			elif fit_outflows==False:
				if print_output:
					print('\n Outflows not detected: disabling outflow components from fit.')
		elif ( (fit_reg[0]>=5008.+5.) & (fit_reg[1]>=6750.0) ): # if Hb/[OIII] region available
			if print_output:
				print(' Testing the Ha/[NII]/[SII] region...')
			fit_outflows,outflow_res,res_sigma = outflow_test_nii(lam_gal,galaxy,noise,run_dir,line_profile,fwhm_gal,feii_options,
																  velscale,n_basinhop,outflow_test_niter,outflow_test_options,fit_comp_options,print_output)
			if fit_outflows==True:
				if print_output:
					print('\n Outflows detected: including outflow components in fit.')
			elif fit_outflows==False:
				if print_output:
					print('\n Outflows not detected: disabling outflow components from fit.')

	# Generate FeII templates and host-galaxy template for emcee
	# By generating the galaxy and FeII templates before, instead of generating them with each iteration, we save a lot of time
	gal_temp = galaxy_template(lam_gal,age=10.0,print_output=print_output) 
	if (fit_feii==True):
		feii_tab = initialize_feii(lam_gal,feii_options)#,line_profile,fwhm_gal,velscale,fit_reg,feii_options,run_dir)
	elif (fit_feii==False):
		feii_tab = None

	# Initialize maximum likelihood parameters
	if print_output:
		print('\n Initializing parameters for Maximum Likelihood Fitting.')
		print('----------------------------------------------------------------------------------------------------')
	param_dict = initialize_mcmc(lam_gal,galaxy,line_profile,fwhm_gal,velscale,feii_options,run_dir,fit_reg=fit_reg,fit_type='init',
								 fit_feii=fit_feii,fit_losvd=fit_losvd,fit_host=fit_host,
								 fit_power=fit_power,fit_broad=fit_broad,
								 fit_narrow=fit_narrow,fit_outflows=fit_outflows,
								 tie_narrow=tie_narrow,print_output=print_output)

	# Peform maximum likelihood
	result_dict, comp_dict, sn = max_likelihood(param_dict,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
									 temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
									 test_outflows=False,n_basinhop=n_basinhop,outflow_test_niter=outflow_test_niter,max_like_niter=max_like_niter,
									 print_output=print_output)

	# If the total continuum level is < min_sn_losvd, disable the host-galaxy and losvd components 
	# (there is no point in fitting them because the fits will be horrible)
	if (sn<min_sn_losvd):
		if print_output:
			print('\n Continuum S/N = %0.2f' % sn)
			print('\n	 - Total continuum level < %0.1f.  Disabling host_galaxy/LOSVD components...\n' % (min_sn_losvd))
		fit_host  = True
		fit_losvd = False

	if mcmc_fit==False:
		# If not performing MCMC fitting, terminate BADASS here and write 
		# parameters, uncertainties, and components to a fits file
		write_max_like_results(result_dict,comp_dict,run_dir)
		print(' - Done fitting %s! \n' % file.split('/')[-1][:-5])
		return None

	#######################################################################################################

	# Initialize parameters for emcee
	if print_output:
		print('\n Initializing parameters for MCMC.')
		print('----------------------------------------------------------------------------------------------------')
	param_dict = initialize_mcmc(lam_gal,galaxy,line_profile,fwhm_gal,velscale,feii_options,run_dir,fit_reg=fit_reg,fit_type='final',
								 fit_feii=fit_feii,fit_losvd=fit_losvd,fit_host=fit_host,
								 fit_power=fit_power,fit_broad=fit_broad,
								 fit_narrow=fit_narrow,fit_outflows=fit_outflows,tie_narrow=tie_narrow,print_output=print_output)

	# Replace initial conditions with best fit max. likelihood parameters (the old switcharoo)
	for key in result_dict:
		if key in param_dict:
			param_dict[key]['init']=result_dict[key]['med']
	# We make an exception for FeII temperature if Kovadevic et al. (2010) templates are used because 
	# temperature is not every sensitive > 8,000 K.  This causes temperature parameter to blow up
	# during the initial max. likelihood fitting, causing it to be initialized for MCMC at an 
	# unreasonable value.  We therefroe re-initializethe FeiI temp start value to 10,000 K.
	if 'feii_temp' in param_dict:
		param_dict['feii_temp']['init']=10000.0
	
	#######################################################################################################



	# Run emcee
	if print_output:
		print('\n Performing MCMC iterations...')
		print('----------------------------------------------------------------------------------------------------')

	# Extract relevant stuff from dicts
	param_names  = [param_dict[key]['name'] for key in param_dict ]
	init_params  = [param_dict[key]['init'] for key in param_dict ]
	bounds	   = [param_dict[key]['plim'] for key in param_dict ]

	# Check number of walkers
	# If number of walkers < 2*(# of params) (the minimum required), then set it to that
	if nwalkers<2*len(param_names):
		print('\n Number of walkers < 2 x (# of parameters)!  Setting nwalkers = %d' % (2*len(param_names)))
	
	ndim, nwalkers = len(init_params), nwalkers # minimum walkers = 2*len(params)
	# initialize walker starting positions based on parameter estimation from Maximum Likelihood fitting
	pos = [init_params + 1.0e-4*np.random.randn(ndim) for i in range(nwalkers)]
	# Run emcee
	# args = arguments of lnprob (log-probability function)
	lnprob_args=(param_names,bounds,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
		  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir)
	
	sampler_chain,burn_in,flux_blob = run_emcee(pos,ndim,nwalkers,run_dir,lnprob_args,init_params,param_names,
									  			auto_stop,conv_type,min_samp,ncor_times,autocorr_tol,write_iter,write_thresh,
									  			burn_in,min_iter,max_iter,threads=threads,print_output=print_output)

	# Add chains to each parameter in param dictionary
	for k,key in enumerate(param_names):
		if key in param_dict:
			param_dict[key]['chain']=sampler_chain[:,:,k]

	if print_output:
		print('\n Fitting chains...')
		print('----------------------------------------------------------------------------------------------------')
	# These three functions produce parameter, flux, and luminosity histograms and chains from the MCMC sampling.
	# Free parameter values, uncertainties, and plots
	param_dict = param_plots(param_dict,burn_in,run_dir,plot_param_hist=plot_param_hist,print_output=print_output)
	# Corner plot
	if (plot_corner==True):
		corner_plot(sampler_chain,param_names,burn_in,run_dir,print_output=print_output)
	# Flux values, uncertainties, and plots
	flux_dict  = flux_plots(flux_blob, burn_in, nwalkers, run_dir, plot_flux_hist=plot_flux_hist,print_output=print_output)
	# Luminosity values, uncertainties, and plots
	lum_dict   = lum_plots(flux_dict, burn_in, nwalkers, z, run_dir, H0=71.0,Om0=0.27,plot_lum_hist=plot_lum_hist,print_output=print_output)
	# If stellar velocity is fit, estimate the systemic velocity of the galaxy;
	# SDSS redshifts are based on average emission line redshifts.
	extra_dict = {}
	if ('stel_vel' in param_dict):
		if print_output:
			print('\n Estimating systemic velocity of galaxy...')
			print('----------------------------------------------------------------------------------------------------')
		z_best, z_dict = systemic_vel_est(z,param_dict,burn_in,run_dir,plot_param_hist=plot_param_hist)
		if print_output:
			print('\n	 Best-fit Systemic Redshift = %0.6f (-%0.6f,+%0.6f)' %  (z_best[0],z_best[1],z_best[2]))
		# Write to log file
		write_log(z_best,'best_sys_vel',run_dir)
		# Add z_dict to extra_dict
		extra_dict['z_best'] = z_dict

	# If broadlines are fit, estimate BH mass
	if ('br_Hb_fwhm' in param_dict) and ('br_Hb_lum' in lum_dict): 
		mbh_hb_dict = {} # dictionary for storing BH masses, uncertainties, and flags
		# add to dict
		if print_output:
			print('\n Estimating black hole mass from Broad H-beta FWHM and luminosity...')
			print('----------------------------------------------------------------------------------------------------')
		L5100_Hb = hbeta_to_agn_lum(lum_dict['br_Hb_lum']['par_best'],lum_dict['br_Hb_lum']['sig_low'],lum_dict['br_Hb_lum']['sig_upp'],
									run_dir,n_samp=1000,plot_mbh_hist=plot_mbh_hist)
		if print_output:
			print('\n	 AGN Luminosity:  log10(L5100) = %0.3f (-%0.3f, +%0.3f)' % (L5100_Hb[0],L5100_Hb[1],L5100_Hb[2]))
		# Determine flags for L5100_Hb
		if ( (L5100_Hb[0]-3.0*L5100_Hb[1])<0 ):
			flag = 1
		else: flag = 0 
		L5100_Hb_dict = {'par_best':L5100_Hb[0],'sig_low':L5100_Hb[1],'sig_upp':L5100_Hb[2],'flag':flag}
		extra_dict['agn_lum_5100_Hb'] = L5100_Hb_dict
		log_MBH_Hbeta = estimate_BH_mass_hbeta(param_dict['br_Hb_fwhm']['par_best'],param_dict['br_Hb_fwhm']['sig_low'],param_dict['br_Hb_fwhm']['sig_upp'],L5100_Hb[3],
											   run_dir,n_samp=1000,plot_mbh_hist=plot_mbh_hist)
		if print_output:
			print('\n	 BH Mass:		 log10(M_BH) = %0.3f (-%0.3f, +%0.3f)' % (log_MBH_Hbeta[0],log_MBH_Hbeta[1],log_MBH_Hbeta[2]))
		# Determine flags for mbh_Hb
		if ( (log_MBH_Hbeta[0]-3.0*log_MBH_Hbeta[1])<0 ):
			flag = 1
		else: flag = 0 
		mbh_hb_dict = {'par_best':log_MBH_Hbeta[0],'sig_low':log_MBH_Hbeta[1],'sig_upp':log_MBH_Hbeta[2],'flag':flag}
		extra_dict['mbh_Hb'] = mbh_hb_dict
		# Write to log file
		write_log((L5100_Hb,log_MBH_Hbeta),'mbh_Hb',run_dir)
	if ('br_Ha_fwhm' in param_dict) and ('br_Ha_lum' in lum_dict):
		if print_output:
			print('\n Estimating black hole mass from Broad H-alpha FWHM and luminosity...')
			print('----------------------------------------------------------------------------------------------------')
		L5100_Ha = halpha_to_agn_lum(lum_dict['br_Ha_lum']['par_best'],lum_dict['br_Ha_lum']['sig_low'],lum_dict['br_Ha_lum']['sig_upp'],
									 run_dir,n_samp=1000,plot_mbh_hist=plot_mbh_hist)
		if print_output:
			print('\n	 AGN Luminosity: log10(L5100) = %0.3f (-%0.3f, +%0.3f)' % (L5100_Ha[0],L5100_Ha[1],L5100_Ha[2]))
		# Determine flags for L5100_Ha
		if ( (L5100_Ha[0]-3.0*L5100_Ha[1])<0 ):
			flag = 1
		else: flag = 0 
		L5100_Ha_dict = {'par_best':L5100_Ha[0],'sig_low':L5100_Ha[1],'sig_upp':L5100_Ha[2],'flag':flag}
		extra_dict['agn_lum_5100_Ha'] = L5100_Ha_dict
		log_MBH_Halpha = estimate_BH_mass_halpha(param_dict['br_Ha_fwhm']['par_best'],param_dict['br_Ha_fwhm']['sig_low'],param_dict['br_Ha_fwhm']['sig_upp'],L5100_Ha[3],
												 run_dir,n_samp=1000,plot_mbh_hist=plot_mbh_hist)
		if print_output:
			print('\n	 BH Mass:		 log10(M_BH) = %0.3f (-%0.3f, +%0.3f)' % (log_MBH_Halpha[0],log_MBH_Halpha[1],log_MBH_Halpha[2]))
		# Determine flags for mbh_Hb
		if ( (log_MBH_Halpha[0]-3.0*log_MBH_Halpha[1])<0 ):
			flag = 1
		else: flag = 0 
		mbh_ha_dict = {'par_best':log_MBH_Halpha[0],'sig_low':log_MBH_Halpha[1],'sig_upp':log_MBH_Halpha[2],'flag':flag}
		extra_dict['mbh_Ha'] = mbh_ha_dict
		# Write to log files
		write_log((L5100_Ha,log_MBH_Halpha),'mbh_Ha',run_dir)

	# If BPT lines are fit, output a BPT diagram
	if all(elem in lum_dict for elem in ('na_Hb_core_lum','na_oiii5007_core_lum','na_Ha_core_lum','na_nii6585_core_lum')) and (plot_bpt==True):
		if print_output:
			print('\n Generating BPT diagram...')
			print('----------------------------------------------------------------------------------------------------')
		BPT1_type, BPT2_type = bpt_diagram(lum_dict,run_dir)
		if print_output:
			if (BPT1_type==BPT2_type):
				print('	 BPT Classification: %s' % (BPT1_type))
			else: 
				print('	 BPT Classification: %s %s' % (BPT1_type, BPT2_type) )
	if print_output:
		print('\n Saving Data...')
		print('----------------------------------------------------------------------------------------------------')
	# Write best fit parameters to fits table
	write_param(param_dict,flux_dict,lum_dict,extra_dict,bounds,run_dir)

	# Write all chains to a fits table
	if (write_chain==True):
		write_chains(param_dict,flux_dict,lum_dict,run_dir)
	# Plot and save the best fit model and all sub-components
	plot_best_model(param_dict,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
						   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir)

	if print_output:
		print('\n Cleaning up...')
		print('----------------------------------------------------------------------------------------------------')
	# Delete redundant files to cut down on space
	cleanup(run_dir)

	# delete variables and manually collect garbage
	del fit_reg
	del good_frac
	del lam_gal
	del galaxy
	del noise
	del velscale
	del vsyst
	del temp_list
	del z
	del ebv
	del npix
	del ntemp
	del temp_fft
	del npad
	del gal_temp
	del feii_tab
	del result_dict
	del comp_dict
	del sn
	del param_names
	del init_params
	del bounds
	del ndim
	del nwalkers
	del pos
	del lnprob_args
	del sampler_chain
	del flux_blob
	del burn_in
	del param_dict
	del flux_dict
	lum_dict = None # won't delete for some reason: SyntaxError()
	
	# Total time
	elap_time = (time.time() - start_time)
	if print_output:
		print("\n Total Runtime = %s" % (time_convert(elap_time)))
	# Write to log
	write_log(elap_time,'total_time',run_dir)
	print(' - Done fitting %s! \n' % file.split('/')[-1][:-5])
	gc.collect()

	return None

##################################################################################

#### Corner Plot #################################################################

def corner_plot(sampler_chain,param_names,burn_in,run_dir,print_output=True):
	"""
	Creates a corner plot of all free parameters using the 
	corner.py (https://corner.readthedocs.io/en/latest/) module.  
	*WARNING* this figure can become very large due to the number of 
	free parameters, and therefore needs to be large, must be in PDF
	format, takes a long time to render, and therefore takes up a LOT
	of space for a single figure.  Output at your own risk!
	"""
	if print_output:
		print('		  \nMaking a corner plot of all free parameters...\n')
	plt.style.use('default')
	if (burn_in >= np.shape(sampler_chain)[1]):
		burn_in = int(0.5*np.shape(sampler_chain)[1])

	samples = sampler_chain[:, burn_in:, :].reshape((-1, len(param_names)))
	try:
		fig = corner.corner(samples, labels=param_names,quantiles=(0.16, 0.84), levels=(1-np.exp(-0.5),))
		plt.savefig(run_dir+'corner_plot.pdf',fmt='pdf')
	except(ValueError):
		if print_output:
			print(' WARNING: Not enough sampling to compute 1-sigma contour levels!')
		fig = corner.corner(samples, labels=param_names,quantiles=(0.16, 0.84))
		plt.savefig(run_dir+'corner_plot.pdf',fmt='pdf')
	#
	plt.close(fig)
	del sampler_chain
	del param_names
	del fig 
	gc.collect()

	plt.style.use('dark_background')

	return None



##################################################################################

#### Calculate Sysetemic Velocity ################################################

def systemic_vel_est(z,param_dict,burn_in,run_dir,plot_param_hist=True):
	"""
	Estimates the systemic (stellar) velocity of the galaxy and corrects 
	the SDSS redshift (which is based on emission lines).
	"""

	c = 299792.458   
	# Get measured stellar velocity
	stel_vel = np.array(param_dict['stel_vel']['chain'])

	# Calculate new redshift
	z_best = (z+1)*(1+stel_vel/c)-1

	# Burned-in + Flattened (along walker axis) chain
	# If burn_in is larger than the size of the chain, then 
	# take 50% of the chain length instead.
	if (burn_in >= np.shape(z_best)[1]):
		burn_in = int(0.5*np.shape(z_best)[1])
		# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

	flat = z_best[:,burn_in:]
	flat = flat.flat

	# Old confidence interval stuff; replaced by np.quantile
	p = np.percentile(flat, [16, 50, 84])
	pdfmax = p[1]
	low1   = p[1]-p[0]
	upp1   = p[2]-p[1]

	if ((pdfmax-(3.0*low1))<0): 
		flag = 1
	else: flag = 0

	if (plot_param_hist==True):
		# Initialize figures and axes
		# Make an updating plot of the chain
		fig = plt.figure(figsize=(10,8)) 
		gs = gridspec.GridSpec(2, 2)
		gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
		ax1  = plt.subplot(gs[0,0])
		ax2  = plt.subplot(gs[0,1])
		ax3  = plt.subplot(gs[1,0:2])
		# Plot 1: Histogram plots
		# Histogram; 'Doane' binning produces the best results from tests.
		n, bins, patches = ax1.hist(flat, bins='doane', density=True, facecolor='xkcd:aqua green', alpha=0.75)
		ax1.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.6f$\n' % pdfmax)
		ax1.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.6f$\n' % low1)
		ax1.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.6f$\n' % upp1)
		# ax1.plot(xvec,yvec,color='white')
		ax1.set_xlabel(r'$z_{\rm{best}}$',fontsize=12)
		ax1.set_ylabel(r'$p(z_{\rm{best}})$',fontsize=12)
		
		# Plot 2: best fit values
		ax2.axvline(pdfmax,linestyle='--',color='black',alpha=0.0,label='$\mu=%0.6f$\n' % pdfmax)
		ax2.axvline(pdfmax-low1,linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.6f$\n' % low1)
		ax2.axvline(pdfmax+upp1,linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.6f$\n' % upp1)
		ax2.legend(loc='center left',frameon=False,fontsize=14)
		ax2.axis('off')
		
		# Plot 3: Chain plot
		for w in range(0,np.shape(z_best)[0],1):
			ax3.plot(range(np.shape(z_best)[1]),z_best[w,:],color='white',linewidth=0.5,alpha=0.5,zorder=0)
		# Calculate median and median absolute deviation of walkers at each iteration; we have depreciated
		# the average and standard deviation because they do not behave well for outlier walkers, which
		# also don't agree with histograms.
		c_med = np.median(z_best,axis=0)
		c_madstd = mad_std(z_best)
		ax3.plot(range(np.shape(z_best)[1]),c_med,color='xkcd:red',alpha=1.,linewidth=2.0,label='Median',zorder=10)
		ax3.fill_between(range(np.shape(z_best)[1]),c_med+c_madstd,c_med-c_madstd,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
		ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in = %d' % burn_in)
		ax3.set_xlim(0,np.shape(z_best)[1])
		ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
		ax3.set_ylabel(r'$z_{\rm{best}}$',fontsize=12)
		ax3.legend(loc='upper left')
		
		# Save the figure
		plt.savefig(run_dir+'histogram_plots/param_histograms/'+'z_best_MCMC.png' ,bbox_inches="tight",dpi=300,fmt='png')
		
		# Close plot window
		fig.clear()
		plt.close()
		# Collect garbage
		del fig
		del ax1
		del ax2
		del ax3
		del flat
		gc.collect()

	z_dict = {'par_best':pdfmax,'sig_low':low1,'sig_upp':upp1,'chain':z_best,'flag':flag}

	#
	z_best = pdfmax
	z_best_low = low1 
	z_best_upp = upp1
	return (z_best,z_best_low,z_best_upp),z_dict

##################################################################################

#### BPT Diagram #################################################################

def bpt_diagram(lum_dict,run_dir):
	"""
	If both the H-beta and H-alpha regions are fit, produce a BPT 
	diagram and output the best classification to the log file.
	"""
	# BPT diagnostic relations
	def Kewley01(x):
		y = 0.61 / (x - 0.47) + 1.19
		return y
	
	def Kauffmann03(x):
		y = 0.61 / (x - 0.05) + 1.3
		return y
	def main_AGN(x): # Kewley et. al (2006)
		y = 0.72 / (x - 0.32) + 1.30
		return y
	
	def LINER_Sy2(x): # Kewley et. al (2006)
		y = 1.89 * x + 0.76
		return y
	# Get luminosities (or fluxes) 
	hb_lum	   = lum_dict['na_Hb_core_lum']['par_best']
	hb_lum_low   = lum_dict['na_Hb_core_lum']['sig_low']
	hb_lum_upp   = lum_dict['na_Hb_core_lum']['sig_upp']
	#
	oiii_lum	 = lum_dict['na_oiii5007_core_lum']['par_best']
	oiii_lum_low = lum_dict['na_oiii5007_core_lum']['sig_low']
	oiii_lum_upp = lum_dict['na_oiii5007_core_lum']['sig_upp']
	#
	ha_lum	   = lum_dict['na_Ha_core_lum']['par_best']
	ha_lum_low   = lum_dict['na_Ha_core_lum']['sig_low']
	ha_lum_upp   = lum_dict['na_Ha_core_lum']['sig_upp']
	#
	nii_lum	 = lum_dict['na_nii6585_core_lum']['par_best']
	nii_lum_low = lum_dict['na_nii6585_core_lum']['sig_low']
	nii_lum_upp = lum_dict['na_nii6585_core_lum']['sig_upp']
	#
	sii_lum	 = lum_dict['na_sii6732_core_lum']['par_best']
	sii_lum_low = lum_dict['na_sii6732_core_lum']['sig_low']
	sii_lum_upp = lum_dict['na_sii6732_core_lum']['sig_upp']
	#
	# Calculate log ratios 
	log_oiii_hb_ratio = np.log10(oiii_lum/hb_lum)
	log_nii_ha_ratio  = np.log10(nii_lum/ha_lum)
	log_sii_ha_ratio  = np.log10(sii_lum/ha_lum)
	# Calculate uncertainnties
	log_oiii_hb_ratio_low = 0.434*((np.sqrt((oiii_lum_low/oiii_lum)**2+(hb_lum_low/hb_lum)**2))/(oiii_lum/hb_lum))
	log_oiii_hb_ratio_upp = 0.434*((np.sqrt((oiii_lum_upp/oiii_lum)**2+(hb_lum_upp/hb_lum)**2))/(oiii_lum/hb_lum))
	log_nii_ha_ratio_low  = 0.434*((np.sqrt((nii_lum_low/nii_lum)**2+(ha_lum_low/ha_lum)**2))/(nii_lum/ha_lum))
	log_nii_ha_ratio_upp  = 0.434*((np.sqrt((nii_lum_upp/nii_lum)**2+(ha_lum_upp/ha_lum)**2))/(nii_lum/ha_lum))
	log_sii_ha_ratio_low  = 0.434*((np.sqrt((sii_lum_low/sii_lum)**2+(ha_lum_low/ha_lum)**2))/(sii_lum/ha_lum))
	log_sii_ha_ratio_upp  = 0.434*((np.sqrt((sii_lum_upp/sii_lum)**2+(ha_lum_upp/ha_lum)**2))/(sii_lum/ha_lum))
	# Plot and save figure
	fig = plt.figure(figsize=(14,7))
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)
	#
	ax1.errorbar(log_nii_ha_ratio,log_oiii_hb_ratio,
				xerr = [[log_nii_ha_ratio_low],[log_nii_ha_ratio_upp]],
				yerr = [[log_oiii_hb_ratio_low],[log_oiii_hb_ratio_upp]],
				# xerr = 0.1,
				# yerr = 0.1,
				color='xkcd:black',marker='*',markersize=15,
				elinewidth=1,ecolor='white',markeredgewidth=1,markerfacecolor='red',
				capsize=5,linestyle='None',zorder=15)
	ax2.errorbar(log_sii_ha_ratio,log_oiii_hb_ratio,
				xerr = [[log_sii_ha_ratio_low],[log_sii_ha_ratio_upp]],
				yerr = [[log_oiii_hb_ratio_low],[log_oiii_hb_ratio_upp]],
				# xerr = 0.1,
				# yerr = 0.1,
				color='xkcd:black',marker='*',markersize=15,
				elinewidth=1,ecolor='white',markeredgewidth=1,markerfacecolor='red',
				capsize=5,linestyle='None',zorder=15)
	#
	# [NII] BPT Diagnostic
	# Kewley et al. 2001
	x_k01 = np.arange(-2.0,0.30,0.01)
	y_k01 = 0.61 / (x_k01 - 0.47) + 1.19
	ax1.plot(x_k01,y_k01,linewidth=2,linestyle='--',color='xkcd:turquoise',label='Kewley et al. 2001')
	# Kauffmann et al. 2003 
	x_k03 = np.arange(-2.0,0.0,0.01)
	y_k03 = 0.61 / (x_k03 - 0.05) + 1.3
	ax1.plot(x_k03,y_k03,linewidth=2,linestyle='--',color='xkcd:lime green',label='Kauffmann et al. 2003')
	#
	ax1.annotate('AGN\nDominated', xy=(0.95, 0.70),  xycoords='axes fraction',
			xytext=(0.95, 0.70), textcoords='axes fraction',
			horizontalalignment='right', verticalalignment='top',
			fontsize=14
			)
	
	ax1.annotate('Starforming\nDominated', xy=(0.25, 0.25),  xycoords='axes fraction',
			xytext=(0.25, 0.25), textcoords='axes fraction',
			horizontalalignment='right', verticalalignment='top',
			fontsize=14
			)
	ax1.annotate('Comp.', xy=(0.72, 0.10),  xycoords='axes fraction',
			xytext=(0.72, 0.10), textcoords='axes fraction',
			horizontalalignment='right', verticalalignment='top',
			fontsize=14
			)
	#
	ax1.set_ylabel(r'$\log_{10}(\rm{[OIII]}/\rm{H}\beta)$',fontsize=14)
	ax1.set_xlabel(r'$\log_{10}(\rm{[NII]}/\rm{H}\alpha)$',fontsize=14)
	ax1.tick_params(axis="x", labelsize=14)
	ax1.tick_params(axis="y", labelsize=14)
	ax1.set_xlim(-2.0,1.0)
	ax1.set_ylim(-1.2,1.5)
	ax1.legend(fontsize=14)
	
	# [SII] BPT Diagnostic
	# Main AGN Curve
	x_agn = np.arange(-2.0,0.30,0.01)
	y_agn = 0.72 / (x_agn - 0.32) + 1.30
	ax2.plot(x_agn,y_agn,linewidth=2,linestyle='--',color='xkcd:yellow',label='Kewley et al. 2006')
	# LINER/Sy2 Line
	x_liner = np.arange(-0.31,0.25,0.01)
	y_liner = 1.89*x_liner + 0.76
	ax2.plot(x_liner,y_liner,linewidth=2,linestyle='--',color='xkcd:yellow')
	#
	ax2.annotate('Starforming\nDominated', xy=(0.25, 0.25),  xycoords='axes fraction',
			xytext=(0.25, 0.25), textcoords='axes fraction',
			horizontalalignment='right', verticalalignment='top',
			fontsize=14
			)
	ax2.annotate('LINER', xy=(0.90, 0.25),  xycoords='axes fraction',
			xytext=(0.90, 0.25), textcoords='axes fraction',
			horizontalalignment='right', verticalalignment='top',
			fontsize=14
			)
	ax2.annotate('Seyfert', xy=(0.35, 0.90),  xycoords='axes fraction',
			xytext=(0.35, 0.90), textcoords='axes fraction',
			horizontalalignment='right', verticalalignment='top',
			fontsize=14
			)
	ax2.set_ylabel(r'$\log_{10}(\rm{[OIII]}/\rm{H}\beta)$',fontsize=14)
	ax2.set_xlabel(r'$\log_{10}(\rm{[SII]}/\rm{H}\alpha)$',fontsize=14)
	ax2.tick_params(axis="x", labelsize=14)
	ax2.tick_params(axis="y", labelsize=14)
	ax2.set_xlim(-1.2,0.8)
	ax2.set_ylim(-1.2,1.5)
	ax2.legend(fontsize=14)
	#
	plt.suptitle('BPT Diagnostic Classification',fontsize=16)
	# 
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig(run_dir+'bpt_class_diagram.png',dpi=150,fmt='png')
	fig.clear()
	plt.close()
	#
	# Determine BPT classifications
	# Determine location of object on BPT diagrams for classification
	# BPT1 ([NII])
	k01_loc = Kewley01(log_nii_ha_ratio)
	k03_loc = Kauffmann03(log_nii_ha_ratio)
	if (log_oiii_hb_ratio>k01_loc):
			BPT1_type = 'AGN'
	elif (k03_loc<=log_oiii_hb_ratio<=k01_loc):
			BPT1_type = 'COMPOSITE'
	elif (log_oiii_hb_ratio<k03_loc):
			BPT1_type = 'STARFORMING'
	# BPT2 ([SII])
	mainAGN_loc  = main_AGN(log_sii_ha_ratio)
	LINERSy2_loc = LINER_Sy2(log_sii_ha_ratio)
	if (log_oiii_hb_ratio<mainAGN_loc):
			BPT2_type = 'STARFORMING'
	elif (log_oiii_hb_ratio>=mainAGN_loc) & (log_oiii_hb_ratio>LINERSy2_loc):
			BPT2_type = 'SEYFERT'
	elif (log_oiii_hb_ratio>=mainAGN_loc) & (log_oiii_hb_ratio<LINERSy2_loc):
			BPT2_type = 'LINER'
	# Write to log file
	write_log((BPT1_type,BPT2_type),'bpt_class',run_dir)
	#
	# Collect garbage
	del fig
	del ax1
	del ax2
	del lum_dict
	gc.collect()

	return BPT1_type, BPT2_type

##################################################################################

#### BH Mass Estimation ##########################################################

# The following functions are required for BH mass estimation
def normal_dist(x,mean,sigma):
	"""
	Function that computes a simple normal distribution (not the same as 
	the gaussian() function for emission line fitting).
	"""
	return 1.0/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mean)**2/(2.0*sigma**2))

def asymmetric_gaussian(mean,sigma_low,sigma_upp):
	"""
	Constructs an asymmetric gaussian from which we can generate samples for 
	formulae that have asymmetric uncertainties.
	"""
	# Create a linearly-space axis that goes out to N-sigma past the mean in either direction
	n_sigma = 3.
	n_samp = 100
	x_low = np.linspace(mean-n_sigma*sigma_low,mean+n_sigma*sigma_low,n_samp)
	x_upp = np.linspace(mean-n_sigma*sigma_upp,mean+n_sigma*sigma_upp,n_samp)
	# Generate gaussian distributions for each side
	g_low = normal_dist(x_low,mean,sigma_low)
	g_upp = normal_dist(x_upp,mean,sigma_upp)
	# Normalize each gaussian to 1.0
	g_low_norm = g_low/np.max(g_low)
	g_upp_norm = g_upp/np.max(g_upp)
	# index closest to the maximum of each gaussian
	g_low_max = find_nearest(g_low_norm,np.max(g_low_norm))[1]
	g_upp_max = find_nearest(g_upp_norm,np.max(g_upp_norm))[1]
	# Split each gaussian into its respective half
	g_low_half = g_low_norm[:g_low_max]
	x_low_half = x_low[:g_low_max]
	g_upp_half = g_upp_norm[g_low_max+1:]
	x_upp_half = x_upp[g_low_max+1:]
	# Concatenate the two halves together 
	g_merged = np.concatenate([g_low_half,g_upp_half])
	x_merged = np.concatenate([x_low_half,x_upp_half])
	# Interpolate the merged gaussian 
	g_interp = interp1d(x_merged,g_merged,kind='linear',fill_value=0.0)
	# Create new x axis to interpolate the new gaussian onto
	x_new = np.linspace(x_merged[0],x_merged[-1],n_samp)
	g_new = g_interp(x_new)
	# truncate
	cutoff = 0
	return g_new[g_new>=cutoff],x_new[g_new>=cutoff]

def hbeta_to_agn_lum(L,L_low,L_upp,run_dir,n_samp=1000,plot_mbh_hist=True):
	"""
	Calculate the AGN luminosity at 5100 A using the luminosity of 
	broad H-beta emission line using the relations from Greene & Ho 2005.
	"""
	# Create a histograms sub-folder
	if (plot_mbh_hist==True):
		if (os.path.exists(run_dir + 'histogram_plots')==False):
			os.mkdir(run_dir + 'histogram_plots')
		os.mkdir(run_dir + 'histogram_plots/BH_mass_histograms')

	# The equation used to convert broad H-beta luminosities to AGN luminosity 
	# can be found in Greene & Ho 2005 (https://ui.adsabs.harvard.edu/abs/2005ApJ...630..122G/abstract)
	# 
	# Eq. (2): L_hbeta = (1.425  +/- 0.007 ) * 1.e+42 * (L_5100  / 1.e+44)^(1.133  +/- 0.005 )
	# 		   L_5100  = (0.7315 +/- 0.0042) * 1.e+44 * (L_hbeta / 1.e+42)^(0.8826 +/- 0.0050)
	#
	# Define variables
	A	  = 0.7315 
	dA_low = 0.0042
	dA_upp = 0.0042
	#
	B	  = 0.8826
	dB_low = 0.0050
	dB_upp = 0.0050
	#
	# Create distibutions
	p_A,x_A = asymmetric_gaussian(A,dA_low,dA_upp) 
	p_A = p_A/p_A.sum() 
	p_B,x_B = asymmetric_gaussian(B,dB_low,dB_upp)
	p_B = p_B/p_B.sum()
	# Choose from distributions
	A_ = np.random.choice(a=x_A,size=n_samp,p=p_A,replace=True)
	B_ = np.random.choice(a=x_B,size=n_samp,p=p_B,replace=True)
	#
	# fig = plt.figure(figsize=(5,5))
	# ax1 = fig.add_subplot(1,1,1)
	# ax1.hist(A_,bins='doane')
	# plt.show()
	#
	# Generate samples of L_hbeta
	p_L,x_L = asymmetric_gaussian(L,L_low,L_upp) 
	p_L = p_L/p_L.sum() 
	L_hbeta = np.random.choice(a=x_L,size=n_samp,p=p_L,replace=True) * 1.e+42
	#
	L5100_ = A_ * 1.e+44 * (L_hbeta/1.e+42)**(B_)
	#
	# Remove non-physical values 
	L5100_ = L5100_[(L5100_/L5100_ == 1) & (L5100_ >= 0.0)]
	#
	# Make distribution and get best-fit MBH and uncertainties
	p = np.percentile(L5100_, [16, 50, 84])
	pdfmax = p[1]
	low1   = p[1]-p[0]
	upp1   = p[2]-p[1]
	# Plot of the L5100_Hb histogram and best-fit values
	if (plot_mbh_hist==True):
		# Initialize figures and axes
		fig = plt.figure(figsize=(10,4)) 
		gs = gridspec.GridSpec(1, 2)
		gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
		ax1  = plt.subplot(gs[0,0])
		ax2  = plt.subplot(gs[0,1])
		# Plot 1: Histogram plots
		n, bins, patches = ax1.hist(np.log10(L5100_), bins='doane', density=True, alpha=0.75,color='xkcd:dodger blue')
		ax1.axvline(np.log10(pdfmax),linestyle='--',color='white',label='$\mu=%0.3f$\n' % np.log10(pdfmax))
		ax1.axvline((np.log10(pdfmax)-0.434*(low1/pdfmax)),linestyle=':',color='white',label='$\sigma_-=%0.3f$\n' % (0.434*(low1/pdfmax)) )
		ax1.axvline((np.log10(pdfmax)+0.434*(upp1/pdfmax)),linestyle=':',color='white',label='$\sigma_+=%0.3f$\n' % (0.434*(upp1/pdfmax)) )
		# ax1.plot(xvec,yvec,color='white')
		ax1.set_xlabel(r'$L_{5100\rm{\;\AA, H}\beta}$',fontsize=8)
		ax1.set_ylabel(r'$p(L_{5100\rm{\;\AA, H}\beta})$',fontsize=8)

		# Plot 2: best fit values
		ax2.axvline(np.log10(pdfmax),linestyle='--',color='black',alpha=0.0,label='$\mu=%0.3f$\n' % np.log10(pdfmax))
		ax2.axvline((np.log10(pdfmax)-0.434*(low1/pdfmax)),linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.3f$\n' % (0.434*(low1/pdfmax)) )
		ax2.axvline((np.log10(pdfmax)+0.434*(upp1/pdfmax)),linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.3f$\n' % (0.434*(upp1/pdfmax)) )
		ax2.legend(loc='center left',frameon=False,fontsize=14)
		ax2.axis('off')

		# Save the figure
		plt.savefig(run_dir+'histogram_plots/BH_mass_histograms/L5100_Hbeta_hist.png',bbox_inches="tight",dpi=150,fmt='png')

		# Close plot
		fig.clear()
		plt.close()
		# Collect garbage
		del fig
		del ax1
		del ax2
		gc.collect
	#
	return np.log10(pdfmax),0.434*(low1/pdfmax),0.434*(upp1/pdfmax),L5100_

def halpha_to_agn_lum(L,L_low,L_upp,run_dir,n_samp=1000,plot_mbh_hist=True):
	"""
	Calculate the AGN luminosity at 5100 A using the luminosity of 
	broad H-alpha emission line using the relations from Greene & Ho 2005.
	"""
	# Create a histograms sub-folder
	if (plot_mbh_hist==True):
		if (os.path.exists(run_dir + 'histogram_plots')==False):
			os.mkdir(run_dir + 'histogram_plots')
		if (os.path.exists(run_dir + 'histogram_plots/BH_mass_histograms')==False):
			os.mkdir(run_dir + 'histogram_plots/BH_mass_histograms')
	# The equation used to convert broad H-alpha luminosities to AGN luminosity 
	# can be found in Greene & Ho 2005 (https://ui.adsabs.harvard.edu/abs/2005ApJ...630..122G/abstract)
	# 
	# Eq. (1): L_alpha = (5.25   +/- 0.02  ) * 1.e+42 * (L_5100  / 1.e+44)^(1.157  +/- 0.005 )
	#		  L_5100  = (0.2385 +/- 0.0023) * 1.e+44 * (L_alpha / 1.e+42)^(0.8643 +/- 0.0050)
	#
	# Define variables
	A	  = 0.2385
	dA_low = 0.0023
	dA_upp = 0.0023
	#
	B	  = 0.8643
	dB_low = 0.0050
	dB_upp = 0.0050
	#
	# Create distibutions
	p_A,x_A = asymmetric_gaussian(A,dA_low,dA_upp) 
	p_A = p_A/p_A.sum() 
	p_B,x_B = asymmetric_gaussian(B,dB_low,dB_upp)
	p_B = p_B/p_B.sum()
	# Choose from distributions
	A_ = np.random.choice(a=x_A,size=n_samp,p=p_A,replace=True)
	B_ = np.random.choice(a=x_B,size=n_samp,p=p_B,replace=True)
	#
	# fig = plt.figure(figsize=(5,5))
	# ax1 = fig.add_subplot(1,1,1)
	# ax1.hist(A_,bins='doane')
	# plt.show()
	#
	# Generate samples of L_hbeta
	p_L,x_L = asymmetric_gaussian(L,L_low,L_upp) 
	p_L = p_L/p_L.sum() 
	L_halpha = np.random.choice(a=x_L,size=n_samp,p=p_L,replace=True) * 1.e+42
	#
	L5100_ = A_ * 1.e+44 * (L_halpha/1.e+42)**(B_)
	#
	# Remove non-physical values 
	L5100_ = L5100_[(L5100_/L5100_ == 1) & (L5100_ >= 0.0)]
	#
	# Make distribution and get best-fit MBH and uncertainties
	p = np.percentile(L5100_, [16, 50, 84])
	pdfmax = p[1]
	low1   = p[1]-p[0]
	upp1   = p[2]-p[1]
	#
	# Plot of the L5100_Ha histogram and best-fit values
	if (plot_mbh_hist==True):
		# Initialize figures and axes
		fig = plt.figure(figsize=(10,4)) 
		gs = gridspec.GridSpec(1, 2)
		gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
		ax1  = plt.subplot(gs[0,0])
		ax2  = plt.subplot(gs[0,1])
		# Plot 1: Histogram plots
		n, bins, patches = ax1.hist(np.log10(L5100_), bins='doane', density=True, alpha=0.75,color='xkcd:red')
		ax1.axvline(np.log10(pdfmax),linestyle='--',color='white',label='$\mu=%0.3f$\n' % np.log10(pdfmax))
		ax1.axvline((np.log10(pdfmax)-0.434*(low1/pdfmax)),linestyle=':',color='white',label='$\sigma_-=%0.3f$\n' % (0.434*(low1/pdfmax)) )
		ax1.axvline((np.log10(pdfmax)+0.434*(upp1/pdfmax)),linestyle=':',color='white',label='$\sigma_+=%0.3f$\n' % (0.434*(upp1/pdfmax)) )
		# ax1.plot(xvec,yvec,color='white')
		ax1.set_xlabel(r'$L_{5100\rm{\;\AA, H}\alpha}$',fontsize=8)
		ax1.set_ylabel(r'$p(L_{5100\rm{\;\AA, H}\alpha})$',fontsize=8)

		# Plot 2: best fit values
		ax2.axvline(np.log10(pdfmax),linestyle='--',color='black',alpha=0.0,label='$\mu=%0.3f$\n' % np.log10(pdfmax))
		ax2.axvline((np.log10(pdfmax)-0.434*(low1/pdfmax)),linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.3f$\n' % (0.434*(low1/pdfmax)) )
		ax2.axvline((np.log10(pdfmax)+0.434*(upp1/pdfmax)),linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.3f$\n' % (0.434*(upp1/pdfmax)) )
		ax2.legend(loc='center left',frameon=False,fontsize=14)
		ax2.axis('off')

		# Save the figure
		plt.savefig(run_dir+'histogram_plots/BH_mass_histograms/L5100_Halpha_hist.png',bbox_inches="tight",dpi=150,fmt='png')

		# Close plot
		fig.clear()
		plt.close()
		# Collect garbage
		del fig
		del ax1
		del ax2
		gc.collect
	#
	return np.log10(pdfmax),0.434*(low1/pdfmax),0.434*(upp1/pdfmax),L5100_

def estimate_BH_mass_hbeta(fwhm,fwhm_low,fwhm_upp,L5100,run_dir,n_samp=1000,plot_mbh_hist=True):
	"""
	Estimate black hole mass based on the broad H-beta luminosity and broad H-beta width using 
	the relation from Sexton et al. 2019.
	"""
	# Generate samples of FWHM_Hbeta
	p_FWHM,x_FWHM = asymmetric_gaussian(fwhm,fwhm_low,fwhm_upp) 
	p_FWHM = p_FWHM/p_FWHM.sum() 
	FWHM_Hb = np.random.choice(a=x_FWHM,size=n_samp,p=p_FWHM,replace=True)
	#
	# Calculate BH Mass using the Sexton et al. 2019 relation (bassed on Woo et al. 2015 recalibration)
	# (https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)
	#
	# Eq. (4): M_BH = 10^(6.867,-0.153,+0.155)*(FWHM_Hb/1.e+3)^2 * (L5100/1.e+44)^(0.533,-0.033,+0.035)
	#
	# Define variables
	A	  = 6.867
	dA_low = 0.153
	dA_upp = 0.155
	#
	B	  = 0.533
	dB_low = 0.033
	dB_upp = 0.035
	#
	# Create distibutions
	p_A,x_A = asymmetric_gaussian(A,dA_low,dA_upp) 
	p_A = p_A/p_A.sum() 
	p_B,x_B = asymmetric_gaussian(B,dB_low,dB_upp)
	p_B = p_B/p_B.sum()
	# Choose from distributions
	A_ = np.random.choice(a=x_A,size=n_samp,p=p_A,replace=True)
	B_ = np.random.choice(a=x_B,size=n_samp,p=p_B,replace=True)
	#
	mask = np.where( (L5100/L5100 == 1) & (L5100 >= 0.0) )
	L5100   = L5100[mask]
	FWHM_Hb = FWHM_Hb[mask]
	A_      = A_[mask]
	B_      = B_[mask]

	MBH_ = 10**(A_) * (FWHM_Hb/1.e+3)**2 * (L5100/1.e+44)**B_
	#
	# Remove non-physical values 
	MBH_ = MBH_[(MBH_/MBH_ == 1)  & (MBH_ >= 0.0)]
	#
	# Make distribution and get best-fit MBH and uncertainties
	p = np.percentile(MBH_, [16, 50, 84])
	pdfmax = p[1]
	low1   = p[1]-p[0]
	upp1   = p[2]-p[1]
	#
	# Plot of the L5100_Ha histogram and best-fit values
	if (plot_mbh_hist==True):
		# Initialize figures and axes
		fig = plt.figure(figsize=(10,4)) 
		gs = gridspec.GridSpec(1, 2)
		gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
		ax1  = plt.subplot(gs[0,0])
		ax2  = plt.subplot(gs[0,1])
		# Plot 1: Histogram plots
		n, bins, patches = ax1.hist(np.log10(MBH_), bins='doane', density=True, alpha=0.75,color='xkcd:turquoise')
		ax1.axvline(np.log10(pdfmax),linestyle='--',color='white',label='$\mu=%0.3f$\n' % np.log10(pdfmax))
		ax1.axvline((np.log10(pdfmax)-0.434*(low1/pdfmax)),linestyle=':',color='white',label='$\sigma_-=%0.3f$\n' % (0.434*(low1/pdfmax)) )
		ax1.axvline((np.log10(pdfmax)+0.434*(upp1/pdfmax)),linestyle=':',color='white',label='$\sigma_+=%0.3f$\n' % (0.434*(upp1/pdfmax)) )
		# ax1.plot(xvec,yvec,color='white')
		ax1.set_xlabel(r'$M_{\rm{BH,\;H}\beta}$',fontsize=8)
		ax1.set_ylabel(r'$p(M_{\rm{BH,\;H}\beta})$',fontsize=8)

		# Plot 2: best fit values
		ax2.axvline(np.log10(pdfmax),linestyle='--',color='black',alpha=0.0,label='$\mu=%0.3f$\n' % np.log10(pdfmax))
		ax2.axvline((np.log10(pdfmax)-0.434*(low1/pdfmax)),linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.3f$\n' % (0.434*(low1/pdfmax)) )
		ax2.axvline((np.log10(pdfmax)+0.434*(upp1/pdfmax)),linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.3f$\n' % (0.434*(upp1/pdfmax)) )
		ax2.legend(loc='center left',frameon=False,fontsize=14)
		ax2.axis('off')

		# Save the figure
		plt.savefig(run_dir+'histogram_plots/BH_mass_histograms/MBH_Hbeta_hist.png',bbox_inches="tight",dpi=150,fmt='png')

		# Close plot
		fig.clear()
		plt.close()
		# Collect garbage
		del fig
		del ax1
		del ax2
		gc.collect
	#
	return np.log10(pdfmax),0.434*(low1/pdfmax),0.434*(upp1/pdfmax)

def estimate_BH_mass_halpha(fwhm,fwhm_low,fwhm_upp,L5100,run_dir,n_samp=1000,plot_mbh_hist=True):
	"""
	Estimate black hole mass based on the broad H-alpha luminosity and broad H-alpha width using 
	the relation from Woo et al. 2015.
	"""
	# Generate samples of FWHM_Halpha
	p_FWHM,x_FWHM = asymmetric_gaussian(fwhm,fwhm_low,fwhm_upp) 
	p_FWHM = p_FWHM/p_FWHM.sum() 
	FWHM_Ha = np.random.choice(a=x_FWHM,size=n_samp,p=p_FWHM,replace=True)
	#
	# Calculate BH Mass using the Woo et al. 2015 relation (and after a bit of math)
	# (https://ui.adsabs.harvard.edu/abs/2015ApJ...801...38W/abstract)
	#
	# M_BH = (0.8437,-0.3232,+0.5121) * 1.e+7 * (FWHM_Ha/1.e+3)^(2.06 +/- 0.06) * (L5100/1.e+44)^(0.533,-0.033,+0.035)
	# 
	#
	# Define variables
	#
	A	  = 0.8437
	dA_low = 0.3232
	dA_upp = 0.5121
	#
	B	  = 2.06
	dB_low = 0.06
	dB_upp = 0.06
	#
	C	  = 0.533
	dC_low = 0.033
	dC_upp = 0.035
	#
	# Create distibutions
	p_A,x_A = asymmetric_gaussian(A,dA_low,dA_upp) 
	p_A = p_A/p_A.sum() 
	p_B,x_B = asymmetric_gaussian(B,dB_low,dB_upp)
	p_B = p_B/p_B.sum()
	p_C,x_C = asymmetric_gaussian(C,dC_low,dC_upp)
	p_C = p_C/p_C.sum()
	# Choose from distributions
	A_ = np.random.choice(a=x_A,size=n_samp,p=p_A,replace=True)
	B_ = np.random.choice(a=x_B,size=n_samp,p=p_B,replace=True)
	C_ = np.random.choice(a=x_C,size=n_samp,p=p_C,replace=True)
	#
	mask = np.where( (L5100/L5100 == 1) & (L5100 >= 0.0) )
	L5100   = L5100[mask]
	FWHM_Ha = FWHM_Ha[mask]
	A_      = A_[mask]
	B_      = B_[mask]
	C_      = C_[mask]

	MBH_ = (A_)* 1.e+7 * (FWHM_Ha/1.e+3)**(B_) * (L5100/1.e+44)**(C_)
	#
	# Remove non-physical values 
	MBH_ = MBH_[(MBH_/MBH_ == 1)  & (MBH_ >= 0.0)]
	#
	# Make distribution and get best-fit MBH and uncertainties
	p = np.percentile(MBH_, [16, 50, 84])
	pdfmax = p[1]
	low1   = p[1]-p[0]
	upp1   = p[2]-p[1]
	#
	# Plot of the L5100_Ha histogram and best-fit values
	if (plot_mbh_hist==True):
		# Initialize figures and axes
		fig = plt.figure(figsize=(10,4)) 
		gs = gridspec.GridSpec(1, 2)
		gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
		ax1  = plt.subplot(gs[0,0])
		ax2  = plt.subplot(gs[0,1])
		# Plot 1: Histogram plots
		n, bins, patches = ax1.hist(np.log10(MBH_), bins='doane', density=True, alpha=0.75,color='xkcd:orange')
		ax1.axvline(np.log10(pdfmax),linestyle='--',color='white',label='$\mu=%0.3f$\n' % np.log10(pdfmax))
		ax1.axvline((np.log10(pdfmax)-0.434*(low1/pdfmax)),linestyle=':',color='white',label='$\sigma_-=%0.3f$\n' % (0.434*(low1/pdfmax)) )
		ax1.axvline((np.log10(pdfmax)+0.434*(upp1/pdfmax)),linestyle=':',color='white',label='$\sigma_+=%0.3f$\n' % (0.434*(upp1/pdfmax)) )
		# ax1.plot(xvec,yvec,color='white')
		ax1.set_xlabel(r'$M_{\rm{BH,\;H}\alpha}$',fontsize=8)
		ax1.set_ylabel(r'$p(M_{\rm{BH,\;H}\alpha})$',fontsize=8)

		# Plot 2: best fit values
		ax2.axvline(np.log10(pdfmax),linestyle='--',color='black',alpha=0.0,label='$\mu=%0.3f$\n' % np.log10(pdfmax))
		ax2.axvline((np.log10(pdfmax)-0.434*(low1/pdfmax)),linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.3f$\n' % (0.434*(low1/pdfmax)) )
		ax2.axvline((np.log10(pdfmax)+0.434*(upp1/pdfmax)),linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.3f$\n' % (0.434*(upp1/pdfmax)) )
		ax2.legend(loc='center left',frameon=False,fontsize=14)
		ax2.axis('off')

		# Save the figure
		plt.savefig(run_dir+'histogram_plots/BH_mass_histograms/MBH_Halpha_hist.png',bbox_inches="tight",dpi=150,fmt='png')

		# Close plot
		fig.clear()
		plt.close()
		# Collect garbage
		del fig
		del ax1
		del ax2
		gc.collect
	#
	return np.log10(pdfmax),0.434*(low1/pdfmax),0.434*(upp1/pdfmax)

##################################################################################

#### Find Nearest Function #######################################################

def find_nearest(array, value):
	"""
	This function finds the nearest value in an array and returns the 
	closest value and the corresponding index.
	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx],idx

##################################################################################


#### Convert Seconds to Minutes ##################################################

# Python Program to Convert seconds 
# into hours, minutes and seconds 
  
def time_convert(seconds): 
	"""
	Converts runtimes in seconds to hours:minutes:seconds format.
	"""
	seconds = seconds % (24. * 3600.) 
	hour = seconds // 3600.
	seconds %= 3600.
	minutes = seconds // 60.
	seconds %= 60.
	  
	return "%d:%02d:%02d" % (hour, minutes, seconds)

##################################################################################


#### Setup Directory Structure ###################################################

def setup_dirs(work_dir,print_output=True):
	"""
	This sets up the BADASS directory structure for each spectra.  It creates
	the "MCMC_output_#" folders.  
	"""

	def atoi(text):
		return int(text) if text.isdigit() else text

	def natural_keys(text):
		'''
		alist.sort(key=natural_keys) sorts in human order
		http://nedbatchelder.com/blog/200712/human_sorting.html
		(See Toothy's implementation in the comments)
		'''
		return [ atoi(c) for c in re.split('(\d+)', text) ]
	
	# Get list of folders in work_dir:
	folders = glob.glob(work_dir+'MCMC_output_*')
	folders.sort(key=natural_keys)
	if (len(folders)==0):
		if print_output:
			print(' Folder has not been created.  Creating MCMC_output folder...')
		# Create the first MCMC_output file starting with index 1
		os.mkdir(work_dir+'MCMC_output_1')
		run_dir = work_dir+'MCMC_output_1/' # running directory
		prev_dir = None
	else: 
		# Get last folder name
		s = folders[-1]
		result = re.search('MCMC_output_(.*)', s)
		# The next folder is named with this number
		fnum = str(int(result.group(1))+1)
		prev_num = str(int(result.group(1)))
		# Create the first MCMC_output file starting with index 1
		new_fold = work_dir+'MCMC_output_'+fnum+'/'
		prev_fold = work_dir+'MCMC_output_'+prev_num+'/'
		os.mkdir(new_fold)
		run_dir = new_fold
		if os.path.exists(prev_fold+'MCMC_chain.csv')==True:
			prev_dir = prev_fold
		else:
			prev_dir = prev_fold
		if print_output:
			print(' Storing MCMC_output in %s' % run_dir)

	return run_dir,prev_dir

##################################################################################


#### Determine fitting region ####################################################

def determine_upper_bound(first_good,last_good):
	"""
	Determines upper bound to the fit region for the determine_fit_reg() function.
	"""
	# Set some rules for the upper spectrum limit
	# Indo-US Library of Stellar Templates has a upper limit of 9464
	if ((last_good>=7000.) & (last_good<=9464.)) and (last_good-first_good>=500.): # cap at 7000 A
		auto_upp = last_good #7000.
	elif ((last_good>=6750.) & (last_good<=7000.)) and (last_good-first_good>=500.): # include Ha/[NII]/[SII] region
		auto_upp = last_good
	elif ((last_good>=6400.) & (last_good<=6750.)) and (last_good-first_good>=500.): # omit H-alpha/[NII] region if we can't fit all lines in region
		auto_upp = 6400.
	elif ((last_good>=5050.) & (last_good<=6400.)) and (last_good-first_good>=500.): # Full MgIb/FeII region
		auto_upp = last_good
	elif ((last_good>=4750.) & (last_good<=5025.)) and (last_good-first_good>=500.): # omit H-beta/[OIII] region if we can't fit all lines in region
		auto_upp = 4750.
	elif ((last_good>=4400.) & (last_good<=4750.)) and (last_good-first_good>=500.):
		auto_upp = last_good
	elif ((last_good>=4300.) & (last_good<=4400.)) and (last_good-first_good>=500.): # omit H-gamma region if we can't fit all lines in region
		auto_upp = 4300.
	elif ((last_good>=3500.) & (last_good<=4300.)) and (last_good-first_good>=500.): # omit H-gamma region if we can't fit all lines in region
		auto_upp = last_good
	elif (last_good-first_good>=500.):
		print('\n Not enough spectrum to fit! ')
		auto_upp = None 
	else:
		auto_upp = last_good
	return auto_upp


def determine_fit_reg(file,good_thresh,run_dir,fit_reg='auto'):
	"""
	Determine fit region based on the user input to ensure certain 
	lines are included or excluded in the fit.  This may change the 
	user-input fit region to ensure certain families of lines are all
	included.  For instance, to ensure  both H-beta and [OIII] have 
	sufficient space of either side for proper fitting.
	"""
	# Open spectrum file
	hdu = fits.open(file)
	specobj = hdu[2].data
	z = specobj['z'][0]
	t = hdu['COADD'].data
	lam_gal = (10**(t['loglam']))/(1+z)

	gal  = t['flux']
	ivar = t['ivar']
	and_mask = t['and_mask']
	# Edges of wavelength vector
	first_good = lam_gal[0]
	last_good  = lam_gal[-1]

	if ((fit_reg=='auto') or (fit_reg is None) or (fit_reg=='full')):
		# The lower limit of the spectrum must be the lower limit of our stellar templates
		auto_low = np.max([3500.,first_good]) # Indo-US Library of Stellar Templates has a lower limit of 3460
		auto_upp = determine_upper_bound(first_good,last_good)
		if (auto_upp is not None):
			new_fit_reg = (int(auto_low),int(auto_upp))	
		elif (auto_upp is None):
			new_fit_reg = None
			return None, None
	elif ((isinstance(fit_reg,tuple)==True) or (isinstance(fit_reg,list)==True) ):
		# Check to see if tuple/list makes sense
		if ((fit_reg[0]>fit_reg[1]) or (fit_reg[1]<fit_reg[0])): # if boundaries overlap
			print('\n Fitting boundary error. \n')
			new_fit_reg = None
			return None, None
		elif ((fit_reg[1]-fit_reg[0])<100.0): # if fitting region is < 500 A
			print('\n Your fitting region is suspiciously small... \n')
			new_fit_reg = None
			return None, None
		else:
			man_low = np.max([3500.,first_good,fit_reg[0]])
			man_upper_bound  = determine_upper_bound(fit_reg[0],fit_reg[1])
			man_upp = np.min([man_upper_bound,fit_reg[1],last_good])
			new_fit_reg = (int(man_low),int(man_upp))

	# Determine number of good pixels in new fitting region
	mask = ((lam_gal >= new_fit_reg[0]) & (lam_gal <= new_fit_reg[1]))
	igood = np.where((gal[mask]>0) & (ivar[mask]>0) & (and_mask[mask]==0))[0]
	ibad  = np.where(and_mask[mask]!=0)[0]
	good_frac = (len(igood)*1.0)/len(gal[mask])

	if 0:
		##################################################################################
		fig = plt.figure(figsize=(14,6))
		ax1 = fig.add_subplot(1,1,1)

		ax1.plot(lam_gal,gal,linewidth=0.5)
		ax1.axvline(new_fit_reg[0],linestyle='--',color='xkcd:yellow')
		ax1.axvline(new_fit_reg[1],linestyle='--',color='xkcd:yellow')

		ax1.scatter(lam_gal[mask][ibad],gal[mask][ibad],color='red')
		ax1.set_ylabel(r'$f_\lambda$ (erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
		ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)')

		plt.tight_layout()
		plt.savefig(run_dir+'good_pixels.pdf',fmt='pdf',dpi=150)
		fig.clear()
		plt.close()
	##################################################################################
	# Close the fits file 
	hdu.close()
	del hdu
	del specobj
	del t
	del lam_gal
	del gal
	del ivar
	del and_mask
	gc.collect()
	##################################################################################

	return new_fit_reg,good_frac

##################################################################################


#### Galactic Extinction Correction ##############################################

def ccm_unred(wave, flux, ebv, r_v=""):
	"""ccm_unred(wave, flux, ebv, r_v="")
	Deredden a flux vector using the CCM 1989 parameterization 
	Returns an array of the unreddened flux
	
	INPUTS:
	wave - array of wavelengths (in Angstroms)
	dec - calibrated flux array, same number of elements as wave
	ebv - colour excess E(B-V) float. If a negative ebv is supplied
		  fluxes will be reddened rather than dereddened	 
	
	OPTIONAL INPUT:
	r_v - float specifying the ratio of total selective
		  extinction R(V) = A(V)/E(B-V). If not specified,
		  then r_v = 3.1
			
	OUTPUTS:
	funred - unreddened calibrated flux array, same number of 
			 elements as wave
			 
	NOTES:
	1. This function was converted from the IDL Astrolib procedure
	   last updated in April 1998. All notes from that function
	   (provided below) are relevant to this function 
	   
	2. (From IDL:) The CCM curve shows good agreement with the Savage & Mathis (1979)
	   ultraviolet curve shortward of 1400 A, but is probably
	   preferable between 1200 and 1400 A.
	3. (From IDL:) Many sightlines with peculiar ultraviolet interstellar extinction 
	   can be represented with a CCM curve, if the proper value of 
	   R(V) is supplied.
	4. (From IDL:) Curve is extrapolated between 912 and 1000 A as suggested by
	   Longo et al. (1989, ApJ, 339,474)
	5. (From IDL:) Use the 4 parameter calling sequence if you wish to save the 
	   original flux vector.
	6. (From IDL:) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
	   curve (3.3 -- 8.0 um-1).	But since their revised curve does
	   not connect smoothly with longer and shorter wavelengths, it is
	   not included here.
	
	7. For the optical/NIR transformation, the coefficients from 
	   O'Donnell (1994) are used
	
	>>> ccm_unred([1000, 2000, 3000], [1, 1, 1], 2 ) 
	array([9.7976e+012, 1.12064e+07, 32287.1])
	"""
	wave = np.array(wave, float)
	flux = np.array(flux, float)
	
	if wave.size != flux.size: raise TypeError( 'ERROR - wave and flux vectors must be the same size')
	
	if not bool(r_v): r_v = 3.1 

	x = 10000.0/wave
	npts = wave.size
	a = np.zeros(npts, float)
	b = np.zeros(npts, float)
	
	###############################
	#Infrared
	
	good = np.where( (x > 0.3) & (x < 1.1) )
	a[good] = 0.574 * x[good]**(1.61)
	b[good] = -0.527 * x[good]**(1.61)
	
	###############################
	# Optical & Near IR

	good = np.where( (x  >= 1.1) & (x < 3.3) )
	y = x[good] - 1.82
	
	c1 = np.array([ 1.0 , 0.104,   -0.609,	0.701,  1.137, \
				  -1.718,   -0.827,	1.647, -0.505 ])
	c2 = np.array([ 0.0,  1.952,	2.908,   -3.989, -7.985, \
				  11.102,	5.491,  -10.805,  3.347 ] )

	a[good] = np.polyval(c1[::-1], y)
	b[good] = np.polyval(c2[::-1], y)

	###############################
	# Mid-UV
	
	good = np.where( (x >= 3.3) & (x < 8) )   
	y = x[good]
	F_a = np.zeros(np.size(good),float)
	F_b = np.zeros(np.size(good),float)
	good1 = np.where( y > 5.9 )	
	
	if np.size(good1) > 0:
		y1 = y[good1] - 5.9
		F_a[ good1] = -0.04473 * y1**2 - 0.009779 * y1**3
		F_b[ good1] =   0.2130 * y1**2  +  0.1207 * y1**3

	a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
	b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b
	
	###############################
	# Far-UV
	
	good = np.where( (x >= 8) & (x <= 11) )   
	y = x[good] - 8.0
	c1 = [ -1.073, -0.628,  0.137, -0.070 ]
	c2 = [ 13.670,  4.257, -0.420,  0.374 ]
	a[good] = np.polyval(c1[::-1], y)
	b[good] = np.polyval(c2[::-1], y)

	# Applying Extinction Correction
	
	a_v = r_v * ebv
	a_lambda = a_v * (a + b/r_v)
	
	funred = flux * 10.0**(0.4*a_lambda)   

	return funred #,a_lambda



#### Prepare SDSS spectrum for pPXF ################################################

def sdss_prepare(file,fit_reg,interp_bad,temp_dir,run_dir,plot=False):
	"""
	Adapted from example from Cappellari's pPXF (Cappellari et al. 2004,2017)
	Prepare an SDSS spectrum for pPXF, returning all necessary 
	parameters. 
	
	file: fully-specified path of the spectrum 
	z: the redshift; we use the SDSS-measured redshift
	fit_reg: (min,max); tuple specifying the minimum and maximum 
			wavelength bounds of the region to be fit. 
	
	"""

	def nan_helper(y):
		"""
		Helper to handle indices and logical indices of NaNs.

		Input:
		    - y, 1d numpy array with possible NaNs
		Output:
		    - nans, logical indices of NaNs
		    - index, a function, with signature indices= index(logical_indices),
		      to convert logical indices of NaNs to 'equivalent' indices
		Example:
		    >>> # linear interpolation of NaNs
		    >>> nans, x= nan_helper(y)
		    >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
		"""

		return np.isnan(y), lambda z: z.nonzero()[0]

	def insert_nan(spec,ibad):
		"""
		Inserts additional NaN values to neighboriing ibad pixels.
		"""
		all_bad = np.unique(np.concatenate([ibad-1,ibad,ibad+1]))
		ibad_new = []
		for i in all_bad:
			if (i>0) & (i<len(spec)):
				ibad_new.append(i)
		ibad_new = np.array(ibad_new)
		try:
			spec[ibad_new] = np.nan
			return spec
		except:
			return spec

	# Load the data
	hdu = fits.open(file)

	specobj = hdu[2].data
	z = specobj['z'][0]
	try:
		ra  = hdu[0].header['RA']
		dec = hdu[0].header['DEC']
	except:
		ra = specobj['PLUG_RA'][0]
		dec = specobj['PLUG_DEC'][0]

	t = hdu['COADD'].data
	hdu.close()
	del hdu

	# Only use the wavelength range in common between galaxy and stellar library.
	# Determine limits of spectrum vs templates
	# mask = ( (t['loglam'] > np.log10(3540)) & (t['loglam'] < np.log10(7409)) )
	fit_min,fit_max = float(fit_reg[0]),float(fit_reg[1])
	mask = ( (t['loglam'] > np.log10(fit_min*(1+z))) & (t['loglam'] < np.log10(fit_max*(1+z))) )
	
	# Unpack the spectra
	galaxy = t['flux'][mask]
	# SDSS spectra are already log10-rebinned
	loglam_gal = t['loglam'][mask] # This is the observed SDSS wavelength range, NOT the rest wavelength range of the galaxy
	lam_gal = 10**loglam_gal
	ivar = t['ivar'][mask] # inverse variance
	noise = np.sqrt(1.0/ivar) # 1-sigma spectral noise
	and_mask = t['and_mask'] # bad pixels 
	ibad  = np.where(and_mask[mask]!=0)[0]

	### Interpolating over bad pixels ############################

	# Get locations of nan or -inf pixels
	nan_gal   = np.where(galaxy/galaxy!=1)[0]
	nan_noise = np.where(noise/noise!=1)[0]
	inan = np.unique(np.concatenate([nan_gal,nan_noise]))

	# Interpolate over nans and infs if in galaxy or noise
	if 1: 
		noise[inan] = np.nan
		noise[inan] = np.nanmedian(noise)

	# Iterpolate over bad pixels
	if interp_bad:
		spec = insert_nan(galaxy,ibad)
		nans, x= nan_helper(galaxy)
		galaxy[nans]= np.interp(x(nans), x(~nans), galaxy[~nans])
		noise = insert_nan(noise,ibad)
		nans, x= nan_helper(noise)
		noise[nans]= np.interp(x(nans), x(~nans), noise[~nans])

	###############################################################

	c = 299792.458				  # speed of light in km/s
	frac = lam_gal[1]/lam_gal[0]	# Constant lambda fraction per pixel
	dlam_gal = (frac - 1)*lam_gal   # Size of every pixel in Angstrom
	# print('\n Size of every pixel: %s (A)' % dlam_gal)
	wdisp = t['wdisp'][mask]		# Intrinsic dispersion of every pixel, in pixels units
	fwhm_gal = 2.355*wdisp*dlam_gal # Resolution FWHM of every pixel, in Angstroms
	velscale = np.log(frac)*c	   # Constant velocity scale in km/s per pixel

	# If the galaxy is at significant redshift, one should bring the galaxy
	# spectrum roughly to the rest-frame wavelength, before calling pPXF
	# (See Sec2.4 of Cappellari 2017). In practice there is no
	# need to modify the spectrum in any way, given that a red shift
	# corresponds to a linear shift of the log-rebinned spectrum.
	# One just needs to compute the wavelength range in the rest-frame
	# and adjust the instrumental resolution of the galaxy observations.
	# This is done with the following three commented lines:
	#
	lam_gal = lam_gal/(1+z)  # Compute approximate restframe wavelength
	fwhm_gal = fwhm_gal/(1+z)   # Adjust resolution in Angstrom
	# We pass this interp1d class to the fit_model function to correct for 
	# the instrumental resolution of emission lines in our model
	# fwhm_gal_ftn = interp1d(lam_gal,fwhm_gal,kind='linear',bounds_error=False,fill_value=(0,0)) 

	val,idx = find_nearest(lam_gal,5175)

	# Read the list of filenames from the Single Stellar Population library
	# by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
	# of the library is included for this example with permission
	# num_temp = 10 # number of templates
	# temp_list = glob.glob(temp_dir + '/Mun1.30Z*.fits')#[:num_temp]
	temp_list = glob.glob(temp_dir + '/*.fits')#[:num_temp]

	temp_list = natsort.natsorted(temp_list) # Sort them in the order they appear in the directory
	# fwhm_tem = 2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
	fwhm_tem = 1.35 # Indo-US Template Library FWHM

	# Extract the wavelength range and logarithmically rebin one spectrum
	# to the same velocity scale of the SDSS galaxy spectrum, to determine
	# the size needed for the array which will contain the template spectra.
	#
	hdu = fits.open(temp_list[0])
	ssp = hdu[0].data
	h2 = hdu[0].header
	hdu.close()
	del hdu
	lam_temp = np.array(h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1']))
	# By cropping the templates we save some fitting time
	mask_temp = ( (lam_temp > (fit_min-200.)) & (lam_temp < (fit_max+200.)) )
	ssp = ssp[mask_temp]
	lam_temp = lam_temp[mask_temp]

	lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
	sspNew = log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
	templates = np.empty((sspNew.size, len(temp_list)))
	
	# Interpolates the galaxy spectral resolution at the location of every pixel
	# of the templates. Outside the range of the galaxy spectrum the resolution
	# will be extrapolated, but this is irrelevant as those pixels cannot be
	# used in the fit anyway.
	fwhm_gal_interp = np.interp(lam_temp, lam_gal, fwhm_gal)
	# Convolve the whole Vazdekis library of spectral templates
	# with the quadratic difference between the SDSS and the
	# Vazdekis instrumental resolution. Logarithmically rebin
	# and store each template as a column in the array TEMPLATES.
	
	# Quadratic sigma difference in pixels Vazdekis --> SDSS
	# The formula below is rigorously valid if the shapes of the
	# instrumental spectral profiles are well approximated by Gaussians.
	#
	# In the line below, the fwhm_dif is set to zero when fwhm_gal < fwhm_tem.
	# In principle it should never happen and a higher resolution template should be used.
	#
	fwhm_dif = np.sqrt((fwhm_gal_interp**2 - fwhm_tem**2).clip(0))
	sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

	for j, fname in enumerate(temp_list):
		hdu = fits.open(fname)
		ssp = hdu[0].data
		ssp = ssp[mask_temp]
		ssp = gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
		sspNew,loglam_temp,velscale_temp = log_rebin(lamRange_temp, ssp, velscale=velscale)#[0]
		templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
		hdu.close()
		del hdu

	# The galaxy and the template spectra do not have the same starting wavelength.
	# For this reason an extra velocity shift DV has to be applied to the template
	# to fit the galaxy spectrum. We remove this artificial shift by using the
	# keyword VSYST in the call to PPXF below, so that all velocities are
	# measured with respect to DV. This assume the redshift is negligible.
	# In the case of a high-redshift galaxy one should de-redshift its
	# wavelength to the rest frame before using the line below (see above).
	#
	dv = np.log(lam_temp[0]/lam_gal[0])*c	# km/s
	vsyst = dv

	# Here the actual fit starts. The best fit is plotted on the screen.
	# Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
	#
	vel = 0.0#c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
	start = [vel, 200.,0.0,0.0]  # (km/s), starting guess for [V, sigma]

	#################### Correct for galactic extinction ##################

	co = coordinates.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
	try: 
		table = IrsaDust.get_query_table(co,section='ebv')
		ebv = table['ext SandF mean'][0]
	except: 
		ebv = 0.04
	galaxy = ccm_unred(lam_gal,galaxy,ebv)

	#######################################################################

	npix = galaxy.shape[0] # number of output pixels
	ntemp = np.shape(templates)[1]# number of templates
	
	# Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
	temp_fft,npad = template_rfft(templates) # we will use this throughout the code

	################################################################################   

	if plot: 
		sdss_prepare_plot(lam_gal,galaxy,noise,loglam_temp,templates,run_dir)

	################################################################################
	# Write to Log 
	write_log((file,ra,dec,z,fit_min,fit_max,velscale,ebv),0,run_dir)
	################################################################################
	# Collect garbage
	del specobj
	del t
	del templates
	gc.collect()
	################################################################################

	return lam_gal,galaxy,noise,velscale,vsyst,temp_list,z,ebv,npix,ntemp,temp_fft,npad,fwhm_gal

##################################################################################

def sdss_prepare_plot(lam_gal,galaxy,noise,loglam_temp,templates,run_dir):
	# Plot the galaxy+ templates
	fig = plt.figure(figsize=(10,6))
	ax1 = fig.add_subplot(2,1,1)
	ax2 = fig.add_subplot(2,1,2)
	ax1.plot(lam_gal,galaxy,label='Galaxy',linewidth=0.5)
	ax1.plot(lam_gal,noise,label='Error Spectrum',linewidth=0.5,color='white')
	ax1.axhline(0.0,color='white',linewidth=0.5,linestyle='--')
	ax2.plot(np.exp(loglam_temp),templates[:,:],alpha=0.5,label='Template',linewidth=0.5)
	ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)',fontsize=12)
	ax2.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)',fontsize=12)
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=10)
	ax2.set_ylabel(r'Normalized Flux',fontsize=10)
	ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
	ax2.set_xlim(np.min(lam_gal),np.max(lam_gal))
	ax1.legend(loc='best')
	plt.tight_layout()
	plt.savefig(run_dir+'sdss_prepare',dpi=150,fmt='png')
	ax1.clear()
	ax2.clear()
	fig.clear()
	plt.close(fig)
	# Collect garbage
	del ax1
	del ax2
	del fig
	del lam_gal
	del galaxy
	del noise
	del loglam_temp
	del templates
	gc.collect()
	#
	return

##################################################################################


#### Initialize Parameters #######################################################


def initialize_mcmc(lam_gal,galaxy,line_profile,fwhm_gal,velscale,feii_options,run_dir,fit_reg,fit_type='init',fit_feii=True,fit_losvd=True,fit_host=True,
					fit_power=True,fit_broad=True,fit_narrow=True,fit_outflows=True,tie_narrow=True,print_output=True):
	"""
	Initializes all free parameters to be fit based on the fitting region and estimate some reasonable
	initial values based on the data itself.
	"""
	# Issue warnings for dumb options
	if ((fit_narrow==False) & (fit_outflows==True)): # why would you fit outflow without narrow lines?
		raise ValueError('\n Why would you fit outflows without narrow lines? Turn on narrow line component! \n')

	# For emission lines fwhm and stellar dispersion, resolution is already corrected for, so the minimum is technically zero.
	# However, this usually leads to unrealistic fitting of noise spikes.  A noise spike is defined as a single pixel spike, which
	# has a minimum velocity width of "velscale".  This corresponds to a minimal dispersion, so the minimal FWHM is 2.3548*velscale.
	#  We include a dictionary key 'min_good' to desinate the minimum value 
	# (resolution limit) so we can flag any values that fall below this at the end.
	min_fwhm = velscale # km/s

	################################################################################
	# Initial conditions for some parameters
	max_flux = np.max(galaxy)
	total_flux_init = np.median(galaxy)#np.median(galaxy[(lam_gal>5025.) & (lam_gal<5800.)])
	# cont_flux_init = 0.01*(np.median(galaxy))
	feii_flux_init= (0.1*np.median(galaxy))

	if (((fit_reg[0]+25) < 6085. < (fit_reg[1]-25))==True):
		fevii6085_amp_init= (np.max(galaxy[(lam_gal>6085.-25.) & (lam_gal<6085.+25.)]))
	if (((fit_reg[0]+25) < 5722. < (fit_reg[1]-25))==True):
		fevii5722_amp_init= (np.max(galaxy[(lam_gal>5722.-25.) & (lam_gal<5722.+25.)]))
	if (((fit_reg[0]+25) < 6302. < (fit_reg[1]-25))==True):
		oi_amp_init= (np.max(galaxy[(lam_gal>6302.-25.) & (lam_gal<6302.+25.)]))
	if (((fit_reg[0]+25) < 3727. < (fit_reg[1]-25))==True):
		oii_amp_init= (np.max(galaxy[(lam_gal>3727.-25.) & (lam_gal<3727.+25.)]))
	if (((fit_reg[0]+25) < 3870. < (fit_reg[1]-25))==True):
		neiii_amp_init= (np.max(galaxy[(lam_gal>3870.-25.) & (lam_gal<3870.+25.)]))
	if (((fit_reg[0]+25) < 4102. < (fit_reg[1]-25))==True):
		hd_amp_init= (np.max(galaxy[(lam_gal>4102.-25.) & (lam_gal<4102.+25.)]))
	if (((fit_reg[0]+25) < 4341. < (fit_reg[1]-25))==True):
		hg_amp_init= (np.max(galaxy[(lam_gal>4341.-25.) & (lam_gal<4341.+25.)]))
	if (((fit_reg[0]+25) < 4862. < (fit_reg[1]-25))==True):
		hb_amp_init= (np.max(galaxy[(lam_gal>4862.-25.) & (lam_gal<4862.+25.)]))
	if (((fit_reg[0]+25) < 5007. < (fit_reg[1]-25))==True):
		oiii5007_amp_init = (np.max(galaxy[(lam_gal>5007.-25.) & (lam_gal<5007.+25.)]))
	if (((fit_reg[0]+25) < 6564. < (fit_reg[1]-25))==True):
		ha_amp_init = (np.max(galaxy[(lam_gal>6564.-25.) & (lam_gal<6564.+25.)]))
	if (((fit_reg[0]+25) < 6725. < (fit_reg[1]-25))==True):
		sii_amp_init = (np.max(galaxy[(lam_gal>6725.-15.) & (lam_gal<6725.+15.)]))
	################################################################################
	
	mcmc_input = {} # dictionary of parameter dictionaries

	if (tie_narrow==True):
		if print_output:
			print('	 - Tying narrow line widths.')

	#### Host Galaxy ###############################################################
	# Galaxy template amplitude
	if ((fit_type=='init') or ((fit_type=='final') and (fit_losvd==False))) and (fit_host==True):
		if print_output:
			print('	 - Fitting a host-galaxy template.')

		mcmc_input['gal_temp_amp'] = ({'name':'gal_temp_amp',
									   'label':'$A_\mathrm{gal}$',
									   'init':0.5*total_flux_init,
									   'plim':(1.0e-3,max_flux),
									   'pcolor':'blue',
									   })

	# Stellar velocity
	if ((fit_type=='final') and (fit_losvd==True)):
		if print_output:
			print('	 - Fitting the stellar LOSVD.')
		mcmc_input['stel_vel'] = ({'name':'stel_vel',
						   		   'label':'$V_*$',
						   		   'init':100. ,
						   		   'plim':(-500.,500.),
						   		   'pcolor':'blue',
						   		   })
		# Stellar velocity dispersion
		mcmc_input['stel_disp'] = ({'name':'stel_disp',
						   			'label':'$\sigma_*$',
						   			'init':100.0,
						   			'plim':(15.0,500.),
						   			'pcolor':'dodgerblue',
						   			'min_width':min_fwhm,
						   			})
	##############################################################################

	#### AGN Power-Law ###########################################################
	if (fit_power==True):
		if print_output:
			print('	 - Fitting AGN power-law continuum.')
		# AGN simple power-law amplitude
		mcmc_input['power_amp'] = ({'name':'power_amp',
						   		   'label':'$A_\mathrm{power}$',
						   		   'init':(0.5*total_flux_init),
						   		   'plim':(1.0e-3,max_flux),
						   		   'pcolor':'orangered',
						   		   })
		# AGN simple power-law slope
		mcmc_input['power_slope'] = ({'name':'power_slope',
						   			 'label':'$m_\mathrm{power}$',
						   			 'init':-1.0  ,
						   			 'plim':(-4.0,2.0),
						   			 'pcolor':'salmon',
						   			 })
		
	##############################################################################

	#### FeII Templates ##########################################################
	if (fit_feii==True) & (feii_options['template']['type']=='VC04'):
		# Veron-Cerry et al. 2004 2-8 Parameter FeII template
		if print_output:
			print('	 - Fitting broad and narrow FeII using Veron-Cetty et al. (2004) optical FeII templates')
		if (feii_options['amp_const']['bool']==False):
			if print_output:
				print('     		* varying FeII amplitudes')
			# Narrow FeII amplitude
			mcmc_input['na_feii_amp'] = ({'name'  :'na_feii_amp',
							   			  'label' :'$A_{\mathrm{Na\;FeII}}$',
							   			  'init'  :feii_flux_init,
							   			  'plim'  :(1.0e-3,total_flux_init),
							   			  'pcolor':'sandybrown',
							   			  })
			# Broad FeII amplitude
			mcmc_input['br_feii_amp'] = ({'name'  :'br_feii_amp',
							   			  'label' :'$A_{\mathrm{Br\;FeII}}$',
							   			  'init'  :feii_flux_init,
							   			  'plim'  :(1.0e-3,total_flux_init),
							   			  'pcolor':'darkorange',
							   			  })
		if (feii_options['fwhm_const']['bool']==False):
			if print_output:
				print('     		* varying FeII fwhm')
			# Narrow FeII FWHM
			mcmc_input['na_feii_fwhm'] = ({'name' :'na_feii_fwhm',
							   			  'label' :'FWHM$_{\mathrm{Na\;FeII}}$',
							   			  'init'  :500.0,
							   			  'plim'  :(100.0,1000.0),
							   			  'pcolor':'sandybrown',
							   			  })
			# Broad FeII FWHM
			mcmc_input['br_feii_fwhm'] = ({'name' :'br_feii_fwhm',
							   			  'label' :'FWHM$_{\mathrm{Br\;FeII}}$',
							   			  'init'  :3000.0,
							   			  'plim'  :(1000.0,5000.0),
							   			  'pcolor':'darkorange',
							   			  })
		if (feii_options['voff_const']['bool']==False):
			if print_output:
				print('     		* varying FeII voff')
			# Narrow FeII VOFF
			mcmc_input['na_feii_voff'] = ({'name' :'na_feii_voff',
							   			  'label' :'VOFF$_{\mathrm{Na\;FeII}}$',
							   			  'init'  :0.0,
							   			  'plim'  :(-1000.0,1000.0),
							   			  'pcolor':'sandybrown',
							   			  })
			# Broad FeII VOFF
			mcmc_input['br_feii_voff'] = ({'name' :'br_feii_voff',
							   			  'label' :'VOFF$_{\mathrm{Br\;FeII}}$',
							   			  'init'  :0.0,
							   			  'plim'  :(-1000.0,1000.0),
							   			  'pcolor':'darkorange',
							   			  })
	elif (fit_feii==True) & (feii_options['template']['type']=='K10'):
		if print_output:
			print('	 - Fitting optical template from Kovacevic et al. (2010)')

		# Kovacevic et al. 2010 7-parameter FeII template (for NLS1s and BAL QSOs)
		# Consits of 7 free parameters
		#	- 4 amplitude parameters for S,F,G,IZw1 line families
		#	- 1 Temperature parameter determines relative intensities (5k-15k Kelvin)
		#	- 1 FWHM parameter
		#	- 1 VOFF parameter
		# 	- all lines modeled as Gaussians
		# Narrow FeII amplitude
		if (feii_options['amp_const']['bool']==False):
			mcmc_input['feii_f_amp'] = ({'name'  :'feii_f_amp',
							   			 'label' :'$A_{\mathrm{FeII,\;}F}$',
							   			 'init'  :feii_flux_init,
							   			 'plim'  :(1.0e-3,total_flux_init),
							   			 'pcolor':'xkcd:rust orange',
							   			})
			mcmc_input['feii_s_amp'] = ({'name'  :'feii_s_amp',
							   			 'label' :'$A_{\mathrm{FeII,\;}S}$',
							   			 'init'  :feii_flux_init,
							   			 'plim'  :(1.0e-3,total_flux_init),
							   			 'pcolor':'xkcd:rust orange',
							   			})
			mcmc_input['feii_g_amp'] = ({'name'  :'feii_g_amp',
							   			 'label' :'$A_{\mathrm{FeII,\;}G}$',
							   			 'init'  :feii_flux_init,
							   			 'plim'  :(1.0e-3,total_flux_init),
							   			 'pcolor':'xkcd:rust orange',
							   			})
			mcmc_input['feii_z_amp'] = ({'name'  :'feii_z_amp',
							   			    'label' :'$A_{\mathrm{FeII,\;IZw1}}$',
							   			    'init'  :feii_flux_init,
							   			    'plim'  :(1.0e-3,total_flux_init),
							   			    'pcolor':'xkcd:rust orange',
							   			   })
		if (feii_options['fwhm_const']['bool']==False):
			# FeII FWHM
			mcmc_input['feii_fwhm'] = ({'name' :'feii_fwhm',
							   		    'label' :'FWHM$_{\mathrm{FeII}}$',
							   		    'init'  :1000.0,
							   		    'plim'  :(100.0,5000.0),
							   		    'pcolor':'xkcd:rust orange',
							   		   })
		if (feii_options['voff_const']['bool']==False):
			# Narrow FeII amplitude
			mcmc_input['feii_voff'] = ({'name' :'feii_voff',
							   			'label' :'VOFF$_{\mathrm{FeII}}$',
							   			'init'  :0.0,
							   			'plim'  :(-1000.0,1000.0),
							   			'pcolor':'xkcd:rust orange',
							   		   })
		if (feii_options['temp_const']['bool']==False):
			mcmc_input['feii_temp'] = ({'name'  :'feii_temp',
							   		 'label' :'$T_{\mathrm{FeII}}$',
							   		 'init'  :10000.0,
							   		 'plim'  :(2000.0,25000.0),
							   		 'pcolor':'xkcd:rust orange',
							   		})


	##############################################################################

	#### Emission Lines ##########################################################

	#### Narrow [OII] Doublet ##############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 3727.092 < (fit_reg[1]-25.))==True))):
		if print_output:
			print('	 - Fitting narrow H-delta emission.')
		# Na. [OII]3727 Core Amplitude
		mcmc_input['na_oii3727_core_amp'] = ({'name':'na_oii3727_core_amp',
						   					'label':'$A_{\mathrm{[OII]3727}}$',
						   					'init':(oii_amp_init-total_flux_init),
						   					'plim':(1.0e-3,max_flux),
						   					'pcolor':'green',
						   					})
		if (tie_narrow==False):
			# Na. [OII]3727 Core FWHM
			mcmc_input['na_oii3727_core_fwhm'] = ({'name':'na_oii3727_core_fwhm',
							   					 'label':'$\mathrm{FWHM}_{\mathrm{[OII]3727}}$',
							   					 'init':250.,
							   					 'plim':(min_fwhm,650.),
							   					 'pcolor':'limegreen',
							   					 'min_width':min_fwhm,
							   					 })
		# Na. [OII]3727 Core VOFF
		mcmc_input['na_oii3727_core_voff'] = ({'name':'na_oii3727_core_voff',
						   					 'label':'$\mathrm{VOFF}_{\mathrm{[OII]3727}}$',
						   					 'init':0.,
						   					 'plim':(-1000.,1000.),
						   					 'pcolor':'palegreen',
						   					 })
		# Na. [OII]3729 Core Amplitude
		mcmc_input['na_oii3729_core_amp'] = ({'name':'na_oii3729_core_amp',
						   					'label':'$A_{\mathrm{[OII]3729}}$',
						   					'init':(oii_amp_init-total_flux_init),
						   					'plim':(1.0e-3,max_flux),
						   					'pcolor':'green',
						   					})

	###################################################################
	#### Narrow [NeIII]3870 ##############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 3869.81 < (fit_reg[1]-25.))==True))):
		if print_output:
			print('	 - Fitting narrow [NIII]3870 emission.')
		# Na. [NIII]3870 Core Amplitude
		mcmc_input['na_neiii_core_amp'] = ({'name':'na_neiii_core_amp',
						   					'label':'$A_{\mathrm{[NeIII]}}$',
						   					'init':(neiii_amp_init-total_flux_init),
						   					'plim':(1.0e-3,max_flux),
						   					'pcolor':'green',
						   					})
		if (tie_narrow==False):
			# Na. [NIII]3870 Core FWHM
			mcmc_input['na_neiii_core_fwhm'] = ({'name':'na_neiii_core_fwhm',
							   					 'label':'$\mathrm{FWHM}_{\mathrm{[NeIII]}}$',
							   					 'init':250.,
							   					 'plim':(min_fwhm,650.),
							   					 'pcolor':'limegreen',
							   					 'min_width':min_fwhm,
							   					 })
		# Na. [NIII]3870 Core VOFF
		mcmc_input['na_neiii_core_voff'] = ({'name':'na_neiii_core_voff',
						   					 'label':'$\mathrm{VOFF}_{\mathrm{[NeIII]}}$',
						   					 'init':0.,
						   					 'plim':(-1000.,1000.),
						   					 'pcolor':'palegreen',
						   					 })

	###################################################################

	#### Narrow H-delta ###############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 4102.89 < (fit_reg[1]-25.))==True))):
		if print_output:
			print('	 - Fitting narrow H-delta emission.')
		# Na. H-delta Core Amplitude
		mcmc_input['na_Hd_amp'] = ({'name':'na_Hd_amp',
						   					'label':'$A_{\mathrm{H}\delta}$',
						   					'init':(hd_amp_init-total_flux_init),
						   					'plim':(1.0e-3,max_flux),
						   					'pcolor':'green',
						   					})
		if (tie_narrow==False):
			# Na. H-delta Core FWHM
			mcmc_input['na_Hd_fwhm'] = ({'name':'na_Hd_fwhm',
							   					 'label':'$\mathrm{FWHM}_{\mathrm{H}\delta}$',
							   					 'init':250.,
							   					 'plim':(min_fwhm,650.),
							   					 'pcolor':'limegreen',
							   					 'min_width':min_fwhm,
							   					 })
		# Na. H-delta Core VOFF
		mcmc_input['na_Hd_voff'] = ({'name':'na_Hd_voff',
						   					 'label':'$\mathrm{VOFF}_{\mathrm{H}\delta}$',
						   					 'init':0.,
						   					 'plim':(-1000.,1000.),
						   					 'pcolor':'palegreen',
						   					 })

	##############################################################################

	#### Narrow H-gamma/[OIII]4363 ###############################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 4341.68 < (fit_reg[1]-25.))==True))):
		if print_output:
			print('	 - Fitting narrow H-gamma/[OIII]4363 emission.')

		if (tie_narrow==False):
			# Na. H-gamma Core FWHM
			mcmc_input['na_Hg_fwhm'] = ({'name':'na_Hg_fwhm',
							   					 'label':'$\mathrm{FWHM}_{\mathrm{H}\gamma}$',
							   					 'init':250.,
							   					 'plim':(0.0,1000.),
							   					 'pcolor':'limegreen',
							   					 'min_width':min_fwhm,
							   					 })
			# Na. [OIII]4363 Core FWHM
			mcmc_input['na_oiii4363_core_fwhm'] = ({'name':'na_oiii4363_core_fwhm',
							   					 'label':'$\mathrm{FWHM}_\mathrm{[OIII]4363\;Core}$',
							   					 'init':250.,
							   					 'plim':(min_fwhm,650.),
							   					 'pcolor':'limegreen',
							   					 'min_width':min_fwhm,
							   					 })
		if (tie_narrow==True) and (fit_reg[1]+25<=4600.00):
			# Na. H-gamma Core FWHM
			mcmc_input['na_Hg_fwhm'] = ({'name':'na_Hg_fwhm',
							   					 'label':'$\mathrm{FWHM}_{\mathrm{H}\gamma}$',
							   					 'init':250.,
							   					 'plim':(min_fwhm,650.),
							   					 'pcolor':'limegreen',
							   					 'min_width':min_fwhm,
							   					 })

		# Na. H-gamma Core Amplitude
		mcmc_input['na_Hg_amp'] = ({'name':'na_Hg_amp',
						   					'label':'$A_{\mathrm{H}\gamma}$',
						   					'init':(hg_amp_init-total_flux_init),
						   					'plim':(1.0e-3,max_flux),
						   					'pcolor':'green',
						   					})

		# Na. H-gamma Core VOFF
		mcmc_input['na_Hg_voff'] = ({'name':'na_Hg_voff',
						   					 'label':'$\mathrm{VOFF}_{\mathrm{H}\gamma}$',
						   					 'init':0.,
						   					 'plim':(-1000.,1000.),
						   					 'pcolor':'palegreen',
						   					 })
		# Na. [OIII]4363 Core Amplitude
		mcmc_input['na_oiii4363_core_amp'] = ({'name':'na_oiii4363_core_amp',
						   					'label':'$A_\mathrm{[OIII]4363\;Core}$',
						   					'init':(hg_amp_init-total_flux_init),
						   					'plim':(1.0e-3,max_flux),
						   					'pcolor':'green',
						   					})			
		# Na. [OIII]4363 Core VOFF
		mcmc_input['na_oiii4363_core_voff'] = ({'name':'na_oiii4363_core_voff',
						   					 'label':'$\mathrm{VOFF}_\mathrm{[OIII]4363\;Core}$',
						   					 'init':0.,
						   					 'plim':(-1000.,1000.),
						   					 'pcolor':'palegreen',
						   					 })

	##############################################################################

	#### Broad Line H-gamma ######################################################
	if ((fit_broad==True) and ((((fit_reg[0]+25.) < 4341.68 < (fit_reg[1]-25.))==True))):
		if print_output:
			print('	 - Fitting broad H-gamma.')
		# Br. H-beta amplitude
		mcmc_input['br_Hg_amp'] = ({'name':'br_Hg_amp',
						   			'label':'$A_{\mathrm{Br.\;Hg}}$' ,
						   			'init':(hg_amp_init-total_flux_init)/2.0  ,
						   			'plim':(1.0e-3,max_flux),
						   			'pcolor':'steelblue',
						   			})
		# Br. H-beta FWHM
		mcmc_input['br_Hg_fwhm'] = ({'name':'br_Hg_fwhm',
					   	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Hg}}$',
					   	   			 'init':2500.,
					   	   			 'plim':(500.,10000.),
					   	   			 'pcolor':'royalblue',
					   	   			 'min_width':min_fwhm,
					   	   			 })
		# Br. H-beta VOFF
		mcmc_input['br_Hg_voff'] = ({'name':'br_Hg_voff',
					   	   		 	 'label':'$\mathrm{VOFF}_{\mathrm{Br.\;Hg}}$',
					   	   		 	 'init':0.,
					   	   		 	 'plim':(-1000.,1000.),
					   	   		 	 'pcolor':'turquoise',
					   	   		 	 })
	##############################################################################



	#### Narrow Hb/[OIII] Core ###########################################################
	if ((fit_narrow==True) and ((((fit_reg[0]+25.) < 5008.240 < (fit_reg[1]-25.))==True))):
		if print_output:
			print('	 - Fitting narrow H-beta/[OIII]4959,5007 emission.')
		# Na. [OIII]5007 Core Amplitude
		mcmc_input['na_oiii5007_core_amp'] = ({'name':'na_oiii5007_core_amp',
						   					'label':'$A_{\mathrm{[OIII]5007\;Core}}$',
						   					'init':(oiii5007_amp_init-total_flux_init),
						   					'plim':(1.0e-3,max_flux),
						   					'pcolor':'green',
						   					})
		# If tie_narrow=True, then all line widths are tied to [OIII]5007, as well as their outflows (currently H-alpha/[NII]/[SII] outflows only)
		# Na. [OIII]5007 Core FWHM
		mcmc_input['na_oiii5007_core_fwhm'] = ({'name':'na_oiii5007_core_fwhm',
						   					 'label':'$\mathrm{FWHM}_{\mathrm{[OIII]5007\;Core}}$',
						   					 'init':250.,
						   					 'plim':(min_fwhm,650.),
						   					 'pcolor':'limegreen',
						   					 'min_width':min_fwhm,
						   					 })
		# Na. [OIII]5007 Core VOFF
		mcmc_input['na_oiii5007_core_voff'] = ({'name':'na_oiii5007_core_voff',
						   					 'label':'$\mathrm{VOFF}_{\mathrm{[OIII]5007\;Core}}$',
						   					 'init':0.,
						   					 'plim':(-1000.,1000.),
						   					 'pcolor':'palegreen',
						   					 })
		if 1:#(line_profile!='Lorentzian'):
			# Na. H-beta amplitude
			mcmc_input['na_Hb_core_amp'] = ({'name':'na_Hb_core_amp',
							   		 		 'label':'$A_{\mathrm{Na.\;Hb}}$' ,
							   		 		 'init':(hb_amp_init-total_flux_init) ,
							   		 		 'plim':(1.0e-3,max_flux),
							   		 		 'pcolor':'gold',
							   		 		 })
			# Na. H-beta FWHM tied to [OIII]5007 FWHM
			# Na. H-beta VOFF
			mcmc_input['na_Hb_core_voff'] = ({'name':'na_Hb_core_voff',
							   				  'label':'$\mathrm{VOFF}_{\mathrm{Na.\;Hb}}$',
							   				  'init':0.,
							   				  'plim':(-1000,1000.),
							   				  'pcolor':'yellow',
							   				  })
	##############################################################################

	#### Hb/[OIII] Outflows ######################################################
	if ((fit_narrow==True) and (fit_outflows==True) and ((((fit_reg[0]+25.) < 5008.240 < (fit_reg[1]-25.))==True))):
		if print_output:
			print('	 - Fitting H-beta/[OIII]4959,5007 outflows.')
		# Br. [OIII]5007 Outflow amplitude
		mcmc_input['na_oiii5007_outflow_amp'] = ({'name':'na_oiii5007_outflow_amp',
						   					   'label':'$A_{\mathrm{[OIII]5007\;Outflow}}$' ,
						   					   'init':(oiii5007_amp_init-total_flux_init)/2.,
						   					   'plim':(1.0e-3,max_flux),
						   					   'pcolor':'mediumpurple',
						   					   })
		# Br. [OIII]5007 Outflow FWHM
		mcmc_input['na_oiii5007_outflow_fwhm'] = ({'name':'na_oiii5007_outflow_fwhm',
						   						'label':'$\mathrm{FWHM}_{\mathrm{[OIII]5007\;Outflow}}$',
						   						'init':450.,
						   						'plim':(min_fwhm,3000.),
						   						'pcolor':'darkorchid',
						   						'min_width':min_fwhm,
						   						})
		# Br. [OIII]5007 Outflow VOFF
		mcmc_input['na_oiii5007_outflow_voff'] = ({'name':'na_oiii5007_outflow_voff',
						   						'label':'$\mathrm{VOFF}_{\mathrm{[OIII]5007\;Outflow}}$',
						   						'init':-50.,
						   						'plim':(-2000.,2000.),
						   						'pcolor':'orchid',
						   						})
		# Br. [OIII]4959 Outflow is tied to all components of [OIII]5007 outflow
	##############################################################################

	#### Broad Line H-beta ############################################################
	if ((fit_broad==True) and ((((fit_reg[0]+25.) < 5008.240 < (fit_reg[1]-25.))==True))):
		if print_output:
			print('	 - Fitting broad H-beta.')
		# Br. H-beta amplitude
		mcmc_input['br_Hb_amp'] = ({'name':'br_Hb_amp',
						   			'label':'$A_{\mathrm{Br.\;Hb}}$' ,
						   			'init':(hb_amp_init-total_flux_init)/2.0  ,
						   			'plim':(1.0e-3,max_flux),
						   			'pcolor':'steelblue',
						   			})
		# Br. H-beta FWHM
		mcmc_input['br_Hb_fwhm'] = ({'name':'br_Hb_fwhm',
					   	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Hb}}$',
					   	   			 'init':2500.,
					   	   			 'plim':(500.,10000.),
					   	   			 'pcolor':'royalblue',
					   	   			 'min_width':min_fwhm,
					   	   			 })
		# Br. H-beta VOFF
		mcmc_input['br_Hb_voff'] = ({'name':'br_Hb_voff',
					   	   		 	 'label':'$\mathrm{VOFF}_{\mathrm{Br.\;Hb}}$',
					   	   		 	 'init':0.,
					   	   		 	 'plim':(-1000.,1000.),
					   	   		 	 'pcolor':'turquoise',
					   	   		 	 })
	##############################################################################

	##############################################################################

	#### Narrow Ha/[NII]/[SII] ###################################################
	if ((fit_narrow==True) and ((((fit_reg[0]+150.) <  6564.61< (fit_reg[1]-150.))==True))):
		if print_output:
			print('	 - Fitting narrow Ha/[NII]/[SII] emission.')
		
		# If we aren't tying all narrow line widths, include the FWHM for H-alpha region
		if (tie_narrow==False) or (fit_reg[0]>=5025.):
			# Na. H-alpha FWHM
			mcmc_input['na_Ha_core_fwhm'] = ({'name':'na_Ha_core_fwhm',
						   				   'label':'$\mathrm{FWHM}_{\mathrm{Na.\;Ha}}$',
						   				   'init': 250,
						   				   'plim':(min_fwhm,650.),
						   				   'pcolor':'limegreen',
						   				   'min_width':min_fwhm,
						   				   })

		# Na. H-alpha amplitude
		mcmc_input['na_Ha_core_amp'] = ({'name':'na_Ha_core_amp',
						   		 	'label':'$A_{\mathrm{Na.\;Ha}}$' ,
						   		 	'init':(ha_amp_init-total_flux_init) ,
						   		 	'plim':(1.0e-3,max_flux),
						   		 	'pcolor':'gold',
						   		 	})
		# Na. H-alpha VOFF
		mcmc_input['na_Ha_core_voff'] = ({'name':'na_Ha_core_voff',
						   			 'label':'$\mathrm{VOFF}_{\mathrm{Na.\;Ha}}$',
						   			 'init':0.,
						   			 'plim':(-1000,1000.),
						   			 'pcolor':'yellow',
						   			 })
		# Na. [NII]6585 Amp.
		mcmc_input['na_nii6585_core_amp'] = ({'name':'na_nii6585_core_amp',
						   				   'label':'$A_{\mathrm{[NII]6585\;Core}}$',
						   				   'init':(ha_amp_init-total_flux_init)*0.75,
						   				   'plim':(1.0e-3,max_flux),
						   				   'pcolor':'green',
						   				   })
		# Na. [SII]6718 Amp.
		mcmc_input['na_sii6718_core_amp'] = ({'name':'na_sii6718_core_amp',
						   				   'label':'$A_{\mathrm{[SII]6718\;Core}}$',
						   				   'init':(sii_amp_init-total_flux_init),
						   				   'plim':(1.0e-3,max_flux),
						   				   'pcolor':'green',
						   				   })
		# Na. [SII]6732 Amp.
		mcmc_input['na_sii6732_core_amp'] = ({'name':'na_sii6732_core_amp',
						   				   'label':'$A_{\mathrm{[SII]6732\;Core}}$',
						   				   'init':(sii_amp_init-total_flux_init),
						   				   'plim':(1.0e-3,max_flux),
						   				   'pcolor':'green',
						   				   })

	#### Ha/[NII]/[SII] Outflows ######################################################
	# As it turns out, the Type 1 H-alpha broad line severely limits the ability of outflows in this region to be 
	# fit, and lead to very inconsistent results.  Even in Type 2 (no broad line) AGNs, the Ha/[NII] lines tend to be
	# blended due to their proximity in wavelength. In order to fit outflows in this region in the presence of broad
	# lines, it is recommended to included the Hb/[OIII] region (lower limit 4400 A) to constrain the Ha/[NII]/[SII]
	# outflow components.  However, if one excludes the Hb/[OIII] region, you can force BADASS to fit outflows, but is
	# is *NOT* recommended.
	# 

	if ( (fit_narrow==True) and (fit_outflows==True) and (fit_reg[0] >= 5025.) and (fit_reg[1] >= 6750.)):
		if print_output:
			print('	 - Fitting Ha/[NII]/[SII] outflows independently of Hb/[OIII] outflows (not recommended)')
		# Br. [OIII]5007 Outflow amplitude
		mcmc_input['na_Ha_outflow_amp'] = ({'name':'na_Ha_outflow_amp',
						   					   'label':'$A_{\mathrm{Na.\;Ha\;Outflow}}$' ,
						   					   'init':(ha_amp_init-total_flux_init)*0.25,
						   					   'plim':(1.0e-3,max_flux),
						   					   'pcolor':'mediumpurple',
						   					   })
		# if (tie_narrow==False):
		# Br. [OIII]5007 Outflow FWHM
		mcmc_input['na_Ha_outflow_fwhm'] = ({'name':'na_Ha_outflow_fwhm',
						   						'label':'$\mathrm{FWHM}_{\mathrm{Na.\;Ha\;Outflow}}$',
						   						'init':450.,
						   						'plim':(min_fwhm,2500.),
						   						'pcolor':'darkorchid',
						   						'min_width':min_fwhm,
						   						})
		# Br. [OIII]5007 Outflow VOFF
		mcmc_input['na_Ha_outflow_voff'] = ({'name':'na_Ha_outflow_voff',
						   						'label':'$\mathrm{VOFF}_{\mathrm{Na.\;Ha\;Outflow}}$',
						   						'init':-50.,
						   						'plim':(-2000.,2000.),
						   						'pcolor':'orchid',
						   						})
		# All components [NII]6585 of outflow are tied to all outflows of the Ha/[NII]/[SII] region

	##############################################################################

	#### Broad Line H-alpha ###########################################################

	if ((fit_broad==True) and ((((fit_reg[0]+150.) < 6564.61 < (fit_reg[1]-150.))==True))):
		if print_output:
			print('	 - Fitting broad H-alpha.')
		# Br. H-alpha amplitude
		mcmc_input['br_Ha_amp'] = ({'name':'br_Ha_amp',
						   			'label':'$A_{\mathrm{Br.\;Ha}}$' ,
						   			'init':(ha_amp_init-total_flux_init)/2.0  ,
						   			'plim':(1.0e-3,max_flux),
						   			'pcolor':'steelblue',
						   			})
		# Br. H-alpha FWHM
		mcmc_input['br_Ha_fwhm'] = ({'name':'br_Ha_fwhm',
					   	   			 'label':'$\mathrm{FWHM}_{\mathrm{Br.\;Ha}}$',
					   	   			 'init':2500.,
					   	   			 'plim':(500.,10000.),
					   	   			 'pcolor':'royalblue',
					   	   			 'min_width':min_fwhm,
					   	   			 })
		# Br. H-alpha VOFF
		mcmc_input['br_Ha_voff'] = ({'name':'br_Ha_voff',
					   	   		 	 'label':'$\mathrm{VOFF}_{\mathrm{Br.\;Ha}}$',
					   	   		 	 'init':0.,
					   	   		 	 'plim':(-1000.,1000.),
					   	   		 	 'pcolor':'turquoise',
					   	   		 	 }) 

	

	##############################################################################
	del lam_gal
	del galaxy
	del fwhm_gal
	gc.collect()
	return mcmc_input

##################################################################################

#### Outflow Tests ################################################################

def ssr_test(resid_outflow,resid_no_outflow,run_dir):
	"""
	Sum-of-Squares of Residuals test:
	The sum-of-squares of the residuals of the no-outflow model
	and the sum-of-squares of the residuals of outflow model for each iteration
	of the outflow test. 
	"""

	# For multiple runs
	ssr_ratio = np.empty(np.shape(resid_outflow)[0])
	ssr_outflow    = np.empty(np.shape(resid_outflow)[0])
	ssr_no_outflow = np.empty(np.shape(resid_outflow)[0])

	for i in range(np.shape(resid_outflow)[0]):
		# Compute median and std of residual standard deviations
		ssr_resid_outflow        = np.sum(resid_outflow[i,:]**2)
		ssr_resid_no_outflow     = np.sum(resid_no_outflow[i,:]**2)
		ssr_ratio[i] = (ssr_resid_no_outflow)/(ssr_resid_outflow) # sum-of-squares ratio
		ssr_outflow[i] = ssr_resid_outflow
		ssr_no_outflow[i] = ssr_resid_no_outflow

	return np.median(ssr_ratio), np.std(ssr_ratio), \
		   np.median(ssr_no_outflow), np.std(ssr_no_outflow), \
		   np.median(ssr_outflow), np.std(ssr_outflow)


def f_test(resid_outflow,resid_no_outflow,run_dir):
	"""
	f-test:
	Perform an f-statistic for model comparison between a single and double-component
	model for the [OIII] line.  The f_oneway test is only accurate for normally-distributed 
	values and should be compared against the Kruskal-Wallis test (non-normal distributions),
	as well as the Bartlett and Levene variance tests.  We use the sum-of-squares of residuals
	for each model for the test. 
	"""

	f_stat = np.empty(np.shape(resid_outflow)[0])
	f_pval = np.empty(np.shape(resid_outflow)[0])

	k1 = 3.0 # nested (simpler) model; single-Gaussian deg. of freedom
	k2 = 6.0 # complex model; double-Gaussian model deg. of freedom

	for i in range(np.shape(resid_outflow)[0]):

		RSS1 = np.sum(resid_no_outflow[i,:]**2) # resid. sum of squares single_Gaussian
		RSS2 = np.sum(resid_outflow[i,:]**2)    # resid. sum of squares double-Gaussian

		n = float(len(resid_outflow[i,:]))
		dfn = k2 - k1 # deg. of freedom numerator
		dfd = n - k2  # deg. of freedom denominator

		f_stat[i] = ((RSS1-RSS2)/(k2-k1))/((RSS2)/(n-k2)) 
		f_pval[i] = 1 - f.cdf(f_stat[i], dfn, dfd)

	# print('f-statistic model comparison = %0.2f +/- %0.2f, p-value = %0.2e +/- %0.2f' % (np.median(f_stat), np.std(f_stat),np.median(f_pval), np.std(f_pval) ))
	# print('f-statistic model comparison = %0.2f ' % (f_stat))

	outflow_conf, outflow_conf_err = 1.0-np.median(f_pval),(1.0-np.median(f_pval))-(1-(np.median(f_pval)+np.std(f_pval)))

	return np.median(f_stat), np.std(f_stat), np.median(f_pval), np.std(f_pval), outflow_conf, outflow_conf_err


def outflow_test_oiii(lam_gal,galaxy,noise,run_dir,line_profile,fwhm_gal,feii_options,velscale,n_basinhop,outflow_test_niter,outflow_test_options,fit_comp_options,print_output=True):
	"""
	Performs outflow tests using the outflow criteria on the [OIII] outflows.
	"""
	# The optimal fitting region for fitting outflows in [OIII] is (4400,5800) with ALL comoponents
	fit_reg = (np.max([np.min(lam_gal),4400]),np.min([np.max(lam_gal),5800]))
	# Initialize FeII and host template
	gal_temp = galaxy_template(lam_gal,age=10.0,print_output=print_output)
	if fit_comp_options['fit_feii']:
		feii_tab = initialize_feii(lam_gal,feii_options)#,line_profile,fwhm_gal,velscale,fit_reg,feii_options,run_dir)
	else: 
		feii_tab = None	
	# Create mask to mask out parts of spectrum; should speed things up
	mask = np.where( (lam_gal > fit_reg[0]) & (lam_gal < fit_reg[1]) )
	lam_gal	      = lam_gal[mask]
	galaxy		  = galaxy[mask]
	noise		  = noise[mask]
	gal_temp	  = gal_temp[mask]
	fwhm_gal      = fwhm_gal[mask]
	# use SDSS spectral noise
	sigma = np.median(noise) 
	# For outflow testing, we override user FeII options so hold fwhm and voff constant
	# because it usually takes a very long time.  If FeII emission is that bad, you can't 
	# fit outflows anyway...
	if feii_options['template']['type']=='VC04':
		feii_options={
		'template':{'type':'VC04'},
		'amp_const':{'bool':False,'br_feii_val':1.0,'na_feii_val':1.0},
		'fwhm_const':{'bool':True,'br_feii_val':3000.0,'na_feii_val':500.0},
		'voff_const':{'bool':True,'br_feii_val':0.0,'na_feii_val':0.0}
		}
	elif feii_options['template']['type']=='K10':
		feii_options={
		'template'  :{'type':'K10'},
		'amp_const' :{'bool':False,'f_feii_val':1.0,'s_feii_val':1.0,'g_feii_val':1.0,'z_feii_val':1.0},
		'fwhm_const':{'bool':True,'val':1500.0},
		'voff_const':{'bool':True,'val':0.0},
		'temp_const':{'bool':True,'val':10000.0} 
		}
	# Perform fitting WITH outflows
	if print_output:
		print('\n Fitting with outflow components...')
	param_dict_outflows = initialize_mcmc(lam_gal,galaxy,line_profile,fwhm_gal,velscale,feii_options,run_dir,fit_reg=fit_reg,fit_type='init',
								 fit_feii=fit_comp_options['fit_feii'],fit_losvd=True,
								 fit_power=fit_comp_options['fit_power'],fit_broad=fit_comp_options['fit_broad'],
								 fit_narrow=True,fit_outflows=True,tie_narrow=False,print_output=print_output)

	mcpars_outflow, mccomps_outflow = max_likelihood(param_dict_outflows,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
										None,None,None,line_profile,fwhm_gal,velscale,None,None,run_dir,test_outflows=True,n_basinhop=n_basinhop,
										outflow_test_niter=outflow_test_niter,print_output=print_output)
	# Perform fitting with NO outflows
	if print_output:
		print('\n Fitting without outflow components...')
	param_dict_no_outflows = initialize_mcmc(lam_gal,galaxy,line_profile,fwhm_gal,velscale,feii_options,run_dir,fit_reg=fit_reg,fit_type='init',
								 fit_feii=fit_comp_options['fit_feii'],fit_losvd=True,
								 fit_power=fit_comp_options['fit_power'],fit_broad=fit_comp_options['fit_broad'],
								 fit_narrow=True,fit_outflows=False,tie_narrow=False,print_output=print_output)

	mcpars_no_outflow, mccomps_no_outflow = max_likelihood(param_dict_no_outflows,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
										None,None,None,line_profile,fwhm_gal,velscale,None,None,run_dir,test_outflows=True,n_basinhop=n_basinhop,
										outflow_test_niter=outflow_test_niter,print_output=print_output)


	# Determine if there is a significant improvement in residuals to warrant inclusion of a second component
	# From the outflow fit parameters, get the median values for the widths and velocity offsets to determine th 
	# optimal window for measuring the residual standard deviation of the fit 
	c = 299792.458 # speed of light (km/s)
	ref_wave = 5008.240 # reference wavelength for [OIII]
	med_outflow_fwhm = np.median(mcpars_outflow['na_oiii5007_outflow_fwhm'])
	med_outflow_voff = np.median(mcpars_outflow['na_oiii5007_outflow_voff'])
	med_core_fwhm    = np.median(mcpars_outflow['na_oiii5007_core_fwhm'])
	med_core_voff    = np.median(mcpars_outflow['na_oiii5007_core_voff'])

	sigma_inc = 3.0 # number of sigmas to include
	min_wave = ref_wave + np.min([(med_outflow_voff - (sigma_inc*med_outflow_fwhm/2.3548))/c*ref_wave,(med_core_voff - (sigma_inc*med_core_fwhm/2.3548))/c*ref_wave])
	max_wave = ref_wave + np.max([(med_outflow_voff + (sigma_inc*med_outflow_fwhm/2.3548))/c*ref_wave,(med_core_voff + (sigma_inc*med_core_fwhm/2.3548))/c*ref_wave])
	# Get indices where we perform f-test
	eval_ind = np.where((lam_gal >= min_wave) & (lam_gal <= max_wave))[0]
	# number of channels in the [OIII] test region 
	nchannel = len( mccomps_outflow['resid'][:,0][eval_ind])
	# if the number of channels < 6 (number of degrees of freedom for double-Gaussian model), then the calculated f-statistic
	# will be zero.  To resolve this, we extend the range by one pixel on each side, i.e. nchannel = 8.
	if nchannel <= 6: 
		add_chan = 7 - nchannel# number of channels to add to each side; minimum is 7 channels since deg. of freedom  = 6
		lower_pad = range(eval_ind[0]-add_chan,eval_ind[0],1)#np.arange(eval_ind[0]-add_chan,eval_ind[0],1)
		upper_pad = range(eval_ind[-1]+1,eval_ind[-1]+1+add_chan,1)
		eval_ind = np.concatenate([lower_pad, eval_ind, upper_pad],axis=0)
		nchannel = len( mccomps_outflow['resid'][:,0][eval_ind])

	# storage arrays for residuals in [OIII] test region
	resid_outflow    = np.empty((outflow_test_niter,nchannel))
	resid_no_outflow = np.empty((outflow_test_niter,nchannel))
	resid_total      = np.empty((outflow_test_niter,len(lam_gal)))
	for i in range(outflow_test_niter):
		resid_outflow[i,:]    = mccomps_outflow['resid'][:,i][eval_ind]
		resid_no_outflow[i,:] = mccomps_no_outflow['resid'][:,i][eval_ind]
		resid_total[i,:]      = mccomps_outflow['resid'][:,i]

	# Calculate sum-of-square of residuals and its uncertainty
	ssr_ratio, ssr_ratio_err, ssr_no_outflow, ssr_no_outflow_err, ssr_outflow, ssr_outflow_err = ssr_test(resid_outflow,resid_no_outflow,run_dir)
	# Perform f-test model comparison(for normally distributed model residuals)
	f_stat, f_stat_err, f_pval, f_pval_err, outflow_conf, outflow_conf_err = f_test(resid_outflow,resid_no_outflow,run_dir)

	# Calculate total residual noise
	resid_std_no_outflow     = np.median([np.std(resid_no_outflow[i,:]) for i in range(np.shape(resid_no_outflow)[0])])
	resid_std_no_outflow_err = np.std([np.std(resid_no_outflow[i,:]) for i in range(np.shape(resid_no_outflow)[0])])
	resid_std_outflow        = np.median([np.std(resid_outflow[i,:]) for i in range(np.shape(resid_outflow)[0])])
	resid_std_outflow_err    = np.std([np.std(resid_outflow[i,:]) for i in range(np.shape(resid_outflow)[0])])
	total_resid_noise        = np.median([np.std(resid_total[i,:]) for i in range(np.shape(resid_total)[0])])
	total_resid_noise_err    = np.std([np.std(resid_total[i,:]) for i in range(np.shape(resid_total)[0])])

	# Iterate through every parameter to determine if the fit is "good" (more than 1-sigma away from bounds)
	# if not, then add 1 to that parameter flag value		
	pdict = {}
	for key in mcpars_outflow:
		mc_med = np.median(mcpars_outflow[key],axis=0)
		mc_std = np.std(mcpars_outflow[key],axis=0)
		param_flags = 0
		if ( (mc_med-mc_std*outflow_test_options['bounds_test']['nsigma']) <= param_dict_outflows[key]['plim'][0]):
			param_flags += 1
		if ( (mc_med+mc_std*outflow_test_options['bounds_test']['nsigma']) >= param_dict_outflows[key]['plim'][1]):
			param_flags += 1
		pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}

	pdict_no_outflow = {}
	for key in mcpars_no_outflow:
		mc_med = np.median(mcpars_no_outflow[key],axis=0)
		mc_std = np.std(mcpars_no_outflow[key],axis=0)
		param_flags = 0
		if ( (mc_med-mc_std*outflow_test_options['bounds_test']['nsigma']) <= param_dict_no_outflows[key]['plim'][0]):
			param_flags += 1
		if ( (mc_med+mc_std*outflow_test_options['bounds_test']['nsigma']) >= param_dict_no_outflows[key]['plim'][1]):
			param_flags += 1
		pdict_no_outflow[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}

	# Outflow vs. No-outflow Model Metrics; these metrics test significance of parameters between models, and are printed to logfile 
	# regardless of whether they are tested or not.
	amp_metric = ((pdict['na_oiii5007_outflow_amp']['med'])/np.sqrt( (pdict['na_oiii5007_outflow_amp']['std'])**2 + (np.median(noise))**2 ))
	fwhm_metric = ( (pdict['na_oiii5007_outflow_fwhm']['med'])-(pdict['na_oiii5007_core_fwhm']['med']) )/np.sqrt( (pdict['na_oiii5007_outflow_fwhm']['std'])**2 + (pdict['na_oiii5007_core_fwhm']['std'])**2 )
	voff_metric =  ( (pdict['na_oiii5007_core_voff']['med'])-(pdict['na_oiii5007_outflow_voff']['med']) )/np.sqrt( (pdict['na_oiii5007_core_voff']['std'])**2 + (pdict['na_oiii5007_outflow_voff']['std'])**2 )
	
	# print('\n Amp. Metric = %0.2f' % amp_metric)
	# print(' FWHM Metric = %0.2f' % fwhm_metric)
	# print(' VOFF Metric = %0.2f' % voff_metric)
	# print(' Resid. Metric = %0.2f' % resid_metric)


	if print_output:
		print('\n{0:<30}{1:<25}{2:<25}{3:<25}'.format('Parameter', 'Best-fit Value', '+/- 1-sigma','Flag'))
		print('--------------------------------------------------------------------------------------')
	# Sort into arrays
	pname = []
	med   = []
	std   = []
	flag  = [] 
	for key in pdict:
		pname.append(key)
		med.append(pdict[key]['med'])
		std.append(pdict[key]['std'])
		flag.append(pdict[key]['flag'])
	i_sort = np.argsort(pname)
	pname = np.array(pname)[i_sort] 
	med   = np.array(med)[i_sort]   
	std   = np.array(std)[i_sort]   
	flag  = np.array(flag)[i_sort]
	if print_output:  
		for i in range(0,len(pname),1):
			print('{0:<30}{1:<25.2f}{2:<25.2f}{3:<25}'.format(pname[i], med[i], std[i], flag[i]))
		print('{0:<30}{1:<25.2f}'.format('median spec noise',sigma))
		print('{0:<30}{1:<25.2f}{2:<25.2e}'.format('total resid noise',total_resid_noise,total_resid_noise_err)) 
		print('{0:<30}{1:<25.2f}'.format('amp metric',amp_metric))
		print('{0:<30}{1:<25.2f}'.format('fwhm metric',fwhm_metric))
		print('{0:<30}{1:<25.2f}'.format('voff metric',voff_metric))

		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('SSR ratio',ssr_ratio,ssr_ratio_err))
		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('SSR no outflow',ssr_no_outflow,ssr_no_outflow_err))
		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('SSR outflow',ssr_outflow,ssr_outflow_err))

		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('f-statistic',f_stat,f_stat_err))
		print('{0:<30}{1:<25.2e}{2:<25.2e}'.format('p-value',f_pval,f_pval_err))
		print('{0:<30}{1:<25.5f}{2:<25.5f}'.format('Outflow confidence',outflow_conf, outflow_conf_err ) )


		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('[OIII] no outflow resid. std.',resid_std_no_outflow,resid_std_no_outflow_err))
		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('[OIII] outflow resid. std.',resid_std_outflow,resid_std_outflow_err))
		print('--------------------------------------------------------------------------------------')

	# Write to log
	write_log(pdict_no_outflow,'no_outflow_test',run_dir)
	write_log((pdict,
			   sigma,total_resid_noise,total_resid_noise_err,
			   amp_metric,fwhm_metric,voff_metric,ssr_ratio,ssr_ratio_err,
			   ssr_no_outflow,ssr_no_outflow_err,ssr_outflow,ssr_outflow_err,
			   f_stat, f_stat_err, f_pval, f_pval_err, outflow_conf, outflow_conf_err,
			   resid_std_no_outflow,resid_std_no_outflow_err,
			   resid_std_outflow,resid_std_outflow_err,
			   ),'outflow_test',run_dir)

	# Determine the significance of outflows
	outflow_conds = []
	# Outflow test criteria:
	if (outflow_test_options['amp_test']['test']==True):
		#	(1) Amp. test: (AMP_outflow - dAMP_outflow) > sigma
		amp_cond = (amp_metric > (outflow_test_options['amp_test']['nsigma']) )
		if (amp_cond==True):
			outflow_conds.append(True)
			outflow_test_options['amp_test']['pass'] = True # Add to outflow_test_options dictionary 
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow amplitude condition:', 'Pass' ) )
		elif (amp_cond==False):
			outflow_conds.append(False)
			outflow_test_options['amp_test']['pass'] = False # Add to outflow_test_options dictionary 
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow amplitude condition:', 'Fail' ) )

	#	(2) FWHM test: (FWHM_outflow - dFWHM_outflow) > (FWHM_core + dFWHM_core)
	if (outflow_test_options['fwhm_test']['test']==True):
		fwhm_cond = ( fwhm_metric > (outflow_test_options['fwhm_test']['nsigma']) )
		if (fwhm_cond==True):
			outflow_conds.append(True)
			outflow_test_options['fwhm_test']['pass'] = True # Add to outflow_test_options dictionary 
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow FWHM condition:'	 , 'Pass') )
		elif (fwhm_cond==False):
			outflow_conds.append(False)
			outflow_test_options['fwhm_test']['pass'] = False # Add to outflow_test_options dictionary 
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow FWHM condition:'	 , 'Fail') )

	#	(3) VOFF test: (VOFF_outflow + dVOFF_outflow) < (VOFF_core - dVOFF_core)
	if (outflow_test_options['voff_test']['test']==True):
		voff_cond = (voff_metric>(outflow_test_options['voff_test']['nsigma']))
		if (voff_cond==True):
			outflow_conds.append(True)
			outflow_test_options['voff_test']['pass'] = True
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow VOFF condition:'	 , 'Pass ') )
		elif (voff_cond==False):
			outflow_conds.append(False)
			outflow_test_options['voff_test']['pass'] = False
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow VOFF condition:'	 , 'Fail' ) )

	#	(4) Outflow Confidence Threshold 
	if (outflow_test_options['outflow_confidence']['test']==True):
		if outflow_conf_err <= 0.05:
			outflow_conf_cond = ((outflow_conf+outflow_conf_err)>=(outflow_test_options['outflow_confidence']['conf']))
		else:
			outflow_conf_cond = ((outflow_conf)>=(outflow_test_options['outflow_confidence']['conf']))
		if (outflow_conf_cond==True):
			outflow_conds.append(True)
			outflow_test_options['outflow_confidence']['pass'] = True
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow Confidence condition:'	 , 'Pass ') )
		elif (outflow_conf_cond==False):
			outflow_conds.append(False)
			outflow_test_options['outflow_confidence']['pass'] = False
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow Confidence condition:'	 , 'Fail' ) )

	#	(5) Parameter bounds: if parameter reached the parameter limits (bounds), then its uncertainty is zero, and something went wrong with the fit.
	if (outflow_test_options['bounds_test']['test']==True):
		bounds_cond = all([i==0 for i in [ pdict['na_oiii5007_outflow_fwhm']['flag'],pdict['na_oiii5007_outflow_voff']['flag'],pdict['na_oiii5007_outflow_amp']['flag'],
			     	  pdict['na_oiii5007_core_fwhm']['flag'],pdict['na_oiii5007_core_voff']['flag'],pdict['na_oiii5007_core_amp']['flag'] ] ])
		if (bounds_cond==True):
			outflow_conds.append(True)
			outflow_test_options['bounds_test']['pass'] = True
			if print_output:
				print('{0:<35}{1:<30}'.format('Parameter bounds condition:', 'Pass' ) )
		elif (bounds_cond==False):
			outflow_conds.append(False)
			outflow_test_options['bounds_test']['pass'] = False
			if print_output:
				print('{0:<35}{1:<30}'.format('Parameter bounds condition:', 'Fail' ) )

	# Make plot 
	# Get best fit model components for each model 
	param_names = [key for key in pdict ]
	params	    = [pdict[key]['med']  for key in pdict ]
	fit_type	 = 'outflow_test'
	output_model = False
	comp_dict_outflow = fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,
						  feii_tab,feii_options,
						  None,None,None,line_profile,fwhm_gal,velscale,None,None,run_dir,
						  fit_type,output_model)
	param_names = [key for key in pdict_no_outflow ]
	params	    = [pdict_no_outflow[key]['med']  for key in pdict_no_outflow ]
	fit_type	 = 'outflow_test'
	output_model = False
	comp_dict_no_outflow = fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,
						  feii_tab,feii_options,
						  None,None,None,line_profile,fwhm_gal,velscale,None,None,run_dir,
						  fit_type,output_model)
	# Make comparison plots of outflow and no-outflow models
	outflow_test_plot_oiii(comp_dict_outflow,comp_dict_no_outflow,run_dir)

	# collect garbage
	del lam_gal	      
	del galaxy		  
	del noise		  
	del fwhm_gal  
	del gal_temp	  
	del feii_tab
	del param_dict_outflows
	del mcpars_outflow
	del mccomps_outflow
	del param_dict_no_outflows
	del mcpars_no_outflow
	del mccomps_no_outflow
	del pdict_no_outflow
	del comp_dict_outflow
	del comp_dict_no_outflow
	gc.collect()

	if not outflow_conds: # outflow_conds is empty because no tests were performed but we tested for outflows anyway
		return True,pdict,sigma
	elif (all(outflow_conds)==True):
		# Write to log
		write_log(outflow_test_options,'outflow_test_pass',run_dir)
		return True,pdict,sigma
	elif (all(outflow_conds)==False):
		# Write to log
		write_log(outflow_test_options,'outflow_test_fail',run_dir)
		return False,pdict,sigma

##################################################################################

def outflow_test_plot_oiii(comp_dict_outflow,comp_dict_no_outflow,run_dir):
	"""
	The plotting function for outflow_test_oiii().  It plots both the outflow
	and no_outflow results.
	"""
	# Creat plot window and axes
	fig = plt.figure(figsize=(14,11)) 
	gs = gridspec.GridSpec(9,1)
	ax1  = fig.add_subplot(gs[0:3,0]) # No outflow
	ax2  = fig.add_subplot(gs[3:4,0]) # No outflow residuals
	ax3  = fig.add_subplot(gs[5:8,0]) # Outflow
	ax4  = fig.add_subplot(gs[8:9,0]) # Outflow residuals
	gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
	# No outflow model (ax1,ax2)
	norm = np.median(comp_dict_no_outflow['data']['comp'])
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['data']['comp']            , color='xkcd:white'      , linewidth=0.5, linestyle='-' , label='Data'         )  
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['model']['comp']           , color='xkcd:red'        , linewidth=1.0, linestyle='-' , label='Model'        ) 
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['host_galaxy']['comp']     , color='xkcd:lime green' , linewidth=1.0, linestyle='-' , label='Galaxy'       )
	if ('power' in comp_dict_no_outflow):
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['power']['comp']           , color='xkcd:orange red' , linewidth=1.0, linestyle='--', label='AGN Cont.'    )
	if ('na_feii_template' in comp_dict_no_outflow) and ('br_feii_template' in comp_dict_no_outflow):
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_feii_template']['comp'], color='xkcd:yellow'     , linewidth=1.0, linestyle='-' , label='Na. FeII'     )
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['br_feii_template']['comp'], color='xkcd:orange'     , linewidth=1.0, linestyle='-' , label='Br. FeII'     )
	elif ('F_feii_template' in comp_dict_no_outflow) and ('S_feii_template' in comp_dict_no_outflow) and ('G_feii_template' in comp_dict_no_outflow) and ('Z_feii_template' in comp_dict_no_outflow):
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['F_feii_template']['comp'], color='xkcd:yellow'     , linewidth=1.0, linestyle='-' , label='F-transition FeII'     )
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['S_feii_template']['comp'], color='xkcd:mustard'     , linewidth=1.0, linestyle='-' , label='S_transition FeII'     )
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['G_feii_template']['comp'], color='xkcd:orange'     , linewidth=1.0, linestyle='-' , label='G_transition FeII'     )
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['Z_feii_template']['comp'], color='xkcd:rust'     , linewidth=1.0, linestyle='-' , label='Z_transition FeII'     )
	if ('br_Hb' in comp_dict_no_outflow):
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['br_Hb']['comp']           , color='xkcd:turquoise'  , linewidth=1.0, linestyle='-' , label='Br. H-beta'       )
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_Hb_core']['comp']      , color='xkcd:dodger blue', linewidth=1.0, linestyle='-' , label='Core comp.'   )
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_oiii4959_core']['comp'], color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                        )
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_oiii5007_core']['comp'], color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                        )
	ax1.axvline(4862.680, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax1.axvline(4960.295, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax1.axvline(5008.240, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax1.axvline(5176.700, color='xkcd:white' , linewidth=0.5, linestyle='--')    
	# ax1.plot(comp_dict_no_outflow['wave']['comp'], 1*comp_dict_no_outflow['noise']['comp'], color='xkcd:dodger blue' , linewidth=0.5, linestyle='--')
	# ax1.plot(comp_dict_no_outflow['wave']['comp'], 2*comp_dict_no_outflow['noise']['comp'], color='xkcd:lime green'  , linewidth=0.5, linestyle='--')
	# ax1.plot(comp_dict_no_outflow['wave']['comp'], 3*comp_dict_no_outflow['noise']['comp'], color='xkcd:orange red'  , linewidth=0.5, linestyle='--')
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)')
	ax1.set_xticklabels([])
	ax1.legend(loc='upper left',fontsize=6)
	ax1.set_xlim(np.min(comp_dict_outflow['wave']['comp']),np.max(comp_dict_outflow['wave']['comp']))
	ax1.set_ylim(0.0,np.max(comp_dict_no_outflow['model']['comp'])+3*np.median(comp_dict_no_outflow['noise']['comp']))
	ax1.set_title('No Outflow Model')
	# No Outflow Residuals
	ax2.plot(comp_dict_no_outflow['wave']['comp'],3*(comp_dict_no_outflow['data']['comp']-comp_dict_no_outflow['model']['comp']), color='xkcd:white' , linewidth=0.5, linestyle='-')
	ax2.axvline(4862.680, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.axvline(4960.295, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.axvline(5008.240, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.axvline(5176.700, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.axhline(0.0, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.plot(comp_dict_no_outflow['wave']['comp'], 3*1*comp_dict_no_outflow['noise']['comp'], color='xkcd:bright aqua' , linewidth=0.5, linestyle='-')
	# ax2.plot(comp_dict_no_outflow['wave']['comp'], 3*2*comp_dict_no_outflow['noise']['comp'], color='xkcd:lime green'  , linewidth=0.5, linestyle='--')
	# ax2.plot(comp_dict_no_outflow['wave']['comp'], 3*3*comp_dict_no_outflow['noise']['comp'], color='xkcd:orange red'  , linewidth=0.5, linestyle='--')
	ax2.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)')
	ax2.set_ylabel(r'$\Delta f_\lambda$')
	ax2.set_xlim(np.min(comp_dict_outflow['wave']['comp']),np.max(comp_dict_outflow['wave']['comp']))
	ax2.set_ylim(0.0-9*np.median(comp_dict_no_outflow['noise']['comp']),ax1.get_ylim()[1])
    # Outlfow models (ax3,ax4)
	norm = np.median(comp_dict_outflow['data']['comp'])
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['data']['comp']               , color='xkcd:white'      , linewidth=0.5, linestyle='-' , label='Data'         )  
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['model']['comp']              , color='xkcd:red'        , linewidth=1.0, linestyle='-' , label='Model'        ) 
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['host_galaxy']['comp']        , color='xkcd:lime green' , linewidth=1.0, linestyle='-' , label='Galaxy'       )
	if ('power' in comp_dict_outflow):
		ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['power']['comp']              , color='xkcd:orange red' , linewidth=1.0, linestyle='--', label='AGN Cont.'    )
	if ('na_feii_template' in comp_dict_outflow) and ('br_feii_template' in comp_dict_outflow):
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_feii_template']['comp'], color='xkcd:yellow'     , linewidth=1.0, linestyle='-' , label='Na. FeII'     )
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['br_feii_template']['comp'], color='xkcd:orange'     , linewidth=1.0, linestyle='-' , label='Br. FeII'     )
	elif ('F_feii_template' in comp_dict_outflow) and ('S_feii_template' in comp_dict_outflow) and ('G_feii_template' in comp_dict_outflow) and ('Z_feii_template' in comp_dict_outflow):
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['F_feii_template']['comp'], color='xkcd:yellow'     , linewidth=1.0, linestyle='-' , label='F-transition FeII'     )
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['S_feii_template']['comp'], color='xkcd:mustard'     , linewidth=1.0, linestyle='-' , label='S_transition FeII'     )
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['G_feii_template']['comp'], color='xkcd:orange'     , linewidth=1.0, linestyle='-' , label='G_transition FeII'     )
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['Z_feii_template']['comp'], color='xkcd:rust'     , linewidth=1.0, linestyle='-' , label='Z_transition FeII'     )
	if ('br_Hb' in comp_dict_outflow):
		ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['br_Hb']['comp']              , color='xkcd:turquoise'  , linewidth=1.0, linestyle='-' , label='Br. H-beta'       )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_Hb_core']['comp']         , color='xkcd:dodger blue', linewidth=1.0, linestyle='-' , label='Core comp.'   )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_oiii4959_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                        )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_oiii5007_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                        )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_Hb_outflow']['comp']      , color='xkcd:magenta'    , linewidth=1.0, linestyle='-' , label='Outflow comp.')
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_oiii4959_outflow']['comp'], color='xkcd:magenta'    , linewidth=1.0, linestyle='-'                        )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_oiii5007_outflow']['comp'], color='xkcd:magenta'    , linewidth=1.0, linestyle='-'                        )
	ax3.axvline(4862.680, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax3.axvline(4960.295, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax3.axvline(5008.240, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax3.axvline(5176.700, color='xkcd:white' , linewidth=0.5, linestyle='--')    
	# ax3.plot(comp_dict_outflow['wave']['comp'], 1*comp_dict_outflow['noise']['comp'], color='xkcd:dodger blue' , linewidth=0.5, linestyle='--')
	# ax3.plot(comp_dict_outflow['wave']['comp'], 2*comp_dict_outflow['noise']['comp'], color='xkcd:lime green'  , linewidth=0.5, linestyle='--')
	# ax3.plot(comp_dict_outflow['wave']['comp'], 3*comp_dict_outflow['noise']['comp'], color='xkcd:orange red'  , linewidth=0.5, linestyle='--')
	ax3.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)')
	ax3.set_xticklabels([])
	ax3.legend(loc='upper left',fontsize=6)
	ax3.set_xlim(np.min(comp_dict_outflow['wave']['comp']),np.max(comp_dict_outflow['wave']['comp']))
	ax3.set_ylim(0.0,np.max(comp_dict_outflow['model']['comp'])+3*np.median(comp_dict_outflow['noise']['comp']))
	ax3.set_title('Outflow Model')
	# Outflow Residuals
	ax4.plot(comp_dict_outflow['wave']['comp'],3*(comp_dict_outflow['data']['comp']-comp_dict_outflow['model']['comp']), color='xkcd:white' , linewidth=0.5, linestyle='-')
	ax4.axvline(4862.680, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axvline(4960.295, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axvline(5008.240, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axvline(5176.700, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axhline(0.0, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.plot(comp_dict_outflow['wave']['comp'], 3*1*comp_dict_outflow['noise']['comp'], color='xkcd:bright aqua' , linewidth=0.5, linestyle='-')
	# ax4.plot(comp_dict_outflow['wave']['comp'], 3*2*comp_dict_outflow['noise']['comp'], color='xkcd:lime green'  , linewidth=0.5, linestyle='--')
	# ax4.plot(comp_dict_outflow['wave']['comp'], 3*3*comp_dict_outflow['noise']['comp'], color='xkcd:orange red'  , linewidth=0.5, linestyle='--')
	ax4.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)')
	ax4.set_ylabel(r'$\Delta f_\lambda$')
	ax4.set_xlim(np.min(comp_dict_outflow['wave']['comp']),np.max(comp_dict_outflow['wave']['comp']))
	ax4.set_ylim(0.0-9*np.median(comp_dict_outflow['noise']['comp']),ax3.get_ylim()[1])
    
	fig.tight_layout()
	plt.savefig(run_dir+'outflow_test.pdf',fmt='pdf',dpi=150)

	plt.close()
	# Collect garbage
	del ax1
	del ax2
	del ax3
	del ax4
	del fig 
	del comp_dict_outflow
	del comp_dict_no_outflow
	gc.collect()

	return None

##################################################################################

def outflow_test_nii(lam_gal,galaxy,noise,run_dir,line_profile,fwhm_gal,feii_options,velscale,n_basinhop,outflow_test_niter,outflow_test_options,fit_comp_options,print_output=True):
	"""
	Performs outflow tests using the outflow criteria on the [OIII] outflows.
	"""
	# The optimal fitting region for fitting outflows in [OIII] is (4400,5800) with ALL comoponents
	# fit_reg = (6200,7000)
	fit_reg = (np.max([np.min(lam_gal),6200]),np.min([np.max(lam_gal),7000]))

	# Initialize FeII and host template
	gal_temp = galaxy_template(lam_gal,age=10.0,print_output=print_output)
	if fit_comp_options['fit_feii']:
		feii_tab = initialize_feii(lam_gal,feii_options)#,line_profile,fwhm_gal,velscale,fit_reg,feii_options,run_dir)
	else: 
		feii_tab = None
	# Create mask to mask out parts of spectrum; should speed things up
	mask = np.where( (lam_gal > fit_reg[0]) & (lam_gal < fit_reg[1]) )
	lam_gal	      = lam_gal[mask]
	galaxy		  = galaxy[mask]
	noise		  = noise[mask]
	gal_temp	  = gal_temp[mask]
	fwhm_gal      = fwhm_gal[mask]
	# use SDSS spectral noise
	sigma = np.median(noise) 
	# For outflow testing, we override user FeII options so hold fwhm and voff constant
	# because it usually takes a very long time.  If FeII emission is that bad, you can't 
	# fit outflows anyway...
	if feii_options['template']['type']=='VC04':
		feii_options={
		'template':{'type':'VC04'},
		'amp_const':{'bool':False,'br_feii_val':1.0,'na_feii_val':1.0},
		'fwhm_const':{'bool':True,'br_feii_val':3000.0,'na_feii_val':500.0},
		'voff_const':{'bool':True,'br_feii_val':0.0,'na_feii_val':0.0}
		}
	elif feii_options['template']['type']=='K10':
		feii_options={
		'template'  :{'type':'K10'},
		'amp_const' :{'bool':False,'f_feii_val':1.0,'s_feii_val':1.0,'g_feii_val':1.0,'z_feii_val':1.0},
		'fwhm_const':{'bool':True,'val':1500.0},
		'voff_const':{'bool':True,'val':0.0},
		'temp_const':{'bool':True,'val':10000.0} 
		}

	# Perform fitting WITH outflows
	if print_output:
		print('\n Fitting with outflow components...')
	param_dict_outflows = initialize_mcmc(lam_gal,galaxy,line_profile,fwhm_gal,velscale,feii_options,run_dir,fit_reg=fit_reg,fit_type='init',
								 fit_feii=fit_comp_options['fit_feii'],fit_losvd=True,
								 fit_power=fit_comp_options['fit_power'],fit_broad=fit_comp_options['fit_broad'],
								 fit_narrow=True,fit_outflows=True,tie_narrow=False,print_output=print_output)

	mcpars_outflow, mccomps_outflow = max_likelihood(param_dict_outflows,lam_gal,galaxy,noise,gal_temp,
										feii_tab,feii_options,
										None,None,None,line_profile,fwhm_gal,velscale,None,None,run_dir,test_outflows=True,n_basinhop=n_basinhop,
										outflow_test_niter=outflow_test_niter,print_output=print_output)

	# Perform fitting with NO outflows
	if print_output:
		print('\n Fitting without outflow components...')
	param_dict_no_outflows = initialize_mcmc(lam_gal,galaxy,line_profile,fwhm_gal,velscale,feii_options,run_dir,fit_reg=fit_reg,fit_type='init',
								 fit_feii=fit_comp_options['fit_feii'],fit_losvd=True,
								 fit_power=fit_comp_options['fit_power'],fit_broad=fit_comp_options['fit_broad'],
								 fit_narrow=True,fit_outflows=False,tie_narrow=False,print_output=print_output)

	mcpars_no_outflow, mccomps_no_outflow = max_likelihood(param_dict_no_outflows,lam_gal,galaxy,noise,gal_temp,
										feii_tab,feii_options,
										None,None,None,line_profile,fwhm_gal,velscale,None,None,run_dir,test_outflows=True,n_basinhop=n_basinhop,
										outflow_test_niter=outflow_test_niter,print_output=print_output)

	# Determine if there is a significant improvement in residuals to warrant inclusion of a second component
	# From the outflow fit parameters, get the median values for the widths and velocity offsets to determine th 
	# optimal window for measuring the residual standard deviation of the fit 
	c = 299792.458 # speed of light (km/s)
	ref_wave = 6564.610 # reference wavelength for [OIII]
	med_outflow_fwhm = np.median(mcpars_outflow['na_Ha_outflow_fwhm'])
	med_outflow_voff = np.median(mcpars_outflow['na_Ha_outflow_voff'])
	med_core_fwhm    = np.median(mcpars_outflow['na_Ha_core_fwhm'])
	med_core_voff    = np.median(mcpars_outflow['na_Ha_core_voff'])
	
	sigma_inc = 3.0 # number of sigmas to include
	min_wave = 6500. #ref_wave + np.min([(med_outflow_voff - (sigma_inc*med_outflow_fwhm/2.3548))/c*ref_wave,(med_core_voff - (sigma_inc*med_core_fwhm/2.3548))/c*ref_wave])
	max_wave = 6600. #ref_wave + np.max([(med_outflow_voff + (sigma_inc*med_outflow_fwhm/2.3548))/c*ref_wave,(med_core_voff + (sigma_inc*med_core_fwhm/2.3548))/c*ref_wave])
	# Get indices where we perform f-test
	eval_ind = np.where((lam_gal >= min_wave) & (lam_gal <= max_wave))[0]
	# number of channels in the [OIII] test region 
	nchannel = len( mccomps_outflow['resid'][:,0][eval_ind])
	# if the number of channels < 6 (number of degrees of freedom for double-Gaussian model), then the calculated f-statistic
	# will be zero.  To resolve this, we extend the range by one pixel on each side, i.e. nchannel = 8.
	if nchannel <= 6: 
		add_chan = 7 - nchannel# number of channels to add to each side; minimum is 7 channels since deg. of freedom  = 6
		lower_pad = range(eval_ind[0]-add_chan,eval_ind[0],1)#np.arange(eval_ind[0]-add_chan,eval_ind[0],1)
		upper_pad = range(eval_ind[-1]+1,eval_ind[-1]+1+add_chan,1)
		eval_ind = np.concatenate([lower_pad, eval_ind, upper_pad],axis=0)
		nchannel = len( mccomps_outflow['resid'][:,0][eval_ind])

	# storage arrays for residuals in [OIII] test region
	resid_outflow    = np.empty((outflow_test_niter,nchannel))
	resid_no_outflow = np.empty((outflow_test_niter,nchannel))
	resid_total      = np.empty((outflow_test_niter,len(lam_gal)))
	for i in range(outflow_test_niter):
		resid_outflow[i,:]    = mccomps_outflow['resid'][:,i][eval_ind]
		resid_no_outflow[i,:] = mccomps_no_outflow['resid'][:,i][eval_ind]
		resid_total[i,:]      = mccomps_outflow['resid'][:,i]

	# Calculate sum-of-square of residuals and its uncertainty
	ssr_ratio, ssr_ratio_err, ssr_no_outflow, ssr_no_outflow_err, ssr_outflow, ssr_outflow_err = ssr_test(resid_outflow,resid_no_outflow,run_dir)
	# Perform f-test model comparison(for normally distributed model residuals)
	f_stat, f_stat_err, f_pval, f_pval_err, outflow_conf, outflow_conf_err = f_test(resid_outflow,resid_no_outflow,run_dir)

	# Calculate total residual noise
	resid_std_no_outflow     = np.median([np.std(resid_no_outflow[i,:]) for i in range(np.shape(resid_no_outflow)[0])])
	resid_std_no_outflow_err = np.std([np.std(resid_no_outflow[i,:]) for i in range(np.shape(resid_no_outflow)[0])])
	resid_std_outflow        = np.median([np.std(resid_outflow[i,:]) for i in range(np.shape(resid_outflow)[0])])
	resid_std_outflow_err    = np.std([np.std(resid_outflow[i,:]) for i in range(np.shape(resid_outflow)[0])])
	total_resid_noise        = np.median([np.std(resid_total[i,:]) for i in range(np.shape(resid_total)[0])])
	total_resid_noise_err    = np.std([np.std(resid_total[i,:]) for i in range(np.shape(resid_total)[0])])

	# Iterate through every parameter to determine if the fit is "good" (more than 1-sigma away from bounds)
	# if not, then add 1 to that parameter flag value		
	pdict = {}
	for key in mcpars_outflow:
		mc_med = np.median(mcpars_outflow[key],axis=0)
		mc_std = np.std(mcpars_outflow[key],axis=0)
		param_flags = 0
		if ( (mc_med-mc_std*outflow_test_options['bounds_test']['nsigma']) <= param_dict_outflows[key]['plim'][0]):
			param_flags += 1
		if ( (mc_med+mc_std*outflow_test_options['bounds_test']['nsigma']) >= param_dict_outflows[key]['plim'][1]):
			param_flags += 1
		pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}

	pdict_no_outflow = {}
	for key in mcpars_no_outflow:
		mc_med = np.median(mcpars_no_outflow[key],axis=0)
		mc_std = np.std(mcpars_no_outflow[key],axis=0)
		param_flags = 0
		if ( (mc_med-mc_std*outflow_test_options['bounds_test']['nsigma']) <= param_dict_no_outflows[key]['plim'][0]):
			param_flags += 1
		if ( (mc_med+mc_std*outflow_test_options['bounds_test']['nsigma']) >= param_dict_no_outflows[key]['plim'][1]):
			param_flags += 1
		pdict_no_outflow[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}

	# Outflow vs. No-outflow Model Metrics; these metrics test significance of parameters between models, and are printed to logfile 
	# regardless of whether they are tested or not.
	amp_metric = ((pdict['na_Ha_outflow_amp']['med'])/np.sqrt( (pdict['na_Ha_outflow_amp']['std'])**2 + (np.median(noise))**2 ))
	fwhm_metric = ( (pdict['na_Ha_outflow_fwhm']['med'])-(pdict['na_Ha_core_fwhm']['med']) )/np.sqrt( (pdict['na_Ha_outflow_fwhm']['std'])**2 + (pdict['na_Ha_core_fwhm']['std'])**2 )
	voff_metric =  ( (pdict['na_Ha_core_voff']['med'])-(pdict['na_Ha_outflow_voff']['med']) )/np.sqrt( (pdict['na_Ha_core_voff']['std'])**2 + (pdict['na_Ha_outflow_voff']['std'])**2 )

	# print('\n Amp. Metric = %0.2f' % amp_metric)
	# print(' FWHM Metric = %0.2f' % fwhm_metric)
	# print(' VOFF Metric = %0.2f' % voff_metric)
	# print(' Resid. Metric = %0.2f' % resid_metric)

	if print_output:
		print('\n{0:<30}{1:<25}{2:<25}{3:<25}'.format('Parameter', 'Best-fit Value', '+/- 1-sigma','Flag'))
		print('--------------------------------------------------------------------------------------')
	# Sort into arrays
	pname = []
	med   = []
	std   = []
	flag  = [] 
	for key in pdict:
		pname.append(key)
		med.append(pdict[key]['med'])
		std.append(pdict[key]['std'])
		flag.append(pdict[key]['flag'])
	i_sort = np.argsort(pname)
	pname = np.array(pname)[i_sort] 
	med   = np.array(med)[i_sort]   
	std   = np.array(std)[i_sort]   
	flag  = np.array(flag)[i_sort]
	if print_output:  
		for i in range(0,len(pname),1):
			print('{0:<30}{1:<25.2f}{2:<25.2f}{3:<25}'.format(pname[i], med[i], std[i], flag[i]))
		print('{0:<30}{1:<25.2f}'.format('median spec noise',sigma))
		print('{0:<30}{1:<25.2f}{2:<25.2e}'.format('total resid noise',total_resid_noise,total_resid_noise_err)) 
		print('{0:<30}{1:<25.2f}'.format('amp metric',amp_metric))
		print('{0:<30}{1:<25.2f}'.format('fwhm metric',fwhm_metric))
		print('{0:<30}{1:<25.2f}'.format('voff metric',voff_metric))

		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('SSR ratio',ssr_ratio,ssr_ratio_err))
		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('SSR no outflow',ssr_no_outflow,ssr_no_outflow_err))
		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('SSR outflow',ssr_outflow,ssr_outflow_err))

		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('f-statistic',f_stat,f_stat_err))
		print('{0:<30}{1:<25.2e}{2:<25.2e}'.format('p-value',f_pval,f_pval_err))
		print('{0:<30}{1:<25.5f}{2:<25.5f}'.format('Outflow confidence', outflow_conf, outflow_conf_err ) )


		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('[OIII] no outflow resid. std.',resid_std_no_outflow,resid_std_no_outflow_err))
		print('{0:<30}{1:<25.2f}{2:<25.2f}'.format('[OIII] outflow resid. std.',resid_std_outflow,resid_std_outflow_err))
		print('--------------------------------------------------------------------------------------')

	# Write to log
	write_log(pdict_no_outflow,'no_outflow_test',run_dir)
	write_log((pdict,
			   sigma,total_resid_noise,total_resid_noise_err,
			   amp_metric,fwhm_metric,voff_metric,ssr_ratio,ssr_ratio_err,
			   ssr_no_outflow, ssr_no_outflow_err, ssr_outflow, ssr_outflow_err,
			   f_stat, f_stat_err, f_pval, f_pval_err, outflow_conf, outflow_conf_err,
			   resid_std_no_outflow,resid_std_no_outflow_err,
			   resid_std_outflow,resid_std_outflow_err,
			   ),'outflow_test',run_dir)

	# Determine the significance of outflows
	outflow_conds = []
	# Outflow test criteria:
	if (outflow_test_options['amp_test']['test']==True):
		#	(1) Amp. test: (AMP_outflow - dAMP_outflow) > sigma
		amp_cond = (amp_metric > (outflow_test_options['amp_test']['nsigma']) )
		if (amp_cond==True):
			outflow_conds.append(True)
			outflow_test_options['amp_test']['pass'] = True # Add to outflow_test_options dictionary 
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow amplitude condition:', 'Pass' ) )
		elif (amp_cond==False):
			outflow_conds.append(False)
			outflow_test_options['amp_test']['pass'] = False # Add to outflow_test_options dictionary 
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow amplitude condition:', 'Fail' ) )

	#	(2) FWHM test: (FWHM_outflow - dFWHM_outflow) > (FWHM_core + dFWHM_core)
	if (outflow_test_options['fwhm_test']['test']==True):
		fwhm_cond = ( fwhm_metric > (outflow_test_options['fwhm_test']['nsigma']) )
		if (fwhm_cond==True):
			outflow_conds.append(True)
			outflow_test_options['fwhm_test']['pass'] = True # Add to outflow_test_options dictionary 
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow FWHM condition:'	 , 'Pass') )
		elif (fwhm_cond==False):
			outflow_conds.append(False)
			outflow_test_options['fwhm_test']['pass'] = False # Add to outflow_test_options dictionary 
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow FWHM condition:'	 , 'Fail') )

	#	(3) VOFF test: (VOFF_outflow + dVOFF_outflow) < (VOFF_core - dVOFF_core)
	if (outflow_test_options['voff_test']['test']==True):
		voff_cond = (voff_metric>(outflow_test_options['voff_test']['nsigma']))
		if (voff_cond==True):
			outflow_conds.append(True)
			outflow_test_options['voff_test']['pass'] = True
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow VOFF condition:'	 , 'Pass ') )
		elif (voff_cond==False):
			outflow_conds.append(False)
			outflow_test_options['voff_test']['pass'] = False
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow VOFF condition:'	 , 'Fail' ) )

	#	(4) Outflow Confidence Threshold 
	if (outflow_test_options['outflow_confidence']['test']==True):
		if outflow_conf_err <= 0.05:
			outflow_conf_cond = ((outflow_conf+outflow_conf_err)>=(outflow_test_options['outflow_confidence']['conf']))
		else:
			outflow_conf_cond = ((outflow_conf)>=(outflow_test_options['outflow_confidence']['conf']))
		if (outflow_conf_cond==True):
			outflow_conds.append(True)
			outflow_test_options['outflow_confidence']['pass'] = True
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow Confidence condition:'	 , 'Pass ') )
		elif (outflow_conf_cond==False):
			outflow_conds.append(False)
			outflow_test_options['outflow_confidence']['pass'] = False
			if print_output:
				print('{0:<35}{1:<30}'.format('Outflow Confidence condition:'	 , 'Fail' ) )

	#	(5) Parameter bounds: if parameter reached the parameter limits (bounds), then its uncertainty is zero, and something went wrong with the fit.
	if (outflow_test_options['bounds_test']['test']==True):
		bounds_cond = all([i==0 for i in [ pdict['na_Ha_outflow_fwhm']['flag'],pdict['na_Ha_outflow_voff']['flag'],pdict['na_Ha_outflow_amp']['flag'],
			     	  pdict['na_Ha_core_fwhm']['flag'],pdict['na_Ha_core_voff']['flag'],pdict['na_Ha_core_amp']['flag'] ] ])
		if (bounds_cond==True):
			outflow_conds.append(True)
			outflow_test_options['bounds_test']['pass'] = True
			if print_output:
				print('{0:<35}{1:<30}'.format('Parameter bounds condition:', 'Pass' ) )
		elif (bounds_cond==False):
			outflow_conds.append(False)
			outflow_test_options['bounds_test']['pass'] = False
			if print_output:
				print('{0:<35}{1:<30}'.format('Parameter bounds condition:', 'Fail' ) )

	# Make plot 
	# Get best fit model components for each model 
	param_names = [key for key in pdict ]
	params	    = [pdict[key]['med']  for key in pdict ]
	fit_type	 = 'outflow_test'
	output_model = False
	comp_dict_outflow = fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
						  None,None,None,line_profile,fwhm_gal,velscale,None,None,run_dir,
						  fit_type,output_model)
	param_names = [key for key in pdict_no_outflow ]
	params	    = [pdict_no_outflow[key]['med']  for key in pdict_no_outflow ]
	fit_type	 = 'outflow_test'
	output_model = False
	comp_dict_no_outflow = fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
						  None,None,None,line_profile,fwhm_gal,velscale,None,None,run_dir,
						  fit_type,output_model)
	# Make comparison plots of outflow and no-outflow models
	outflow_test_plot_nii(comp_dict_outflow,comp_dict_no_outflow,run_dir)

	# collect garbage
	del lam_gal	      
	del galaxy		  
	del noise		  
	del fwhm_gal  
	del gal_temp	  
	del feii_tab
	del param_dict_outflows
	del mcpars_outflow
	del mccomps_outflow
	del param_dict_no_outflows
	del mcpars_no_outflow
	del mccomps_no_outflow
	del pdict_no_outflow
	del comp_dict_outflow
	del comp_dict_no_outflow
	gc.collect()

	if not outflow_conds: # outflow_conds is empty because no tests were performed but we tested for outflows anyway
		return True,pdict,sigma
	elif (all(outflow_conds)==True):
		# Write to log
		write_log(outflow_test_options,'outflow_test_pass',run_dir)
		return True,pdict,sigma
	elif (all(outflow_conds)==False):
		# Write to log
		write_log(outflow_test_options,'outflow_test_fail',run_dir)
		return False,pdict,sigma

##################################################################################

def outflow_test_plot_nii(comp_dict_outflow,comp_dict_no_outflow,run_dir):
	"""
	The plotting function for outflow_test_oiii().  It plots both the outflow
	and no_outflow results.
	"""

	# Creat plot window and axes
	fig = plt.figure(figsize=(14,11)) 
	gs = gridspec.GridSpec(9,1)
	ax1  = fig.add_subplot(gs[0:3,0]) # No outflow
	ax2  = fig.add_subplot(gs[3:4,0]) # No outflow residuals
	ax3  = fig.add_subplot(gs[5:8,0]) # Outflow
	ax4  = fig.add_subplot(gs[8:9,0]) # Outflow residuals
	gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
	# No outflow model (ax1,ax2)
	norm = np.median(comp_dict_no_outflow['data']['comp'])
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['data']['comp']              , color='xkcd:white'      , linewidth=0.5, linestyle='-' , label='Data'        )  
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['model']['comp']             , color='xkcd:red'        , linewidth=1.0, linestyle='-' , label='Model'       ) 
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['host_galaxy']['comp']       , color='xkcd:lime green' , linewidth=1.0, linestyle='-' , label='Galaxy'      )
	if ('power' in comp_dict_no_outflow):
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['power']['comp']             , color='xkcd:orange red' , linewidth=1.0, linestyle='--', label='AGN Cont.'   )
	if ('na_feii_template' in comp_dict_no_outflow) and ('br_feii_template' in comp_dict_no_outflow):
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_feii_template']['comp'], color='xkcd:yellow'     , linewidth=1.0, linestyle='-' , label='Na. FeII'     )
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['br_feii_template']['comp'], color='xkcd:orange'     , linewidth=1.0, linestyle='-' , label='Br. FeII'     )
	elif ('F_feii_template' in comp_dict_no_outflow) and ('S_feii_template' in comp_dict_no_outflow) and ('G_feii_template' in comp_dict_no_outflow) and ('Z_feii_template' in comp_dict_no_outflow):
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['F_feii_template']['comp'], color='xkcd:yellow'     , linewidth=1.0, linestyle='-' , label='F-transition FeII'     )
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['S_feii_template']['comp'], color='xkcd:mustard'     , linewidth=1.0, linestyle='-' , label='S_transition FeII'     )
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['G_feii_template']['comp'], color='xkcd:orange'     , linewidth=1.0, linestyle='-' , label='G_transition FeII'     )
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['Z_feii_template']['comp'], color='xkcd:rust'     , linewidth=1.0, linestyle='-' , label='Z_transition FeII'     )
	if ('br_Ha' in comp_dict_no_outflow):
		ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['br_Ha']['comp']             , color='xkcd:turquoise'  , linewidth=1.0, linestyle='-' , label='Br. H-beta'      )
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_Ha_core']['comp']        , color='xkcd:dodger blue', linewidth=1.0, linestyle='-' , label='Core comp.'  )
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_nii6549_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                       )
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_nii6585_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                       )
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_sii6718_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                       )
	ax1.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_sii6732_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                       )
	ax1.axvline(6549.86, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax1.axvline(6564.61, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax1.axvline(6585.27, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax1.axvline(6718.29, color='xkcd:white' , linewidth=0.5, linestyle='--') 
	ax1.axvline(6732.67, color='xkcd:white' , linewidth=0.5, linestyle='--')     
	# ax1.plot(comp_dict_no_outflow['wave']['comp'], 1*comp_dict_no_outflow['noise']['comp'], color='xkcd:dodger blue' , linewidth=0.5, linestyle='--')
	# ax1.plot(comp_dict_no_outflow['wave']['comp'], 2*comp_dict_no_outflow['noise']['comp'], color='xkcd:lime green'  , linewidth=0.5, linestyle='--')
	# ax1.plot(comp_dict_no_outflow['wave']['comp'], 3*comp_dict_no_outflow['noise']['comp'], color='xkcd:orange red'  , linewidth=0.5, linestyle='--')
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)')
	ax1.set_xticklabels([])
	ax1.legend(loc='upper left',fontsize=6)
	ax1.set_xlim(np.min(comp_dict_outflow['wave']['comp']),np.max(comp_dict_outflow['wave']['comp']))
	ax1.set_ylim(0.0,np.max(comp_dict_no_outflow['model']['comp'])+3*np.median(comp_dict_no_outflow['noise']['comp']))
	ax1.set_title('No Outflow Model')
	# No Outflow Residuals
	ax2.plot(comp_dict_no_outflow['wave']['comp'],3*(comp_dict_no_outflow['data']['comp']-comp_dict_no_outflow['model']['comp']), color='xkcd:white' , linewidth=0.5, linestyle='-')
	ax2.axvline(6549.86, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.axvline(6564.61, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.axvline(6585.27, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.axvline(6718.29, color='xkcd:white' , linewidth=0.5, linestyle='--') 
	ax2.axvline(6732.67, color='xkcd:white' , linewidth=0.5, linestyle='--')  
	ax2.axhline(0.0, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax2.plot(comp_dict_no_outflow['wave']['comp'], 3*1*comp_dict_no_outflow['noise']['comp'], color='xkcd:bright aqua' , linewidth=0.5, linestyle='-')
	# ax2.plot(comp_dict_no_outflow['wave']['comp'], 3*2*comp_dict_no_outflow['noise']['comp'], color='xkcd:lime green'  , linewidth=0.5, linestyle='--')
	# ax2.plot(comp_dict_no_outflow['wave']['comp'], 3*3*comp_dict_no_outflow['noise']['comp'], color='xkcd:orange red'  , linewidth=0.5, linestyle='--')
	ax2.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)')
	ax2.set_ylabel(r'$\Delta f_\lambda$')
	ax2.set_xlim(np.min(comp_dict_outflow['wave']['comp']),np.max(comp_dict_outflow['wave']['comp']))
	ax2.set_ylim(0.0-9*np.std(comp_dict_no_outflow['resid']['comp']),ax1.get_ylim()[1])
    # Outlfow models (ax3,ax4)
	norm = np.median(comp_dict_outflow['data']['comp'])
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['data']['comp']              , color='xkcd:white'      , linewidth=0.5, linestyle='-' , label='Data'        )  
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['model']['comp']             , color='xkcd:red'        , linewidth=1.0, linestyle='-' , label='Model'       ) 
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['host_galaxy']['comp']       , color='xkcd:lime green' , linewidth=1.0, linestyle='-' , label='Galaxy'      )
	if ('power' in comp_dict_outflow):
		ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['power']['comp']             , color='xkcd:orange red' , linewidth=1.0, linestyle='--', label='AGN Cont.'   )
	if ('na_feii_template' in comp_dict_outflow) and ('br_feii_template' in comp_dict_outflow):
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['na_feii_template']['comp'], color='xkcd:yellow'     , linewidth=1.0, linestyle='-' , label='Na. FeII'     )
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['br_feii_template']['comp'], color='xkcd:orange'     , linewidth=1.0, linestyle='-' , label='Br. FeII'     )
	elif ('F_feii_template' in comp_dict_outflow) and ('S_feii_template' in comp_dict_outflow) and ('G_feii_template' in comp_dict_outflow) and ('Z_feii_template' in comp_dict_outflow):
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['F_feii_template']['comp'], color='xkcd:yellow'     , linewidth=1.0, linestyle='-' , label='F-transition FeII'     )
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['S_feii_template']['comp'], color='xkcd:mustard'     , linewidth=1.0, linestyle='-' , label='S_transition FeII'     )
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['G_feii_template']['comp'], color='xkcd:orange'     , linewidth=1.0, linestyle='-' , label='G_transition FeII'     )
		ax3.plot(comp_dict_no_outflow['wave']['comp'], comp_dict_no_outflow['Z_feii_template']['comp'], color='xkcd:rust'     , linewidth=1.0, linestyle='-' , label='Z_transition FeII'     )
	if ('br_Ha' in comp_dict_outflow):
		ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['br_Ha']['comp']             , color='xkcd:turquoise'  , linewidth=1.0, linestyle='-' , label='Br. H-beta'      )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_Ha_core']['comp']        , color='xkcd:dodger blue', linewidth=1.0, linestyle='-' , label='Core comp.'  )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_nii6549_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                       )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_nii6585_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                       )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_sii6718_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                       )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_sii6732_core']['comp']   , color='xkcd:dodger blue', linewidth=1.0, linestyle='-'                       )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_Ha_outflow']['comp']     , color='xkcd:magenta'    , linewidth=1.0, linestyle='-', label='Outflow comp.')
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_nii6549_outflow']['comp'], color='xkcd:magenta'    , linewidth=1.0, linestyle='-'                       )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_nii6585_outflow']['comp'], color='xkcd:magenta'    , linewidth=1.0, linestyle='-'                       )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_sii6718_outflow']['comp'], color='xkcd:magenta'    , linewidth=1.0, linestyle='-'                       )
	ax3.plot(comp_dict_outflow['wave']['comp'], comp_dict_outflow['na_sii6732_outflow']['comp'], color='xkcd:magenta'    , linewidth=1.0, linestyle='-'                       )
	ax3.axvline(6549.86, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax3.axvline(6564.61, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax3.axvline(6585.27, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax3.axvline(6718.29, color='xkcd:white' , linewidth=0.5, linestyle='--') 
	ax3.axvline(6732.67, color='xkcd:white' , linewidth=0.5, linestyle='--')    
	# ax3.plot(comp_dict_outflow['wave']['comp'], 1*comp_dict_outflow['noise']['comp'], color='xkcd:dodger blue' , linewidth=0.5, linestyle='--')
	# ax3.plot(comp_dict_outflow['wave']['comp'], 2*comp_dict_outflow['noise']['comp'], color='xkcd:lime green'  , linewidth=0.5, linestyle='--')
	# ax3.plot(comp_dict_outflow['wave']['comp'], 3*comp_dict_outflow['noise']['comp'], color='xkcd:orange red'  , linewidth=0.5, linestyle='--')
	ax3.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)')
	ax3.set_xticklabels([])
	ax3.legend(loc='upper left',fontsize=6)
	ax3.set_xlim(np.min(comp_dict_outflow['wave']['comp']),np.max(comp_dict_outflow['wave']['comp']))
	ax3.set_ylim(0.0,np.max(comp_dict_outflow['model']['comp'])+3*np.median(comp_dict_outflow['noise']['comp']))
	ax3.set_title('Outflow Model')
	# Outflow Residuals
	ax4.plot(comp_dict_outflow['wave']['comp'],3*(comp_dict_outflow['data']['comp']-comp_dict_outflow['model']['comp']), color='xkcd:white' , linewidth=0.5, linestyle='-')
	ax4.axvline(6549.86, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axvline(6564.61, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axvline(6585.27, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axvline(6718.29, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axvline(6732.67, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.axhline(0.0, color='xkcd:white' , linewidth=0.5, linestyle='--')
	ax4.plot(comp_dict_outflow['wave']['comp'], 3*1*comp_dict_outflow['noise']['comp'], color='xkcd:bright aqua' , linewidth=0.5, linestyle='-')
	# ax4.plot(comp_dict_outflow['wave']['comp'], 3*2*comp_dict_outflow['noise']['comp'], color='xkcd:lime green'  , linewidth=0.5, linestyle='--')
	# ax4.plot(comp_dict_outflow['wave']['comp'], 3*3*comp_dict_outflow['noise']['comp'], color='xkcd:orange red'  , linewidth=0.5, linestyle='--')
	ax4.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)')
	ax4.set_ylabel(r'$\Delta f_\lambda$')
	ax4.set_xlim(np.min(comp_dict_outflow['wave']['comp']),np.max(comp_dict_outflow['wave']['comp']))
	ax4.set_ylim(0.0-9*np.std(comp_dict_outflow['resid']['comp']),ax3.get_ylim()[1])
    
	fig.tight_layout()
	plt.savefig(run_dir+'outflow_test.pdf',fmt='pdf',dpi=150)

	plt.close()
	# Collect garbage
	del ax1
	del ax2
	del ax3
	del ax4
	del fig 
	del comp_dict_outflow
	del comp_dict_no_outflow
	gc.collect()

	return None

##################################################################################

#### Maximum Likelihood Fitting ##################################################

def max_likelihood(param_dict,lam_gal,galaxy,noise,gal_temp,
				   feii_tab,feii_options,
				   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
				   fit_type='init',output_model=False,
				   test_outflows=False,n_basinhop=10,outflow_test_niter=10,max_like_niter=10,
				   print_output=True):

	"""
	This function performs an initial maximum likelihood estimation to acquire robust
	initial parameters.  It performs the monte carlo bootstrapping for both 
	testing outflows and fit for final initial parameters for emcee.

	*NOTE*: As of v7.4.1, we have started using the scipy.optimize.basinhopping algorithm with SLSQP
	minimization.  The produces a far superior fit due to the fact that SLSQP alone 
	get stuck in local minima, namely for the power_slope parameter.  While the 
	basinhopping algorithm requires more time, it provides a significantly better
	fit for the power-law component.  We then only use SLSQP algorithm (with no 
	basinhopping) for the monte carlo bootstrapping iterations.  
	"""
	param_names  = [param_dict[key]['name'] for key in param_dict ]
	params	     = [param_dict[key]['init'] for key in param_dict ]
	bounds	     = [param_dict[key]['plim'] for key in param_dict ]

	# Constraints for Outflow components
	def oiii_amp_constraint(p):
		# (core_amp >= outflow_amp) OR (core_amp - outflow_amp >= 0)
		return p[param_names.index('na_oiii5007_core_amp')]-p[param_names.index('na_oiii5007_outflow_amp')]
	def oiii_fwhm_constraint(p):
		# (outflow_fwhm >= core_fwhm) OR (outflow_fwhm - core_fwhm >= 0)
		return p[param_names.index('na_oiii5007_outflow_fwhm')]-p[param_names.index('na_oiii5007_core_fwhm')]
	# def oiii_voff_constraint(p):
		# (core_voff >= outflow_voff) OR (core_voff - outflow_voff >= 0)
		# return p[param_names.index('na_oiii5007_core_voff')]-p[param_names.index('na_oiii5007_outflow_voff')]
	def nii_amp_constraint(p):
		# (core_amp >= outflow_amp) OR (core_amp - outflow_amp >= 0)
		return p[param_names.index('na_Ha_core_amp')]-p[param_names.index('na_Ha_outflow_amp')]
	def nii_fwhm_constraint(p):
		# (outflow_fwhm >= core_fwhm) OR (outflow_fwhm - core_fwhm >= 0)
		return p[param_names.index('na_Ha_outflow_fwhm')]-p[param_names.index('na_Ha_core_fwhm')]
	# def nii_voff_constraint(p):
		# (core_voff >= outflow_voff) OR (core_voff - outflow_voff >= 0)
		# return p[param_names.index('na_Ha_core_voff')]-p[param_names.index('na_Ha_outflow_voff')]
	# Constraints for narrow and broad lines 
	# Broad H-beta have a greater width than na. Hb/[OIII]
	def hb_fwhm_constraint(p):
		# (br_Hb_fwhm >= na_Hb_fwhm) OR (br_Hb_fwhm - na_Hb_fwhm >= 0)
		# but we use [OIII]5007 core FWHM to constrain na. Hb. FWHM, so
		return p[param_names.index('br_Hb_fwhm')]-p[param_names.index('na_oiii5007_core_fwhm')]
	# Broad H-alpha must have a greater width than na. Ha/[NII]/[SII]
	def ha_fwhm_constraint(p):
		# (br_Ha_fwhm >= na_Ha_fwhm) OR (br_Ha_fwhm - na_Ha_fwhm >= 0)
		return p[param_names.index('br_Ha_fwhm')]-p[param_names.index('na_Ha_core_fwhm')]
	# For TIED option
	def ha_tied_constraint(p): # Constraint for Model 12 if narrow lines are TIED
		# (br_Ha_fwhm >= na_oiii5007_core_fwhm) OR (br_Ha_fwhm - na_oiii5007_core_fwhm >= 0)
		return p[param_names.index('br_Ha_fwhm')]-p[param_names.index('na_oiii5007_core_fwhm')]

	# Constraints 1: [OIII]/na. Hb outflows, NO broad lines
	cons1 = [{'type':'ineq','fun': oiii_amp_constraint  },
			 {'type':'ineq','fun': oiii_fwhm_constraint },]
			 # {'type':'ineq','fun': oiii_voff_constraint }]
	# Constraints 2: [OIII]/na. Hb outflows, WITH broad lines
	cons2 = [{'type':'ineq','fun': oiii_amp_constraint  },
			 {'type':'ineq','fun': oiii_fwhm_constraint },
			 # {'type':'ineq','fun': oiii_voff_constraint },
			 {'type':'ineq','fun': hb_fwhm_constraint   }]	  
	# Constraints 3: [NII]/[SII]/na Ha. outflows, NO broad lines
	cons3 = [{'type':'ineq','fun': nii_amp_constraint   },
			 {'type':'ineq','fun': nii_fwhm_constraint  },]
			 # {'type':'ineq','fun': nii_voff_constraint  }]
	# Constraints 4: [NII]/[SII]/na Ha. outflows, NO broad lines
	cons4 = [{'type':'ineq','fun': nii_amp_constraint   },
			 {'type':'ineq','fun': nii_fwhm_constraint  },
			 # {'type':'ineq','fun': nii_voff_constraint  },
			 {'type':'ineq','fun': ha_fwhm_constraint   }]
	# Constraints 5: Hb/[OIII] and Ha/[NII]/[SII], NO broad lines
	cons5 = [{'type':'ineq','fun': oiii_amp_constraint  },
			 {'type':'ineq','fun': oiii_fwhm_constraint },]
			 # {'type':'ineq','fun': oiii_voff_constraint }]
	# Constraints 6: Hb/[OIII] and Ha/[NII]/[SII], WITH broad lines
	cons6 = [{'type':'ineq','fun': oiii_amp_constraint  },
			 {'type':'ineq','fun': oiii_fwhm_constraint },
			 # {'type':'ineq','fun': oiii_voff_constraint },
			 {'type':'ineq','fun': hb_fwhm_constraint   },
			 {'type':'ineq','fun': ha_fwhm_constraint   }]
	# Constraints  7: No outflows, NO Broad lines Hb/[OIII] -> No constraints
	# Constraints  8: No outflows, WITH Broad lines Hb/[OIII]
	cons8 = [{'type':'ineq','fun': hb_fwhm_constraint   }]
	# Constraints  9: No outflows, NO Broad lines  Ha/[NII]/[SII] -> No constraints
	# Constraints 10: No outflows, WITH Broad lines  Ha/[NII]/[SII]
	cons10 = [{'type':'ineq','fun': ha_fwhm_constraint   }]
	# Constraints 11: No outflows, NO Broad lines  Hb/[OIII] + Ha/[NII]/[SII] -> No constraints
	# Constraints 12: No outflows, WITH Broad lines  Hb/[OIII] + Ha/[NII]/[SII]
	cons12 = [{'type':'ineq','fun': hb_fwhm_constraint   },
			  {'type':'ineq','fun': ha_fwhm_constraint   }]
	# Constraints 13: narrow lines TIED, with outflows, constraint for broad H-alpha (modifies constraints 6)
	cons13 = [{'type':'ineq','fun': oiii_amp_constraint  },
			  {'type':'ineq','fun': oiii_fwhm_constraint },
			  # {'type':'ineq','fun': oiii_voff_constraint },
			  {'type':'ineq','fun': hb_fwhm_constraint   },
			  {'type':'ineq','fun': ha_tied_constraint   }]	
	# Constraints 14: narrow lines TIED, no outflows, constraint for broad H-alpha (modifies constraints 12)
	cons14 = [{'type':'ineq','fun': hb_fwhm_constraint   },
			  {'type':'ineq','fun': ha_tied_constraint   }]		 
	#
	# Perform maximum likelihood estimation for initial guesses of MCMC fit
	# Below are 12 Models used for determination of constraints.
	# Model 1: Hb/[OIII] Region with outflows, NO broad lines (excludes Ha/[NII]/[SII] region and outflows)
	if (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											 'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True) and \
	   	(all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
	   											 'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   											 'nii6585_core_amp',
	   											 'sii6732_core_amp','sii6718_core_amp',
	   											 'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
	   											 'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  Constraining outflow components for Hb/[OIII] region. \n ')
			print('\n Model 1: Hb/[OIII] Region with outflows, NO broad lines (excludes Ha/[NII]/[SII] region and outflows) \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = cons1
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# Model 2: Hb/[OIII] Region with outflows AND broad lines (excludes Ha/[NII]/[SII] region and outflows)
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
											   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff',])==True) and \
		 (all(comp not in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   											 'nii6585_core_amp',
	   											 'sii6732_core_amp','sii6718_core_amp',
	   											 'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
	   											 'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  Constraining outflow components for Hb/[OIII] region. \n ')
			print('\n Model 2: Hb/[OIII] Region with outflows AND broad lines (excludes Ha/[NII]/[SII] region and outflows) \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = cons2
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# Model 3: Ha/[NII]/[SII] region with outflows, no broad lines (excludes Hb/[OIII] region and outflows)
	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  Constraining outflow components for Ha/[NII]/[SII] region. \n ')
			print('\n Model 3: Ha/[NII]/[SII] region with outflows, no broad lines (excludes Hb/[OIII] region and outflows) \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = cons3
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# Model 4: Ha/[NII]/[SII] region with outflows, WITH broad lines (excludes Hb/[OIII] region and outflows)
	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   										   'na_nii6585_core_amp',
	   										   'na_sii6732_core_amp','na_sii6718_core_amp',
	   										   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
	   										   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True):
		if print_output: 
			print('\n Performing max. likelihood fitting.  Constraining outflow components for Ha/[NII]/[SII] region. \n ')
			print('\n Model 4: Ha/[NII]/[SII] region with outflows, WITH broad lines (excludes Hb/[OIII] region and outflows. \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = cons4
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# Model 5: Hb/[OIII] + Ha/[NII]/[SII] regions with outflows, NO broad lines
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
											   'na_Ha_core_amp','na_Ha_core_voff',#,'na_Ha_core_fwhm'
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp'])==True) and \
	 	 (all(comp not in param_names for comp in ['na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
	 	 										   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  Constraining outflow components for Hb/[OIII] + Ha/[NII]/[SII] regions. \n ')
			print('\n Model 5: Hb/[OIII] + Ha/[NII]/[SII] regions with outflows, NO broad lines \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = cons5
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# Model 6: Hb/[OIII] + Ha/[NII]/[SII] regions with outflows, WITH broad lines
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
											   'na_Ha_core_amp','na_Ha_core_voff',#,'na_Ha_core_fwhm'
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
											   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True) and \
		 (all(comp not in param_names for comp in ['na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  Constraining outflow components for Hb/[OIII] + Ha/[NII]/[SII] regions. \n ')
			print('\n Model 6: Hb/[OIII] + Ha/[NII]/[SII] regions with outflows, WITH broad lines \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		# If narrow components are NOT tied, use na_Ha_core_fwhm (constraint 12), 
		# otherwise use na_oiii5007_core_fwhm (constraint 13)
		if 'na_Ha_core_fwhm' in param_names: 
			cons = cons6
		elif 'na_Ha_core_fwhm' not in param_names:
			cons = cons13
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# Model 7: NO outflows, NO broad lines, Hb/[OIII]
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
		 										   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
		 										   'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
												   'na_nii6585_core_amp',
												   'na_sii6732_core_amp','na_sii6718_core_amp',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  No constraints for Ha/[NII]/[SII] region. \n ')
			print('\n Model 7: NO outflows, NO broad lines, Hb/[OIII] region \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = None
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds},disp=False,niter_success=n_basinhop)
	#
	# Model 8: NO outflows, WITH broad lines, Hb/[OIII]
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff'])==True) and \
		 (all(comp not in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
		 										   'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
												   'na_nii6585_core_amp',
												   'na_sii6732_core_amp','na_sii6718_core_amp',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  No constraints for Ha/[NII]/[SII] region. \n ')
			print('\n Model 8: NO outflows, WITH broad lines, Hb/[OIII] region \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = cons8
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# Model 9: NO outflows, NO broad lines, Ha/[NII]/[SII]
	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  No constraints for Ha/[NII]/[SII] region. \n ')
			print('\n Model 9: NO outflows, NO broad lines, Ha/[NII]/[SII] region \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = None
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds},disp=False,niter_success=n_basinhop)
	#
	# Model 10: NO outflows, WITH broad lines, Ha/[NII]/[SII]
	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  Constraining broad and narrow line widths for Ha/[NII]/[SII] region. \n ')
			print('\n Model 10: NO outflows, WITH broad lines, Ha/[NII]/[SII] region \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = cons10
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# Model 11: NO outflows, NO broad lines, Hb/[OIII] + Ha/[NII]/[SII]
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_Ha_core_amp','na_Ha_core_voff',#,'na_Ha_core_fwhm'
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  No constraints for for Hb/[OIII] + Ha/[NII]/[SII] region. \n ')
			print('\n Model 11: NO outflows, NO broad lines, Hb/[OIII] + Ha/[NII]/[SII] region \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		cons = None
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 		feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds},disp=False,niter_success=n_basinhop)
	#
	# Model 12: NO outflows, WITH broad lines, Hb/[OIII] + Ha/[NII]/[SII]
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
											   'na_Ha_core_amp','na_Ha_core_voff',#,'na_Ha_core_fwhm'
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True) and \
		 (all(comp not in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if print_output:
			print('\n Performing max. likelihood fitting.  Constraining broad and narrow line widths for Hb/[OIII] + Ha/[NII]/[SII] region. \n ')
			print('\n Model 12: NO outflows, WITH broad lines, Hb/[OIII] + Ha/[NII]/[SII] region \n')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		# If narrow components are NOT tied, use na_Ha_core_fwhm (constraint 12), 
		# otherwise use na_oiii5007_core_fwhm (constraint 13)
		if 'na_Ha_core_fwhm' in param_names: 
			cons = cons12
		elif 'na_Ha_core_fwhm' not in param_names:
			cons = cons14
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 					  feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds, 'constraints':cons},disp=False,niter_success=n_basinhop)
	#
	# No constraint model:
	else: 
		if print_output:
			print('\n Performing max. likelihood fitting.  No parameter constraints. \n ')
			print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d \n ' % (n_basinhop))
		# No constraint model, only bounds are used.
		cons = None
		start_time = time.time()
		nll = lambda *args: -lnlike(*args)

		result = op.basinhopping(func = nll, x0 = params, \
			 				 minimizer_kwargs = {'args':(param_names,lam_gal,galaxy,noise,gal_temp,
			 				 					  feii_tab,feii_options,
			 				 	   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   fit_type,output_model),\
			 				 'method':'SLSQP', 'bounds':bounds},disp=False,niter_success=n_basinhop)


	elap_time = (time.time() - start_time)

	###### Monte Carlo Bootstrapping for Outflow Test ###########################################################

	if ((test_outflows==True) ):

		# Get component dictionary of best-fit
		fit_type	 = 'outflow_test'
		output_model = False # do not output the model, just the component dictionary
		comp_dict = fit_model(result['x'],param_names,lam_gal,galaxy,noise,gal_temp,
							  feii_tab,feii_options,
							  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
							  fit_type,output_model)

		# Construct random normally-distributed noise
		# How we do the monte carlo bootstrapping (i.e., the proper way):
		# (1) The 1-sigma uncertainty (spectral "noise") from inverse variance of the SDSS spectra is 
		# 	  the pixel-to-pixel variation in the spectrum when rows of pixels are added to form the final 1-d spectrum. 
		# 	  This is always an underestimate of the true noise in the spectrum. 
		# (2) The residual noise from a fit, taken to be the median absolute deviation of the residuals from a fit.  This 
		#	 is always greater than the "noise" from (1), but closer to the actual value of the noise across the fitting 
		#	  region.  
		#  We add (1) and (2) in quadrature to simulate the noise at /every/ pixel in the fitting region.

		mcnoise = np.array(noise)#np.sqrt(np.array(noise)**2 + (resid_std)**2) # the residual noise and spectral noise added in quadrature
		# Initialize a parameter dictionary, which for each parameter, there will be an array of parameters
		mcpars = {}
		for par in param_names:
			mcpars[par] =  np.empty(outflow_test_niter)
		# Initialize a components dictionary, which for each components, there will be an array of components
		mccomps = {}
		for key in comp_dict:
			mccomps[key] = np.empty((len(comp_dict[key]['comp']), outflow_test_niter)) 
		if print_output:
			print( '\n Performing Monte Carlo bootstrapping...')
			# print( '\n Approximate time for %d iterations: %s \n' % (outflow_test_niter,time_convert(elap_time*outflow_test_niter))  )

		for n in range(0,outflow_test_niter,1):
			# Generate a simulated galaxy spectrum with noise added at each pixel
			mcgal  = np.random.normal(galaxy,mcnoise)
			# Get rid of any infs or nan if there are none; this will cause scipy.optimize to fail
			mcgal[mcgal/mcgal!=1] = np.median(mcgal)
			fit_type	 = 'init'
			output_model = False

			if cons is not None:

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, x0 = result['x'], \
			 				 		 args=(param_names,lam_gal,mcgal,mcnoise,gal_temp,
			 				 		 	   feii_tab,feii_options,
			 				 	   		   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   		   fit_type,output_model),\
			 				 		 method='SLSQP', bounds = bounds, constraints=cons, options={'maxiter':2500,'disp': False})
			elif cons is None:

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, x0 = result['x'], \
			 				 		 args=(param_names,lam_gal,mcgal,mcnoise,gal_temp,
			 				 		 	   feii_tab,feii_options,
			 				 	   		   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   		   fit_type,output_model),\
			 				 		 method='SLSQP', bounds = bounds, options={'maxiter':2500,'disp': False})
			# Append parameters to parameter dictionary
			for p,par in enumerate(param_names):	
				mcpars[par][n] = resultmc['x'][p]
			# Generate model components for each monte carlo iteration
			fit_type	 = 'outflow_test'
			output_model = False
			comp_dict = fit_model(resultmc['x'],param_names,lam_gal,galaxy,noise,gal_temp,
								  feii_tab,feii_options,
								  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
								  fit_type,output_model)
			# Append comp_dict for this monte-carlo iteration to the mccomps array
			for key in comp_dict:
				mccomps[key][:,n] = comp_dict[key]['comp']

			if print_output:
				print('	   Completed %d of %d iterations.' % (n+1,outflow_test_niter) )

		# Collect garbage
		del param_dict
		del lam_gal
		del galaxy
		del noise
		del gal_temp
		del feii_tab
		del temp_list
		del temp_fft
		del params	   
		del bounds	 
		del result
		del mcgal
		del resultmc
		gc.collect()

		# Return to outflow_test function to determine if outflows are present
		return mcpars, mccomps

	elif (test_outflows==False): # max-likelihoof fitting for initial parameters for emcee

		par_best	 = result['x']
		fit_type	 = 'init'
		output_model = True

		comp_dict = fit_model(par_best,param_names,lam_gal,galaxy,noise,gal_temp,
							  feii_tab,feii_options,
							  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
							  fit_type,output_model)
	
		# Construct random normally-distributed noise
		# How we do the monte carlo bootstrapping (i.e., the proper way):
		# (1) The 1-sigma uncertainty (spectral "noise") from inverse variance of the SDSS spectra is 
		# 	  the pixel-to-pixel variation in the spectrum when rows of pixels are added to form the final 1-d spectrum. 
		# 	  This is always an underestimate of the true noise in the spectrum. 
		# (2) The residual noise from a fit, taken to be the median absolute deviation of the residuals from a fit.  This 
		#	 is always greater than the "noise" from (1), but closer to the actual value of the noise across the fitting 
		#	  region.  
		#  We add (1) and (2) in quadrature to simulate the noise at /every/ pixel in the fitting region.

		resid_std = mad_std(comp_dict['resid']['comp']) # the residual noise
		mcnoise = np.array(noise)#np.sqrt(np.array(noise)**2 + (resid_std)**2) # the residual noise and spectral noise added in quadrature
		mcpars = np.empty((len(par_best), max_like_niter)) # stores best-fit parameters of each MC iteration

		if print_output:
			print( '\n Performing Monte Carlo bootstrapping...')
			# print( '\n	   Approximate time for %d iterations: %s \n' % (max_like_niter,time_convert(elap_time*max_like_niter))  )

		for n in range(0,max_like_niter,1):
			# Generate a simulated galaxy spectrum with noise added at each pixel
			mcgal  = np.random.normal(galaxy,mcnoise)
			# Get rid of any infs or nan if there are none; this will cause scipy.optimize to fail
			mcgal[mcgal/mcgal!=1] = np.median(mcgal)
			fit_type	 = 'init'
			output_model = False

			if (cons is not None):

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, x0 = result['x'], \
			 				 		 args=(param_names,lam_gal,mcgal,mcnoise,gal_temp,
			 				 		 	   feii_tab,feii_options,
			 				 	   		   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   		   fit_type,output_model),\
			 				 		 method='SLSQP', bounds = bounds, constraints=cons, options={'maxiter':2500,'disp': False})

			elif (cons is None):

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, x0 = result['x'], \
			 				 		 args=(param_names,lam_gal,mcgal,mcnoise,gal_temp,
			 				 		 	   feii_tab,feii_options,
			 				 	   		   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			 				 	   		   fit_type,output_model),\
			 				 		 method='SLSQP', bounds = bounds, options={'maxiter':2500,'disp': False})
			mcpars[:,n] = resultmc['x']
			if print_output:
				print('	   Completed %d of %d iterations.' % (n+1,max_like_niter) )
			# For testing: plots every max. likelihood iteration
	
		# create dictionary of param names, means, and standard deviations
		mc_med = np.median(mcpars,axis=1)
		mc_std = mad_std(mcpars,axis=1)
		# Iterate through every parameter to determine if the fit is "good" (more than 1-sigma away from bounds)
		# if not, then add 1 to that parameter flag value			
		pdict = {}
		for k in range(0,len(param_names),1):
			param_flags = 0
			if (mc_med[k]-mc_std[k] <= bounds[k][0]):
				param_flags += 1
			if (mc_med[k]+mc_std[k] >= bounds[k][1]):
				param_flags += 1
			pdict[param_names[k]] = {'med':mc_med[k],'std':mc_std[k],'flag':param_flags}

		if 1:
			output_model = True
			comp_dict = fit_model([pdict[key]['med'] for key in pdict],pdict.keys(),
								  lam_gal,galaxy,noise,gal_temp,
								  feii_tab,feii_options,
								  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
								  fit_type,output_model)
			fig = plt.figure(figsize=(10,6)) 
			gs = gridspec.GridSpec(4, 1)
			gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
			ax1  = plt.subplot(gs[0:3,0])
			ax2  = plt.subplot(gs[3,0])

			for key in comp_dict:
				# Galaxy + Best-fit Model
				if (key is not 'resid') and (key is not 'noise') and (key is not 'wave') and (key is not 'data'):
					ax1.plot(lam_gal,comp_dict[key]['comp'],linewidth=comp_dict[key]['linewidth'],color=comp_dict[key]['pcolor'],label=key,zorder=15)
				if (key not in ['resid','noise','wave','data','model','na_feii_template','br_feii_template','host_galaxy','power']):
					ax1.axvline(lam_gal[np.where(comp_dict[key]['comp']==np.max(comp_dict[key]['comp']))[0][0]],color='xkcd:white',linestyle='--',linewidth=0.5)
					ax2.axvline(lam_gal[np.where(comp_dict[key]['comp']==np.max(comp_dict[key]['comp']))[0][0]],color='xkcd:white',linestyle='--',linewidth=0.5)
		
			ax1.plot(lam_gal,comp_dict['data']['comp'],linewidth=0.5,color='white',label='data',zorder=0)
		
			ax1.set_xticklabels([])
			ax1.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
			ax1.set_ylim(-0.5*np.median(comp_dict['model']['comp']),np.max([comp_dict['data']['comp'],comp_dict['model']['comp']]))
			ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=12)
			# Residuals
			sigma_resid = np.std(comp_dict['data']['comp']-comp_dict['model']['comp'])
			sigma_noise = np.median(comp_dict['noise']['comp'])
			ax2.plot(lam_gal,(comp_dict['noise']['comp'])*3,linewidth=comp_dict['noise']['linewidth'],color=comp_dict['noise']['pcolor'],label='$\sigma_{\mathrm{noise}}=%0.4f$' % (sigma_noise))
			ax2.plot(lam_gal,(comp_dict['resid']['comp'])*3,linewidth=comp_dict['resid']['linewidth'],color=comp_dict['resid']['pcolor'],label='$\sigma_{\mathrm{resid}}=%0.4f$' % (sigma_resid))
			ax2.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
			ax2.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
			ax2.set_ylim(ax1.get_ylim())
			ax2.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
			ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
			ax2.set_yticks([0.0])
			ax2.legend(loc='upper right',fontsize=8)
			plt.savefig(run_dir+'max_likelihood_fit.pdf',fmt='pdf',dpi=150)
			# Close plot
			fig.clear()
			plt.close()

			# 
			if print_output:
				print('\n Maximum Likelihood Parameters & Initial Guesses:')
				print('--------------------------------------------------------------------------------------')
				print('\n{0:<30}{1:<25}{2:<25}{3:<25}'.format('Parameter', 'Best-fit Value', '+/- 1-sigma','Flag'))
				print('--------------------------------------------------------------------------------------')
			# Sort into arrays
			pname = []
			med   = []
			std   = []
			flag  = [] 
			for key in pdict:
				pname.append(key)
				med.append(pdict[key]['med'])
				std.append(pdict[key]['std'])
				flag.append(pdict[key]['flag'])
			i_sort = np.argsort(pname)
			pname = np.array(pname)[i_sort] 
			med   = np.array(med)[i_sort]   
			std   = np.array(std)[i_sort]   
			flag  = np.array(flag)[i_sort]  
			if print_output:
				for i in range(0,len(pname),1):
					print('{0:<30}{1:<25.2f}{2:<25.2f}{3:<25}'.format(pname[i], med[i], std[i], flag[i] ))
			del pname
			del med
			del std
			del flag
			if print_output:
				print('{0:<30}{1:<25.2f}{2:<25}{3:<25}'.format('noise_std', sigma_noise, ' ',' '))
				print('{0:<30}{1:<25.2f}{2:<25}{3:<25}'.format('resid_std', sigma_resid, ' ',' '))
				print('--------------------------------------------------------------------------------------')

		# Get S/N of region (5050,5800) to determine if LOSVD/host-galaxy should be fit
		if all(key in comp_dict for key in ['gal_temp','power'])==True:
			sn = np.median(comp_dict['gal_temp']['comp']+comp_dict['power']['comp']/np.median(noise))
			if print_output:
				print('\n Signal-to-noise of host-galaxy continuum: %0.2f' % sn)
		else:	
			sn = np.median(comp_dict['data']['comp']/np.median(noise))
			if print_output:
				print('\n Signal-to-noise of object continuum: %0.2f' % sn)
		# 
		# Write to log 
		write_log((pdict,sn,sigma_noise,sigma_resid),'max_like_fit',run_dir)

		# Collect garbage
		del param_dict
		del lam_gal
		del galaxy
		del noise
		del gal_temp
		del feii_tab
		del temp_list
		del temp_fft
		del params	   
		del bounds	 
		del result
		del par_best
		del mcpars  
		del mcgal
		del mcnoise
		del resultmc
		del fig
		del ax1
		del ax2
		gc.collect()

		#
		if (test_outflows==True):
			return pdict, sigma_resid
		elif (test_outflows==False):
			return pdict, comp_dict ,sn

#### Likelihood function #########################################################

# Maximum Likelihood (initial fitting), Prior, and log Probability functions
def lnlike(params,param_names,lam_gal,galaxy,noise,gal_temp,
		   feii_tab,feii_options,
		   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
		   fit_type,output_model):
	"""
	Log-likelihood function.
	"""
	# Create model
	if (fit_type=='final') and (output_model==False):
		model, blobs = fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,
								 feii_tab,feii_options,
						  		 temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
						  		 fit_type,output_model)

		# Calculate log-likelihood
		l = -0.5*(galaxy-model)**2/(noise)**2
		l = np.sum(l,axis=0)
		return l, blobs

	else:
		model = fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,
						  feii_tab,feii_options,
						  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
						  fit_type,output_model)
		# Calculate log-likelihood
		l = -0.5*(galaxy-model)**2/(noise)**2
		l = np.sum(l,axis=0)
		return l

##################################################################################

#### Priors ######################################################################
# These priors are the same constraints used for outflow testing and maximum likelihood
# fitting, simply formatted for use by emcee. 
# To relax a constrain, simply comment out the condition (*not recommended*).

def lnprior(params,param_names,bounds):
	"""
	Log-prior function.
	"""
	lower_lim = []
	upper_lim = []
	for i in range(0,len(bounds),1):
		lower_lim.append(bounds[i][0])
		upper_lim.append(bounds[i][1])

	pdict = {}
	for k in range(0,len(param_names),1):
			pdict[param_names[k]] = {'p':params[k]}

		# Model 1: [OIII]/Hb Region with outflows, NO broad lines (excludes Ha/[NII]/[SII] region and outflows)
	if (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											 'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True) and \
	   (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
	   											 'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   											 'nii6585_core_amp',
	   											 'sii6732_core_amp','sii6718_core_amp',
	   											 'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
	   											 'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['na_oiii5007_core_amp']['p'] >= pdict['na_oiii5007_outflow_amp']['p']) & \
			(pdict['na_oiii5007_outflow_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']):
			return 0.0
		else: return -np.inf
	#
	# Model 2: [OIII]/Hb Region with outflows AND broad lines (excludes Ha/[NII]/[SII] region and outflows)
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
											   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff',])==True) and \
	   (all(comp not in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   											 'nii6585_core_amp',
	   											 'sii6732_core_amp','sii6718_core_amp',
	   											 'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
	   											 'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		# (br_Hb_fwhm) >= (core_fwhm)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['na_oiii5007_core_amp']['p'] >= pdict['na_oiii5007_outflow_amp']['p']) & \
			(pdict['na_oiii5007_outflow_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
			(pdict['br_Hb_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']):
			return 0.0
		else: return -np.inf
	#
	# Model 3: Ha/[NII]/[SII] region with outflows, no broad lines (excludes [OIII]/Hb region and outflows)
	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['na_Ha_core_amp']['p'] >= pdict['na_Ha_outflow_amp']['p']) & \
			(pdict['na_Ha_outflow_fwhm']['p'] >= pdict['na_Ha_core_fwhm']['p']):
			return 0.0
		else: return -np.inf
		
	#
	# Model 4: Ha/[NII]/[SII] region with outflows, WITH broad lines (excludes [OIII]/Hb region and outflows)
	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
	   										   'na_nii6585_core_amp',
	   										   'na_sii6732_core_amp','na_sii6718_core_amp',
	   										   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
	   										   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True):
		# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		# (br_Ha_fwhm) >= (core_fwhm)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['na_Ha_core_amp']['p'] >= pdict['na_Ha_outflow_amp']['p']) & \
			(pdict['na_Ha_outflow_fwhm']['p'] >= pdict['na_Ha_core_fwhm']['p']) & \
			(pdict['br_Ha_fwhm']['p'] >= pdict['na_Ha_core_fwhm']['p']):
			return 0.0
		else: return -np.inf
		
	#
	# Model 5: Hb/[OIII] + Ha/[NII]/[SII] regions with outflows, NO broad lines
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
											   'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp'])==True) and \
	 	 (all(comp not in param_names for comp in ['na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
	 	 										   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
	 	# Remember that if we are fitting outflows in Hb/[OIII] + Ha/[NII]/[SII] simultaneously
	 	# that we use [OIII] to constrain the outflows in Ha/[NII]/[SII]. 
	 	# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['na_oiii5007_core_amp']['p'] >= pdict['na_oiii5007_outflow_amp']['p']) & \
			(pdict['na_oiii5007_outflow_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']):
			return 0.0
		else: return -np.inf
		
	#
	# Model 6: Hb/[OIII] + Ha/[NII]/[SII] regions with outflows, WITH broad lines
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
											   'na_Ha_core_amp','na_Ha_core_voff',#,'na_Ha_core_fwhm'
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
											   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True) and \
		 (all(comp not in param_names for comp in ['na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		# Remember that if we are fitting outflows in Hb/[OIII] + Ha/[NII]/[SII] simultaneously
	 	# that we use [OIII] to constrain the outflows in Ha/[NII]/[SII]. 
	 	# (core_amp >= outflow_amp)
		# (outflow_fwhm >= core_fwhm)
		# (core_voff >= outflow_voff)
		if 'na_Ha_core_fwhm' in param_names:
			if np.all((params >= lower_lim) & (params <= upper_lim)) & \
				(pdict['na_oiii5007_core_amp']['p'] >= pdict['na_oiii5007_outflow_amp']['p']) & \
				(pdict['na_oiii5007_outflow_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
				(pdict['br_Hb_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
				(pdict['br_Ha_fwhm']['p'] >= pdict['na_Ha_core_fwhm']['p']):
				return 0.0
			else: return -np.inf
		elif 'na_Ha_core_fwhm' not in param_names:
			if np.all((params >= lower_lim) & (params <= upper_lim)) & \
				(pdict['na_oiii5007_core_amp']['p'] >= pdict['na_oiii5007_outflow_amp']['p']) & \
				(pdict['na_oiii5007_outflow_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
				(pdict['br_Hb_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
				(pdict['br_Ha_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']):
				return 0.0
			else: return -np.inf

		
	#
	# Model 7: NO outflows, NO broad lines, Hb/[OIII] region
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
		 										   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
		 										   'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
												   'na_nii6585_core_amp',
												   'na_sii6732_core_amp','na_sii6718_core_amp',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		# No constraints except for bounds
		if np.all((params >= lower_lim) & (params <= upper_lim)):
			return 0.0
		else: return -np.inf
		
	#
	# Model 8: NO outflows, WITH broad lines, Hb/[OIII] region
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff'])==True) and \
		 (all(comp not in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
		 										   'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
												   'na_nii6585_core_amp',
												   'na_sii6732_core_amp','na_sii6718_core_amp',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['br_Hb_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']):
			return 0.0
		else: return -np.inf
	#
	# Model 9: NO outflows, NO broad lines, Ha/[NII]/[SII]
	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		# No constraints except for bounds
		if np.all((params >= lower_lim) & (params <= upper_lim)):
			return 0.0
		else: return -np.inf
	#
	# Model 10: NO outflows, WITH broad lines, Ha/[NII]/[SII]
	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if np.all((params >= lower_lim) & (params <= upper_lim)) & \
			(pdict['br_Ha_fwhm']['p'] >= pdict['na_Ha_core_fwhm']['p']):
			return 0.0
		else: return -np.inf
		
	#
	# Model 11: NO outflows, NO broad lines, Hb/[OIII] + Ha/[NII]/[SII]
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp'])==True) and \
		 (all(comp not in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
												   'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff',
												   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True):
		# No constraints except for bounds
		if np.all((params >= lower_lim) & (params <= upper_lim)):
			return 0.0
		else: return -np.inf
	#
	# Model 12: NO outflows, WITH broad lines, Hb/[OIII] + Ha/[NII]/[SII]
	elif (all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff',
											   'br_Hb_amp','br_Hb_fwhm','br_Hb_voff',
											   'na_Ha_core_amp','na_Ha_core_voff',#,'na_Ha_core_fwhm'
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True) and \
		 (all(comp not in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff',
												   'na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		if 'na_Ha_core_fwhm' in param_names:
			if np.all((params >= lower_lim) & (params <= upper_lim)) & \
				(pdict['br_Hb_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
				(pdict['br_Ha_fwhm']['p'] >= pdict['na_Ha_core_fwhm']['p']):
				return 0.0
			else: return -np.inf
		elif 'na_Ha_core_fwhm' not in param_names:
			if np.all((params >= lower_lim) & (params <= upper_lim)) & \
				(pdict['br_Hb_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']) & \
				(pdict['br_Ha_fwhm']['p'] >= pdict['na_oiii5007_core_fwhm']['p']):
				return 0.0
			else: return -np.inf
	#
	# No constraint model.
	else:
		# No constraints except for bounds
		if np.all((params >= lower_lim) & (params <= upper_lim)):
			return 0.0
		else: return -np.inf

##################################################################################

def lnprob(params,param_names,bounds,lam_gal,galaxy,noise,gal_temp,
		   feii_tab,feii_options,
		   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir):
	"""
	Log-probability function.
	"""
	# lnprob (params,args)

	fit_type	 = 'final'
	output_model = False
	ll, blobs 	 = lnlike(params,param_names,lam_gal,galaxy,noise,gal_temp,
						  feii_tab,feii_options,
		   				  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
		   				  fit_type,output_model)

	lp = lnprior(params,param_names,bounds)

	if not np.isfinite(lp):
		return -np.inf, blobs
	elif (np.isfinite(lp)==True):
		return lp + ll, blobs

####################################################################################

#### Model Function ##############################################################

def fit_model(params,param_names,lam_gal,galaxy,noise,gal_temp,
			  feii_tab,feii_options,
			  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			  fit_type,output_model):
	"""
	Constructs galaxy model by convolving templates with a LOSVD given by 
	a specified set of velocity parameters. 
	
	Parameters:
		pars: parameters of Markov-chain
		lam_gal: wavelength vector used for continuum model
		temp_fft: the Fourier-transformed templates
		npad: 
		velscale: the velocity scale in km/s/pixel
		npix: number of output pixels; must be same as galaxy
		vsyst: dv; the systematic velocity fr
	"""

	# Construct dictionary of parameter names and their respective parameter values
	# param_names  = [param_dict[key]['name'] for key in param_dict ]
	# params	   = [param_dict[key]['init'] for key in param_dict ]
	keys = param_names
	values = params
	p = dict(zip(keys, values))
	c = 299792.458 # speed of light
	host_model = np.copy(galaxy)
	comp_dict  = {} 

	# Perform linear interpolation on the fwhm_gal array as a function of wavelength 
	# We will use this to determine the fwhm resolution as a fucntion of wavelenth for each 
	# emission line so we can correct for the resolution at every iteration.
	fwhm_gal_ftn = interp1d(lam_gal,fwhm_gal,kind='linear',bounds_error=False,fill_value=(0,0))

	# Re-directed line_profile function 
	def line_model(line_profile,*args):
		"""
		This function maps the user-chosen line profile
		to the correct line_model
		"""
		if (line_profile=='Gaussian'):
			line = gaussian(*args)
			return line
		elif (line_profile=='Lorentzian'):
			line = lorentzian(*args)
			return line

	############################# Power-law Component ######################################################

	# if all(comp in param_names for comp in ['power_amp','power_slope','power_break'])==True:
	if all(comp in param_names for comp in ['power_amp','power_slope'])==True:

		# Create a template model for the power-law continuum
		# power = simple_power_law(lam_gal,p['power_amp'],p['power_slope'],p['power_break']) # 
		power = simple_power_law(lam_gal,p['power_amp'],p['power_slope']) # 

		host_model = (host_model) - (power) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['power'] = {'comp':power,'pcolor':'xkcd:orange red','linewidth':1.0}

	########################################################################################################

	 ############################# Fe II Component ##########################################################

	if (feii_tab is not None):

		if (feii_options['template']['type']=='VC04'):
			#  Unpack feii_tab
			na_feii_tab = (feii_tab[0],feii_tab[1])
			br_feii_tab = (feii_tab[2],feii_tab[3])
			# Parse FeII options
			if (feii_options['amp_const']['bool']==False): # if amp not constant
				na_feii_amp = p['na_feii_amp']
				br_feii_amp = p['br_feii_amp']
			elif (feii_options['amp_const']['bool']==True): # if amp constant
				na_feii_amp = feii_options['amp_const']['na_feii_val']
				br_feii_amp = feii_options['amp_const']['br_feii_val']
			if (feii_options['fwhm_const']['bool']==False): # if amp not constant
				na_feii_fwhm = p['na_feii_fwhm']
				br_feii_fwhm = p['br_feii_fwhm']
			elif (feii_options['fwhm_const']['bool']==True): # if amp constant
				na_feii_fwhm = feii_options['fwhm_const']['na_feii_val']
				br_feii_fwhm = feii_options['fwhm_const']['br_feii_val']
			if (feii_options['voff_const']['bool']==False): # if amp not constant
				na_feii_voff = p['na_feii_voff']
				br_feii_voff = p['br_feii_voff']
			elif (feii_options['voff_const']['bool']==True): # if amp constant
				na_feii_voff = feii_options['voff_const']['na_feii_val']
				br_feii_voff = feii_options['voff_const']['br_feii_val']

			na_feii_template = VC04_feii_template(lam_gal,fwhm_gal,na_feii_tab,na_feii_amp,na_feii_fwhm,na_feii_voff,velscale,run_dir)
			br_feii_template = VC04_feii_template(lam_gal,fwhm_gal,br_feii_tab,br_feii_amp,br_feii_fwhm,br_feii_voff,velscale,run_dir)
			 
			host_model = (host_model) - (na_feii_template) - (br_feii_template)
			comp_dict['na_feii_template'] = {'comp':na_feii_template,'pcolor':'xkcd:yellow','linewidth':1.0}
			comp_dict['br_feii_template'] = {'comp':br_feii_template,'pcolor':'xkcd:orange','linewidth':1.0}

		elif (feii_options['template']['type']=='K10'):
			# Unpack tables for each template
			f_trans_tab = (feii_tab[0],feii_tab[1],feii_tab[2])
			s_trans_tab = (feii_tab[3],feii_tab[4],feii_tab[5])
			g_trans_tab = (feii_tab[6],feii_tab[7],feii_tab[8])
			z_trans_tab = (feii_tab[9],feii_tab[10])
			# Parse FeII options
			if (feii_options['amp_const']['bool']==False): # if amp not constant
				f_feii_amp  = p['feii_f_amp']
				s_feii_amp  = p['feii_s_amp']
				g_feii_amp  = p['feii_g_amp']
				z_feii_amp  = p['feii_z_amp']
			elif (feii_options['amp_const']['bool']==True): # if amp constant
				f_feii_amp  = feii_options['amp_const']['f_feii_val']
				s_feii_amp  = feii_options['amp_const']['s_feii_val']
				g_feii_amp  = feii_options['amp_const']['g_feii_val']
				z_feii_amp  = feii_options['amp_const']['z_feii_val']
			#
			if (feii_options['fwhm_const']['bool']==False): # if fwhm not constant
				feii_fwhm = p['feii_fwhm']
			elif (feii_options['fwhm_const']['bool']==True): # if fwhm constant
				feii_fwhm = feii_options['fwhm_const']['val']
			#
			if (feii_options['voff_const']['bool']==False): # if voff not constant
				feii_voff = p['feii_voff']
			elif (feii_options['voff_const']['bool']==True): # if voff constant
				feii_voff = feii_options['voff_const']['val']
			#
			if (feii_options['temp_const']['bool']==False): # if temp not constant
				feii_temp = p['feii_temp']
			elif (feii_options['temp_const']['bool']==True): # if temp constant
				feii_temp = feii_options['temp_const']['val']
			
			f_trans_feii_template = K10_feii_template(lam_gal,'F',fwhm_gal,f_trans_tab,f_feii_amp,feii_temp,feii_fwhm,feii_voff,velscale,run_dir)
			s_trans_feii_template = K10_feii_template(lam_gal,'S',fwhm_gal,s_trans_tab,s_feii_amp,feii_temp,feii_fwhm,feii_voff,velscale,run_dir)
			g_trans_feii_template = K10_feii_template(lam_gal,'G',fwhm_gal,g_trans_tab,g_feii_amp,feii_temp,feii_fwhm,feii_voff,velscale,run_dir)
			z_trans_feii_template = K10_feii_template(lam_gal,'IZw1',fwhm_gal,z_trans_tab,z_feii_amp,feii_temp,feii_fwhm,feii_voff,velscale,run_dir)

			host_model = (host_model) - (f_trans_feii_template) - (s_trans_feii_template) - (g_trans_feii_template) - (z_trans_feii_template)
			comp_dict['F_feii_template'] = {'comp':f_trans_feii_template,'pcolor':'xkcd:rust orange','linewidth':1.0}
			comp_dict['S_feii_template'] = {'comp':s_trans_feii_template,'pcolor':'xkcd:rust orange','linewidth':1.0}
			comp_dict['G_feii_template'] = {'comp':g_trans_feii_template,'pcolor':'xkcd:rust orange','linewidth':1.0}
			comp_dict['Z_feii_template'] = {'comp':z_trans_feii_template,'pcolor':'xkcd:rust orange','linewidth':1.0}


	 ########################################################################################################

	 ############################# Emission Line Components #################################################	
	 # Narrow lines
	 #### [OII]3727,3729 #################################################################################

	if all(comp in param_names for comp in ['na_oii3727_core_amp','na_oii3727_core_fwhm','na_oii3727_core_voff','na_oii3729_core_amp'])==True:
		# Narrow [OII]3727
		na_oii3727_core_center		= 3727.092 # Angstroms
		na_oii3727_core_amp		   = p['na_oii3727_core_amp'] # flux units
		na_oii3727_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oii3727_core_center,p['na_oii3727_core_voff'])
		na_oii3727_core_fwhm		  = np.sqrt(p['na_oii3727_core_fwhm']**2+(na_oii3727_core_fwhm_res)**2) # km/s
		na_oii3727_core_voff		  = p['na_oii3727_core_voff']  # km/s
		na_oii3727_core			   = gaussian(lam_gal,na_oii3727_core_center,na_oii3727_core_amp,na_oii3727_core_fwhm,na_oii3727_core_voff,velscale)
		host_model					= host_model - na_oii3727_core
		comp_dict['na_oii3727_core']  = {'comp':na_oii3727_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OII]3729
		na_oii3729_core_center		= 3729.875 # Angstroms
		na_oii3729_core_amp		   = p['na_oii3729_core_amp'] # flux units
		na_oii3729_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oii3729_core_center,na_oii3727_core_voff)
		na_oii3729_core_fwhm		  = np.sqrt(p['na_oii3727_core_fwhm']**2+(na_oii3729_core_fwhm_res)**2) # km/s # km/s
		na_oii3729_core_voff		  = na_oii3727_core_voff  # km/s
		na_oii3729_core			   = gaussian(lam_gal,na_oii3729_core_center,na_oii3729_core_amp,na_oii3729_core_fwhm,na_oii3729_core_voff,velscale)
		host_model					= host_model - na_oii3729_core
		comp_dict['na_oii3729_core']  = {'comp':na_oii3729_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	# If tie_narrow=True, and includes [OIII]5007	
	elif (all(comp in param_names for comp in ['na_oii3727_core_amp','na_oii3727_core_voff','na_oii3729_core_amp','na_oiii5007_core_fwhm'])==True) & \
		 (all(comp not in param_names for comp in ['na_neiii_core_fwhm','na_Hg_fwhm','oiii4363_core_fwhm'])==True):
		# Narrow [OII]3727
		na_oii3727_core_center		= 3727.092 # Angstroms
		na_oii3727_core_amp		   = p['na_oii3727_core_amp'] # flux units
		na_oii3727_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oii3727_core_center,p['na_oii3727_core_voff'])
		na_oii3727_core_fwhm		  = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_oii3727_core_fwhm_res)**2) # km/s
		na_oii3727_core_voff		  = p['na_oii3727_core_voff']  # km/s
		na_oii3727_core			   = gaussian(lam_gal,na_oii3727_core_center,na_oii3727_core_amp,na_oii3727_core_fwhm,na_oii3727_core_voff,velscale)
		host_model					= host_model - na_oii3727_core
		comp_dict['na_oii3727_core']  = {'comp':na_oii3727_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OII]3729 
		na_oii3729_core_center		= 3729.875 # Angstroms
		na_oii3729_core_amp		   = p['na_oii3729_core_amp'] # flux units
		na_oii3729_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oii3729_core_center,na_oii3727_core_voff)
		na_oii3729_core_fwhm		  = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_oii3729_core_fwhm_res)**2) # km/s
		na_oii3729_core_voff		  = na_oii3727_core_voff  # km/s
		na_oii3729_core			   = gaussian(lam_gal,na_oii3729_core_center,na_oii3729_core_amp,na_oii3729_core_fwhm,na_oii3729_core_voff,velscale)
		host_model					= host_model - na_oii3729_core
		comp_dict['na_oii3729_core'] = {'comp':na_oii3729_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	# If tie_narrow=True, but doesn't include [OIII]5007
	elif (all(comp in param_names for comp in ['na_oii3727_core_amp','na_oii3727_core_voff','na_oii3729_core_amp','na_Hg_fwhm'])==True) & \
		 (all(comp not in param_names for comp in ['na_neiii_core_fwhm','oiii4363_core_fwhm','na_oiii5007_core_fwhm'])==True):
		# Narrow [OII]3727
		na_oii3727_core_center		= 3727.092 # Angstroms
		na_oii3727_core_amp		   = p['na_oii3727_core_amp'] # flux units
		na_oii3727_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oii3727_core_center,p['na_oii3727_core_voff'])
		na_oii3727_core_fwhm		  = np.sqrt(p['na_Hg_fwhm']**2+(na_oii3727_core_fwhm_res)**2) # km/s
		na_oii3727_core_voff		  = p['na_oii3727_core_voff']  # km/s
		na_oii3727_core			   = gaussian(lam_gal,na_oii3727_core_center,na_oii3727_core_amp,na_oii3727_core_fwhm,na_oii3727_core_voff,velscale)
		host_model					= host_model - na_oii3727_core
		comp_dict['na_oii3727_core']  = {'comp':na_oii3727_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OII]3729 
		na_oii3729_core_center		= 3729.875 # Angstroms
		na_oii3729_core_amp		   = p['na_oii3729_core_amp'] # flux units
		na_oii3729_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oii3729_core_center,na_oii3727_core_voff)
		na_oii3729_core_fwhm		  = np.sqrt(p['na_Hg_fwhm']**2+(na_oii3729_core_fwhm_res)**2) # km/s
		na_oii3729_core_voff		  = na_oii3727_core_voff  # km/s
		na_oii3729_core			   = gaussian(lam_gal,na_oii3729_core_center,na_oii3729_core_amp,na_oii3729_core_fwhm,na_oii3729_core_voff,velscale)
		host_model					= host_model - na_oii3729_core
		comp_dict['na_oii3729_core']  = {'comp':na_oii3729_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	 #### [NeIII]3870 #################################################################################
	if all(comp in param_names for comp in ['na_neiii_core_amp','na_neiii_core_fwhm','na_neiii_core_voff'])==True:
		# Narrow H-gamma
		na_neiii_core_center		  = 3869.810 # Angstroms
		na_neiii_core_amp			 = p['na_neiii_core_amp'] # flux units
		na_neiii_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_neiii_core_center,p['na_neiii_core_voff'])
		na_neiii_core_fwhm			= np.sqrt(p['na_neiii_core_fwhm']**2+(na_neiii_core_fwhm_res)**2) # km/s
		na_neiii_core_voff			= p['na_neiii_core_voff']  # km/s
		na_neiii_core				 = gaussian(lam_gal,na_neiii_core_center,na_neiii_core_amp,na_neiii_core_fwhm,na_neiii_core_voff,velscale)
		host_model					= host_model - na_neiii_core
		comp_dict['na_neiii_core']	= {'comp':na_neiii_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	# If tie_narrow=True, and includes [OIII]5007
	elif (all(comp in param_names for comp in ['na_neiii_core_amp','na_neiii_core_voff','na_oiii5007_core_fwhm'])==True) & \
		 (all(comp not in param_names for comp in ['na_neiii_core_fwhm','na_Hg_fwhm','oiii4363_core_fwhm'])==True):
		# Narrow H-gamma
		na_neiii_core_center		  = 3869.810 # Angstroms
		na_neiii_core_amp			 = p['na_neiii_core_amp'] # flux units
		na_neiii_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_neiii_core_center,p['na_neiii_core_voff'])
		na_neiii_core_fwhm			= np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_neiii_core_fwhm_res)**2) # km/s
		na_neiii_core_voff			= p['na_neiii_core_voff']  # km/s
		na_neiii_core				 = gaussian(lam_gal,na_neiii_core_center,na_neiii_core_amp,na_neiii_core_fwhm,na_neiii_core_voff,velscale)
		host_model					= host_model - na_neiii_core
		comp_dict['na_neiii_core']	= {'comp':na_neiii_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	# If tie_narrow=True, but doesn't include [OIII]5007
	elif (all(comp in param_names for comp in ['na_neiii_core_amp','na_neiii_core_voff','na_Hg_fwhm'])==True) & \
		 (all(comp not in param_names for comp in ['na_neiii_core_fwhm','oiii4363_core_fwhm','na_oiii5007_core_fwhm'])==True):
		# Narrow H-gamma
		na_neiii_core_center		  = 3869.810 # Angstroms
		na_neiii_core_amp			 = p['na_neiii_core_amp'] # flux units
		na_neiii_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_neiii_core_center,p['na_neiii_core_voff'])
		na_neiii_core_fwhm			= np.sqrt(p['na_Hg_fwhm']**2+(na_neiii_core_fwhm_res)**2) # km/s
		na_neiii_core_voff			= p['na_neiii_core_voff']  # km/s
		na_neiii_core				 = gaussian(lam_gal,na_neiii_core_center,na_neiii_core_amp,na_neiii_core_fwhm,na_neiii_core_voff,velscale)
		host_model					= host_model - na_neiii_core
		comp_dict['na_neiii_core']	= {'comp':na_neiii_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	 #### H-delta #####################################################################################
	if all(comp in param_names for comp in ['na_Hd_amp','na_Hd_fwhm','na_Hd_voff'])==True:
		# Narrow H-gamma
		na_hd_core_center			 = 4102.890 # Angstroms
		na_hd_core_amp				= p['na_Hd_amp'] # flux units
		na_hd_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_hd_core_center,p['na_Hd_voff'])
		na_hd_core_fwhm			   = np.sqrt(p['na_Hd_fwhm']**2+(na_hd_core_fwhm_res)**2) # km/s
		na_hd_core_voff			   = p['na_Hd_voff']  # km/s
		na_Hd_core					= gaussian(lam_gal,na_hd_core_center,na_hd_core_amp,na_hd_core_fwhm,na_hd_core_voff,velscale)
		host_model					= host_model - na_Hd_core
		comp_dict['na_Hd_core']	   = {'comp':na_Hd_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	# If tie_narrow=True, and includes [OIII]5007
	elif (all(comp in param_names for comp in ['na_Hd_amp','na_Hd_voff','na_oiii5007_core_fwhm'])==True) & \
		 (all(comp not in param_names for comp in ['na_Hd_fwhm','na_Hg_fwhm','oiii4363_core_fwhm'])==True):
		# Narrow H-gamma
		na_hd_core_center			 = 4102.890 # Angstroms
		na_hd_core_amp				= p['na_Hd_amp'] # flux units
		na_hd_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_hd_core_center,p['na_Hd_voff'])
		na_hd_core_fwhm			   = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_hd_core_fwhm_res)**2) # km/s
		na_hd_core_voff			   = p['na_Hd_voff']  # km/s
		na_Hd_core					= gaussian(lam_gal,na_hd_core_center,na_hd_core_amp,na_hd_core_fwhm,na_hd_core_voff,velscale)
		host_model					= host_model - na_Hd_core
		comp_dict['na_Hd_core']	   = {'comp':na_Hd_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	# If tie_narrow=True, but doesn't include [OIII]5007
	elif (all(comp in param_names for comp in ['na_Hd_amp','na_Hd_voff','na_Hg_fwhm'])==True) & \
		 (all(comp not in param_names for comp in ['na_Hg_fwhm','oiii4363_core_fwhm','na_oiii5007_core_fwhm'])==True):
		# Narrow H-gamma
		na_hd_core_center			 = 4102.890 # Angstroms
		na_hd_core_amp				= p['na_Hd_amp'] # flux units
		na_hd_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_hd_core_center,p['na_Hd_voff'])
		na_hd_core_fwhm			   = np.sqrt(p['na_Hg_fwhm']**2+(na_hd_core_fwhm_res)**2) # km/s
		na_hd_core_voff			   = p['na_Hd_voff']  # km/s
		na_Hd_core					= gaussian(lam_gal,na_hd_core_center,na_hd_core_amp,na_hd_core_fwhm,na_hd_core_voff,velscale)
		host_model					= host_model - na_Hd_core
		comp_dict['na_Hd_core']	   = {'comp':na_Hd_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	 #### H-gamma/[OIII]4363 ##########################################################################
	if all(comp in param_names for comp in ['na_Hg_amp','na_Hg_fwhm','na_Hg_voff','na_oiii4363_core_amp','na_oiii4363_core_fwhm','na_oiii4363_core_voff'])==True:
		# Narrow H-gamma
		na_hg_core_center			 = 4341.680 # Angstroms
		na_hg_core_amp				= p['na_Hg_amp'] # flux units
		na_hg_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_hg_core_center,p['na_Hg_voff'])
		na_hg_core_fwhm			   = np.sqrt(p['na_Hg_fwhm']**2+(na_hg_core_fwhm_res)**2) # km/s
		na_hg_core_voff			   = p['na_Hg_voff']  # km/s
		na_Hg_core					= gaussian(lam_gal,na_hg_core_center,na_hg_core_amp,na_hg_core_fwhm,na_hg_core_voff,velscale)
		host_model					= host_model - na_Hg_core
		comp_dict['na_Hg_core']	   = {'comp':na_Hg_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OIII]4363 core
		na_oiii4363_core_center	   = 4364.436 # Angstroms
		na_oiii4363_core_amp		  = p['na_oiii4363_core_amp'] # flux units
		na_oiii4363_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oiii4363_core_center,p['na_oiii4363_core_voff'])
		na_oiii4363_core_fwhm		 = np.sqrt(p['na_oiii4363_core_fwhm']**2+(na_oiii4363_core_fwhm_res)**2) # km/s
		na_oiii4363_core_voff		 = p['na_oiii4363_core_voff'] # km/s
		na_oiii4363_core			  = gaussian(lam_gal,na_oiii4363_core_center,na_oiii4363_core_amp,na_oiii4363_core_fwhm,na_oiii4363_core_voff,velscale)
		host_model					= host_model - na_oiii4363_core
		comp_dict['na_oiii4363_core'] = {'comp':na_oiii4363_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	# If tie_narrow=True, and includes [OIII]5007
	elif (all(comp in param_names for comp in ['na_Hg_amp','na_Hg_voff','na_oiii4363_core_amp','na_oiii4363_core_voff','na_oiii5007_core_fwhm'])==True) & \
		 (all(comp not in param_names for comp in ['na_Hg_fwhm','oiii4363_core_fwhm'])==True):
		# Narrow H-gamma
		na_hg_core_center			 = 4341.680 # Angstroms
		na_hg_core_amp				= p['na_Hg_amp'] # flux units
		na_hg_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_hg_core_center,p['na_Hg_voff'])
		na_hg_core_fwhm			   = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_hg_core_fwhm_res)**2) # km/s
		na_hg_core_voff			   = p['na_Hg_voff']  # km/s
		na_Hg_core					= gaussian(lam_gal,na_hg_core_center,na_hg_core_amp,na_hg_core_fwhm,na_hg_core_voff,velscale)
		host_model					= host_model - na_Hg_core
		comp_dict['na_Hg_core']	   = {'comp':na_Hg_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OIII]4363 core
		na_oiii4363_core_center	   = 4364.436 # Angstroms
		na_oiii4363_core_amp		  = p['na_oiii4363_core_amp'] # flux units
		na_oiii4363_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oiii4363_core_center,p['na_oiii4363_core_voff'])
		na_oiii4363_core_fwhm		 = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_oiii4363_core_fwhm_res)**2) # km/s
		na_oiii4363_core_voff		 = p['na_oiii4363_core_voff'] # km/s
		na_oiii4363_core			  = gaussian(lam_gal,na_oiii4363_core_center,na_oiii4363_core_amp,na_oiii4363_core_fwhm,na_oiii4363_core_voff,velscale)
		host_model					= host_model - na_oiii4363_core
		comp_dict['na_oiii4363_core'] = {'comp':na_oiii4363_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	# If tie_narrow=True, but doesn't include [OIII]5007
	elif (all(comp in param_names for comp in ['na_Hg_amp','na_Hg_fwhm','na_Hg_voff','na_oiii4363_core_amp','na_oiii4363_core_voff'])==True) & \
		 (all(comp not in param_names for comp in ['oiii4363_core_fwhm','na_oiii5007_core_fwhm'])==True):
		# Narrow H-gamma
		na_hg_core_center			 = 4341.680 # Angstroms
		na_hg_core_amp				= p['na_Hg_amp'] # flux units
		na_hg_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_hg_core_center,p['na_Hg_voff'])
		na_hg_core_fwhm			   = np.sqrt(p['na_Hg_fwhm']**2+(na_hg_core_fwhm_res)**2) # km/s
		na_hg_core_voff			   = p['na_Hg_voff']  # km/s
		na_Hg_core					= gaussian(lam_gal,na_hg_core_center,na_hg_core_amp,na_hg_core_fwhm,na_hg_core_voff,velscale)
		host_model					= host_model - na_Hg_core
		comp_dict['na_Hg_core']	   = {'comp':na_Hg_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [OIII]4363 core
		na_oiii4363_core_center	   = 4364.436 # Angstroms
		na_oiii4363_core_amp		  = p['na_oiii4363_core_amp'] # flux units
		na_oiii4363_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oiii4363_core_center,p['na_oiii4363_core_voff'])
		na_oiii4363_core_fwhm		 = np.sqrt(p['na_Hg_fwhm']**2+(na_oiii4363_core_fwhm_res)**2) # km/s
		na_oiii4363_core_voff		 = p['na_oiii4363_core_voff'] # km/s
		na_oiii4363_core			  = gaussian(lam_gal,na_oiii4363_core_center,na_oiii4363_core_amp,na_oiii4363_core_fwhm,na_oiii4363_core_voff,velscale)
		host_model					= host_model - na_oiii4363_core
		comp_dict['na_oiii4363_core'] = {'comp':na_oiii4363_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}

	 #### H-beta/[OIII] #########################################################################################
	if all(comp in param_names for comp in ['na_oiii5007_core_amp','na_oiii5007_core_fwhm','na_oiii5007_core_voff'])==True:
		# Narrow [OIII]5007 Core
		na_oiii5007_core_center	      = 5008.240 # Angstroms
		na_oiii5007_core_amp		  = p['na_oiii5007_core_amp'] # flux units
		na_oiii5007_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_oiii5007_core_center,p['na_oiii5007_core_voff'])
		na_oiii5007_core_fwhm		  = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_oiii5007_core_fwhm_res)**2) # km/s
		na_oiii5007_core_voff		  = p['na_oiii5007_core_voff']  # km/s
		na_oiii5007_core			  = gaussian(lam_gal,na_oiii5007_core_center,na_oiii5007_core_amp,na_oiii5007_core_fwhm,na_oiii5007_core_voff,velscale)
		# na_oiii5007_core			  = line_model(line_profile,lam_gal,na_oiii5007_core_center,na_oiii5007_core_amp,na_oiii5007_core_fwhm,na_oiii5007_core_voff,velscale)
		host_model					  = host_model - na_oiii5007_core
		comp_dict['na_oiii5007_core'] = {'comp':na_oiii5007_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	 	# Narrow [OIII]4959 Core
		na_oiii4959_core_center	      = 4960.295 # Angstroms
		na_oiii4959_core_amp		  = (1.0/3.0)*na_oiii5007_core_amp # flux units
		na_oiii4959_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_oiii4959_core_center,na_oiii5007_core_voff)
		na_oiii4959_core_fwhm		  = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_oiii4959_fwhm_res)**2) # km/s
		na_oiii4959_core_voff		  = na_oiii5007_core_voff  # km/s
		na_oiii4959_core			  = gaussian(lam_gal,na_oiii4959_core_center,na_oiii4959_core_amp,na_oiii4959_core_fwhm,na_oiii4959_core_voff,velscale)
		# na_oiii4959_core			  = line_model(line_profile,lam_gal,na_oiii4959_core_center,na_oiii4959_core_amp,na_oiii4959_core_fwhm,na_oiii4959_core_voff,velscale)
		host_model					  = host_model - na_oiii4959_core
		comp_dict['na_oiii4959_core'] = {'comp':na_oiii4959_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}

	if all(comp in param_names for comp in ['na_Hb_core_amp','na_Hb_core_voff'])==True:
		# Narrow H-beta
		na_hb_core_center			 = 4862.680 # Angstroms
		na_hb_core_amp				 = p['na_Hb_core_amp'] # flux units
		na_hb_core_fwhm_res 	  	 = get_fwhm_res(fwhm_gal_ftn,na_hb_core_center,p['na_Hb_core_voff'])
		na_hb_core_fwhm			     = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_hb_core_fwhm_res)**2) # km/s
		na_hb_core_voff			     = p['na_Hb_core_voff']  # km/s
		na_Hb_core					 = gaussian(lam_gal,na_hb_core_center,na_hb_core_amp,na_hb_core_fwhm,na_hb_core_voff,velscale)
		# na_Hb_core					 = line_model(line_profile,lam_gal,na_hb_core_center,na_hb_core_amp,na_hb_core_fwhm,na_hb_core_voff,velscale)
		host_model					 = host_model - na_Hb_core
		comp_dict['na_Hb_core']	     = {'comp':na_Hb_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}

	#### H-alpha/[NII]/[SII] ####################################################################################
	if all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_fwhm','na_Ha_core_voff',
											'na_nii6585_core_amp',
											'na_sii6732_core_amp','na_sii6718_core_amp'])==True:
		# Narrow H-alpha
		na_ha_core_center			 = 6564.610 # Angstroms
		na_ha_core_amp				= p['na_Ha_core_amp'] # flux units
		na_ha_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_ha_core_center,p['na_Ha_core_voff'])
		na_ha_core_fwhm			   = np.sqrt(p['na_Ha_core_fwhm']**2+(na_ha_core_fwhm_res)**2) # km/s
		na_ha_core_voff			   = p['na_Ha_core_voff']  # km/s
		na_Ha_core					= gaussian(lam_gal,na_ha_core_center,na_ha_core_amp,na_ha_core_fwhm,na_ha_core_voff,velscale)
		host_model					= host_model - na_Ha_core
		comp_dict['na_Ha_core']	   = {'comp':na_Ha_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [NII]6585 Core
		na_nii6585_core_center 		  = 6585.270 # Angstroms
		na_nii6585_core_amp			  = p['na_nii6585_core_amp'] # flux units
		na_nii6585_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_nii6585_core_center,na_ha_core_voff)
		na_nii6585_core_fwhm   		  = np.sqrt(p['na_Ha_core_fwhm']**2+(na_nii6585_core_fwhm_res)**2) # km/s
		na_nii6585_core_voff   		  = na_ha_core_voff
		na_nii6585_core   			  = gaussian(lam_gal,na_nii6585_core_center,na_nii6585_core_amp,na_nii6585_core_fwhm,na_nii6585_core_voff,velscale)
		host_model					  = host_model - na_nii6585_core
		comp_dict['na_nii6585_core']  = {'comp':na_nii6585_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	 	# Narrow [NII]6549 Core
		na_nii6549_core_center		= 6549.860 # Angstroms
		na_nii6549_core_amp		   = (1.0/2.93)*na_nii6585_core_amp # flux units
		na_nii6549_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_nii6549_core_center,na_ha_core_voff)
		na_nii6549_core_fwhm		  = np.sqrt(p['na_Ha_core_fwhm']**2+(na_nii6549_core_fwhm_res)**2) # km/s # km/s
		na_nii6549_core_voff		  = na_ha_core_voff
		na_nii6549_core			   = gaussian(lam_gal,na_nii6549_core_center,na_nii6549_core_amp,na_nii6549_core_fwhm,na_nii6549_core_voff,velscale)
		host_model					= host_model - na_nii6549_core
		comp_dict['na_nii6549_core']  = {'comp':na_nii6549_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6718
		na_sii6718_core_center		= 6718.290 # Angstroms
		na_sii6718_core_amp		   = p['na_sii6718_core_amp'] # flux units
		na_sii6718_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_sii6718_core_center,na_ha_core_voff)
		na_sii6718_core_fwhm		  = np.sqrt(p['na_Ha_core_fwhm']**2+(na_sii6718_core_fwhm_res)**2) # km/s #na_sii6732_fwhm # km/s
		na_sii6718_core_voff		  = na_ha_core_voff
		na_sii6718_core			   = gaussian(lam_gal,na_sii6718_core_center,na_sii6718_core_amp,na_sii6718_core_fwhm,na_sii6718_core_voff,velscale)
		host_model					= host_model - na_sii6718_core
		comp_dict['na_sii6718_core']  = {'comp':na_sii6718_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6732
		na_sii6732_core_center		= 6732.670 # Angstroms
		na_sii6732_core_amp		   = p['na_sii6732_core_amp'] # flux units
		na_sii6732_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_sii6732_core_center,na_ha_core_voff)
		na_sii6732_core_fwhm		  = np.sqrt(p['na_Ha_core_fwhm']**2+(na_sii6732_core_fwhm_res)**2) # km/s 
		na_sii6732_core_voff		  = na_ha_core_voff
		na_sii6732_core			   = gaussian(lam_gal,na_sii6732_core_center,na_sii6732_core_amp,na_sii6732_core_fwhm,na_sii6732_core_voff,velscale)
		host_model					= host_model - na_sii6732_core
		comp_dict['na_sii6732_core']  = {'comp':na_sii6732_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}

	elif (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_voff',
											   'na_nii6585_core_amp',
											   'na_sii6732_core_amp','na_sii6718_core_amp',
											   'na_oiii5007_core_fwhm'])==True) & ('na_Ha_core_fwhm' not in param_names):

		# If all narrow line widths are tied to [OIII]5007 FWHM...
		# Narrow H-alpha
		na_ha_core_center			 = 6564.610 # Angstroms
		na_ha_core_amp				= p['na_Ha_core_amp'] # flux units
		na_ha_core_fwhm_res 	  	  = get_fwhm_res(fwhm_gal_ftn,na_ha_core_center,p['na_Ha_core_voff'])
		na_ha_core_fwhm			   = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_ha_core_fwhm_res)**2) # km/s
		na_ha_core_voff			   = p['na_Ha_core_voff']  # km/s
		na_Ha_core					= gaussian(lam_gal,na_ha_core_center,na_ha_core_amp,na_ha_core_fwhm,na_ha_core_voff,velscale)
		host_model					= host_model - na_Ha_core
		comp_dict['na_Ha_core']	   = {'comp':na_Ha_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [NII]6585 Core
		na_nii6585_core_center		= 6585.270 # Angstroms
		na_nii6585_core_amp		   = p['na_nii6585_core_amp'] # flux units
		na_nii6585_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_nii6585_core_center,na_ha_core_voff)
		na_nii6585_core_fwhm		  = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_nii6585_core_fwhm_res)**2) # km/s
		na_nii6585_core_voff		  = na_ha_core_voff
		na_nii6585_core			   = gaussian(lam_gal,na_nii6585_core_center,na_nii6585_core_amp,na_nii6585_core_fwhm,na_nii6585_core_voff,velscale)
		host_model					= host_model - na_nii6585_core
		comp_dict['na_nii6585_core']  = {'comp':na_nii6585_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	 	# Narrow [NII]6549 Core
		na_nii6549_core_center		= 6549.860 # Angstroms
		na_nii6549_core_amp		   = (1.0/2.93)*na_nii6585_core_amp # flux units
		na_nii6549_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_nii6549_core_center,na_ha_core_voff)
		na_nii6549_core_fwhm		  = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_nii6549_core_fwhm_res)**2) # km/s
		na_nii6549_core_voff		  = na_ha_core_voff
		na_nii6549_core			   = gaussian(lam_gal,na_nii6549_core_center,na_nii6549_core_amp,na_nii6549_core_fwhm,na_nii6549_core_voff,velscale)
		host_model					= host_model - na_nii6549_core
		comp_dict['na_nii6549_core']  = {'comp':na_nii6549_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6732
		na_sii6732_core_center		= 6732.670 # Angstroms
		na_sii6732_core_amp		   = p['na_sii6732_core_amp'] # flux units
		na_sii6732_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_sii6732_core_center,na_ha_core_voff)
		na_sii6732_core_fwhm		  = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_sii6732_core_fwhm_res)**2) # km/s
		na_sii6732_core_voff		  = na_ha_core_voff
		na_sii6732_core			   = gaussian(lam_gal,na_sii6732_core_center,na_sii6732_core_amp,na_sii6732_core_fwhm,na_sii6732_core_voff,velscale)
		host_model					= host_model - na_sii6732_core
		comp_dict['na_sii6732_core']  = {'comp':na_sii6732_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
		# Narrow [SII]6718
		na_sii6718_core_center		= 6718.290 # Angstroms
		na_sii6718_core_amp		   = p['na_sii6718_core_amp'] # flux units
		na_sii6718_core_fwhm_res 	  = get_fwhm_res(fwhm_gal_ftn,na_sii6718_core_center,na_ha_core_voff)
		na_sii6718_core_fwhm		  = np.sqrt(p['na_oiii5007_core_fwhm']**2+(na_sii6718_core_fwhm_res)**2) # km/s
		na_sii6718_core_voff		  = na_ha_core_voff
		na_sii6718_core			   = gaussian(lam_gal,na_sii6718_core_center,na_sii6718_core_amp,na_sii6718_core_fwhm,na_sii6718_core_voff,velscale)
		host_model					= host_model - na_sii6718_core
		comp_dict['na_sii6718_core']  = {'comp':na_sii6718_core,'pcolor':'xkcd:dodger blue','linewidth':1.0}
	########################################################################################################

	# Outflow Components
	 #### Hb/[OIII] outflows ################################################################################
	if (all(comp in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True):
		# Broad [OIII]5007 Outflow;
		na_oiii5007_outflow_center	   = 5008.240 # Angstroms
		na_oiii5007_outflow_amp		  = p['na_oiii5007_outflow_amp'] # flux units
		na_oiii5007_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_oiii5007_outflow_center,p['na_oiii5007_outflow_voff'])
		na_oiii5007_outflow_fwhm		 = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(na_oiii5007_outflow_fwhm_res)**2) # km/s
		na_oiii5007_outflow_voff		 = p['na_oiii5007_outflow_voff']  # km/s
		na_oiii5007_outflow			  = gaussian(lam_gal,na_oiii5007_outflow_center,na_oiii5007_outflow_amp,na_oiii5007_outflow_fwhm,na_oiii5007_outflow_voff,velscale)
		host_model					   = host_model - na_oiii5007_outflow
		comp_dict['na_oiii5007_outflow'] = {'comp':na_oiii5007_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
	 	# Broad [OIII]4959 Outflow; 
		na_oiii4959_outflow_center	   = 4960.295 # Angstroms
		na_oiii4959_outflow_amp		  = na_oiii4959_core_amp*na_oiii5007_outflow_amp/na_oiii5007_core_amp # flux units
		na_oiii4959_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_oiii4959_outflow_center,na_oiii5007_outflow_voff)
		na_oiii4959_outflow_fwhm		 = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(na_oiii4959_outflow_fwhm_res)**2) # km/s
		na_oiii4959_outflow_voff		 = na_oiii5007_outflow_voff  # km/s
		if (na_oiii4959_outflow_amp!=na_oiii4959_outflow_amp/1.0) or (na_oiii4959_outflow_amp==np.inf): na_oiii4959_outflow_amp=0.0
		na_oiii4959_outflow			  = gaussian(lam_gal,na_oiii4959_outflow_center,na_oiii4959_outflow_amp,na_oiii4959_outflow_fwhm,na_oiii4959_outflow_voff,velscale)
		host_model					   = host_model - na_oiii4959_outflow
		comp_dict['na_oiii4959_outflow'] = {'comp':na_oiii4959_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
	if (all(comp in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff','na_Hb_core_amp','na_Hb_core_voff'])==True): 
		# Broad H-beta Outflow; only a model, no free parameters, tied to [OIII]5007
		na_hb_core_center			 	 = 4862.680 # Angstroms
		na_hb_outflow_amp				= na_hb_core_amp*na_oiii5007_outflow_amp/na_oiii5007_core_amp
		na_hb_outflow_fwhm_res 		 	 = get_fwhm_res(fwhm_gal_ftn,na_hb_core_center,na_hb_core_voff+na_oiii5007_outflow_voff)
		na_hb_outflow_fwhm			   = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(na_hb_outflow_fwhm_res)**2) # km/s
		na_hb_outflow_voff			   = na_hb_core_voff+na_oiii5007_outflow_voff  # km/s
		if (na_hb_outflow_amp!=na_hb_outflow_amp/1.0) or (na_hb_outflow_amp==np.inf): na_hb_outflow_amp=0.0
		na_Hb_outflow					= gaussian(lam_gal,na_hb_core_center,na_hb_outflow_amp,na_hb_outflow_fwhm,na_hb_outflow_voff,velscale)
		host_model					   = host_model - na_Hb_outflow
		comp_dict['na_Hb_outflow']	   = {'comp':na_Hb_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
	#### Ha/[NII]/[SII] outflows ###########################################################################
	# Outflows in H-alpha/[NII] are poorly constrained due to the presence of a broad line and/or blending of narrow lines
	# First, we check if the fit includes Hb/[OIII] outflows, if it does, we use the outflow in [OIII] to constrain the outflows
	# in the Ha/[NII]/[SII] region.  If the fi does NOT include Hb/[OIII] outflows (*not recommended*), we then allow the outflows 
	# in the Ha/[NII]/[SII] region to be fit as free parameters.
	if (all(comp in param_names for comp in ['na_Ha_core_amp','na_Ha_core_voff','na_nii6585_core_amp','na_sii6732_core_amp','na_sii6718_core_amp',
											 'na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True) and \
	   (all(comp not in param_names for comp in ['na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True):
		# H-alpha Outflow; 
		na_ha_outflow_center			 = 6564.610 # Angstroms
		na_ha_outflow_amp				= p['na_Ha_core_amp']*p['na_oiii5007_outflow_amp']/p['na_oiii5007_core_amp'] # flux units
		na_ha_outflow_fwhm_res 	  		 = get_fwhm_res(fwhm_gal_ftn,na_ha_outflow_center,p['na_oiii5007_outflow_voff'])
		na_ha_outflow_fwhm			   = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(na_ha_outflow_fwhm_res)**2) # km/s
		na_ha_outflow_voff			   = p['na_oiii5007_outflow_voff']  # km/s  # km/s
		if (na_ha_outflow_amp!=na_ha_outflow_amp/1.0) or (na_ha_outflow_amp==np.inf): na_ha_outflow_amp=0.0
		na_Ha_outflow					= gaussian(lam_gal,na_ha_outflow_center,na_ha_outflow_amp,na_ha_outflow_fwhm,na_ha_outflow_voff,velscale)
		host_model					   = host_model - na_Ha_outflow
		comp_dict['na_Ha_outflow']	   = {'comp':na_Ha_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [NII]6585 Outflow;
		na_nii6585_outflow_center		= 6585.270 # Angstroms
		na_nii6585_outflow_amp		   = na_nii6585_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_nii6585_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_nii6585_outflow_center,na_ha_outflow_voff)
		na_nii6585_outflow_fwhm		  = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(na_nii6585_outflow_fwhm_res)**2)
		na_nii6585_outflow_voff		  = na_ha_outflow_voff
		if (na_nii6585_outflow_amp!=na_nii6585_outflow_amp/1.0) or (na_nii6585_outflow_amp==np.inf): na_nii6585_outflow_amp=0.0
		na_nii6585_outflow			   = gaussian(lam_gal,na_nii6585_outflow_center,na_nii6585_outflow_amp,na_nii6585_outflow_fwhm,na_nii6585_outflow_voff,velscale)
		host_model					   = host_model - na_nii6585_outflow
		comp_dict['na_nii6585_outflow']  = {'comp':na_nii6585_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [NII]6549 Outflow; 
		na_nii6549_outflow_center		= 6549.860 # Angstroms
		na_nii6549_outflow_amp		   = na_nii6549_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_nii6549_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_nii6549_outflow_center,na_ha_outflow_voff)
		na_nii6549_outflow_fwhm		  = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(na_nii6549_outflow_fwhm_res)**2) # km/s
		na_nii6549_outflow_voff		  = na_ha_outflow_voff  # km/s
		if (na_nii6549_outflow_amp!=na_nii6549_outflow_amp/1.0) or (na_nii6549_outflow_amp==np.inf): na_nii6549_outflow_amp=0.0
		na_nii6549_outflow			   = gaussian(lam_gal,na_nii6549_outflow_center,na_nii6549_outflow_amp,na_nii6549_outflow_fwhm,na_nii6549_outflow_voff,velscale)
		host_model					   = host_model - na_nii6549_outflow
		comp_dict['na_nii6549_outflow']  = {'comp':na_nii6549_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# Broad [SII]6718 Outflow; 
		na_sii6718_outflow_center		= 6718.290 # Angstroms
		na_sii6718_outflow_amp		   = na_sii6718_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_sii6718_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_sii6718_outflow_center,na_ha_outflow_voff)
		na_sii6718_outflow_fwhm		  = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(na_sii6718_outflow_fwhm_res)**2) # km/s
		na_sii6718_outflow_voff		  = na_ha_outflow_voff  # km/s
		if (na_sii6718_outflow_amp!=na_sii6718_outflow_amp/1.0) or (na_sii6718_outflow_amp==np.inf): na_sii6718_outflow_amp=0.0
		na_sii6718_outflow			   = gaussian(lam_gal,na_sii6718_outflow_center,na_sii6718_outflow_amp,na_sii6718_outflow_fwhm,na_sii6718_outflow_voff,velscale)
		host_model					   = host_model - na_sii6718_outflow
		comp_dict['na_sii6718_outflow']  = {'comp':na_sii6718_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [SII]6732 Outflow; 
		na_sii6732_outflow_center		= 6732.670 # Angstroms
		na_sii6732_outflow_amp		   = na_sii6732_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_sii6732_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_sii6732_outflow_center,na_ha_outflow_voff)
		na_sii6732_outflow_fwhm		  = np.sqrt(p['na_oiii5007_outflow_fwhm']**2+(na_sii6732_outflow_fwhm_res)**2) # km/s
		na_sii6732_outflow_voff		  = na_ha_outflow_voff  # km/s
		if (na_sii6732_outflow_amp!=na_sii6732_outflow_amp/1.0) or (na_sii6732_outflow_amp==np.inf): na_sii6732_outflow_amp=0.0
		na_sii6732_outflow			   = gaussian(lam_gal,na_sii6732_outflow_center,na_sii6732_outflow_amp,na_sii6732_outflow_fwhm,na_sii6732_outflow_voff,velscale)
		host_model					   = host_model - na_sii6732_outflow
		comp_dict['na_sii6732_outflow']  = {'comp':na_sii6732_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
	elif (all(comp in param_names for comp in ['na_Ha_outflow_amp','na_Ha_outflow_fwhm','na_Ha_outflow_voff'])==True) and \
		 (all(comp not in param_names for comp in ['na_oiii5007_outflow_amp','na_oiii5007_outflow_fwhm','na_oiii5007_outflow_voff'])==True):
		# H-alpha Outflow; 
		na_ha_outflow_center			 = 6564.610 # Angstroms
		na_ha_outflow_amp				= p['na_Ha_outflow_amp'] # flux units
		na_ha_outflow_fwhm_res 	  		 = get_fwhm_res(fwhm_gal_ftn,na_ha_outflow_center,p['na_Ha_outflow_voff'])
		na_ha_outflow_fwhm			   = np.sqrt(p['na_Ha_outflow_fwhm']**2+(na_ha_outflow_fwhm_res)**2) # km/s
		na_ha_outflow_voff			   = p['na_Ha_outflow_voff']  # km/s  # km/s
		if (na_ha_outflow_amp!=na_ha_outflow_amp/1.0) or (na_ha_outflow_amp==np.inf): na_ha_outflow_amp=0.0
		na_Ha_outflow					= gaussian(lam_gal,na_ha_outflow_center,na_ha_outflow_amp,na_ha_outflow_fwhm,na_ha_outflow_voff,velscale)
		host_model					   = host_model - na_Ha_outflow
		comp_dict['na_Ha_outflow']	   = {'comp':na_Ha_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [NII]6585 Outflow;
		na_nii6585_outflow_center		= 6585.270 # Angstroms
		na_nii6585_outflow_amp		   = na_nii6585_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_nii6585_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_nii6585_outflow_center,na_ha_outflow_voff)
		na_nii6585_outflow_fwhm		  = np.sqrt(p['na_Ha_outflow_fwhm']**2+(na_nii6585_outflow_fwhm_res)**2)
		na_nii6585_outflow_voff		  = na_ha_outflow_voff
		if (na_nii6585_outflow_amp!=na_nii6585_outflow_amp/1.0) or (na_nii6585_outflow_amp==np.inf): na_nii6585_outflow_amp=0.0
		na_nii6585_outflow			   = gaussian(lam_gal,na_nii6585_outflow_center,na_nii6585_outflow_amp,na_nii6585_outflow_fwhm,na_nii6585_outflow_voff,velscale)
		host_model					   = host_model - na_nii6585_outflow
		comp_dict['na_nii6585_outflow']  = {'comp':na_nii6585_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [NII]6549 Outflow; 
		na_nii6549_outflow_center		= 6549.860 # Angstroms
		na_nii6549_outflow_amp		   = na_nii6549_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_nii6549_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_nii6549_outflow_center,na_ha_outflow_voff)
		na_nii6549_outflow_fwhm		  = np.sqrt(p['na_Ha_outflow_fwhm']**2+(na_nii6549_outflow_fwhm_res)**2) # km/s
		na_nii6549_outflow_voff		  = na_ha_outflow_voff  # km/s
		if (na_nii6549_outflow_amp!=na_nii6549_outflow_amp/1.0) or (na_nii6549_outflow_amp==np.inf): na_nii6549_outflow_amp=0.0
		na_nii6549_outflow			   = gaussian(lam_gal,na_nii6549_outflow_center,na_nii6549_outflow_amp,na_nii6549_outflow_fwhm,na_nii6549_outflow_voff,velscale)
		host_model					   = host_model - na_nii6549_outflow
		comp_dict['na_nii6549_outflow']  = {'comp':na_nii6549_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# Broad [SII]6718 Outflow; 
		na_sii6718_outflow_center		= 6718.290 # Angstroms
		na_sii6718_outflow_amp		   = na_sii6718_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_sii6718_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_sii6718_outflow_center,na_ha_outflow_voff)
		na_sii6718_outflow_fwhm		  = np.sqrt(p['na_Ha_outflow_fwhm']**2+(na_sii6718_outflow_fwhm_res)**2) # km/s
		na_sii6718_outflow_voff		  = na_ha_outflow_voff  # km/s
		if (na_sii6718_outflow_amp!=na_sii6718_outflow_amp/1.0) or (na_sii6718_outflow_amp==np.inf): na_sii6718_outflow_amp=0.0
		na_sii6718_outflow			   = gaussian(lam_gal,na_sii6718_outflow_center,na_sii6718_outflow_amp,na_sii6718_outflow_fwhm,na_sii6718_outflow_voff,velscale)
		host_model					   = host_model - na_sii6718_outflow
		comp_dict['na_sii6718_outflow']  = {'comp':na_sii6718_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}
		# [SII]6732 Outflow; 
		na_sii6732_outflow_center		= 6732.670 # Angstroms
		na_sii6732_outflow_amp		   = na_sii6732_core_amp*na_ha_outflow_amp/na_ha_core_amp # flux units
		na_sii6732_outflow_fwhm_res 	 = get_fwhm_res(fwhm_gal_ftn,na_sii6732_outflow_center,na_ha_outflow_voff)
		na_sii6732_outflow_fwhm		  = np.sqrt(p['na_Ha_outflow_fwhm']**2+(na_sii6732_outflow_fwhm_res)**2) # km/s
		na_sii6732_outflow_voff		  = na_ha_outflow_voff  # km/s
		if (na_sii6732_outflow_amp!=na_sii6732_outflow_amp/1.0) or (na_sii6732_outflow_amp==np.inf): na_sii6732_outflow_amp=0.0
		na_sii6732_outflow			   = gaussian(lam_gal,na_sii6732_outflow_center,na_sii6732_outflow_amp,na_sii6732_outflow_fwhm,na_sii6732_outflow_voff,velscale)
		host_model					   = host_model - na_sii6732_outflow
		comp_dict['na_sii6732_outflow']  = {'comp':na_sii6732_outflow,'pcolor':'xkcd:magenta','linewidth':1.0}


	########################################################################################################

	# Broad Lines
	#### Br. H-gamma #######################################################################################
	if all(comp in param_names for comp in ['br_Hg_amp','br_Hg_fwhm','br_Hg_voff'])==True:
		br_hg_center	   = 4341.680 # Angstroms
		br_hg_amp		   = p['br_Hg_amp'] # flux units
		br_hg_fwhm_res 	   = get_fwhm_res(fwhm_gal_ftn,br_hg_center,p['br_Hg_voff'])
		br_hg_fwhm		   = np.sqrt(p['br_Hg_fwhm']**2+(br_hg_fwhm_res)**2) # km/s
		br_hg_voff		   = p['br_Hg_voff']  # km/s
		# br_Hg			   = gaussian(lam_gal,br_hg_center,br_hg_amp,br_hg_fwhm,br_hg_voff,velscale)
		br_Hg			   = line_model(line_profile,lam_gal,br_hg_center,br_hg_amp,br_hg_fwhm,br_hg_voff,velscale)
		host_model		   = host_model - br_Hg
		comp_dict['br_Hg'] = {'comp':br_Hg,'pcolor':'xkcd:turquoise','linewidth':1.0}
	#### Br. H-beta ########################################################################################
	if all(comp in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff'])==True:
		br_hb_center	   = 4862.68 # Angstroms
		br_hb_amp		   = p['br_Hb_amp'] # flux units
		br_hb_fwhm_res 	   = get_fwhm_res(fwhm_gal_ftn,br_hb_center,p['br_Hb_voff'])
		br_hb_fwhm		   = np.sqrt(p['br_Hb_fwhm']**2+(br_hb_fwhm_res)**2) # km/s
		br_hb_voff		   = p['br_Hb_voff']  # km/s
		# br_Hb			   = gaussian(lam_gal,br_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)
		br_Hb			   = line_model(line_profile,lam_gal,br_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)
		host_model		   = host_model - br_Hb
		comp_dict['br_Hb'] = {'comp':br_Hb,'pcolor':'xkcd:turquoise','linewidth':1.0}
	
	#### Br. H-alpha #######################################################################################
	if all(comp in param_names for comp in ['br_Ha_amp','br_Ha_fwhm','br_Ha_voff'])==True:
		br_ha_center	   = 6564.610 # Angstroms
		br_ha_amp		   = p['br_Ha_amp'] # flux units
		br_ha_fwhm_res 	   = get_fwhm_res(fwhm_gal_ftn,br_ha_center,p['br_Ha_voff'])
		br_ha_fwhm		   = np.sqrt(p['br_Ha_fwhm']**2+(br_ha_fwhm_res)**2) # km/s
		br_ha_voff		   = p['br_Ha_voff']  # km/s
		# br_Ha			   = gaussian(lam_gal,br_ha_center,br_ha_amp,br_ha_fwhm,br_ha_voff,velscale)
		br_Ha			   = line_model(line_profile,lam_gal,br_ha_center,br_ha_amp,br_ha_fwhm,br_ha_voff,velscale)
		host_model		   = host_model - br_Ha
		comp_dict['br_Ha'] = {'comp':br_Ha,'pcolor':'xkcd:turquoise','linewidth':1.0}

	########################################################################################################

	########################################################################################################

	############################# Host-galaxy Component ######################################################

	if all(comp in param_names for comp in ['gal_temp_amp'])==True:
		gal_temp = p['gal_temp_amp']*(gal_temp)

		host_model = (host_model) - (gal_temp) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['host_galaxy'] = {'comp':gal_temp,'pcolor':'xkcd:lime green','linewidth':1.0}

	########################################################################################################   

	############################# LOSVD Component ####################################################

	if all(comp in param_names for comp in ['stel_vel','stel_disp'])==True:
		# Convolve the templates with a LOSVD
		losvd_params = [p['stel_vel'],p['stel_disp']] # ind 0 = velocity*, ind 1 = sigma*
		conv_temp	= convolve_gauss_hermite(temp_fft,npad,float(velscale),\
					   losvd_params,npix,velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
		
		# Fitted weights of all templates using Non-negative Least Squares (NNLS)
		host_model[host_model/host_model!=1] = 0 
		weights	 = nnls(conv_temp,host_model) # scipy.optimize Non-negative Least Squares
		host_galaxy = (np.sum(weights*conv_temp,axis=1)) 
		comp_dict['host_galaxy'] = {'comp':host_galaxy,'pcolor':'xkcd:lime green','linewidth':1.0}

	 ########################################################################################################

	# The final model
	gmodel = np.sum((d['comp'] for d in comp_dict.values() if d),axis=0)
	
	########################## Measure Emission Line Fluxes #################################################

	# Fluxes of components are stored in a dictionary and returned to emcee as metadata blob.  
	# This is a vast improvement over the previous method, which was storing fluxes in an 
	# output file at each iteration, which is computationally expensive for opening, writing to, and closing 
	# a file nwalkers x niter times.
	if (fit_type=='final') and (output_model==False):
		fluxes = {}
		for key in comp_dict:
			# compute the integrated flux 
			flux = simps(comp_dict[key]['comp'],lam_gal)
			# add key/value pair to dictionary
			fluxes[key+'_flux'] = flux
		
	##################################################################################

	# Add last components to comp_dict for plotting purposes 
	# Add galaxy, sigma, model, and residuals to comp_dict
	comp_dict['data']	   = {'comp':galaxy		   ,'pcolor':'xkcd:white', 'linewidth':0.5}
	comp_dict['wave']	   = {'comp':lam_gal 	   ,'pcolor':'xkcd:black', 'linewidth':0.5}
	comp_dict['noise']	   = {'comp':noise		   ,'pcolor':'xkcd:cyan' , 'linewidth':0.5}
	comp_dict['model']	   = {'comp':gmodel		   ,'pcolor':'xkcd:red'  , 'linewidth':1.0}
	comp_dict['resid']     = {'comp':galaxy-gmodel ,'pcolor':'xkcd:white', 'linewidth':0.5}
	
	##################################################################################

	##################################################################################

	if (fit_type=='init') and (output_model==False): # For max. likelihood fitting
		return gmodel
	if (fit_type=='init') and (output_model==True): # For max. likelihood fitting
		return comp_dict
	elif (fit_type=='outflow_test'):
		return comp_dict
	elif (fit_type=='final') and (output_model==False): # For emcee
		return gmodel, fluxes
	elif (fit_type=='final') and (output_model==True): # output all models for best-fit model
		return comp_dict

########################################################################################################


#### Host-Galaxy Template##############################################################################

def galaxy_template(lam,age=10.0,print_output=True):
	"""
	This function is used if we use the Maraston et al. 2009 SDSS composite templates for fitting
	the host-galaxy for the outflow test, maximum likelihood estimation, and the emcee model if 
	fit_losvd=False.  Default = 5.0 Gyr.
	"""
	ages = [0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,14.0]
	if ((age<0.1) or (age>14.0)):
		if (np.min(lam)>=3540.5) and (np.max(lam)<=7409.6):
			fname = 'MILES_'+str(age)+'.csv'
			df = pd.read_csv('badass_data_files/MILES_ssp_templates/'+fname,skiprows=54,sep='   ',names=['lam','','flam'], skipinitialspace=True,header=0,engine='python')
			wave = np.array(df['lam'])
			flux = np.array(df['flam'])
			val,idx = find_nearest(wave,5500.0)
			flux = flux/flux[idx] # Normalize to 1.0 at 5500 A
			# Interpolate the template 
			gal_interp = interp1d(wave,flux,kind='linear',bounds_error=False,fill_value=(0,0))
			gal_temp = gal_interp(lam)
			# Normalize by median
			gal_temp = gal_temp/np.median(gal_temp)
			return gal_temp
		elif (np.min(lam)<3540.5) or (np.max(lam)>7409.6):
			print('\n You must choose an age between (1 Gyr <= age <= 14 Gyr)! Using 10.0 Gyr template instead... \n')
			fname = 'M09_'+str(age)+'.csv'
			df = pd.read_csv('badass_data_files/M09_ssp_templates/'+fname,skiprows=5,sep=',',names=['t_Gyr','Z','lam','flam'])
			wave = np.array(df['lam'])
			flux = np.array(df['flam'])
			flux = flux/flux[wave==5500.] # Normalize to 1.0 at 5500 A
			# Interpolate the template 
			gal_interp = interp1d(wave,flux,kind='linear',bounds_error=False,fill_value=(0,0))
			gal_temp = gal_interp(lam)
			# Normalize by median
			gal_temp = gal_temp/np.median(gal_temp)
			return gal_temp
	elif ((age>=0.1) and (age<=15.0)):
		# Get nearest user-input age
		age, aidx =find_nearest(ages,age)
		if (np.min(lam)>=3540.5) and (np.max(lam)<=7409.6):
			fname = 'MILES_'+str(age)+'.csv'
			df = pd.read_csv('badass_data_files/MILES_ssp_templates/'+fname,skiprows=54,sep='   ',names=['lam','','flam'], skipinitialspace=True,header=0,engine='python')
			wave = np.array(df['lam'])
			flux = np.array(df['flam'])
			val,idx = find_nearest(wave,5500.0)
			flux = flux/flux[idx] # Normalize to 1.0 at 5500 A
			# Interpolate the template 
			gal_interp = interp1d(wave,flux,kind='linear',bounds_error=False,fill_value=(0,0))
			gal_temp = gal_interp(lam)
			# Normalize by median
			gal_temp = gal_temp/np.median(gal_temp)
			return gal_temp
		elif (np.min(lam)<3540.5) or (np.max(lam)>7409.6):
			if print_output:
				print('\n Using Maraston et al. 2009 %0.1f Gyr template... \n ' % age)
			fname = 'M09_'+str(age)+'.csv'
			df = pd.read_csv('badass_data_files/M09_ssp_templates/'+fname,skiprows=5,sep=',',names=['t_Gyr','Z','lam','flam'])
			wave = np.array(df['lam'])
			flux = np.array(df['flam'])
			flux = flux/flux[wave==5500.] # Normalize to 1.0 at 5500 A
			# Interpolate the template 
			gal_interp = interp1d(wave,flux,kind='linear',bounds_error=False,fill_value=(0,0))
			gal_temp = gal_interp(lam)
			# Normalize by median
			gal_temp = gal_temp/np.median(gal_temp)
			return gal_temp

##################################################################################


#### FeII Templates ##############################################################

def initialize_feii(lam_gal,feii_options):
	"""
	Generate FeII templates.  Options:

	'VC04' : Veron-Cetty et al. (2004) template, which utilizes a single broad 
			 and single narrow line template with fixed relative intensities. 
			 One can choose to fix FWHM and VOFF for each, and only vary 
			 amplitudes (2 free parameters), or vary amplitude, FWHM, and VOFF
			 for each template (6 free parameters)

	'K10'  : Kovacevic et al. (2010) template, which treats the F, S, and G line 
			 groups as independent templates (each amplitude is a free parameter)
			 and whose relative intensities are temperature dependent (1 free 
			 parameter).  There are additonal lines from IZe1 that only vary in 
			 amplitude.  All 4 line groups share the same FWHM and VOFF, for a 
			 total of 7 free parameters.  This template is only recommended 
			 for objects with very strong FeII emission, for which the LOSVD
			 cannot be determined at all.
			 """
	if (feii_options['template']['type']=='VC04'):
		# Read in template data
		na_feii_table = pd.read_csv('badass_data_files/feii_templates/veron-cetty_2004/na_feii_template.csv')
		br_feii_table = pd.read_csv('badass_data_files/feii_templates/veron-cetty_2004/br_feii_template.csv')

		# Extract relative intensities and line centers
		na_feii_rel_int  = np.array(na_feii_table['na_relative_intensity'])
		na_feii_center   = np.array(na_feii_table['na_wavelength'])
		br_feii_rel_int  = np.array(br_feii_table['br_relative_intensity'])
		br_feii_center   = np.array(br_feii_table['br_wavelength'])
		# Prune templates 25 angstroms from edges
		edge_pad = 0 # angstroms
		na_feii_rel_int = na_feii_rel_int[(na_feii_center>=(lam_gal[0]+edge_pad)) & (na_feii_center<=(lam_gal[-1]-edge_pad))]
		na_feii_center_  = na_feii_center[(na_feii_center>=(lam_gal[0]+edge_pad)) & (na_feii_center<=(lam_gal[-1]-edge_pad))]
		br_feii_rel_int = br_feii_rel_int[(br_feii_center>=(lam_gal[0]+edge_pad)) & (br_feii_center<=(lam_gal[-1]-edge_pad))]
		br_feii_center_  = br_feii_center[(br_feii_center>=(lam_gal[0]+edge_pad)) & (br_feii_center<=(lam_gal[-1]-edge_pad))]

		# Store as as tuples of lists to be reconstructed by fit_model()
		return (na_feii_center_,na_feii_rel_int,br_feii_center_,br_feii_rel_int)

	elif (feii_options['template']['type']=='K10'):
		# Read in template data
		F_trans_table    = pd.read_csv('badass_data_files/feii_templates/kovacevic_2010/k10_F_transitions.csv')
		S_trans_table    = pd.read_csv('badass_data_files/feii_templates/kovacevic_2010/k10_S_transitions.csv')
		G_trans_table    = pd.read_csv('badass_data_files/feii_templates/kovacevic_2010/k10_G_transitions.csv')
		IZe1_trans_table = pd.read_csv('badass_data_files/feii_templates/kovacevic_2010/k10_IZw1_transitions.csv')
		
		# Extract relative intensities and line centers
		f_trans_center  = np.array(F_trans_table['wavelength']) 
		f_trans_gf      = np.array(F_trans_table['gf']) # gf ratio
		f_trans_e2      = np.array(F_trans_table['E2_J']) # upper energy level (Joules)
		s_trans_center  = np.array(S_trans_table['wavelength']) 
		s_trans_gf      = np.array(S_trans_table['gf']) # gf ratio
		s_trans_e2      = np.array(S_trans_table['E2_J']) # upper energy level (Joules)
		g_trans_center  = np.array(G_trans_table['wavelength']) 
		g_trans_gf      = np.array(G_trans_table['gf']) # gf ratio
		g_trans_e2      = np.array(G_trans_table['E2_J']) # upper energy level (Joules)
		z_trans_center  = np.array(IZe1_trans_table['wavelength'])
		z_trans_rel_int = np.array(IZe1_trans_table['rel_int']) # IZw1 NOT temperature dependent
		# Prune templates 25 angstroms from edges
		edge_pad = 0 # angstroms
		f_trans_center_  = f_trans_center[(f_trans_center>=(lam_gal[0]+edge_pad)) & (f_trans_center<=(lam_gal[-1]-edge_pad))]
		f_trans_gf      = f_trans_gf[(f_trans_center>=(lam_gal[0]+edge_pad)) & (f_trans_center<=(lam_gal[-1]-edge_pad))]
		f_trans_e2      = f_trans_e2[(f_trans_center>=(lam_gal[0]+edge_pad)) & (f_trans_center<=(lam_gal[-1]-edge_pad))]
		s_trans_center_  = s_trans_center[(s_trans_center>=(lam_gal[0]+edge_pad)) & (s_trans_center<=(lam_gal[-1]-edge_pad))]
		s_trans_gf      = s_trans_gf[(s_trans_center>=(lam_gal[0]+edge_pad)) & (s_trans_center<=(lam_gal[-1]-edge_pad))]
		s_trans_e2      = s_trans_e2[(s_trans_center>=(lam_gal[0]+edge_pad)) & (s_trans_center<=(lam_gal[-1]-edge_pad))]
		g_trans_center_  = g_trans_center[(g_trans_center>=(lam_gal[0]+edge_pad)) & (g_trans_center<=(lam_gal[-1]-edge_pad))]
		g_trans_gf      = g_trans_gf[(g_trans_center>=(lam_gal[0]+edge_pad)) & (g_trans_center<=(lam_gal[-1]-edge_pad))]
		g_trans_e2      = g_trans_e2[(g_trans_center>=(lam_gal[0]+edge_pad)) & (g_trans_center<=(lam_gal[-1]-edge_pad))]
		z_trans_center_  = z_trans_center[(z_trans_center>=(lam_gal[0]+edge_pad)) & (z_trans_center<=(lam_gal[-1]-edge_pad))]
		z_trans_rel_int = z_trans_rel_int[(z_trans_center>=(lam_gal[0]+edge_pad)) & (z_trans_center<=(lam_gal[-1]-edge_pad))]

		# Return a list of arrays which will be unpacked during the fitting process
		return (f_trans_center_,f_trans_gf,f_trans_e2,
		        s_trans_center_,s_trans_gf,s_trans_e2,
		        g_trans_center_,g_trans_gf,g_trans_e2,
		        z_trans_center_,z_trans_rel_int)




def get_fwhm_res(fwhm_gal_ftn,line_center,line_voff):
		c = 299792.458
		fwhm_res = (fwhm_gal_ftn(line_center + 
			   	   (line_voff*line_center/c))/(line_center + 
				   (line_voff*line_center/c))*c)
		return fwhm_res

def VC04_feii_template(lam_gal,fwhm_gal,tab,amp,fwhm,voff,velscale,run_dir):
	"""
	Constructs an FeII template using a series of Gaussians and ensures
	no lines are created at the edges of the fitting region.
	"""
	# Interpolation function for the instrumental fwhm
	fwhm_gal_ftn = interp1d(lam_gal,fwhm_gal,kind='linear',bounds_error=False,fill_value=(0,0))
	
	# Unpack the tables
	center, rel_int = tab
	center  = np.array(center)
	rel_int = np.array(rel_int)
	amp = rel_int*amp
	# Get the fwhm resolution (in km/s) at every line center shifted by the velocity offset
	# start_time = time.time()
	fwhm_res = get_fwhm_res(fwhm_gal_ftn,center,voff)
	# Calculate the observed width (intrinsic + instrumental added in quadrature)
	fwhm_obs = np.sqrt((fwhm)**2 + (fwhm_res)**2)
	# print("--- %s seconds ---" % (time.time() - start_time))

	# start_time = time.time()
	template = gaussian(lam_gal,center,amp,fwhm_obs,voff,velscale)
	# print("--- %s seconds ---" % (time.time() - start_time))

	# If the summation results in 0.0, it means that features were too close 
	# to the edges of the fitting region (usually because the region is too 
	# small), then simply return an array of zeros.
	if (isinstance(template,int)) or (isinstance(template,float)):
		template=np.zeros(len(lam_gal))
	
	# fig = plt.figure(figsize=(10,3))
	# ax1 = fig.add_subplot(1,1,1)
	# ax1.plot(lam_gal,template)
	# plt.tight_layout()
	# plt.savefig(run_dir+'feii_plot.pdf')
	# plt.close()

	# sys.exit()
	return template

def K10_feii_template(lam_gal,transition,fwhm_gal,tab,amp,temp,fwhm,voff,velscale,run_dir):
    """
    Constructs an Kovacevic et al. 2010 FeII template using a series of Gaussians and ensures
    no lines are created at the edges of the fitting region.
    """
    # Interpolation function for the instrumental fwhm
    fwhm_gal_ftn = interp1d(lam_gal,fwhm_gal,kind='linear',bounds_error=False,fill_value=(0,0))
    if transition=='IZw1':
        # Unpack the tables
        center, rel_int = tab
        center  = np.array(center)
        rel_int = np.array(rel_int)
        amp = rel_int*amp
        # Get the fwhm resolution (in km/s) at every line center shifted by the velocity offset
        # start_time = time.time()
        fwhm_res = get_fwhm_res(fwhm_gal_ftn,center,voff)
        # Calculate the observed width (intrinsic + instrumental added in quadrature)
        fwhm_obs = np.sqrt((fwhm)**2 + (fwhm_res)**2)
        # Get summed template
        template = gaussian(lam_gal,center,amp,fwhm_obs,voff,velscale)
        # If the summation results in 0.0, it means that features were too close 
        # to the edges of the fitting region (usually because the region is too 
        # small), then simply return an array of zeros.
        if (isinstance(template,int)) or (isinstance(template,float)):
            template=np.zeros(len(lam_gal))
    elif (transition=='S') or (transition=='F') or (transition=='G'):
        # Calculate temperature dependent relative intensities 
        center,gf,e2 = tab
        center = np.array(center) # line centers
        gf     = np.array(gf) # statistical weights 
        e2     = np.array(e2) # upper leavel energies of transitions
        temp   = float(temp) # temperature
        rel_int = calculate_k10_rel_int(transition,center,gf,e2,amp,temp)
        amp = rel_int*amp
        # Get the fwhm resolution (in km/s) at every line center shifted by the velocity offset
        # start_time = time.time()
        fwhm_res = get_fwhm_res(fwhm_gal_ftn,center,voff)
        # Calculate the observed width (intrinsic + instrumental added in quadrature)
        fwhm_obs = np.sqrt((fwhm)**2 + (fwhm_res)**2)
        # Get summed template
        template = gaussian(lam_gal,center,amp,fwhm_obs,voff,velscale)
        # If the summation results in 0.0, it means that features were too close 
        # to the edges of the fitting region (usually because the region is too 
        # small), then simply return an array of zeros.
        if (isinstance(template,int)) or (isinstance(template,float)):
            template=np.zeros(len(lam_gal))
    return template
    
def calculate_k10_rel_int(transition,center,gf,e2,I2,temp):
    """
    Calculate relative intensities for the S, F, and G FeII line groups
    from Kovacevic et al. 2010 template as a fucntion a temperature.
    """
    c = 2.99792458e+8  # speed of light; m/s
    h = 6.62607004e-34 # Planck's constant; m2 kg s-1
    k = 1.38064852e-23 # Boltzmann constant; m2 kg s-2 K-1
    if (transition=='F'):
        # For the F transition, we normalize to the values of 4549.474 
        rel_int = I2*(4549.474/center)**3 * (gf/1.10e-02) * np.exp(-1.0/(k*temp) * (e2 - 8.896255e-19))
        return rel_int
    elif (transition=='S'):
        # For the S transition, we normalize to the values of 5018.440
        rel_int = I2*(5018.440/center)**3 * (gf/3.98e-02) * np.exp(-1.0/(k*temp) * (e2 - 8.589111e-19))
        return rel_int
    elif (transition=='G'):
        # For the G transition, we normalize to the values of 5316.615
        rel_int = I2*(5316.615/center)**3 * (gf/1.17e-02) * np.exp(-1.0/(k*temp) * (e2 - 8.786549e-19))
        return rel_int


##################################################################################


#### Power-Law Template ##########################################################

def simple_power_law(x,amp,alpha):#,xb)
	"""
	Simple power-low function to model
	the AGNs continuum.

	Parameters
	----------
	x	 : array_like
			wavelength vector (angstroms)
	amp   : float 
			continuum amplitude (flux density units)
	alpha : float
			power-law slope
	xb	: float
			location of break in the power-law (angstroms)

	Returns
	----------
	C	 : array
			AGN continuum model the same length as x
	"""
	# This works
	xb = np.max(x)-(0.5*(np.max(x)-np.min(x))) # take to be half of the wavelength range
	C = amp*(x/xb)**alpha # un-normalized

	return C

##################################################################################


##################################################################################

def gaussian(x,center,amp,fwhm,voff,velscale):
	"""
	Produces a gaussian vector the length of
	x with the specified parameters.
	
	Parameters
	----------
	x		: array_like
	           the wavelength vector in angstroms.

	center   : float
	           the mean or center wavelength of the gaussian in angstroms.
	fwhm 	: float
	           the full-width half max of the gaussian in km/s.
	amp	  : float
	           the amplitude of the gaussian in flux units.
	voff	 : the velocity offset (in km/s) from the rest-frame 
	           line-center (taken from SDSS rest-frame emission
	           line wavelengths)
	velscale : velocity scale; km/s/pixel

	Returns
	-------
	g		: array_like
	           a one-dimensional gaussian as a function of x,
	           where x is measured in PIXELS.
	"""
	x_pix = np.array(range(len(x)))
	# Interpolation function that maps x (in angstroms) to pixels so we can 
	pix_interp_ftn = interp1d(x,x_pix,kind='linear',bounds_error=False,fill_value=(0,0))

	center_pix = pix_interp_ftn(center) # pixel value corresponding to line center
	sigma = fwhm/2.3548 # Gaussian dispersion in km/s
	sigma_pix = sigma/velscale # dispersion in pixels (velscale = km/s/pixel)
	voff_pix = voff/velscale # velocity offset in pixels
	center_pix = center_pix + voff_pix # shift the line center by voff in pixels

	# start_time = time.time()
	# if not isinstance(center,float):
	x_pix = x_pix.reshape((len(x_pix),1))
	g = amp*np.exp(-0.5*(x_pix-(center_pix))**2/(sigma_pix)**2) # construct gaussian
	g = np.sum(g,axis=1)

	# Make sure edges of gaussian are zero to avoid wierd things
	g[g<1.0e-6] = 0.0
	# Replace the ends with the same value 
	g[0]  = g[1]
	g[-1] = g[-2]
	# print("--- %s seconds ---" % (time.time() - start_time))

	return g

##################################################################################

##################################################################################

def lorentzian(x,center,amp,fwhm,voff,velscale):
	"""
	Produces a lorentzian vector the length of
	x with the specified parameters.
	(See: https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Lorentz1D.html)
	
	Parameters
	----------
	x		: array_like
	           the wavelength vector in angstroms.

	center   : float
	           the mean or center wavelength of the gaussian in angstroms.
	fwhm 	: float
	           the full-width half max of the gaussian in km/s.
	amp	  : float
	           the amplitude of the gaussian in flux units.
	voff	 : the velocity offset (in km/s) from the rest-frame 
	           line-center (taken from SDSS rest-frame emission
	           line wavelengths)
	velscale : velocity scale; km/s/pixel

	Returns
	-------
	g		: array_like
	           a one-dimensional gaussian as a function of x,
	           where x is measured in PIXELS.
	"""
	x_pix = np.array(range(len(x)))
	# Interpolation function that maps x (in angstroms) to pixels so we can 
	pix_interp_ftn = interp1d(x,x_pix,kind='linear',bounds_error=False,fill_value=(0,0))

	center_pix = pix_interp_ftn(center) # pixel value corresponding to line center
	fwhm_pix = fwhm/velscale # dispersion in pixels (velscale = km/s/pixel)
	voff_pix = voff/velscale # velocity offset in pixels
	center_pix = center_pix + voff_pix # shift the line center by voff in pixels

	# start_time = time.time()
	# if not isinstance(center,float):
	x_pix = x_pix.reshape((len(x_pix),1))
	gamma = 0.5*fwhm_pix
	l = amp*( (gamma**2) / (gamma**2+(x_pix-center_pix)**2) ) # construct lorenzian
	l= np.sum(l,axis=1)

	# Make sure edges of gaussian are zero to avoid wierd things
	l[l<1.0e-6] = 0.0
	# print("--- %s seconds ---" % (time.time() - start_time))

	return l

##################################################################################

##################################################################################

# pPXF Routines (from Cappellari 2017)


# NAME:
#   GAUSSIAN_FILTER1D
#
# MODIFICATION HISTORY:
#   V1.0.0: Written as a replacement for the Scipy routine with the same name,
#	   to be used with variable sigma per pixel. MC, Oxford, 10 October 2015

def gaussian_filter1d(spec, sig):
	"""
	Convolve a spectrum by a Gaussian with different sigma for every pixel.
	If all sigma are the same this routine produces the same output as
	scipy.ndimage.gaussian_filter1d, except for the border treatment.
	Here the first/last p pixels are filled with zeros.
	When creating a template library for SDSS data, this implementation
	is 60x faster than a naive for loop over pixels.

	:param spec: vector with the spectrum to convolve
	:param sig: vector of sigma values (in pixels) for every pixel
	:return: spec convolved with a Gaussian with dispersion sig

	"""
	sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
	p = int(np.ceil(np.max(3*sig)))
	m = 2*p + 1  # kernel size
	x2 = np.linspace(-p, p, m)**2

	n = spec.size
	a = np.zeros((m, n))
	for j in range(m):   # Loop over the small size of the kernel
		a[j, p:-p] = spec[j:n-m+j+1]

	gau = np.exp(-x2[:, None]/(2*sig**2))
	gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

	conv_spectrum = np.sum(a*gau, 0)

	return conv_spectrum

##################################################################################

def log_rebin(lamRange, spec, oversample=1, velscale=None, flux=False):
	"""
	Logarithmically rebin a spectrum, while rigorously conserving the flux.
	Basically the photons in the spectrum are simply redistributed according
	to a new grid of pixels, with non-uniform size in the spectral direction.
	
	When the flux keyword is set, this program performs an exact integration 
	of the original spectrum, assumed to be a step function within the 
	linearly-spaced pixels, onto the new logarithmically-spaced pixels. 
	The output was tested to agree with the analytic solution.

	:param lamRange: two elements vector containing the central wavelength
		of the first and last pixels in the spectrum, which is assumed
		to have constant wavelength scale! E.g. from the values in the
		standard FITS keywords: LAMRANGE = CRVAL1 + [0, CDELT1*(NAXIS1 - 1)].
		It must be LAMRANGE[0] < LAMRANGE[1].
	:param spec: input spectrum.
	:param oversample: can be used, not to loose spectral resolution,
		especally for extended wavelength ranges and to avoid aliasing.
		Default: OVERSAMPLE=1 ==> Same number of output pixels as input.
	:param velscale: velocity scale in km/s per pixels. If this variable is
		not defined, then it will contain in output the velocity scale.
		If this variable is defined by the user it will be used
		to set the output number of pixels and wavelength scale.
	:param flux: (boolean) True to preserve total flux. In this case the
		log rebinning changes the pixels flux in proportion to their
		dLam so the following command will show large differences
		beween the spectral shape before and after LOG_REBIN:

		   plt.plot(exp(logLam), specNew)  # Plot log-rebinned spectrum
		   plt.plot(np.linspace(lamRange[0], lamRange[1], spec.size), spec)

		By defaul, when this is False, the above two lines produce
		two spectra that almost perfectly overlap each other.
	:return: [specNew, logLam, velscale] where logLam is the natural
		logarithm of the wavelength and velscale is in km/s.

	"""
	lamRange = np.asarray(lamRange)
	assert len(lamRange) == 2, 'lamRange must contain two elements'
	assert lamRange[0] < lamRange[1], 'It must be lamRange[0] < lamRange[1]'
	assert spec.ndim == 1, 'input spectrum must be a vector'
	n = spec.shape[0]
	m = int(n*oversample)

	dLam = np.diff(lamRange)/(n - 1.)		# Assume constant dLam
	lim = lamRange/dLam + [-0.5, 0.5]		# All in units of dLam
	borders = np.linspace(*lim, num=n+1)	 # Linearly
	logLim = np.log(lim)

	c = 299792.458						   # Speed of light in km/s
	if velscale is None:					 # Velocity scale is set by user
		velscale = np.diff(logLim)/m*c	   # Only for output
	else:
		logScale = velscale/c
		m = int(np.diff(logLim)/logScale)	# Number of output pixels
		logLim[1] = logLim[0] + m*logScale

	newBorders = np.exp(np.linspace(*logLim, num=m+1)) # Logarithmically
	k = (newBorders - lim[0]).clip(0, n-1).astype(int)

	specNew = np.add.reduceat(spec, k)[:-1]  # Do analytic integral
	specNew *= np.diff(k) > 0	# fix for design flaw of reduceat()
	specNew += np.diff((newBorders - borders[k])*spec[k])

	if not flux:
		specNew /= np.diff(newBorders)

	# Output log(wavelength): log of geometric mean
	logLam = np.log(np.sqrt(newBorders[1:]*newBorders[:-1])*dLam)

	return specNew, logLam, velscale

###############################################################################

def rebin(x, factor):
	"""
	Rebin a vector, or the first dimension of an array,
	by averaging within groups of "factor" adjacent values.

	"""
	if factor == 1:
		xx = x
	else:
		xx = x.reshape(len(x)//factor, factor, -1).mean(1).squeeze()

	return xx

###############################################################################

def template_rfft(templates):
	npix_temp = templates.shape[0]
	templates = templates.reshape(npix_temp, -1)
	npad = fftpack.next_fast_len(npix_temp)
	templates_rfft = np.fft.rfft(templates, npad, axis=0)
	
	return templates_rfft,npad

##################################################################################

def convolve_gauss_hermite(templates_rfft,npad, velscale, start, npix,
						   velscale_ratio=1, sigma_diff=0, vsyst=0):
	"""
	Convolve a spectrum, or a set of spectra, arranged into columns of an array,
	with a LOSVD parametrized by the Gauss-Hermite series.

	This is intended to reproduce what pPXF does for the convolution and it
	uses the analytic Fourier Transform of the LOSVD introduced in

		Cappellari (2017) http://adsabs.harvard.edu/abs/2017MNRAS.466..798C

	EXAMPLE:
		...
		pp = ppxf(templates, galaxy, noise, velscale, start,
				  degree=4, mdegree=4, velscale_ratio=ratio, vsyst=dv)

		spec = convolve_gauss_hermite(templates, velscale, pp.sol, galaxy.size,
									  velscale_ratio=ratio, vsyst=dv)

		# The spectrum below is equal to pp.bestfit to machine precision

		spectrum = (spec @ pp.weights)*pp.mpoly + pp.apoly

	:param spectra: log rebinned spectra
	:param velscale: velocity scale c*dLogLam in km/s
	:param start: parameters of the LOSVD [vel, sig, h3, h4,...]
	:param npix: number of output pixels
	:return: vector or array with convolved spectra

	"""
#	 npix_temp = templates.shape[0]
#	 templates = templates.reshape(npix_temp, -1)
	start = np.array(start)  # make copy
	start[:2] /= velscale
	vsyst /= velscale

#	 npad = fftpack.next_fast_len(npix_temp)
#	 templates_rfft = np.fft.rfft(templates, npad, axis=0)
	lvd_rfft = losvd_rfft(start, 1, start.shape, templates_rfft.shape[0],
						  1, vsyst, velscale_ratio, sigma_diff)

	conv_temp = np.fft.irfft(templates_rfft*lvd_rfft[:, 0], npad, axis=0)
	conv_temp = rebin(conv_temp[:npix*velscale_ratio, :], velscale_ratio)

	return conv_temp

##################################################################################

def losvd_rfft(pars, nspec, moments, nl, ncomp, vsyst, factor, sigma_diff):
	"""
	Analytic Fourier Transform (of real input) of the Gauss-Hermite LOSVD.
	Equation (38) of Cappellari M., 2017, MNRAS, 466, 798
	http://adsabs.harvard.edu/abs/2017MNRAS.466..798C

	"""
	losvd_rfft = np.empty((nl, ncomp, nspec), dtype=complex)
	p = 0
	for j, mom in enumerate(moments):  # loop over kinematic components
		for k in range(nspec):  # nspec=2 for two-sided fitting, otherwise nspec=1
			s = 1 if k == 0 else -1  # s=+1 for left spectrum, s=-1 for right one
			vel, sig = vsyst + s*pars[0 + p], pars[1 + p]
			a, b = [vel, sigma_diff]/sig
			w = np.linspace(0, np.pi*factor*sig, nl)
			losvd_rfft[:, j, k] = np.exp(1j*a*w - 0.5*(1 + b**2)*w**2)

			if mom > 2:
				n = np.arange(3, mom + 1)
				nrm = np.sqrt(special.factorial(n)*2**n)   # vdMF93 Normalization
				coeff = np.append([1, 0, 0], (s*1j)**n * pars[p - 1 + n]/nrm)
				poly = hermite.hermval(w, coeff)
				losvd_rfft[:, j, k] *= poly
		p += mom

	return np.conj(losvd_rfft)

##################################################################################

def nnls(A,b,npoly=0):
	"""
	Non-negative least squares.  
	A nobel prize shall be awarded to whomever makes this 
	way faster, because it is the choke point of the entire code.
	"""
	m, n = A.shape
	AA = np.hstack([A, -A[:, :npoly]])
	x = optimize.nnls(AA, b)[0]
	x[:npoly] -= x[n:]

	return np.array(x[:n])

####################################################################################

def run_emcee(pos,ndim,nwalkers,run_dir,lnprob_args,init_params,param_names,
			  auto_stop,conv_type,min_samp,ncor_times,autocorr_tol,write_iter,write_thresh,burn_in,min_iter,max_iter,threads,
			  print_output=True):
	"""
	Runs MCMC using emcee on all final parameters and checks for autocorrelation convergence 
	every write_iter iterations.
	"""
	# Keep original burn_in and max_iter to reset convergence if jumps out of convergence
	orig_burn_in  = burn_in
	orig_max_iter = max_iter
	# Sorted parameter names
	param_names = np.array(param_names)
	i_sort = np.argsort(param_names) # this array gives the ordered indices of parameter names (alphabetical)
	# Create MCMC_chain.csv if it doesn't exist
	if os.path.exists(run_dir+'log/MCMC_chain.csv')==False:
		f = open(run_dir+'log/MCMC_chain.csv','w')
		param_string = ', '.join(str(e) for e in param_names)
		f.write('# iter, ' + param_string) # Write initial parameters
		best_str = ', '.join(str(e) for e in init_params)
		f.write('\n 0, '+best_str)
		f.close()


	# initialize the sampler
	dtype = [('fluxes',dict)] # necessary change from Python2 -> Python3
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args,threads=threads,blobs_dtype=dtype) # blobs_dtype=dtype added for Python2 -> Python3

	start_time = time.time() # start timer

	write_log((ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter,threads),'emcee_options',run_dir)

	# Initialize stuff for autocorrelation analysis
	if (auto_stop==True):
		autocorr_times_all = [] # storage array for autocorrelation times
		autocorr_tols_all  = [] # storage array for autocorrelation tolerances
		old_tau = np.full(len(param_names),np.inf)
		min_samp	 = min_samp # minimum iterations to use past convergence
		ncor_times   = ncor_times # multiplicative tolerance; number of correlation times before which we stop sampling	
		autocorr_tol = autocorr_tol	
		stop_iter	= max_iter # stopping iteration; changes once convergence is reached
		converged  = False
		# write_log((min_samp,autocorr_tol,ncor_times,conv_type),'autocorr_options',run_dir)

	# If one provides a list of parameters for autocorrelation, it needs to be in the 
	# form of a tuple.  If one only provides one paraemeter, it needs to be converted to a tuple:
	if (auto_stop==True) and (conv_type != 'all') and (conv_type != 'mean') and (conv_type != 'median'):
		if not isinstance(conv_type, tuple):
			conv_type = (conv_type,) #

	# Check auto_stop convergence type:
	if (auto_stop==True) and (isinstance(conv_type,tuple)==True) :
		if all(elem in param_names  for elem in conv_type)==True:
			if (print_output):
				print('\n Only considering convergence of following parameters: ')
				for c in conv_type:	
					print('		  %s' % c)
				pass
		# check to see that all param_names are in conv_type, if not, remove them 
		# from conv_type
		else:
			try:
				conv_type_list = list(conv_type)
				for c in conv_type:
					if c not in param_names:
						conv_type_list.remove(c)
				conv_type = tuple(conv_type_list)
				if all(elem in conv_type  for elem in param_names)==True:
					if (print_output):
						print('\n Only considering convergence of following parameters: ')
						for c in conv_type:	
							print('		  %s' % c)
						pass
					else:
						if (print_output):
							print('\n One of more parameters in conv_type is not a valid parameter. Defaulting to median convergence type../.\n')
						conv_type='median'

			except:
				print('\n One of more parameters in conv_type is not a valid parameter. Defaulting to median convergence type../.\n')
				conv_type='median'

	if (auto_stop==True):
		write_log((min_samp,autocorr_tol,ncor_times,conv_type),'autocorr_options',run_dir)
	# Run emcee
	for k, result in enumerate(sampler.sample(pos, iterations=max_iter)):
		
		best = [] # For storing current chain positions (median of parameter values at write_iter iterations)
		if ((k+1) % write_iter == 0) and ((k+1)>=write_thresh): # Write every [write_iter] iteration
			# Chain location for each parameter
			# Median of last 100 positions for each walker.
			nwalkers = np.shape(sampler.chain)[0]
			npar = np.shape(sampler.chain)[2]
			
			sampler_chain = sampler.chain[:,:k+1,:]
			new_sampler_chain = []
			for i in range(0,np.shape(sampler_chain)[2],1):
				pflat = sampler_chain[:,:,i] # flattened along parameter
				flat  = np.concatenate(np.stack(pflat,axis=1),axis=0)
				new_sampler_chain.append(flat)
			# best = []
			for pp in range(0,npar,1):
				data = new_sampler_chain[pp][-int(nwalkers*write_iter):]
				med = np.median(data)
				best.append(med)
			# write to file
			f = open(run_dir+'log/MCMC_chain.csv','a')
			best_str = ', '.join(str(e) for e in best)
			f.write('\n'+str(k+1)+', '+best_str)
			f.close()
		# Checking autocorrelation times for convergence
		if ((k+1) % write_iter == 0) and ((k+1)>=min_iter) and ((k+1)>=write_thresh) and (auto_stop==True):
			# Autocorrelation analysis of chain to determine convergence; the minimum autocorrelation time is 1.0, which results when a time cannot be accurately calculated.
			tau = autocorr_convergence(sampler.chain,param_names,plot=False) # Calculate autocorrelation times for each parameter
			autocorr_times_all.append(tau) # append tau to storage array
			# Calculate tolerances
			tol = (np.abs(tau-old_tau)/old_tau) * 100.0
			autocorr_tols_all.append(tol) # append tol to storage array
			# If convergence for mean autocorrelation time 
			if (auto_stop==True) & (conv_type == 'mean'):
				par_conv = [] # converged parameter indices
				par_not_conv  = [] # non-converged parameter indices
				for x in range(0,len(param_names),1):
					if (round(tau[x],1)>1.0):# & (0.0<round(tol[x],1)<autocorr_tol):
						par_conv.append(x) # Append index of parameter for which an autocorrelation time can be calculated; we use these to calculate the mean
					else: par_not_conv.append(x)
				# Calculate mean of parameters for which an autocorrelation time could be calculated
				par_conv = np.array(par_conv) # Explicitly convert to array
				par_not_conv = np.array(par_not_conv) # Explicitly convert to array

				if (par_conv.size == 0) and (stop_iter == orig_max_iter):
					if print_output:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Not enough iterations for any autocorrelation times!')
				elif ( (par_conv.size > 0) and (k+1)>(np.mean(tau[par_conv]) * ncor_times) and (np.mean(tol[par_conv])<autocorr_tol) and (stop_iter == max_iter) ):
					if print_output:
						print('\n ---------------------------------------------')
						print(' | Converged at %d iterations.			  | ' % (k+1))
						print(' | Performing %d iterations of sampling... | ' % min_samp )
						print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
						print(' ---------------------------------------------')
					burn_in = (k+1)
					stop_iter = (k+1)+min_samp
					conv_tau = tau
					converged = True
				elif ((par_conv.size == 0) or ( (k+1)<(np.mean(tau[par_conv]) * ncor_times)) or (np.mean(tol[par_conv])>autocorr_tol)) and (stop_iter < orig_max_iter):
					if print_output:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Jumped out of convergence! Resetting convergence criteria...')
						# Reset convergence criteria
						print('- Resetting burn_in = %d' % orig_burn_in)
						print('- Resetting max_iter = %d' % orig_max_iter)
					burn_in = orig_burn_in
					stop_iter = orig_max_iter
					converged = False

				if (par_conv.size>0):
					pnames_sorted = param_names[i_sort]
					tau_sorted	= tau[i_sort]
					tol_sorted	= tol[i_sort]
					best_sorted   = np.array(best)[i_sort]
					if print_output:
						print('{0:<30}{1:<40}{2:<30}'.format('\nIteration = %d' % (k+1),'%d x Mean Autocorr. Time = %0.2f' % (ncor_times,np.mean(tau[par_conv]) * ncor_times),'Mean Tolerance = %0.2f' % np.mean(tol[par_conv])))
						print('--------------------------------------------------------------------------------------------------------')
						print('{0:<30}{1:<20}{2:<20}{3:<20}{4:<20}'.format('Parameter','Current Value','Autocorr. Time','Tolerance','Converged?'))
						print('--------------------------------------------------------------------------------------------------------')
						for i in range(0,len(pnames_sorted),1):
							if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol) and (tau_sorted[i]>1.0) ):
								conv_bool = 'True'
							else: conv_bool = 'False'
							if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
								print('{0:<30}{1:<20.4f}{2:<20.4f}{3:<20.4f}{4:<20}'.format(pnames_sorted[i],best_sorted[i],tau_sorted[i],tol_sorted[i],conv_bool))
							else: 
								print('{0:<30}{1:<20.4f}{2:<20}{3:<20}{4:<20}'.format(pnames_sorted[i],best_sorted[i],' -------- ',' -------- ',' -------- '))
						print('--------------------------------------------------------------------------------------------------------')

			# If convergence for median autocorrelation time 
			if (auto_stop==True) & (conv_type == 'median'):
				par_conv = [] # converged parameter indices
				par_not_conv  = [] # non-converged parameter indices
				for x in range(0,len(param_names),1):
					if (round(tau[x],1)>1.0):# & (tol[x]<autocorr_tol):
						par_conv.append(x) # Append index of parameter for which an autocorrelation time can be calculated; we use these to calculate the mean
					else: par_not_conv.append(x)
				# Calculate mean of parameters for which an autocorrelation time could be calculated
				par_conv = np.array(par_conv) # Explicitly convert to array
				par_not_conv = np.array(par_not_conv) # Explicitly convert to array

				if (par_conv.size == 0) and (stop_iter == orig_max_iter):
					if print_output:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Not enough iterations for any autocorrelation times!')
				elif ( (par_conv.size > 0) and (k+1)>(np.median(tau[par_conv]) * ncor_times) and (np.median(tol[par_conv])<autocorr_tol) and (stop_iter == max_iter) ):
					if print_output:
						print('\n ---------------------------------------------')
						print(' | Converged at %d iterations.			  |' % (k+1))
						print(' | Performing %d iterations of sampling... |' % min_samp )
						print(' | Sampling will finish at %d iterations.  |' % ((k+1)+min_samp) )
						print(' ---------------------------------------------')
					burn_in = (k+1)
					stop_iter = (k+1)+min_samp
					conv_tau = tau
					converged = True
				elif ((par_conv.size == 0) or ( (k+1)<(np.median(tau[par_conv]) * ncor_times)) or (np.median(tol[par_conv])>autocorr_tol)) and (stop_iter < orig_max_iter):
					if print_output:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Jumped out of convergence! Resetting convergence criteria...')
						# Reset convergence criteria
						print('- Resetting burn_in = %d' % orig_burn_in)
						print('- Resetting max_iter = %d' % orig_max_iter)
					burn_in = orig_burn_in
					stop_iter = orig_max_iter
					converged = False

				if (par_conv.size>0):
					pnames_sorted = param_names[i_sort]
					tau_sorted	= tau[i_sort]
					tol_sorted	= tol[i_sort]
					best_sorted   = np.array(best)[i_sort]
					if print_output:
						print('{0:<30}{1:<40}{2:<30}'.format('\nIteration = %d' % (k+1),'%d x Median Autocorr. Time = %0.2f' % (ncor_times,np.median(tau[par_conv]) * ncor_times),'Med. Tolerance = %0.2f' % np.median(tol[par_conv])))
						print('--------------------------------------------------------------------------------------------------------')
						print('{0:<30}{1:<20}{2:<20}{3:<20}{4:<20}'.format('Parameter','Current Value','Autocorr. Time','Tolerance','Converged?'))
						print('--------------------------------------------------------------------------------------------------------')
						for i in range(0,len(pnames_sorted),1):
							if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol) and (tau_sorted[i]>1.0)):
								conv_bool = 'True'
							else: conv_bool = 'False'
							if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):	
								print('{0:<30}{1:<20.4f}{2:<20.4f}{3:<20.4f}{4:<20}'.format(pnames_sorted[i],best_sorted[i],tau_sorted[i],tol_sorted[i],conv_bool))
							else: 
								print('{0:<30}{1:<20.4f}{2:<20}{3:<20}{4:<20}'.format(pnames_sorted[i],best_sorted[i],' -------- ',' -------- ',' -------- '))
						print('--------------------------------------------------------------------------------------------------------')
				
			# If convergence for ALL autocorrelation times 
			if (auto_stop==True) & (conv_type == 'all'):
				if ( all( (x==1.0) for x in tau) ) and (stop_iter == orig_max_iter):
					if print_output:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Not enough iterations for any autocorrelation times!')
				elif all( ((k+1)>(x * ncor_times)) for x in tau) and all( (x>1.0) for x in tau) and all(y<autocorr_tol for y in tol) and (stop_iter == max_iter):
					if print_output:
						print('\n ---------------------------------------------')
						print(' | Converged at %d iterations.			  | ' % (k+1))
						print(' | Performing %d iterations of sampling... | ' % min_samp )
						print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
						print(' ---------------------------------------------')
					burn_in = (k+1)
					stop_iter = (k+1)+min_samp
					conv_tau = tau
					converged = True
				elif (any( ((k+1)<(x * ncor_times)) for x in tau) or any( (x==1.0) for x in tau) or any(y>autocorr_tol for y in tol)) and (stop_iter < orig_max_iter):
					if print_output:
						print('\n Iteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Jumped out of convergence! Resetting convergence criteria...')
						# Reset convergence criteria
						print('- Resetting burn_in = %d' % orig_burn_in)
						print('- Resetting max_iter = %d' % orig_max_iter)
					burn_in = orig_burn_in
					stop_iter = orig_max_iter
					converged = False
				if 1:
					pnames_sorted = param_names[i_sort]
					tau_sorted	= tau[i_sort]
					tol_sorted	= tol[i_sort]
					best_sorted   = np.array(best)[i_sort]
					if print_output:
						print('{0:<30}'.format('\nIteration = %d' % (k+1)))
						print('--------------------------------------------------------------------------------------------------------------------------------------------')
						print('{0:<30}{1:<20}{2:<20}{3:<25}{4:<20}{5:<20}'.format('Parameter','Current Value','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
						print('--------------------------------------------------------------------------------------------------------------------------------------------')
						for i in range(0,len(pnames_sorted),1):
							if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol) and (tau_sorted[i]>1.0) ):
								conv_bool = 'True'
							else: conv_bool = 'False'
							if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
								print('{0:<30}{1:<20.4f}{2:<20.4f}{3:<25.4f}{4:<20.4f}{5:<20}'.format(pnames_sorted[i],best_sorted[i],tau_sorted[i],tau_sorted[i]*ncor_times,tol_sorted[i],str(conv_bool)))
							else: 
								print('{0:<30}{1:<20.4f}{2:<20}{3:<25}{4:<20}{5:<20}'.format(pnames_sorted[i],best_sorted[i],' -------- ',' -------- ',' -------- ',' -------- '))
						print('--------------------------------------------------------------------------------------------------------------------------------------------')

			# If convergence for a specific set of parameters
			if (auto_stop==True) & (isinstance(conv_type,tuple)==True):
				# Get indices of parameters for which we want to converge; these will be the only ones we care about
				par_ind = np.array([i for i, item in enumerate(param_names) if item in set(conv_type)])
				# Get list of parameters, autocorrelation times, and tolerances for the ones we care about
				param_interest   = param_names[par_ind]
				tau_interest = tau[par_ind]
				tol_interest = tol[par_ind]
				best_interest = np.array(best)[par_ind]
				# New sort for selected parameters
				i_sort = np.argsort(param_interest) # this array gives the ordered indices of parameter names (alphabetical)
				if ( all( (x==1.0) for x in tau_interest) ) and (stop_iter == orig_max_iter):
					if print_output:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Not enough iterations for any autocorrelation times!')
				elif all( ((k+1)>(x * ncor_times)) for x in tau_interest) and all( (x>1.0) for x in tau_interest) and all(y<autocorr_tol for y in tol_interest) and (stop_iter == max_iter):
					if print_output:
						print('\n ---------------------------------------------')
						print(' | Converged at %d iterations.			  | ' % (k+1))
						print(' | Performing %d iterations of sampling... | ' % min_samp )
						print(' | Sampling will finish at %d iterations.  | ' % ((k+1)+min_samp) )
						print(' ---------------------------------------------')
					burn_in = (k+1)
					stop_iter = (k+1)+min_samp
					conv_tau = tau
					converged = True
				elif (any( ((k+1)<(x * ncor_times)) for x in tau_interest) or any( (x==1.0) for x in tau_interest) or any(y>autocorr_tol for y in tol_interest)) and (stop_iter < orig_max_iter):
					if print_output:
						print('\n Iteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Jumped out of convergence! Resetting convergence criteria...')
						# Reset convergence criteria
						print('- Resetting burn_in = %d' % orig_burn_in)
						print('- Resetting max_iter = %d' % orig_max_iter)
					burn_in = orig_burn_in
					stop_iter = orig_max_iter
					converged = False
				if 1:
					pnames_sorted = param_interest[i_sort]
					tau_sorted	= tau_interest[i_sort]
					tol_sorted	= tol_interest[i_sort]
					best_sorted   = np.array(best_interest)[i_sort]
					if print_output:
						print('{0:<30}'.format('\nIteration = %d' % (k+1)))
						print('--------------------------------------------------------------------------------------------------------------------------------------------')
						print('{0:<30}{1:<20}{2:<20}{3:<25}{4:<20}{5:<20}'.format('Parameter','Current Value','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
						print('--------------------------------------------------------------------------------------------------------------------------------------------')
						for i in range(0,len(pnames_sorted),1):
							if (((k+1)>tau_sorted[i]*ncor_times) and (tol_sorted[i]<autocorr_tol) and (tau_sorted[i]>1.0) ):
								conv_bool = 'True'
							else: conv_bool = 'False'
							if (round(tau_sorted[i],1)>1.0):# & (tol[i]<autocorr_tol):
								print('{0:<30}{1:<20.4f}{2:<20.4f}{3:<25.4f}{4:<20.4f}{5:<20}'.format(pnames_sorted[i],best_sorted[i],tau_sorted[i],tau_sorted[i]*ncor_times,tol_sorted[i],str(conv_bool)))
							else: 
								print('{0:<30}{1:<20.4f}{2:<20}{3:<25}{4:<20}{5:<20}'.format(pnames_sorted[i],best_sorted[i],' -------- ',' -------- ',' -------- ',' -------- '))
						print('--------------------------------------------------------------------------------------------------------------------------------------------')

			# Stop
			if ((k+1) == stop_iter):
				break

			old_tau = tau	

		# If auto_stop=False, simply print out the parameters and their best values at that iteration
		if ((k+1) % write_iter == 0) and ((k+1)>=min_iter) and ((k+1)>=write_thresh) and (auto_stop==False):
			pnames_sorted = param_names[i_sort]
			best_sorted   = np.array(best)[i_sort]
			if print_output:
				print('{0:<30}'.format('\nIteration = %d' % (k+1)))
				print('------------------------------------------------')
				print('{0:<30}{1:<20}'.format('Parameter','Current Value'))
				print('------------------------------------------------')
				for i in range(0,len(pnames_sorted),1):
						print('{0:<30}{1:<20.4f}'.format(pnames_sorted[i],best_sorted[i]))
				print('------------------------------------------------')

	elap_time = (time.time() - start_time)	   
	run_time = time_convert(elap_time)
	if print_output:
		print("\n emcee Runtime = %s. \n" % (run_time))

	# Write to log file
	if (auto_stop==True):
		# Write autocorrelation chain to log 
		# np.save(run_dir+'/log/autocorr_times_all',autocorr_times_all)
		# np.save(run_dir+'/log/autocorr_tols_all',autocorr_tols_all)
		# Create a dictionary with parameter names as keys, and contains
		# the autocorrelation times and tolerances for each parameter
		autocorr_times_all = np.stack(autocorr_times_all,axis=1)
		autocorr_tols_all  = np.stack(autocorr_tols_all,axis=1)
		autocorr_dict = {}
		for k in range(0,len(param_names),1):
			if (np.shape(autocorr_times_all)[0] > 1):
				autocorr_dict[param_names[k]] = {'tau':autocorr_times_all[k],
											 	 'tol':autocorr_tols_all[k]} 
		np.save(run_dir+'/log/autocorr_dict.npy',autocorr_dict)


		if (converged == True):
			write_log((burn_in,stop_iter,param_names,conv_tau,autocorr_tol,tol,ncor_times),'autocorr_results',run_dir)
		elif (converged == False):
			unconv_tol = (np.abs((old_tau) - (tau)) / (tau))
			write_log((burn_in,stop_iter,param_names,tau,autocorr_tol,unconv_tol,ncor_times),'autocorr_results',run_dir)
	write_log(run_time,'emcee_time',run_dir) 

	# Remove excess zeros from sampler chain if emcee converged on a solution
	# in fewer iterations than max_iter
	# Remove zeros from all chains
	a = [] # the zero-trimmed sampler.chain
	for p in range(0,np.shape(sampler.chain)[2],1):
		c = sampler.chain[:,:,p]
		c_trimmed = [np.delete(c[i,:],np.argwhere(c[i,:]==0)) for i in range(np.shape(c)[0])] # delete any occurence of zero 
		a.append(c_trimmed)
	a = np.swapaxes(a,1,0) 
	a = np.swapaxes(a,2,1)

	# Collect garbage
	del lnprob_args
	if (auto_stop==True):
		del tau
		del tol
	gc.collect()

	return a, burn_in, sampler.blobs


##################################################################################

# Autocorrelation analysis 
##################################################################################

def autocorr_convergence(emcee_chain,param_names,plot=False):
	"""
	My own recipe for convergence.
	"""
	# Remove zeros from all chains
	sampler_chain = []
	for p in range(0,np.shape(emcee_chain)[2],1):
		c = emcee_chain[:,:,p]
		c_trimmed = [np.delete(c[i,:],np.argwhere(c[i,:]==0)) for i in range(np.shape(c)[0])] # delete any occurence of zero 
		sampler_chain.append(c_trimmed)
	sampler_chain = np.swapaxes(sampler_chain,1,0) 
	sampler_chain = np.swapaxes(sampler_chain,2,1)


		
	nwalker = np.shape(sampler_chain)[0] # Number of walkers
	niter   = np.shape(sampler_chain)[1] # Number of iterations
	npar	= np.shape(sampler_chain)[2] # Number of parameters
		
	def autocorr_func(c_x):
		""""""
		acf = []
		for p in range(0,np.shape(c_x)[1],1):
			x = c_x[:,p]
			# Subtract mean value
			rms_x = np.median(x)
			x = x - rms_x
			cc = np.correlate(x,x,mode='full')
			cc = cc[cc.size // 2:]
			cc = cc/np.max(cc)
			acf.append(cc)
		# Flip the array 
		acf = np.swapaxes(acf,1,0)
		return acf
			
	def auto_window(taus, c):
		"""
		(Adapted from https://github.com/dfm/emcee/blob/master/emcee/autocorr.py)
		"""
		m = np.arange(len(taus)) < c * taus
		if np.any(m):
			return np.argmin(m)
		return len(taus) - 1
	
	def integrated_time(acf, c=5, tol=0):
		"""Estimate the integrated autocorrelation time of a time series.
		This estimate uses the iterative procedure described on page 16 of
		`Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
		determine a reasonable window size.
		Args:
			acf: The time series. If multidimensional, set the time axis using the
				``axis`` keyword argument and the function will be computed for
				every other axis.
			c (Optional[float]): The step size for the window search. (default:
				``5``)
			tol (Optional[float]): The minimum number of autocorrelation times
				needed to trust the estimate. (default: ``0``)
		Returns:
			float or array: An estimate of the integrated autocorrelation time of
				the time series ``x`` computed along the axis ``axis``.
		(Adapted from https://github.com/dfm/emcee/blob/master/emcee/autocorr.py)
		"""
		tau_est = np.empty(np.shape(acf)[1])
		windows = np.empty(np.shape(acf)[1], dtype=int)

		# Loop over parameters
		for p in range(0,np.shape(acf)[1],1):
			taus = 2.0*np.cumsum(acf[:,p])-1.0
			windows[p] = auto_window(taus, c)
			tau_est[p] = taus[windows[p]]

		return tau_est

	c_x = np.mean(sampler_chain[:,:,:],axis=0)
	
	acf = autocorr_func(c_x)
	tau_est = integrated_time(acf)
		
	if (plot==True):
		fig = plt.figure(figsize=(14,4))
		ax1 = fig.add_subplot(2,1,1)
		ax2 = fig.add_subplot(2,1,2)
		for c in range(0,np.shape(c_x)[1],1):
			cn = (c_x[:,c])/(np.median(c_x[:,c]))
			ax1.plot(cn,alpha=1.,linewidth=0.5)
		ax1.axhline(1.0,alpha=1.,linewidth=0.5,color='black',linestyle='--')  
		ax1.set_xlim(0,np.shape(c_x)[0])
		ax2.plot(range(np.shape(acf)[0]),acf,alpha=1.,linewidth=0.5,label='ACF')
		ax2.axhline(0.0,alpha=1.,linewidth=0.5)
		ax2.set_xlim(np.min(range(np.shape(acf)[0])),np.max(range(np.shape(acf)[0])))
		plt.tight_layout()
	
	# Collect garbage
	del emcee_chain
	gc.collect()
		
	return tau_est

##################################################################################


# Plotting Routines
##################################################################################

def param_plots(param_dict,burn_in,run_dir,plot_param_hist=True,print_output=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	free parameters from MCMC sample chains.
	"""
	# Create a histograms sub-folder
	if (plot_param_hist==True):
		if (os.path.exists(run_dir + 'histogram_plots')==False):
			os.mkdir(run_dir + 'histogram_plots')
		os.mkdir(run_dir + 'histogram_plots/param_histograms')

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,8)) 
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0])
	ax2  = plt.subplot(gs[0,1])
	ax3  = plt.subplot(gs[1,0:2])

	for key in param_dict:
		ax1.clear()
		ax2.clear()
		ax3.clear()
		if print_output:
			print('		  %s' % key)
		chain = param_dict[key]['chain'] # shape = (nwalkers,niter)
		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])
			# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

		flat = chain[:,burn_in:]
		flat = flat.flat

		pname = param_dict[key]['label']
		pcol  = param_dict[key]['pcolor']

		# Histogram; 'Doane' binning produces the best results from tests.
		n, bins, patches = ax1.hist(flat, bins='doane', density=True, facecolor=pcol, alpha=0.75)

		# Old confidence interval stuff; replaced by np.quantile
		p = np.percentile(flat, [16, 50, 84])
		pdfmax = p[1]
		low1   = p[1]-p[0]
		upp1   = p[2]-p[1]
		# Store values in dictionary
		param_dict[key]['par_best']  = pdfmax # median (50th percentile)
		param_dict[key]['sig_low']   = low1 # -1-sigma
		param_dict[key]['sig_upp']   = upp1 # +1-sigma
		param_dict[key]['hist_bins'] = bins # bins used for histogram; used for corner plot
		# param_dict[key]['flat_samp'] = flat # flattened samples used for histogram.



		if (plot_param_hist==True):
			# Plot 1: Histogram plots
			ax1.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.3f$\n' % pdfmax)
			ax1.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.3f$\n' % low1)
			ax1.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.3f$\n' % upp1)
			# ax1.plot(xvec,yvec,color='white')
			ax1.set_xlabel(r'%s' % pname,fontsize=12)
			ax1.set_ylabel(r'$p$(%s)' % pname,fontsize=12)
	
			# Plot 2: best fit values
			ax2.axvline(pdfmax,linestyle='--',color='black',alpha=0.0,label='$\mu=%0.3f$\n' % pdfmax)
			ax2.axvline(pdfmax-low1,linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.3f$\n' % low1)
			ax2.axvline(pdfmax+upp1,linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.3f$\n' % upp1)
			ax2.legend(loc='center left',frameon=False,fontsize=14)
			ax2.axis('off')
	
			# Plot 3: Chain plot
			for w in range(0,np.shape(chain)[0],1):
				ax3.plot(range(np.shape(chain)[1]),chain[w,:],color='white',linewidth=0.5,alpha=0.5,zorder=0)
			# Calculate median and median absolute deviation of walkers at each iteration; we have depreciated
			# the average and standard deviation because they do not behave well for outlier walkers, which
			# also don't agree with histograms.
			c_med = np.median(chain,axis=0)
			c_madstd = mad_std(chain)
			ax3.plot(range(np.shape(chain)[1]),c_med,color='xkcd:red',alpha=1.,linewidth=2.0,label='Median',zorder=10)
			ax3.fill_between(range(np.shape(chain)[1]),c_med+c_madstd,c_med-c_madstd,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
			ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in = %d' % burn_in)
			ax3.set_xlim(0,np.shape(chain)[1])
			ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
			ax3.set_ylabel(r'%s' % pname,fontsize=12)
			ax3.legend(loc='upper left')
	
			# Save the figure
			figname = param_dict[key]['name']
			plt.savefig(run_dir+'histogram_plots/param_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=300,fmt='png')
			
	# Close plot window
	fig.clear()
	plt.close()
	# Collect garbage
	del fig
	del ax1
	del ax2
	del ax3
	del flat
	gc.collect()

	return param_dict


def flux_plots(flux_blob, burn_in, nwalkers, run_dir, plot_flux_hist=True,print_output=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	component fluxes from MCMC sample chains.
	"""
	# Create a histograms sub-folder
	if (plot_flux_hist==True):
		if (os.path.exists(run_dir + 'histogram_plots')==False): 
			os.mkdir(run_dir + 'histogram_plots')
		os.mkdir(run_dir + 'histogram_plots/flux_histograms')
	
	# Create a flux dictionary
	niter	= np.shape(flux_blob)[0]
	nwalkers = np.shape(flux_blob)[1]
	flux_dict = {}
	for key in flux_blob[0][0][0]:
		flux_dict[key] = {'chain':np.empty([nwalkers,niter])}

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,8)) 
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0])
	ax2  = plt.subplot(gs[0,1])
	ax3  = plt.subplot(gs[1,0:2])

	# Restructure the flux_blob for the flux_dict
	for i in range(niter):
		for j in range(nwalkers):
			for key in flux_blob[0][0][0]:
				flux_dict[key]['chain'][j,i] = flux_blob[i][j][0][key]

	for key in flux_dict:
		ax1.clear()
		ax2.clear()
		ax3.clear()
		if print_output:
			print('		  %s' % key)
		chain = flux_dict[key]['chain'] # shape = (nwalkers,niter)

		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])

		# Remove burn_in iterations and flatten for histogram
		flat = chain[:,burn_in:]
		flat = flat.flat

		# Histogram
		n, bins, patches = ax1.hist(flat, bins='doane', density=True, alpha=0.75)

		p = np.percentile(flat, [16, 50, 84])
		pdfmax = p[1]
		low1   = p[1]-p[0]
		upp1   = p[2]-p[1]

		# Store values in dictionary
		flux_dict[key]['par_best']  = pdfmax # median (50th percentile)
		flux_dict[key]['sig_low']   = low1 # -1-sigma
		flux_dict[key]['sig_upp']   = upp1 # +1-sigma
		flux_dict[key]['hist_bins'] = bins # bins used for histogram; used for corner plot
		flux_dict[key]['flat_samp'] = flat # flattened samples used for histogram.

		if (plot_flux_hist==True):
			# Plot 1: Histogram plots
			ax1.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.3f$\n' % pdfmax)
			ax1.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.3f$\n' % low1)
			ax1.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.3f$\n' % upp1)
			# ax1.plot(xvec,yvec,color='white')
			ax1.set_xlabel(r'%s' % key,fontsize=8)
			ax1.set_ylabel(r'$p$(%s)' % key,fontsize=8)

			# Plot 2: best fit values
			ax2.axvline(pdfmax,linestyle='--',color='black',alpha=0.0,label='$\mu=%0.3f$\n' % pdfmax)
			ax2.axvline(pdfmax-low1,linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.3f$\n' % low1)
			ax2.axvline(pdfmax+upp1,linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.3f$\n' % upp1)
			ax2.legend(loc='center left',frameon=False,fontsize=14)
			ax2.axis('off')

			# Plot 3: Chain plot
			for i in range(0,np.shape(chain)[0],1):
				ax3.plot(range(np.shape(chain)[1]),chain[i],color='white',linewidth=0.5,alpha=0.5,zorder=0)
			c_med = np.median(chain,axis=0)
			c_madstd = mad_std(chain)
			ax3.plot(range(np.shape(chain)[1]),c_med,color='xkcd:red',alpha=1.,linewidth=2.0,label='Median',zorder=10)
			ax3.fill_between(range(np.shape(chain)[1]),c_med+c_madstd,c_med-c_madstd,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
			ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in = %d' % burn_in)
			ax3.set_xlim(0,np.shape(chain)[1])
			ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
			ax3.set_ylabel(r'%s' % key,fontsize=8)
			ax3.legend(loc='upper left')

			# Save the figure
			figname = key
			plt.savefig(run_dir+'histogram_plots/flux_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=150,fmt='png')
		
	# Close plot
	fig.clear()
	plt.close()
	# Collect garbage
	del fig
	del ax1
	del ax2
	del ax3
	del flat
	del flux_blob
	gc.collect()

	return flux_dict

def lum_plots(flux_dict,burn_in,nwalkers,z,run_dir,H0=71.0,Om0=0.27,plot_lum_hist=True,print_output=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	component luminosities from MCMC sample chains.
	"""
	# Create a histograms sub-folder
	if (plot_lum_hist==True):
		if (os.path.exists(run_dir + 'histogram_plots')==False): 
			os.mkdir(run_dir + 'histogram_plots')
		os.mkdir(run_dir + 'histogram_plots/lum_histograms')
	
	# Create a flux dictionary
	lum_dict = {}
	for key in flux_dict:
		flux = (flux_dict[key]['chain']) * 1.0E-17
		# Compute luminosity distance (in cm) using FlatLambdaCDM cosmology
		cosmo = FlatLambdaCDM(H0, Om0)
		d_mpc = cosmo.luminosity_distance(z).value
		d_cm  = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
		# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
		lum   = (flux * 4*np.pi * d_cm**2	) / 1.0E+42
		lum_dict[key[:-4]+'lum']= {'chain':lum}

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,8)) 
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0])
	ax2  = plt.subplot(gs[0,1])
	ax3  = plt.subplot(gs[1,0:2])

	for key in lum_dict:
		ax1.clear()
		ax2.clear()
		ax3.clear()
		if print_output:
			print('		  %s' % key)
		chain = lum_dict[key]['chain'] # shape = (nwalkers,niter)

		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])
			# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

		# Remove burn_in iterations and flatten for histogram
		flat = chain[:,burn_in:]
		flat = flat.flat

		# Histogram
		n, bins, patches = ax1.hist(flat, bins='doane', density=True, alpha=0.75)

		p = np.percentile(flat, [16, 50, 84])
		pdfmax = p[1]
		low1   = p[1]-p[0]
		upp1   = p[2]-p[1]
		# Store values in dictionary
		lum_dict[key]['par_best']  = pdfmax # median (50th percentile)
		lum_dict[key]['sig_low']   = low1 # -1-sigma
		lum_dict[key]['sig_upp']   = upp1 # +1-sigma
		lum_dict[key]['hist_bins'] = bins # bins used for histogram; used for corner plot
		lum_dict[key]['flat_samp'] = flat # flattened samples used for histogram.

		if (plot_lum_hist==True):
			# Plot 1: Histogram plots
			ax1.axvline(pdfmax,linestyle='--',color='white',label='$\mu=%0.3f$\n' % pdfmax)
			ax1.axvline(pdfmax-low1,linestyle=':',color='white',label='$\sigma_-=%0.3f$\n' % low1)
			ax1.axvline(pdfmax+upp1,linestyle=':',color='white',label='$\sigma_+=%0.3f$\n' % upp1)
			# ax1.plot(xvec,yvec,color='white')
			ax1.set_xlabel(r'%s' % key,fontsize=8)
			ax1.set_ylabel(r'$p$(%s)' % key,fontsize=8)

			# Plot 2: best fit values
			ax2.axvline(pdfmax,linestyle='--',color='black',alpha=0.0,label='$\mu=%0.3f$\n' % pdfmax)
			ax2.axvline(pdfmax-low1,linestyle=':',color='black',alpha=0.0,label='$\sigma\_=%0.3f$\n' % low1)
			ax2.axvline(pdfmax+upp1,linestyle=':',color='black',alpha=0.0,label='$\sigma_{+}=%0.3f$\n' % upp1)
			ax2.legend(loc='center left',frameon=False,fontsize=14)
			ax2.axis('off')

			# Plot 3: Chain plot
			for i in range(0,np.shape(chain)[0],1):
				ax3.plot(range(np.shape(chain)[1]),chain[i],color='white',linewidth=0.5,alpha=0.5,zorder=0)
			c_med = np.median(chain,axis=0)
			c_madstd = mad_std(chain)
			ax3.plot(range(np.shape(chain)[1]),c_med,color='xkcd:red',alpha=1.,linewidth=2.0,label='Median',zorder=10)
			ax3.fill_between(range(np.shape(chain)[1]),c_med+c_madstd,c_med-c_madstd,color='xkcd:aqua',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
			ax3.axvline(burn_in,linestyle='--',color='xkcd:orange',label='burn-in = %d' % burn_in)
			ax3.set_xlim(0,np.shape(chain)[1])
			ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
			ax3.set_ylabel(r'%s' % key,fontsize=8)
			ax3.legend(loc='upper left')

			# Save the figure
			figname = key
			plt.savefig(run_dir+'histogram_plots/lum_histograms/'+'%s_MCMC.png' % (figname) ,bbox_inches="tight",dpi=150,fmt='png')
		
	# Close plot	
	fig.clear()
	plt.close()
	# Collect garbage
	del fig
	del ax1
	del ax2
	del ax3
	del flat
	del flux
	del flux_dict
	del lum
	del chain
	gc.collect()

	return lum_dict
		

def write_param(param_dict,flux_dict,lum_dict,extra_dict,bounds,run_dir):
	"""
	Writes all measured parameters, fluxes, luminosities, and extra stuff 
	(black hole mass, systemic redshifts) and all flags to a FITS table.
	"""
	# Extract elements from dictionaries
	par_names = []
	par_best  = []
	sig_low   = []
	sig_upp   = []
	flags = []
	# Param dict
	for key in param_dict:
		param_dict[key]['flag'] = 0  
		if (param_dict[key]['par_best']-param_dict[key]['sig_low'] <= param_dict[key]['plim'][0]):
			param_dict[key]['flag']+=1
		if (param_dict[key]['par_best']+param_dict[key]['sig_upp'] >= param_dict[key]['plim'][1]):
			param_dict[key]['flag']+=1
		if 'min_width' in param_dict[key]:
			if (param_dict[key]['par_best']-param_dict[key]['sig_low']  <= param_dict[key]['min_width']):
				param_dict[key]['flag']+=1
		par_names.append(key)
		par_best.append(param_dict[key]['par_best'])
		sig_low.append(param_dict[key]['sig_low'])
		sig_upp.append(param_dict[key]['sig_upp'])
		flags.append(param_dict[key]['flag'])
	# Flux dict
	for key in flux_dict:
		flux_dict[key]['flag'] = 0 
		if (flux_dict[key]['par_best']-flux_dict[key]['sig_low'] <= 0.0):
			flux_dict[key]['flag']+=1
		par_names.append(key)
		par_best.append(flux_dict[key]['par_best'])
		sig_low.append(flux_dict[key]['sig_low'])
		sig_upp.append(flux_dict[key]['sig_upp'])   
		flags.append(flux_dict[key]['flag']) 
	# Luminosity dict
	for key in lum_dict:
		lum_dict[key]['flag'] = 0 
		if (lum_dict[key]['par_best']-lum_dict[key]['sig_low'] <= 0.0):
			lum_dict[key]['flag']+=1
		par_names.append(key)
		par_best.append(lum_dict[key]['par_best'])
		sig_low.append(lum_dict[key]['sig_low'])
		sig_upp.append(lum_dict[key]['sig_upp']) 
		flags.append(lum_dict[key]['flag']) 
	# Extra dict
	if extra_dict:
		for key in extra_dict:
			par_names.append(key)
			par_best.append(extra_dict[key]['par_best'])
			sig_low.append(extra_dict[key]['sig_low'])
			sig_upp.append(extra_dict[key]['sig_upp']) 
			flags.append(extra_dict[key]['flag'])
	# Sort param_names alphabetically
	i_sort	= np.argsort(par_names)
	par_names = np.array(par_names)[i_sort] 
	par_best  = np.array(par_best)[i_sort]  
	sig_low   = np.array(sig_low)[i_sort]   
	sig_upp   = np.array(sig_upp)[i_sort]   
	flags	 = np.array(flags)[i_sort]	 

	# Write best-fit parameters to FITS table
	col1 = fits.Column(name='parameter', format='30A', array=par_names)
	col2 = fits.Column(name='best_fit', format='E', array=par_best)
	col3 = fits.Column(name='sigma_low', format='E', array=sig_low)
	col4 = fits.Column(name='sigma_upp', format='E', array=sig_upp)
	col5 = fits.Column(name='flag', format='E', array=flags)
	cols = fits.ColDefs([col1,col2,col3,col4,col5])
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'log/par_table.fits',overwrite=True)
	del hdu
	# Write full param dict to log file
	write_log((par_names,par_best,sig_low,sig_upp,flags),'emcee_results',run_dir)
	# Collect garbage
	del param_dict
	del flux_dict
	del lum_dict
	del flags
	gc.collect()
	return None

def write_chains(param_dict,flux_dict,lum_dict,run_dir):
	"""
	Writes all MCMC chains to a FITS table.
	"""
	# Save parameter dict as a npy file
	# np.save(run_dir + '/log/param_dict.npy',param_dict)

	cols = []
	# Construct a column for each parameter and chain
	for key in param_dict:
		cols.append(fits.Column(name=key, format='E',array=param_dict[key]['chain'].flat))
	for key in flux_dict:
		cols.append(fits.Column(name=key, format='E', array=flux_dict[key]['chain'].flat))
	for key in lum_dict:
		cols.append(fits.Column(name=key, format='E', array=lum_dict[key]['chain'].flat))
	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'log/MCMC_chains.fits',overwrite=True)
	del hdu
	# Collect garbage
	del param_dict
	del flux_dict
	del lum_dict
	gc.collect()
	return None
	
def plot_best_model(param_dict,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
						   temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir):
	"""
	Plots the best fig model and outputs the components to a FITS file for reproduction.
	"""
	param_names  = [param_dict[key]['name'] for key in param_dict ]
	par_best	   = [param_dict[key]['par_best'] for key in param_dict ]


	output_model = True
	fit_type	 = 'final'
	comp_dict = fit_model(par_best,param_names,lam_gal,galaxy,noise,gal_temp,feii_tab,feii_options,
			  temp_list,temp_fft,npad,line_profile,fwhm_gal,velscale,npix,vsyst,run_dir,
			  fit_type,output_model)

	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,6)) 
	gs = gridspec.GridSpec(4, 1)
	gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0:3,0])
	ax2  = plt.subplot(gs[3,0])
	
	for key in comp_dict:
		# Galaxy + Best-fit Model
		if (key is not 'resid') and (key is not 'noise') and (key is not 'wave') and (key is not 'data'):
			ax1.plot(lam_gal,comp_dict[key]['comp'],linewidth=comp_dict[key]['linewidth'],color=comp_dict[key]['pcolor'],label=key,zorder=15)
		if (key not in ['resid','noise','wave','data','model','na_feii_template','br_feii_template','host_galaxy','power']):
			ax1.axvline(lam_gal[np.where(comp_dict[key]['comp']==np.max(comp_dict[key]['comp']))[0][0]],color='xkcd:white',linestyle='--',linewidth=0.5)
			ax2.axvline(lam_gal[np.where(comp_dict[key]['comp']==np.max(comp_dict[key]['comp']))[0][0]],color='xkcd:white',linestyle='--',linewidth=0.5)
		if (key=='power'):
			ax1.plot(lam_gal,comp_dict[key]['comp'],linewidth=comp_dict[key]['linewidth'],color=comp_dict[key]['pcolor'],linestyle='--',label=key,zorder=15)

	ax1.plot(lam_gal,comp_dict['data']['comp'],linewidth=0.5,color='white',label='data',zorder=0)

	ax1.set_xticklabels([])
	sigma_resid = np.std(comp_dict['data']['comp']-comp_dict['model']['comp'])
	sigma_noise = np.median(comp_dict['noise']['comp'])
	ax1.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
	ax1.set_ylim(0-3.0*sigma_resid,np.max(comp_dict['data']['comp'])+3.0*sigma_resid)
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=12)
	# ax1.legend(loc='best',fontsize=8)
	# Residuals plot
	ax2.plot(lam_gal,(comp_dict['noise']['comp'])*3,linewidth=comp_dict['noise']['linewidth'],color=comp_dict['noise']['pcolor'],label='$\sigma_{\mathrm{noise}}=%0.4f$' % (sigma_noise))
	ax2.plot(lam_gal,(comp_dict['resid']['comp'])*3,linewidth=comp_dict['resid']['linewidth'],color=comp_dict['resid']['pcolor'],label='$\sigma_{\mathrm{resid}}=%0.4f$' % (sigma_resid))
	ax2.axhline(0.0,linewidth=1.0,color='black',linestyle='--')
	ax2.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
	ax2.set_ylim(ax1.get_ylim())
	ax2.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
	ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
	ax2.set_yticks([0.0])
	ax2.legend(loc='upper right',fontsize=8)
	plt.savefig(run_dir+'bestfit_model.pdf',dpi=150,fmt='png')

	cols = []
	# Construct a column for each parameter and chain
	for key in comp_dict:
		cols.append(fits.Column(name=key, format='E', array=comp_dict[key]['comp']))

	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'log/best_model_components.fits',overwrite=True)
	del hdu
	# Close plot
	fig.clear()
	plt.close()
	# Collect garbage
	del fig
	del ax1
	del ax2
	del param_dict
	del lam_gal
	del galaxy
	del noise
	del gal_temp
	del feii_tab
	del temp_list
	del temp_fft
	gc.collect()

	return None

def write_max_like_results(result_dict,comp_dict,run_dir):
	"""
	Write maximum likelihood fit results to FITS table
	if MCMC is not performed. 
	"""
	# Extract elements from dictionaries
	par_names = []
	par_best  = []
	sig	   = []
	for key in result_dict:
		par_names.append(key)
		par_best.append(result_dict[key]['med'])
		sig.append(result_dict[key]['std'])
	if 0: 
		for i in range(0,len(par_names),1):
			print(par_names[i],par_best[i],sig[i])
	# Write best-fit parameters to FITS table
	col1 = fits.Column(name='parameter', format='30A', array=par_names)
	col2 = fits.Column(name='best_fit' , format='E'  , array=par_best)
	col3 = fits.Column(name='sigma'	, format='E'  , array=sig)
	cols = fits.ColDefs([col1,col2,col3])
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'log/par_table.fits',overwrite=True)
	del hdu
	# Write best-fit components to FITS file
	cols = []
	# Construct a column for each parameter and chain
	for key in comp_dict:
		cols.append(fits.Column(name=key, format='E', array=comp_dict[key]['comp']))
	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir+'log/best_model_components.fits',overwrite=True)
	del hdu
	# Collect garbage
	del result_dict
	del comp_dict
	del par_names
	del par_best
	del sig
	del cols
	gc.collect()

	return None

# Clean-up Routine
##################################################################################

def cleanup(run_dir):
	"""
	Cleans up the run directory.
	"""
	# Remove param_plots folder if empty
	if os.path.exists(run_dir + 'histogram_plots') and not os.listdir(run_dir + 'histogram_plots'):
		shutil.rmtree(run_dir + 'histogram_plots')
	# If sdss_prepare.png is still there, get rid of it
	if os.path.exists(run_dir + 'sdss_prepare.png'):
		os.remove(run_dir + 'sdss_prepare.png')
	# If run_dir is empty because there aren't enough good pixels, remove it
	if not os.listdir(run_dir):
		shutil.rmtree(run_dir)
	gc.collect()

	return None

##################################################################################

def write_log(output_val,output_type,run_dir):
	"""
	This function writes values to a log file as the code runs.
	"""
	# Check if log folder has been created, if not, create it
	if os.path.exists(run_dir+'/log/')==False:
		os.mkdir(run_dir+'/log/')
		# Create log file 
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n############################### BADASS v7.7.4 LOGFILE ####################################\n')
		logfile.close()


	# sdss_prepare
	# output_val=(file,ra,dec,z,fit_min,fit_max,velscale,ebv), output_type=0
	if (output_type==0):
		file,ra,dec,z,fit_min,fit_max,velscale,ebv = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n-----------------------------------------------------------')
		logfile.write('\n{0:<25}{1:<30}'.format('file:'		   , file.split('/')[-1]			))
		logfile.write('\n{0:<25}{1:<30}'.format('(RA, DEC):'	  , '(%0.6f,%0.6f)' % (ra,dec)	 ))
		logfile.write('\n{0:<25}{1:<30}'.format('SDSS redshift:'  , '%0.5f' % z					))
		logfile.write('\n{0:<25}{1:<30}'.format('fitting region:' , '(%d,%d)' % (fit_min,fit_max)  ))
		logfile.write('\n{0:<25}{1:<30}'.format('velocity scale:' , '%0.2f (km/s/pixel)' % velscale))
		logfile.write('\n{0:<25}{1:<30}'.format('galactic E(B-V):', '%0.3f' % ebv))
		logfile.write('\n-----------------------------------------------------------')
		logfile.close()

	if (output_type=='outflow_test'):
		(rdict,sigma,total_resid_noise,total_resid_noise_err,amp_metric,fwhm_metric,voff_metric,ssr_ratio,ssr_ratio_err,
		ssr_no_outflow, ssr_no_outflow_err, ssr_outflow, ssr_outflow_err,
		f_stat, f_stat_err, f_pval, f_pval_err, outflow_conf, outflow_conf_err,
		resid_std_no_outflow,resid_dstd_no_outflow,resid_std_outflow,resid_dstd_outflow) = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Outflow Model Fitting Results ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Best-fit Value','+/- 1-sigma','Flag'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		# Sort into arrays
		pname = []
		med   = []
		std   = []
		flag  = [] 
		for key in rdict:
			pname.append(key)
			med.append(rdict[key]['med'])
			std.append(rdict[key]['std'])
			flag.append(rdict[key]['flag'])
		i_sort = np.argsort(pname)
		pname = np.array(pname)[i_sort] 
		med   = np.array(med)[i_sort]   
		std   = np.array(std)[i_sort]   
		flag  = np.array(flag)[i_sort]  
		for i in range(0,len(pname),1):
			logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}{3:<30}'.format(pname[i], med[i], std[i], flag[i]))
		logfile.write('\n{0:<30}{1:<30.4f}'.format('median noise',sigma))
		logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4e}'.format('total resid. noise',total_resid_noise,total_resid_noise_err))
		logfile.write('\n{0:<30}{1:<30.4f}'.format('amp metric',amp_metric))
		logfile.write('\n{0:<30}{1:<30.4f}'.format('fwhm metric',fwhm_metric))
		logfile.write('\n{0:<30}{1:<30.4f}'.format('voff metric',voff_metric))

		logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}'.format('SSR ratio',ssr_ratio,ssr_ratio_err))
		logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}'.format('SSR no outflow',ssr_no_outflow,ssr_no_outflow_err))
		logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}'.format('SSR outflow',ssr_outflow,ssr_outflow_err))

		logfile.write('\n{0:<30}{1:<25.4f}{2:<25.4f}'.format('f-statistic',f_stat,f_stat_err))
		logfile.write('\n{0:<30}{1:<25.4e}{2:<25.4e}'.format('p-value',f_pval,f_pval_err))
		logfile.write('\n{0:<30}{1:<25.6f}{2:<25.6f}'.format('Outflow confidence',outflow_conf, outflow_conf_err ) )


		logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}'.format('no outflow resid. std.',resid_std_no_outflow,resid_dstd_no_outflow))
		logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}'.format('outflow resid. std.',resid_std_outflow,resid_dstd_outflow))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()

	if (output_type=='no_outflow_test'):
		rdict = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### No-Outflow Model Fitting Results ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Best-fit Value','+/- 1-sigma','Flag'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		# Sort into arrays
		pname = []
		med   = []
		std   = []
		flag  = [] 
		for key in rdict:
			pname.append(key)
			med.append(rdict[key]['med'])
			std.append(rdict[key]['std'])
			flag.append(rdict[key]['flag'])
		i_sort = np.argsort(pname)
		pname = np.array(pname)[i_sort] 
		med   = np.array(med)[i_sort]   
		std   = np.array(std)[i_sort]   
		flag  = np.array(flag)[i_sort]  
		for i in range(0,len(pname),1):
			logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}{3:<30}'.format(pname[i], med[i], std[i], flag[i]))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()

	# Results of outflow_test
	# write_log((cond1,cond2,cond3),20)
	if (output_type=='outflow_test_pass'):
		outflow_test_options = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Tests for Outflows ###')
		logfile.write('\n---------------------------------------------------------------------')
		if (outflow_test_options['amp_test']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Outflow amplitude condition:'   , str(bool(outflow_test_options['amp_test']['pass'])) ))
		if (outflow_test_options['fwhm_test']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Outflow FWHM condition:'	     , str(bool(outflow_test_options['fwhm_test']['pass'])) ))
		if (outflow_test_options['voff_test']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Outflow VOFF condition:'	     , str(bool(outflow_test_options['voff_test']['pass'])) ))
		if (outflow_test_options['outflow_confidence']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Outflow Confidence condition:'	     , str(bool(outflow_test_options['outflow_confidence']['pass'])) ))
		if (outflow_test_options['bounds_test']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Parameter bounds condition:'	 , str(bool(outflow_test_options['bounds_test']['pass'])) ))
		logfile.write('\n---------------------------------------------------------------------')
		logfile.write('\n	----> Setting fit_outflows=True')
		logfile.close()
	elif (output_type=='outflow_test_fail'):
		outflow_test_options = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Tests for Outflows ###')
		logfile.write('\n---------------------------------------------------------------------')
		if (outflow_test_options['amp_test']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Outflow amplitude condition:'   , str(bool(outflow_test_options['amp_test']['pass'])) ))
		if (outflow_test_options['fwhm_test']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Outflow FWHM condition:'	     , str(bool(outflow_test_options['fwhm_test']['pass'])) ))
		if (outflow_test_options['voff_test']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Outflow VOFF condition:'	     , str(bool(outflow_test_options['voff_test']['pass'])) ))
		if (outflow_test_options['outflow_confidence']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Outflow Confidence condition:'	     , str(bool(outflow_test_options['outflow_confidence']['pass'])) ))
		if (outflow_test_options['bounds_test']['test']==True):
			logfile.write('\n{0:<35}{1:<30}'.format('Parameter bounds condition:'	 , str(bool(outflow_test_options['bounds_test']['pass'])) ))
		logfile.write('\n---------------------------------------------------------------------')
		logfile.write('\n	----> Setting fit_outflows=False')
		logfile.close()
	# Maximum likelihood/Initial parameters
	if (output_type=='max_like_fit'):
		pdict,sn,noise_std,resid_std = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Parameters to-be-fit and their Initial Values ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Max. Like. Value','+/- 1-sigma', 'Flag') )
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		# Sort into arrays
		pname = []
		med   = []
		std   = []
		flag  = [] 
		for key in pdict:
			pname.append(key)
			med.append(pdict[key]['med'])
			std.append(pdict[key]['std'])
			flag.append(pdict[key]['flag'])
		i_sort = np.argsort(pname)
		pname = np.array(pname)[i_sort] 
		med   = np.array(med)[i_sort]   
		std   = np.array(std)[i_sort]   
		flag  = np.array(flag)[i_sort]  
		for i in range(0,len(pname),1):
			logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}{3:<30}'.format(pname[i], med[i], std[i], flag[i]))
		logfile.write('\n{0:<30}{1:<30.4f}'.format('cont. S/N',sn ))
		logfile.write('\n{0:<30}{1:<30.4f}'.format('Noise Std. Dev.', noise_std ))
		logfile.write('\n{0:<30}{1:<30.4f}'.format('Resid Std. Dev.', resid_std ))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()

	# run_emcee
	if (output_type=='emcee_options'): # write user input emcee options
		ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter,threads = output_val
		# write_log((ndim,nwalkers,auto_stop,burn_in,write_iter,write_thresh,min_iter,max_iter,threads),40)
		a = str(datetime.datetime.now())
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Emcee Options ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}'.format('ndim'		, ndim ))
		logfile.write('\n{0:<30}{1:<30}'.format('nwalkers'	, nwalkers ))
		logfile.write('\n{0:<30}{1:<30}'.format('auto_stop'   , str(auto_stop) ))
		logfile.write('\n{0:<30}{1:<30}'.format('user burn_in', burn_in ))
		logfile.write('\n{0:<30}{1:<30}'.format('write_iter'  , write_iter ))
		logfile.write('\n{0:<30}{1:<30}'.format('write_thresh', write_thresh ))
		logfile.write('\n{0:<30}{1:<30}'.format('min_iter'	, min_iter ))
		logfile.write('\n{0:<30}{1:<30}'.format('max_iter'	, max_iter ))
		logfile.write('\n{0:<30}{1:<30}'.format('threads'	 , threads ))
		logfile.write('\n{0:<30}{1:<30}'.format('start_time'  , a ))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()
	if (output_type=='autocorr_options'): # write user input auto_stop options
		min_samp,autocorr_tol,ncor_times,conv_type = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
		logfile.write('\n')
		logfile.write('\n### Autocorrelation Options ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}'.format('min_samp'  , min_samp	 ))
		logfile.write('\n{0:<30}{1:<30}'.format('tolerance%', autocorr_tol ))
		logfile.write('\n{0:<30}{1:<30}'.format('ncor_times', ncor_times   ))
		logfile.write('\n{0:<30}{1:<30}'.format('conv_type' , conv_type	))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()
	if (output_type=='autocorr_results'): # write autocorrelation results to log
		# write_log((k+1,burn_in,stop_iter,param_names,tau),42,run_dir)
		burn_in,stop_iter,param_names,tau,autocorr_tol,tol,ncor_times = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
		i_sort = np.argsort(param_names)
		param_names = np.array(param_names)[i_sort]
		tau = np.array(tau)[i_sort]
		tol = np.array(tol)[i_sort]
		logfile.write('\n')
		logfile.write('\n### Autocorrelation Results ###')
		logfile.write('\n----------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}'.format('conv iteration', burn_in   ))
		logfile.write('\n{0:<30}{1:<30}'.format('stop iteration', stop_iter ))
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format('Parameter','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		for i in range(0,len(param_names),1):
			if (burn_in > (tau[i]*ncor_times)) and (0 < tol[i] < autocorr_tol):
				c = 'True'
			elif (burn_in < (tau[i]*ncor_times)) or (tol[i]>= 0.0):
				c = 'False'
			else: 
				c = 'False'
			logfile.write('\n{0:<30}{1:<30.5f}{2:<30.5f}{3:<30.5f}{4:<30}'.format(param_names[i],tau[i],(tau[i]*ncor_times),tol[i],c))
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		logfile.close()
	if (output_type=='emcee_time'): # write autocorrelation results to log
		# write_log(run_time,43,run_dir)
		run_time = output_val
		a = str(datetime.datetime.now())
		logfile = open(run_dir+'log/log_file.txt','a')
		# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
		logfile.write('\n{0:<30}{1:<30}'.format('emcee_runtime',run_time ))
		logfile.write('\n{0:<30}{1:<30}'.format('end_time',  a ))
		logfile.write('\n----------------------------------------------------')
		logfile.close()
	if (output_type=='emcee_results'): # write best fit parameters results to log
		par_names,par_best,sig_low,sig_upp,flags = output_val 
		# write_log((par_names,par_best,sig_low,sig_upp),50,run_dir)
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Best-fit Parameters & Uncertainties ###')
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<25}{2:<25}{3:<25}{4:<25}'.format('Parameter','Best-fit Value','- 1-sigma','+ 1-sigma','Flag'))
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		for par in range(0,len(par_names),1):
			logfile.write('\n{0:<30}{1:<25.4f}{2:<25.4f}{3:<25.4f}{4:<25}'.format(par_names[par],par_best[par],sig_low[par],sig_upp[par],flags[par]))
		logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------')
		logfile.close()

	# BH mass estimates 
	if (output_type=='mbh_Hb'):
		L5100_Hb, MBH_Hb = output_val 
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### H-beta AGN Luminosity & Black Hole Estimate ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format(' ','Best estimate','-sigma (dex)','+sigma (dex)'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30.3f}{2:<30.3f}{3:<30.3f}'.format('log10(L5100)',L5100_Hb[0],L5100_Hb[1],L5100_Hb[2]))
		logfile.write('\n{0:<30}{1:<30.3f}{2:<30.3f}{3:<30.3f}'.format('log10(M_BH)',MBH_Hb[0],MBH_Hb[1],MBH_Hb[2]))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()
	if (output_type=='mbh_Ha'):
		L5100_Ha, MBH_Ha = output_val 
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### H-alpha AGN Luminosity & Black Hole Estimate ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format(' ','Best estimate','-sigma (dex)','+sigma (dex)'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30.3f}{2:<30.3f}{3:<30.3f}'.format('log10(L5100)',L5100_Ha[0],L5100_Ha[1],L5100_Ha[2]))
		logfile.write('\n{0:<30}{1:<30.3f}{2:<30.3f}{3:<30.3f}'.format('log10(M_BH)',MBH_Ha[0],MBH_Ha[1],MBH_Ha[2]))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()
	# Systemic Redshift
	if (output_type=='best_sys_vel'):
		z_best = output_val 
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### Best-fitting Systemic Redshift ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format(' ','Best estimate','-sigma (dex)','+sigma (dex)'))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30.8f}{2:<30.8f}{3:<30.8f}'.format('z_systemic',z_best[0],z_best[1],z_best[2]))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()
	# BPT classification
	if (output_type=='bpt_class'):
		BPT1_type, BPT2_type = output_val
		logfile = open(run_dir+'log/log_file.txt','a')
		logfile.write('\n')
		logfile.write('\n### BPT Diagnostic Classification ###')
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('[OIII]-[NII] Class.',' ','[OIII]-[SII] Class.',' '))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format(BPT1_type,' ',BPT2_type,' '))
		logfile.write('\n---------------------------------------------------------------------------------------------------------')
		logfile.close()
	if (output_type=='total_time'): # write total time to log
		# write_log(run_time,43,run_dir)
		tot_time = output_val
		a = str(datetime.datetime.now())
		logfile = open(run_dir+'log/log_file.txt','a')
		# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
		logfile.write('\n{0:<30}{1:<30}'.format('total_runtime',time_convert(tot_time) ))
		logfile.write('\n{0:<30}{1:<30}'.format('end_time',a ))

		logfile.write('\n----------------------------------------------------')
		logfile.close()


	return None

##################################################################################
