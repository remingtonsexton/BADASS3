#!/usr/bin/env python

"""Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS3), Version 9.1.1, Python 3.8 Version

BADASS is an open-source spectral analysis tool designed for detailed decomposition
of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the 
fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical. 
The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte 
Carlo sampler emcee for robust parameter and uncertainty estimation, as well 
as autocorrelation analysis to access parameter chain convergence.
"""

import numpy as np
from numpy.polynomial import hermite
from numpy import linspace, meshgrid
import scipy.optimize as op
import pandas as pd
import numexpr as ne
import matplotlib.pyplot as plt 
from matplotlib import cm
import matplotlib.gridspec as gridspec
from scipy import optimize, linalg, special, fftpack
from scipy.interpolate import griddata, interp1d
from scipy.stats import f, chisquare
from scipy import stats
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
from astropy.stats import mad_std
from scipy.special import wofz
import emcee
from astroquery.irsa_dust import IrsaDust
import astropy.units as u
from astropy import coordinates
from astropy.cosmology import FlatLambdaCDM
import re
import natsort
import copy
# import StringIO
import psutil
import pathlib
import importlib
import multiprocessing as mp
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

__author__	 = "Remington O. Sexton (GMU/USNO), Sara M. Doan (GMU), Michael A. Reefe (GMU), William Matzko (GMU), Nicholas Darden (UCR)"
__copyright__  = "Copyright (c) 2021 Remington Oliver Sexton"
__credits__	= ["Remington O. Sexton (GMU/USNO)", "Sara M. Doan (GMU)", "Michael A. Reefe (GMU)", "William Matzko (GMU)", "Nicholas Darden (UCR)"]
__license__	= "MIT"
__version__	= "9.1.1"
__maintainer__ = "Remington O. Sexton"
__email__	  = "rsexton2@gmu.edu"
__status__	 = "Release"

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
# - While is is *not recommended*, one can now test for outflows in the H-alpha/[NII] region independently of the H-beta/[OIII] region, as well as
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

# Version 7.7.2  - 7.7.6
# - Fixed problem with FeII emission lines at the edge of the fitting region
#   This is done by setting the variable edge_pad=0.
# - Fixed F-test NaN confidence bug
# - Updated initial fitting parameters in Jupyter notebook
# - Bug fixes and fixes to plots

# Version 8.0.0 - 8.0.13 major updates
# - Added smoothly broken power-law spectrum for high-z objects
# - Optimized FeII template fitting by utilizing PPXF framework
# - Added UV FeII+FeIII template from Vestergaard & Wilkes (2001)
# - Added Balmer continuum component
# - Added equivalent width calculations
# - Added additional chisquared fit statistic for outflow test
# - Voigt and Gauss-Hermite line profile options, with 
#   any number of higher order moments
# - Emission line list options (default and user-specified)
# - Control over soft- and hard constraints
# - Option for non-SDSS spectrum input
# - interpolation over metal absorption lines
# - masking of bad pixels, strong emission+absorption lines (automated), and user-defined masks
# - Various bug fixes, plotting improvements
# - new hypothesis testing for lines and outflows (F-test remains unchanged)
# - Continuum luminosities at 1350 Å, 3000 Å, and 5100 Å.
# - pathlib support
# - corner plots (corner.py) no longer supported; user should make their own corner plots with fewer free parameters
# - removed BPT diagram function; user should make BPT diagrams post processing.

# Version 8.0.14 - 8.0.15
# - Regular expressions now supported for soft constraints
# - IFU support for MANGA and MUSE (General) datasets

# Version 9.0.0 - 9.1.1
# - options for likelihood function
# - consolidated outflow and line testing routines


##########################################################################################################


#### Run BADASS ##################################################################

def run_BADASS(data,
			   nobj=None,
			   nprocesses=None,
			   options_file=None,
			   dust_cache=None,
			   fit_options=False,
			   mcmc_options=False,
			   comp_options=False,
			   user_lines=None,
			   user_constraints=None,
			   user_mask=None,
			   combined_lines={},
			   losvd_options=False,
			   host_options=False,
			   power_options=False,
			   opt_feii_options=False,
			   uv_iron_options=False,
			   balmer_options=False,
			   outflow_test_options=False,
			   plot_options=False,
			   output_options=False,
			   sdss_spec=True,
			   ifu_spec=False,
			   spec=None,
			   wave=None,
			   err=None,
			   fwhm=None,
			   z=None,
			   ebv=None,
			   ):
	"""
	The top-level BADASS function that handles the multiprocessing workers making calls to run_single_thread
	"""
	# Determine the number of processes based on CPU count, if unspecified
	if nprocesses is None:
		nprocesses = int(np.ceil(mp.cpu_count()/2))

	if os.path.isdir(data):
		# Get locations of sub-directories for each fit within the parent data directory
		spec_loc = natsort.natsorted(glob.glob(os.path.join(data, '*')))
		if nobj is not None:
			spec_loc = spec_loc[nobj[0]:nobj[1]]
		work_dirs = [si + os.sep for si in spec_loc]
		print(f"Fitting {len(spec_loc)} 1D spectra")

		# Print memory of the python process at the start
		process = psutil.Process(os.getpid())
		print(f"Start process memory: {process.memory_info().rss/1e9:<30.8f}")

		files = [glob.glob(os.path.join(wd, '*.fits'))[0] for wd in work_dirs]
		arguments = [(pathlib.Path(file), options_file, dust_cache, fit_options, mcmc_options, comp_options, user_lines, user_constraints, user_mask,
					  combined_lines, losvd_options, host_options, power_options, opt_feii_options, uv_iron_options, balmer_options,
					  outflow_test_options, plot_options, output_options, sdss_spec, ifu_spec, spec, wave, err, fwhm, z, ebv) for file in files]

		# map arguments to function
		if len(files) > 1:
			pool = mp.Pool(processes=nprocesses, maxtasksperchild=1)
			pool.starmap(run_single_thread, arguments, chunksize=1)
			pool.close()
			pool.join()
		else:
			run_single_thread(*arguments[0])

	elif os.path.isfile(data):
		# Print memory of the python process at the start
		process = psutil.Process(os.getpid())
		print(f"Start process memory: {process.memory_info().rss/1e9:<30.8f}")

		run_single_thread(pathlib.Path(data), options_file, dust_cache, fit_options, mcmc_options, comp_options,
						  user_lines, user_constraints, user_mask, combined_lines, losvd_options, host_options, power_options,
						  opt_feii_options, uv_iron_options, balmer_options, outflow_test_options, plot_options, output_options,
						  sdss_spec, ifu_spec, spec, wave, err, fwhm, z, ebv)

	# Print memory at the end
	print(f"End process memory: {process.memory_info().rss / 1e9:<30.8f}")




def run_single_thread(fits_file,
			   options_file = None,
			   dust_cache=None,
			   fit_options=False,
			   mcmc_options=False,
			   comp_options=False,
			   user_lines=None,
			   user_constraints=None,
			   user_mask=None,
			   combined_lines={},
			   losvd_options=False,
			   host_options=False,
			   power_options=False,
			   opt_feii_options=False,
			   uv_iron_options=False,
			   balmer_options=False,
			   outflow_test_options=False,
			   plot_options=False,
			   output_options=False,
			   sdss_spec =True,
			   ifu_spec  =False,
			   spec = None,
			   wave = None,
			   err  = None,
			   fwhm = None,
			   z	= None,
			   ebv  = None,
			   ):
			   
	"""
	This is the main function calls all other sub-functions in order. 
	"""

	if dust_cache != None:
		IrsaDust.cache_location = str(dust_cache)


	# Import options if options_file given
	if options_file is not None:
		try:
			options = importlib.import_module(options_file)
			# print("\n Successfully imported options file!\n")
			if hasattr(options,"fit_options"):
				fit_options			 = options.fit_options
			if hasattr(options,"comp_options"):
				comp_options		 = options.comp_options
			if hasattr(options,"mcmc_options"):
				mcmc_options		 = options.mcmc_options
			if hasattr(options,"user_lines"):
				user_lines			 = options.user_lines
			if hasattr(options,"user_constraints"):
				user_constraints	 = options.user_constraints
			if hasattr(options,"user_mask"):
				user_mask			 = options.user_mask
			if hasattr(options,"losvd_options"):
				losvd_options		 = options.losvd_options
			if hasattr(options,"host_options"):
				host_options		 = options.host_options
			if hasattr(options,"power_options"):
				power_options		 = options.power_options
			if hasattr(options,"opt_feii_options"):
				opt_feii_options	 = options.opt_feii_options
			if hasattr(options,"uv_iron_options"):
				uv_iron_options		 = options.uv_iron_options
			if hasattr(options,"balmer_options"):
				balmer_options		 = options.balmer_options
			if hasattr(options,"plot_options"):
				plot_options		 = options.plot_options
			if hasattr(options,"output_options"):
				output_options		 = options.output_options
			if hasattr(options,"line_list"):
				user_lines		     = options.user_lines
			if hasattr(options,"soft_cons"):
				user_constraints     = options.user_constraints
			if hasattr(options,"combined_lines"):
				combined_lines	     = options.combined_lines
		except ImportError:
			print("\n Error in importing options file! Options file must be a .py file!\n ")

	# Check inputs; raises exception if user input is invalid.
	fit_options			 = badass_utils.check_fit_options(fit_options,comp_options)
	comp_options		 = badass_utils.check_comp_options(comp_options)
	mcmc_options		 = badass_utils.check_mcmc_options(mcmc_options)
	user_lines			 = badass_utils.check_user_lines(user_lines)
	user_constraints	 = badass_utils.check_user_constraints(user_constraints)
	user_mask			 = badass_utils.check_user_mask(user_mask)
	losvd_options		 = badass_utils.check_losvd_options(losvd_options)
	host_options		 = badass_utils.check_host_options(host_options)
	power_options		 = badass_utils.check_power_options(power_options)
	opt_feii_options	 = badass_utils.check_opt_feii_options(opt_feii_options)
	uv_iron_options		 = badass_utils.check_uv_iron_options(uv_iron_options)
	balmer_options		 = badass_utils.check_balmer_options(balmer_options)
	plot_options		 = badass_utils.check_plot_options(plot_options)
	output_options		 = badass_utils.check_output_options(output_options)
	verbose				 = output_options["verbose"]
	# Check user input spectrum if sdss_spec=False
	if (not sdss_spec) and (not ifu_spec):
		# If user does not provide a error spectrum one will be provided for them!
		if err is None:
			err = np.abs(0.1*spec)
		spec, wave, err, fwhm, z, ebv = badass_utils.check_user_input_spec(spec,wave,err,fwhm,z,ebv)

	# Unpack input
	# fit_options
	fit_reg				= fit_options["fit_reg"]
	good_thresh			= fit_options["good_thresh"]
	mask_bad_pix	  	= fit_options["mask_bad_pix"]
	mask_emline			= fit_options["mask_emline"]
	mask_metal			= fit_options["mask_metal"]
	fit_stat		  	= fit_options["fit_stat"]
	n_basinhop	   		= fit_options["n_basinhop"]
	test_outflows		= fit_options["test_outflows"]
	test_line			= fit_options["test_line"]
	max_like_niter		= fit_options["max_like_niter"]
	output_pars			= fit_options["output_pars"]
	cosmology		    = fit_options["cosmology"]
	# mcmc_options
	mcmc_fit 			= mcmc_options["mcmc_fit"]
	nwalkers 			= mcmc_options["nwalkers"]
	auto_stop 			= mcmc_options["auto_stop"]
	conv_type 			= mcmc_options["conv_type"]
	min_samp			= mcmc_options["min_samp"]
	ncor_times 			= mcmc_options["ncor_times"]
	autocorr_tol 		= mcmc_options["autocorr_tol"]
	write_iter			= mcmc_options["write_iter"]
	write_thresh		= mcmc_options["write_thresh"]
	burn_in 			= mcmc_options["burn_in"]
	min_iter			= mcmc_options["min_iter"]
	max_iter			= mcmc_options["max_iter"]
	# comp_options
	fit_opt_feii		= comp_options["fit_opt_feii"]
	fit_uv_iron			= comp_options["fit_uv_iron"]
	fit_balmer			= comp_options["fit_balmer"]
	fit_losvd			= comp_options["fit_losvd"]
	fit_host			= comp_options["fit_host"]
	fit_power			= comp_options["fit_power"]
	fit_narrow			= comp_options["fit_narrow"]
	fit_broad			= comp_options["fit_broad"]
	fit_outflow			= comp_options["fit_outflow"]
	fit_absorp			= comp_options["fit_absorp"]
	tie_line_fwhm		= comp_options["tie_line_fwhm"]
	tie_line_voff		= comp_options["tie_line_voff"]
	n_moments 			= comp_options["n_moments"]
	# plot_options
	plot_param_hist		= plot_options["plot_param_hist"]
	plot_flux_hist		= plot_options["plot_flux_hist"]
	plot_lum_hist		= plot_options["plot_lum_hist"]
	plot_eqwidth_hist   = plot_options["plot_eqwidth_hist"]
	plot_HTML			= plot_options["plot_HTML"]

	# Set up run ('MCMC_output_#') directory
	work_dir = os.path.dirname(fits_file)+"/"
	run_dir,prev_dir = setup_dirs(work_dir,output_options['verbose'])
	run_dir = pathlib.Path(run_dir)

	# Check to make sure plotly is installed for HTML interactive plots:
	if plot_HTML==True:
		if importlib.util.find_spec('plotly'):
			pass
	else: plot_HTML=False

	# output_options
	write_chain			= output_options["write_chain"]
	verbose				= output_options["verbose"]
	#
	# Start fitting process
	print('\n > Starting fit for %s' % fits_file.parent.name)
	sys.stdout.flush()
	# Start a timer to record the total runtime
	start_time = time.time()
	#
	# Determine validity of fitting region
	min_fit_reg = 25 # in Å; set the minimum fitting region size here
	if (sdss_spec) or (ifu_spec): # if user-input spectrum is an SDSS spectrum
		#
		fit_reg,good_frac = determine_fit_reg_sdss(fits_file, run_dir, fit_reg, good_thresh, fit_losvd, losvd_options, verbose)
		if (fit_reg is None) or ((fit_reg[1]-fit_reg[0]) < min_fit_reg):
			print('\n Fitting region too small! The fitting region must be at least %d A!  Moving to next object... \n' % (min_fit_reg))
			cleanup(run_dir)
			return None
		elif (good_frac < fit_options["good_thresh"]) and (fit_reg is not None): # if fraction of good pixels is less than good_threshold, then move to next object
			print('\n Not enough good channels above threshold! Moving onto next object...')
			cleanup(run_dir)
			return None
		elif (good_frac >= fit_options["good_thresh"]) and (fit_reg is not None):
			pass
	elif (not sdss_spec): # if user-input spectrum is not an SDSS spectrum
		fit_reg,good_frac = determine_fit_reg_user(wave, z, run_dir, fit_reg, good_thresh, fit_losvd, losvd_options, verbose)
		if (fit_reg is None) or ((fit_reg[1]-fit_reg[0]) < min_fit_reg):
			print('\n Fitting region too small! The fitting region must be at least %d A!  Moving to next object... \n' % (min_fit_reg))
			cleanup(run_dir)
			return None
		elif (fit_reg is not None):
			pass

	# Prepare spectrum for fitting
	# SDSS spectrum
	if (sdss_spec):
		lam_gal,galaxy,noise,z,ebv,velscale,fwhm_gal,fit_mask = prepare_sdss_spec(fits_file, fit_reg, mask_bad_pix, mask_emline, user_mask, mask_metal, cosmology, run_dir, verbose=verbose, plot=True)
		binnum = spaxelx = spaxely = None
	# ifu spectrum
	elif (ifu_spec):
		lam_gal,galaxy,noise,z,ebv,velscale,fwhm_gal,fit_mask,binnum,spaxelx,spaxely = prepare_ifu_spec(fits_file, fit_reg, mask_bad_pix, mask_emline, user_mask, mask_metal, cosmology, run_dir, verbose=verbose, plot=True)
	# non-SDSS spectrum
	elif (not sdss_spec):
		lam_gal,galaxy,noise,z,ebv,velscale,fwhm_gal,fit_mask = prepare_user_spec(fits_file, spec, wave, err, fwhm, z, ebv, fit_reg, mask_emline, user_mask, mask_metal, cosmology, run_dir, verbose=verbose, plot=True)
		binnum = spaxelx = spaxely = None

	# Write to Log 
	write_log((fit_options,mcmc_options,comp_options,losvd_options,host_options,power_options,opt_feii_options,uv_iron_options,balmer_options,
			   plot_options,output_options),'fit_information',run_dir)

	####################################################################################################################################################################################

	# Generate host-galaxy template
	if (fit_host==True) & (lam_gal[0]>1680.2):
		host_template = generate_host_template(lam_gal, host_options, fwhm_gal,fit_mask, velscale, verbose=verbose)
	elif (fit_host==True) & (lam_gal[0]<1680.2):
		host_template = None
		fit_host = False
		comp_options["fit_host"]=False
		if verbose:
			print('\n - Host galaxy SSP template disabled because template is outside of fitting region.')
	elif (fit_host==False):
		host_template = None
	# Load stellar templates if fit_losvd=True 
	if (fit_losvd==True):
		stel_templates = prepare_stellar_templates(galaxy, lam_gal, fit_reg, velscale, fwhm_gal,fit_mask, losvd_options, run_dir)
	elif (fit_losvd==False):
		stel_templates = None

	# For the Optical FeII, UV Iron, and Balmer templates, we disable the templates if the fitting region
	# is entirely outside of the range of the templates.  This saves resources.

	# Check conditions for and generate Optical FeII templates
	# Veron-Cetty et al. (2004)
	if (fit_opt_feii==True) & (opt_feii_options["opt_template"]["type"]=="VC04") & (lam_gal[-1]>=3400.0) & (lam_gal[0]<=7200.0):
		opt_feii_templates = initialize_opt_feii(lam_gal,opt_feii_options,fwhm_gal,fit_mask,velscale)
	elif (fit_opt_feii==True) & (opt_feii_options["opt_template"]["type"]=="VC04") & ((lam_gal[-1]<3400.0) | (lam_gal[0]>7200.0)):
		if verbose:
			print('\n - Optical FeII template disabled because template is outside of fitting region.')
		fit_opt_feii = False
		comp_options["fit_opt_feii"]=False
		opt_feii_templates = None
		write_log((),'update_opt_feii',run_dir)
	# Kovacevic et al. (2010)
	elif (fit_opt_feii==True) & (opt_feii_options["opt_template"]["type"]=="K10") & (lam_gal[-1]>=4400.0) & (lam_gal[0]<=5500.0):
		opt_feii_templates = initialize_opt_feii(lam_gal,opt_feii_options,fwhm_gal,fit_mask,velscale)
	elif (fit_opt_feii==True) & (opt_feii_options["opt_template"]["type"]=="K10") & ((lam_gal[-1]<4400.0) | (lam_gal[0]>5500.0)):
		if verbose:
			print('\n - Optical FeII template disabled because template is outside of fitting region.')
		opt_feii_templates = None
		fit_opt_feii = False
		comp_options["fit_opt_feii"]=False
		opt_feii_templates = None
		write_log((),'update_opt_feii',run_dir)
	elif (fit_opt_feii==False):
		opt_feii_templates = None
		
	# Generate UV Iron template - Vestergaard & Wilkes (2001)
	if (fit_uv_iron==True) & (lam_gal[-1]>=1074.0) & (lam_gal[0]<=3100.0):
		uv_iron_template = initialize_uv_iron(lam_gal,uv_iron_options,fwhm_gal,fit_mask,velscale)
	elif (fit_uv_iron==True) & ((lam_gal[-1]<1074.0) | (lam_gal[0]>3100.0)):
		if verbose:
			print('\n - UV Iron template disabled because template is outside of fitting region.')
		uv_iron_template = None
		fit_uv_iron = False
		comp_options["fit_uv_iron"]=False
		uv_iron_template = None
		write_log((),'update_uv_iron',run_dir)
	elif (fit_uv_iron==False):
		uv_iron_template = None

	# Generate Balmer continuum
	if (fit_balmer==True) & (lam_gal[0]<3500.0):
		balmer_template = initialize_balmer(lam_gal,balmer_options,fwhm_gal,fit_mask,velscale)
	elif (fit_balmer==True) & (lam_gal[0]>=3500.0):
		if verbose:
			print('\n - Balmer continuum disabled because template is outside of fitting region.')
		balmer_template = None
		fit_balmer = False
		comp_options["fit_balmer"]=False
		balmer_template = None
		write_log((),'update_balmer',run_dir)
	elif (fit_balmer==False):
		balmer_template = None

	####################################################################################################################################################################################

	# Initialize maximum likelihood parameters
	if verbose:
		print('\n Initializing parameters for Maximum Likelihood Fitting.')
		print('----------------------------------------------------------------------------------------------------')

	param_dict, line_list, combined_line_list, soft_cons = initialize_pars(lam_gal,galaxy,noise,fit_reg,fwhm_gal,fit_mask,velscale,
								 comp_options,user_lines,user_constraints,combined_lines,losvd_options,host_options,power_options,opt_feii_options,uv_iron_options,balmer_options,
								 run_dir,fit_type='init',fit_stat=fit_stat,
								 fit_opt_feii=fit_opt_feii,fit_uv_iron=fit_uv_iron,fit_balmer=fit_balmer,
								 fit_losvd=fit_losvd,fit_host=fit_host,fit_power=fit_power,
								 fit_narrow=fit_narrow,fit_broad=fit_broad,fit_outflow=fit_outflow,fit_absorp=fit_absorp,
								 tie_line_fwhm=tie_line_fwhm,tie_line_voff=tie_line_voff,verbose=verbose)

	# Output all free parameters of fit prior to fitting (useful for diagnostics)
	if output_pars and verbose:
		output_free_pars(line_list,param_dict,soft_cons)
		write_log((line_list,param_dict,soft_cons),'output_line_list',run_dir)
		return 
	elif not output_pars and verbose:
		output_free_pars(line_list,param_dict,soft_cons)
		write_log((line_list,param_dict,soft_cons),'output_line_list',run_dir)
	elif not output_pars and not verbose:
		write_log((line_list,param_dict,soft_cons),'output_line_list',run_dir)

	#### Line Testing ################################################################################################################################################################################
	
	if (test_line["bool"]==True):
		# If line test, check to make sure line is in line list
		if (isinstance(test_line["line"],str)) and (test_line["line"] not in line_list):
			shutil.rmtree(run_dir)
			print("\n Line to test not found in line list! Make sure line is within fitting region for test.\n")
			return
		elif (isinstance(test_line["line"],list)) and not (np.all([False if line not in line_list else True for line in test_line["line"]])):
			shutil.rmtree(run_dir)
			print("\n Line to test not found in line list! Make sure line is within fitting region for test.\n")
			return
			

		if verbose:
			print("\n Testing for %s" % (test_line["line"]))


		line_test(param_dict,
				  line_list,
				  combined_line_list,
				  soft_cons,
				  lam_gal,
				  galaxy,
				  noise,
				  z,
				  cosmology,
				  fit_reg,
				  user_lines,
				  user_constraints,
				  combined_lines,
				  test_line,
				  comp_options,
				  losvd_options,
				  host_options,
				  power_options,
				  opt_feii_options,
				  uv_iron_options,
				  balmer_options,
				  outflow_test_options,
				  host_template,
				  opt_feii_templates,
				  uv_iron_template,
				  balmer_template,
				  stel_templates,
				  fwhm_gal,
				  fit_mask,
				  velscale,
				  run_dir,
				  fit_type='init',
				  fit_stat=fit_stat,
				  output_model=False,
				  test_outflows=False,
				  n_basinhop=n_basinhop,
				  max_like_niter=max_like_niter,
				  verbose=verbose,
				  binnum=binnum,
				  spaxelx=spaxelx,
				  spaxely=spaxely)
		# Exit BADASS
		print(' - Line testing complete for %s! \n' % fits_file.parent.name)
		return

	####################################################################################################################################################################################

	#### Outflow Testing ################################################################################################################################################################################
	
	if (test_outflows==True):
		# If test_outflow, check to make sure the line list has outflow lines in it
		if (len([line for line in line_list if line_list[line]["line_type"]=="out"])==0):
			shutil.rmtree(run_dir)
			print("\n There are no outflow lines in the line list to test! Make sure fit_outflow = True and are within fitting range.\n")
			return

		if verbose:
			print("\n Testing for outflows...")


		line_test(param_dict,
				  line_list,
				  combined_line_list,
				  soft_cons,
				  lam_gal,
				  galaxy,
				  noise,
				  z,
				  cosmology,
				  fit_reg,
				  user_lines,
				  user_constraints,
				  combined_lines,
				  test_line,
				  comp_options,
				  losvd_options,
				  host_options,
				  power_options,
				  opt_feii_options,
				  uv_iron_options,
				  balmer_options,
				  outflow_test_options,
				  host_template,
				  opt_feii_templates,
				  uv_iron_template,
				  balmer_template,
				  stel_templates,
				  fwhm_gal,
				  fit_mask,
				  velscale,
				  run_dir,
				  fit_type='init',
				  fit_stat=fit_stat,
				  output_model=False,
				  test_outflows=True,
				  n_basinhop=n_basinhop,
				  max_like_niter=max_like_niter,
				  verbose=verbose,
				  binnum=binnum,
				  spaxelx=spaxelx,
				  spaxely=spaxely)
		# Exit BADASS
		print(' - Outflow testing complete for %s! \n' % fits_file.parent.name)
		return

	####################################################################################################################################################################################
	# Peform maximum likelihood
	result_dict, comp_dict     = max_likelihood(param_dict,
												line_list,
												combined_line_list,
												soft_cons,
												lam_gal,
												galaxy,
												noise,
												z,
												cosmology,
												comp_options,
												losvd_options,
												host_options,
												power_options,
												opt_feii_options,
												uv_iron_options,
												balmer_options,
												outflow_test_options,
												host_template,
												opt_feii_templates,
												uv_iron_template,
												balmer_template,
												stel_templates,
												fwhm_gal,
												fit_mask,
												velscale,
												run_dir,
												fit_type='init',
												fit_stat=fit_stat,
												output_model=False,
												test_outflows=False,
												n_basinhop=n_basinhop,
												max_like_niter=max_like_niter,
												verbose=verbose)
	
	if (mcmc_fit==False):
		# If not performing MCMC fitting, terminate BADASS here and write 
		# parameters, uncertainties, and components to a fits file
		# Write final parameters to file
		# Header information
		header_dict = {}
		header_dict["z_sdss"] = z
		header_dict["med_noise"] = np.median(noise)
		header_dict["velscale"]  = velscale
		#
		write_max_like_results(result_dict,comp_dict,header_dict,fit_mask,run_dir,binnum,spaxelx,spaxely)
		
		# Make interactive HTML plot 
		if plot_HTML:
			plotly_best_fit(fits_file.parent.name,line_list,fit_mask,run_dir)

		print(' - Done fitting %s! \n' % fits_file.parent.name)
		sys.stdout.flush()
		return

	#######################################################################################################
 
	# Initialize parameters for emcee
	if verbose:
		print('\n Initializing parameters for MCMC.')
		print('----------------------------------------------------------------------------------------------------')
	param_dict, line_list, combined_line_list, soft_cons = initialize_pars(lam_gal,galaxy,noise,fit_reg,fwhm_gal,fit_mask,velscale,
														   comp_options,user_lines,user_constraints,combined_lines,losvd_options,host_options,power_options,opt_feii_options,uv_iron_options,balmer_options,
														   run_dir,fit_type='final',fit_stat=fit_stat,
														   fit_opt_feii=fit_opt_feii,fit_uv_iron=fit_uv_iron,fit_balmer=fit_balmer,
														   fit_losvd=fit_losvd,fit_host=fit_host,fit_power=fit_power,
														   fit_narrow=fit_narrow,fit_broad=fit_broad,fit_outflow=fit_outflow,fit_absorp=fit_absorp,
														   tie_line_fwhm=tie_line_fwhm,tie_line_voff=tie_line_voff,
														   remove_lines=False,verbose=verbose)
	#
	if verbose:
		output_free_pars(line_list,param_dict,soft_cons)
	#
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
	if verbose:
		print('\n Performing MCMC iterations...')
		print('----------------------------------------------------------------------------------------------------')

	# Extract relevant stuff from dicts
	param_names  = [key for key in param_dict ]
	init_params  = [param_dict[key]['init'] for key in param_dict ]
	bounds		 = [param_dict[key]['plim'] for key in param_dict ]

	# Check number of walkers
	# If number of walkers < 2*(# of params) (the minimum required), then set it to that
	if nwalkers<2*len(param_names):
		if verbose:
			print('\n Number of walkers < 2 x (# of parameters)!  Setting nwalkers = %d' % (2.0*len(param_names)))
		nwalkers = int(2.0*len(param_names))
	
	ndim, nwalkers = len(init_params), nwalkers # minimum walkers = 2*len(params)

	# initialize walker starting positions based on parameter estimation from Maximum Likelihood fitting
	pos = initialize_walkers(init_params,param_names,bounds,soft_cons,nwalkers,ndim)

	# Run emcee
	# args = arguments of lnprob (log-probability function)
	lnprob_args=(param_names,
				 bounds,
				 line_list,
				 combined_line_list,
				 soft_cons,
				 lam_gal,
				 galaxy,
				 noise,
				 comp_options,
				 losvd_options,
				 host_options,
				 power_options,
				 opt_feii_options,
				 uv_iron_options,
				 balmer_options,
				 outflow_test_options,
				 host_template,
				 opt_feii_templates,
				 uv_iron_template,
				 balmer_template,
				 stel_templates,
				 fwhm_gal,
				 fit_mask,
				 fit_stat,
				 velscale,
				 run_dir)
	
	emcee_data = run_emcee(pos,ndim,nwalkers,run_dir,lnprob_args,init_params,param_names,
							auto_stop,conv_type,min_samp,ncor_times,autocorr_tol,write_iter,write_thresh,
							burn_in,min_iter,max_iter,verbose=verbose)

	sampler_chain, burn_in, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob, log_like_blob = emcee_data
	# Add chains to each parameter in param dictionary
	for k,key in enumerate(param_names):
		if key in param_dict:
			param_dict[key]['chain']=sampler_chain[:,:,k]

	if verbose:
		print('\n > Fitting MCMC chains...')
	# These three functions produce parameter, flux, and luminosity histograms and chains from the MCMC sampling.
	# Free parameter values, uncertainties, and plots
	param_dict = param_plots(param_dict,burn_in,run_dir,plot_param_hist=plot_param_hist,verbose=verbose)
	# Add tied parameters
	param_dict = add_tied_parameters(param_dict, line_list)
	# Log Like Function values plots
	log_like_dict = log_like_plot(log_like_blob, burn_in, nwalkers, run_dir, plot_param_hist=plot_param_hist,verbose=verbose)
	# Flux values, uncertainties, and plots
	flux_dict = flux_plots(flux_blob, burn_in, nwalkers, run_dir, plot_flux_hist=plot_flux_hist,verbose=verbose)
	# Luminosity values, uncertainties, and plots
	lum_dict = lum_plots(flux_dict, burn_in, nwalkers, z, run_dir, H0=cosmology["H0"],Om0=cosmology["Om0"],plot_lum_hist=plot_lum_hist,verbose=verbose)
	# Continuum luminosity 
	cont_lum_dict = cont_lum_plots(cont_flux_blob, burn_in, nwalkers, z, run_dir, H0=cosmology["H0"],Om0=cosmology["Om0"],plot_lum_hist=plot_lum_hist,verbose=verbose)
	# Equivalent widths, uncertainties, and plots
	eqwidth_dict = eqwidth_plots(eqwidth_blob, burn_in, nwalkers, run_dir, plot_eqwidth_hist=plot_eqwidth_hist, verbose=verbose)
	# Auxiliary Line Dict (Combined FWHMs and Fluxes of MgII and CIV)
	int_vel_disp_dict = int_vel_disp_plots(int_vel_disp_blob, burn_in, nwalkers, z, run_dir, H0=cosmology["H0"],Om0=cosmology["Om0"],plot_param_hist=plot_param_hist,verbose=verbose)

	# If stellar velocity is fit, estimate the systemic velocity of the galaxy;
	# SDSS redshifts are based on average emission line redshifts.
	extra_dict = {}
	extra_dict["LOG_LIKE"] = log_like_dict

	if ('stel_vel' in param_dict):
		if verbose:
			print('\n > Estimating systemic velocity of galaxy...')
		z_dict = systemic_vel_est(z,param_dict,burn_in,run_dir,plot_param_hist=plot_param_hist)
		extra_dict = {**extra_dict, **z_dict}

	if verbose:
		print('\n > Saving Data...')

	# Write all chains to a fits table
	if (write_chain==True):
		write_chains({**param_dict,**flux_dict,**lum_dict,**cont_lum_dict,**eqwidth_dict,**int_vel_disp_dict},run_dir)
	# Plot and save the best fit model and all sub-components
	comp_dict = plot_best_model(param_dict,
					line_list,
					combined_line_list,
					lam_gal,
					galaxy,
					noise,
					comp_options,
					losvd_options,
					host_options,
					power_options,
					opt_feii_options,
					uv_iron_options,
					balmer_options,
					outflow_test_options,
					host_template,
					opt_feii_templates,
					uv_iron_template,
					balmer_template,
					stel_templates,
					fwhm_gal,
					fit_mask,
					fit_stat,
					velscale,
					run_dir)

	# Calculate some fit quality parameters which will be added to the dictionary
	# These will be appended to result_dict and need to be in the same format {"med": , "std", "flag":}
	fit_quality_dict = fit_quality_pars(param_dict,line_list,combined_line_list,comp_dict,fit_type="mcmc",fit_stat=fit_stat)
	param_dict = {**param_dict,**fit_quality_dict}

	# Write best fit parameters to fits table
	# Header information
	header_dict = {}
	header_dict["Z_SDSS"]	= z
	header_dict["MED_NOISE"] = np.median(noise)
	header_dict["VELSCALE"]  = velscale
	#
	param_dict = {**param_dict,**flux_dict,**lum_dict,**eqwidth_dict,**cont_lum_dict,**int_vel_disp_dict,**extra_dict}
	write_params(param_dict,header_dict,bounds,run_dir,binnum,spaxelx,spaxely)

	# Make interactive HTML plot 
	if plot_HTML:
		plotly_best_fit(fits_file.parent.name,line_list,fit_mask,run_dir)

	if verbose:
		print('\n Cleaning up...')
		print('----------------------------------------------------------------------------------------------------')
	# Delete redundant files to cut down on space
	cleanup(run_dir)
	
	# Total time
	elap_time = (time.time() - start_time)
	if verbose:
		print("\n Total Runtime = %s" % (time_convert(elap_time)))
	# Write to log
	write_log(elap_time,'total_time',run_dir)
	print(' - Done fitting %s! \n' % fits_file.stem)
	sys.stdout.flush()
	return

##################################################################################


def initialize_walkers(init_params,param_names,bounds,soft_cons,nwalkers,ndim):
	"""
	Initializes the MCMC walkers within bounds and soft constraints.
	"""
	# Create refereence dictionary for numexpr
	pdict = {}
	for k in range(0,len(param_names),1):
	        pdict[param_names[k]] = init_params[k]
	        
	pos = init_params + 1.e-3 * np.random.randn(nwalkers,ndim)
 	# First iterate through bounds
	for j in range(np.shape(pos)[1]): # iterate through parameter
	    for i in range(np.shape(pos)[0]): # iterate through walker
	        if (pos[i][j]<bounds[j][0]) | (pos[i][j]>bounds[j][1]):
	            while (pos[i][j]<bounds[j][0]) | (pos[i][j]>bounds[j][1]):
	                pos[i][j] = init_params[j] + 1.e-3*np.random.randn(1)
	#    
	return pos

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

	# Subsample the data into a manageable size for the kde and HDI
	subsampled = np.random.choice(flat,size=10000)

	# Histogram; 'Doane' binning produces the best results from tests.
	hist, bin_edges = np.histogram(subsampled, bins='doane', density=False)

	# Generate pseudo-data on the ends of the histogram; this prevents the KDE
	# from weird edge behavior.
	n_pseudo = 3 # number of pseudo-bins 
	bin_width=bin_edges[1]-bin_edges[0]
	lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
	upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

	# Calculate bandwidth for KDE (Silverman method)
	h = kde_bandwidth(flat)

	# Create a subsampled grid for the KDE based on the subsampled data; by
	# default, we subsample by a factor of 10.
	xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

	# Calculate KDE
	kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
	p68 = compute_HDI(subsampled,0.68)
	p95 = compute_HDI(subsampled,0.95)

	post_max  = xs[kde.argmax()] # posterior max estimated from KDE
	post_mean = np.mean(flat)
	post_med  = np.median(flat)
	low_68    = post_max - p68[0]
	upp_68    = p68[1] - post_max
	low_95    = post_max - p95[0]
	upp_95    = p95[1] - post_max
	post_std  = np.std(flat)
	post_mad  = stats.median_abs_deviation(flat)

	if ((post_max-(3.0*low_68))<0): 
		flag = 1
	else: flag = 0

	z_dict = {}
	z_dict["z_sys"] = {}
	z_dict["z_sys"]["par_best"]    = post_max
	z_dict["z_sys"]["ci_68_low"]   = low_68
	z_dict["z_sys"]["ci_68_upp"]   = upp_68
	z_dict["z_sys"]["ci_95_low"]   = low_95
	z_dict["z_sys"]["ci_95_upp"]   = upp_95
	z_dict["z_sys"]["mean"] 	   = post_mean
	z_dict["z_sys"]["std_dev"] 	   = post_std
	z_dict["z_sys"]["median"]	   = post_med
	z_dict["z_sys"]["med_abs_dev"] = post_mad
	z_dict["z_sys"]["flat_chain"]  = flat
	z_dict["z_sys"]["flag"] 	   = flag
	
	return z_dict

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

def setup_dirs(work_dir,verbose=True):
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
		if verbose:
			print(' Folder has not been created.  Creating MCMC_output folder...')
		# Create the first MCMC_output file starting with index 1
		os.mkdir(work_dir+'MCMC_output_1')
		run_dir = os.path.join(work_dir,'MCMC_output_1/') # running directory
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
		if verbose:
			print(' Storing MCMC_output in %s' % run_dir)

	return run_dir,prev_dir

##################################################################################


#### Determine fitting region ####################################################

# SDSS spectra
def determine_fit_reg_sdss(fits_file, run_dir, fit_reg, good_thresh, fit_losvd, losvd_options, verbose):
	"""
	Determines the fitting region for SDSS spectra.
	"""
	# Limits of the stellar template wavelength range
	# The stellar templates packaged with BADASS are from the Indo-US Coude Feed Stellar Template Library
	# with the below wavelength ranges.
	if (losvd_options["library"]=="IndoUS"):
		min_losvd, max_losvd = 3460, 9464
	if (losvd_options["library"]=="Vazdekis2010"):
		min_losvd, max_losvd = 3540.5, 7409.6
	if (losvd_options["library"]=="eMILES"):
		min_losvd, max_losvd = 1680.2, 49999.4
	# Open spectrum file
	hdu = fits.open(fits_file)
	specobj = hdu[2].data
	z = specobj['z'][0]
	# t = hdu['COADD'].data
	t = hdu[1].data	
	lam_gal = (10**(t['loglam']))/(1+z)

	gal  = t['flux']
	ivar = t['ivar']
	and_mask = t['and_mask']
	# Edges of wavelength vector
	first_good = lam_gal[0]
	last_good  = lam_gal[-1]

	if ((fit_reg=='auto') or (fit_reg=='full')):
		# The lower limit of the spectrum must be the lower limit of our stellar templates
		if ((fit_losvd==True) & (first_good < min_losvd)) | ((fit_losvd==True) & (last_good > max_losvd)):
			if verbose:
				print("\n Warning: Fitting LOSVD requires wavelenth range between %d Å and %d Å for stellar templates. BADASS will adjust your fitting range to fit the LOSVD..." % (min_losvd,max_losvd))
				print("		- Available wavelength range: (%d, %d)" % (first_good, last_good) )
			auto_low = np.max([min_losvd,first_good]) # Indo-US Library of Stellar Templates has a lower limit of 3460
			# auto_upp = determine_upper_bound(first_good,last_good)
			auto_upp = np.min([max_losvd,last_good])
			# if (auto_upp is not None):
			new_fit_reg = (np.floor(auto_low),np.ceil(auto_upp))	
			if verbose:
				print("		- New fitting region is (%d, %d). \n" % (new_fit_reg[0], new_fit_reg[1]) )
			# elif (auto_upp is None):
				# new_fit_reg = None
				# return None, None
		elif (fit_losvd==False):
			new_fit_reg = (np.floor(first_good),np.ceil(last_good))

	elif isinstance(fit_reg,(tuple,list)):
		# Check to see if tuple/list makes sense
		if ((fit_reg[0]>fit_reg[1]) | (fit_reg[1]<fit_reg[0])): # if boundaries overlap
			if verbose:
				print('\n Fitting boundaries overlap! \n')
			new_fit_reg = None
			return None, None
		elif (fit_reg[0] > last_good) | (fit_reg[1] < first_good):
			if verbose:
				print('\n Fitting region not available! \n')
			new_fit_reg = None
			return None, None
		elif ((fit_losvd==True) & (fit_reg[0]<min_losvd)) | ((fit_losvd==True) & (fit_reg[1]>max_losvd)):
			if verbose:
				print("\n Warning: Fitting LOSVD requires wavelenth range between 3460 A and 9464 A for stellar templates. BADASS will adjust your fitting range to fit the LOSVD...")
				print("		- Input fitting range: (%d, %d)" % (fit_reg[0], fit_reg[1]) )
				print("		- Available wavelength range: (%d, %d)" % (first_good, last_good) )
			wave_low = np.max([min_losvd,fit_reg[0],first_good])
			wave_upp = np.min([max_losvd,fit_reg[1],last_good])
			new_fit_reg = (np.floor(wave_low),np.ceil(wave_upp))
			if verbose:
				print("		- New fitting region is (%d, %d). \n" % (new_fit_reg[0], new_fit_reg[1]) )
		else:# (fit_losvd==False):
			if (fit_reg[0] < first_good) | (fit_reg[1] > last_good):
				if verbose:
					print("\n Input fitting region exceeds available wavelength range.  BADASS will adjust your fitting range automatically...")
					print("		- Input fitting range: (%d, %d)" % (fit_reg[0], fit_reg[1]) )
					print("		- Available wavelength range: (%d, %d)" % (first_good, last_good) )
				wave_low = np.max([fit_reg[0],first_good])
				wave_upp = np.min([fit_reg[1],last_good])
				new_fit_reg = (np.floor(wave_low),np.ceil(wave_upp))
				if verbose:
					print("		- New fitting region is (%d, %d). \n" % (new_fit_reg[0], new_fit_reg[1]) )
			else:
				new_fit_reg = (np.floor(fit_reg[0]),np.ceil(fit_reg[1]))

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
		ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)')
		ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)')

		plt.tight_layout()
		plt.savefig(run_dir.joinpath('good_pixels.pdf'),fmt='pdf',dpi=150)
		fig.clear()
		plt.close()
	##################################################################################
	# Close the fits file 
	hdu.close()
	##################################################################################

	return new_fit_reg,good_frac

# User (non-SDSS) spectra
def determine_fit_reg_user(wave, z, run_dir, fit_reg, good_thresh, fit_losvd, losvd_options, verbose):
	"""
	Determines valid fitting region for a user-input spectrum.
	"""
	# Limits of the stellar template wavelength range
	# The stellar templates packaged with BADASS are from the Indo-US Coude Feed Stellar Template Library
	# with the below wavelength ranges.
	min_losvd = 3460
	max_losvd = 9464
	lam_gal   = wave/(1+z)		
	# Edges of wavelength vector
	first_good = lam_gal[0]
	last_good  = lam_gal[-1]

	if ((fit_reg=='auto') or (fit_reg=='full')):
		# The lower limit of the spectrum must be the lower limit of our stellar templates
		if ((fit_losvd==True) & (first_good < min_losvd)) | ((fit_losvd==True) & (last_good > max_losvd)):
			if verbose:
				print("\n Warning: Fitting LOSVD requires wavelenth range between %d Å and %d Å for stellar templates. BADASS will adjust your fitting range to fit the LOSVD..." % (min_losvd,max_losvd))
				print("		- Available wavelength range: (%d, %d)" % (first_good, last_good) )
			auto_low = np.max([min_losvd,first_good]) # Indo-US Library of Stellar Templates has a lower limit of 3460
			# auto_upp = determine_upper_bound(first_good,last_good)
			auto_upp = np.min([max_losvd,last_good])
			# if (auto_upp is not None):
			new_fit_reg = (np.floor(auto_low),np.ceil(auto_upp))	
			if verbose:
				print("		- New fitting region is (%d, %d). \n" % (new_fit_reg[0], new_fit_reg[1]) )
			# elif (auto_upp is None):
				# new_fit_reg = None
				# return None, None
		elif (fit_losvd==False):
			new_fit_reg = (np.floor(first_good),np.ceil(last_good))

	elif isinstance(fit_reg,(tuple,list)):
		# Check to see if tuple/list makes sense
		if ((fit_reg[0]>fit_reg[1]) | (fit_reg[1]<fit_reg[0])): # if boundaries overlap
			if verbose:
				print('\n Fitting boundaries overlap! \n')
			new_fit_reg = None
			return None, None
		elif (fit_reg[0] > last_good) | (fit_reg[1] < first_good):
			if verbose:
				print('\n Fitting region not available! \n')
			new_fit_reg = None
			return None, None
		elif ((fit_losvd==True) & (fit_reg[0]<min_losvd)) | ((fit_losvd==True) & (fit_reg[1]>max_losvd)):
			if verbose:
				print("\n Warning: Fitting LOSVD requires wavelenth range between 3460 A and 9464 A for stellar templates. BADASS will adjust your fitting range to fit the LOSVD...")
				print("		- Input fitting range: (%d, %d)" % (fit_reg[0], fit_reg[1]) )
				print("		- Available wavelength range: (%d, %d)" % (first_good, last_good) )
			wave_low = np.max([min_losvd,fit_reg[0],first_good])
			wave_upp = np.min([max_losvd,fit_reg[1],last_good])
			new_fit_reg = (np.floor(wave_low),np.ceil(wave_upp))
			if verbose:
				print("		- New fitting region is (%d, %d). \n" % (new_fit_reg[0], new_fit_reg[1]) )
		else:# (fit_losvd==False):
			if (fit_reg[0] < first_good) | (fit_reg[1] > last_good):
				if verbose:
					print("\n Input fitting region exceeds available wavelength range.  BADASS will adjust your fitting range automatically...")
					print("		- Input fitting range: (%d, %d)" % (fit_reg[0], fit_reg[1]) )
					print("		- Available wavelength range: (%d, %d)" % (first_good, last_good) )
				wave_low = np.max([fit_reg[0],first_good])
				wave_upp = np.min([fit_reg[1],last_good])
				new_fit_reg = (np.floor(wave_low),np.ceil(wave_upp))
				if verbose:
					print("		- New fitting region is (%d, %d). \n" % (new_fit_reg[0], new_fit_reg[1]) )
			else:
				new_fit_reg = (np.floor(fit_reg[0]),np.ceil(fit_reg[1]))

	##################################################################################

	return new_fit_reg,1.0

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
	# Correction invalid for x>11:
	if np.any(x>11):
		return flux 

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

##################################################################################

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


def emline_masker(wave,spec,noise):
	"""
	Runs a multiple moving window median  
	to determine location of emission lines
	to generate an emission line mask for 
	continuum fitting.
	"""
	# Do a series of median filters with window sizes up to 20 
	window_sizes = [2,5,10,50,100,250,500]#np.arange(10,510,10,dtype=int)
	med_spec = np.empty((len(wave),len(window_sizes)))
	# 
	for i in range(len(window_sizes)):
		med_spec[:,i] = window_filter(spec,window_sizes[i])
	#
	mask_bad = np.unique(np.where((np.std(med_spec,axis=1)>noise) | (np.std(med_spec,axis=1)>np.nanmedian(noise)))[0])
	# mask_good = np.unique(np.where((np.std(med_spec,axis=1)<noise) & (np.std(med_spec,axis=1)<np.nanmedian(noise)))[0])
	#
	return mask_bad#,mask_good


def metal_masker(wave,spec,noise):
	"""
	Runs a multiple moving window median  
	to determine location of emission lines
	to generate an emission line mask for 
	continuum fitting.
	"""
	# Do a series of median filters with window sizes up to 25 
	bandwidths = [5, 10, 15, 20, 25,100,500]
	sig_clip = 3.0 # sigma threshold
	nclip = 10 # number of clipping iterations
	buffer = 2 # buffers # pixels each side 
	all_nans = []
	for i in range(len(bandwidths)):
		count=0
		new_spec = np.copy(spec)
		while (count<=nclip):
			med_spec = window_filter(new_spec,bandwidths[i])
			count+=1

			# fig = plt.figure(figsize=(20,10))
			# ax1 = fig.add_subplot(111)
			# ax1.plot(wave,spec)
			# ax1.plot(wave,med_spec)
			# ax1.plot(wave,noise)
			# plt.tight_layout()
			# sys.exit()

			nan_spec = np.where((med_spec>new_spec) & (np.abs(med_spec-new_spec)>np.median(noise)*sig_clip))[0]
			inan_buffer_upp = np.array([range(i,i+buffer+1) for i in nan_spec if (i+buffer) < len(spec)],dtype=int)
			inan_buffer_upp = inan_buffer_upp.flatten()
			inan_buffer_low = np.array([range(i-buffer-1,i) for i in nan_spec if (i-buffer) > 0],dtype=int)
			inan_buffer_low = inan_buffer_low.flatten()
			inan = np.concatenate([nan_spec,inan_buffer_low, inan_buffer_upp])
			all_nans.append(inan)
			new_spec = med_spec

			if len(nan_spec)>0:
				pass

			else:
				break

	mask_bad = np.unique(np.concatenate(all_nans))

	return mask_bad



def window_filter(spec,size):
	"""
	Estimates the median value of the spectrum 
	within a pixel window.
	"""
	med_spec = np.empty(len(spec))
	pix = np.arange(0,len(spec),1)
	for i,p in enumerate(pix):
		# Get n-nearest pixels
		# Calculate distance from i to each pixel
		i_sort =np.argsort(np.abs(i-pix))
		idx = pix[i_sort][:size] # indices we estimate from
		med = np.median(spec[idx])
		med_spec[i] = med
	#
	return med_spec

def interpolate_metal(spec,noise):
	"""
	Interpolates over metal absorption lines for 
	high-redshift spectra using a moving median
	filter.
	"""
	sig_clip = 3.0
	nclip = 10
	bandwidth= 15
	med_spec = window_filter(spec,bandwidth)
	count = 0 
	new_spec = np.copy(spec)
	while (count<=nclip) and ((np.std(new_spec-med_spec)*sig_clip)>np.median(noise)):
		count+=1
		# Get locations of nan or -inf pixels
		nan_spec = np.where((np.abs(new_spec-med_spec)>(np.std(new_spec-med_spec)*sig_clip)) & (new_spec < (med_spec-sig_clip*noise)) )[0]
		if len(nan_spec)>0:
			inan = np.unique(np.concatenate([nan_spec]))
			buffer = 0
			inan_buffer_upp = np.array([(i+buffer) for i in inan if (i+buffer) < len(spec)],dtype=int)
			inan_buffer_low = np.array([(i-buffer) for i in inan if (i-buffer) > 0],dtype=int)
			inan = np.concatenate([inan,inan_buffer_low, inan_buffer_upp])
			# Interpolate over nans and infs if in spec
			new_spec[inan] = np.nan
			new_spec = insert_nan(new_spec,inan)
			nans, x= nan_helper(new_spec)
			new_spec[nans]= np.interp(x(nans), x(~nans), new_spec[~nans])
		else:
			break
	#
	return new_spec


##################################################################################

#### Prepare SDSS spectrum #######################################################

def prepare_sdss_spec(fits_file,fit_reg,mask_bad_pix,mask_emline,user_mask,mask_metal,cosmology,run_dir,verbose=True,plot=False):
	"""
	Adapted from example from Cappellari's pPXF (Cappellari et al. 2004,2017)
	Prepare an SDSS spectrum for pPXF, returning all necessary 
	parameters. 
	"""

	# Load the data
	hdu = fits.open(fits_file)
	header_cols = [i.keyword for i in hdu[0].header.cards]
	# Retrieve redshift from spectrum file (specobj table)
	specobj = hdu[2].data
	z = specobj['z'][0]

	# For featureless objects, we force z = 0
	# fit_reg = (0,20000)

	# Retrieve RA and DEC from spectrum file
	# if RA and DEC not present, assume an average Galactic E(B-V)
	if ("RA" in header_cols) and ("DEC" in header_cols):
		ra  = hdu[0].header['RA']
		dec = hdu[0].header['DEC']
		ebv_corr = True
	else:
		ebv_corr = False

	# t = hdu['COADD'].data
	t = hdu[1].data
	hdu.close()

	# Only use the wavelength range in common between galaxy and stellar library.
	# Determine limits of spectrum vs templates
	# mask = ( (t['loglam'] > np.log10(3540)) & (t['loglam'] < np.log10(7409)) )
	fit_min,fit_max = float(fit_reg[0]),float(fit_reg[1])
	# mask = ( ((t['loglam']) >= np.log10(fit_min*(1+z))) & ((t['loglam']) <= np.log10(fit_max*(1+z))) )

	def generate_mask(fit_min, fit_max, lam):
		"""
		This function generates a mask that includes all
		channnels *including* the user-input fit_min and fit_max.
		"""
		# Get lower limit
		low, low_idx = find_nearest(lam, fit_min) 
		if (low > fit_min) & (low_idx!=0):
			low_idx -= 1
		low_val, _ = find_nearest(lam, lam[low_idx])
		# Get upper limit
		upp, upp_idx = find_nearest(lam, fit_max) 
		if (upp < fit_max) & (upp_idx == len(lam)): 
			upp_idx += 1
		upp_val, _ = find_nearest(lam, lam[upp_idx])

		mask = ( ( ((10**t['loglam'])/(1+z)) >= low_val) & ( ((10**t['loglam'])/(1+z)) <= upp_val) )
		return mask

	mask = generate_mask(fit_min, fit_max, (10**t['loglam'])/(1+z) )
	
	# Unpack the spectra
	galaxy = t['flux'][mask]
	# SDSS spectra are already log10-rebinned
	loglam_gal = t['loglam'][mask] # This is the observed SDSS wavelength range, NOT the rest wavelength range of the galaxy
	lam_gal = 10**loglam_gal
	ivar = t['ivar'][mask] # inverse variance
	noise = np.sqrt(1.0/ivar) # 1-sigma spectral noise
	and_mask = t['and_mask'][mask] # bad pixels 
	bad_pix  = np.where(and_mask!=0)[0]

	### Interpolating over bad pixels ############################

	# Get locations of nan or -inf pixels
	nan_gal   = np.where(~np.isfinite(galaxy))[0]
	nan_noise = np.where(~np.isfinite(noise))[0]
	inan = np.unique(np.concatenate([nan_gal,nan_noise]))
	# Interpolate over nans and infs if in galaxy or noise
	noise[inan] = np.nan
	noise[inan] = 1.0 if all(np.isnan(noise)) else np.nanmedian(noise)

	fit_mask_bad = []
	if mask_bad_pix:
		for b in bad_pix:
			fit_mask_bad.append(b)

	if mask_emline:
		emline_mask_bad = emline_masker(lam_gal,galaxy,noise)
		for b in emline_mask_bad:
			fit_mask_bad.append(b)

	if len(user_mask)>0:
		for i in user_mask:
			ibad = np.where((lam_gal/(1.0+z)>=i[0]) & (lam_gal/(1.0+z)<=i[1]))[0]
			for b in ibad:
				fit_mask_bad.append(b)
	
	if mask_metal:
		# galaxy = interpolate_metal(galaxy,noise)
		metal_mask_bad = metal_masker(lam_gal,galaxy,noise)
		for b in metal_mask_bad:
			fit_mask_bad.append(b)

	fit_mask_bad = np.sort(np.unique(fit_mask_bad))
	fit_mask_good = np.setdiff1d(np.arange(0,len(lam_gal),1,dtype=int),fit_mask_bad)

	###############################################################

	c = 299792.458				  # speed of light in km/s
	frac = lam_gal[1]/lam_gal[0]	# Constant lambda fraction per pixel
	dlam_gal = (frac - 1)*lam_gal   # Size of every pixel in Angstrom
	# print('\n Size of every pixel: %s (A)' % dlam_gal)
	wdisp = t['wdisp'][mask]		# Intrinsic dispersion of every pixel, in pixels units
	fwhm_gal = 2.355*wdisp*dlam_gal # Resolution FWHM of every pixel, in angstroms
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
	lam_gal = lam_gal/(1.0+z)  # Compute approximate restframe wavelength
	fwhm_gal = fwhm_gal/(1.0+z)   # Adjust resolution in Angstrom

	# fwhm_gal = np.full_like(lam_gal,0.0)
	# We pass this interp1d class to the fit_model function to correct for 
	# the instrumental resolution of emission lines in our model
	# fwhm_gal_ftn = interp1d(lam_gal,fwhm_gal,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10)) 

	val,idx = find_nearest(lam_gal,5175)

	################################################################################

	#################### Correct for galactic extinction ##################

	if ebv_corr==True:
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
	elif ebv_corr==False:
		ebv = 0.04 # average Galactic E(B-V)

	galaxy = ccm_unred(lam_gal,galaxy,ebv)

	#######################################################################

	# Write to log
	write_log((fits_file,ra,dec,z,cosmology,fit_min,fit_max,velscale,ebv),'prepare_sdss_spec',run_dir)

	################################################################################

	if plot: 
		prepare_sdss_plot(lam_gal,galaxy,noise,fit_mask_bad,run_dir)
	
	if verbose:

		print('\n')
		print('-----------------------------------------------------------')
		print('{0:<30}{1:<30}'.format(' file:'		   , fits_file.name				  ))
		print('{0:<30}{1:<30}'.format(' SDSS redshift:'  , '%0.5f' % z						  ))
		print('{0:<30}{1:<30}'.format(' fitting region:' , '(%d,%d) [A]' % (fit_reg[0],fit_reg[1])  ))
		print('{0:<30}{1:<30}'.format(' velocity scale:' , '%0.2f [km/s/pixel]' % velscale	  ))
		print('{0:<30}{1:<30}'.format(' Galactic E(B-V):', '%0.3f' % ebv						))
		print('-----------------------------------------------------------')
	################################################################################

	return lam_gal,galaxy,noise,z,ebv,velscale,fwhm_gal,fit_mask_good

##################################################################################

def prepare_sdss_plot(lam_gal,galaxy,noise,ibad,run_dir):
	# Plot the galaxy fitting region
	fig = plt.figure(figsize=(14,4))
	ax1 = fig.add_subplot(1,1,1)
	ax1.step(lam_gal,galaxy,label='Object Fit Region',linewidth=0.5, color='xkcd:bright aqua')
	ax1.step(lam_gal,noise,label='$1\sigma$ Uncertainty',linewidth=0.5,color='xkcd:bright orange')
	ax1.axhline(0.0,color='white',linewidth=0.5,linestyle='--')
	# Plot bad pixels
	if (len(ibad)>0):# and (len(ibad[0])>1):
		bad_wave = [(lam_gal[m],lam_gal[m+1]) for m in ibad if ((m+1)<len(lam_gal))]
		ax1.axvspan(bad_wave[0][0],bad_wave[0][0],alpha=0.25,color='xkcd:lime green',label="bad pixels")
		for i in bad_wave[1:]:
			ax1.axvspan(i[0],i[0],alpha=0.25,color='xkcd:lime green')

	fontsize = 14
	ax1.set_title(r'Fitting Region',fontsize=fontsize)
	ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)',fontsize=fontsize)
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=fontsize)
	ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
	ax1.legend(loc='best')
	plt.tight_layout()
	plt.savefig(run_dir.joinpath('fitting_region.pdf'))
	ax1.clear()
	fig.clear()
	plt.close(fig)
	#
	return

##################################################################################

#### Prepare User Spectrum #######################################################

def prepare_user_spec(fits_file,spec,wave,err,fwhm,z,ebv,fit_reg,mask_emline,user_mask,mask_metal,cosmology,run_dir,verbose=True,plot=True):
	"""
	Prepares user-input spectrum for BADASS fitting.
	"""
	# Normalize the spectrum by the same factor as SDSS
	spec = spec/1.e-17
	err  = err/1.e-17

	# Only use the wavelength range in common between galaxy and stellar library.
	# Determine limits of spectrum vs templates
	# mask = ( (t['loglam'] > np.log10(3540)) & (t['loglam'] < np.log10(7409)) )
	fit_min,fit_max = float(fit_reg[0]),float(fit_reg[1])
	# mask = ( ((t['loglam']) >= np.log10(fit_min*(1+z))) & ((t['loglam']) <= np.log10(fit_max*(1+z))) )

	def generate_mask(fit_min, fit_max, lam):
		"""
		This function generates a mask that includes all
		channnels *including* the user-input fit_min and fit_max.
		"""
		# Get lower limit
		low, low_idx = find_nearest(lam, fit_min) 
		if (low > fit_min) & (low_idx!=0):
			low_idx -= 1
		low_val, _ = find_nearest(lam, lam[low_idx])
		# Get upper limit
		upp, upp_idx = find_nearest(lam, fit_max) 
		if (upp < fit_max) & (upp_idx == len(lam)): 
			upp_idx += 1
		upp_val, _ = find_nearest(lam, lam[upp_idx])

		mask = ( lam >= low_val) & ( lam <= upp_val)
		return mask

	# First, we must log-rebin the linearly-binned input spectrum
	lamRange = (np.min(wave),np.max(wave))
	galaxy, logLam, velscale = log_rebin(lamRange, spec, velscale=None, flux=True)
	noise, _, _ = log_rebin(lamRange, err, velscale=velscale, flux=True)
	lam_gal = np.exp(logLam)	

	mask = generate_mask(fit_min, fit_max, lam_gal/(1+z) )
	
	galaxy = galaxy[mask]
	lam_gal = lam_gal[mask]
	noise = noise[mask]

	### Interpolating over bad pixels ############################

	# Get locations of nan or -inf pixels
	nan_gal   = np.where(~np.isfinite(galaxy))[0]
	nan_noise = np.where(~np.isfinite(noise))[0]
	inan = np.unique(np.concatenate([nan_gal,nan_noise]))
	# Interpolate over nans and infs if in galaxy or noise
	noise[inan] = np.nan
	noise[inan] = 1.0 if all(np.isnan(noise)) else np.nanmedian(noise)

	fit_mask_bad = []
	if mask_emline:
		emline_mask_bad = emline_masker(lam_gal,galaxy,noise)
		for b in emline_mask_bad:
			fit_mask_bad.append(b)

	if len(user_mask)>0:
		for i in user_mask:
			ibad = np.where((lam_gal/(1.0+z)>=i[0]) & (lam_gal/(1.0+z)<=i[1]))[0]
			for b in ibad:
				fit_mask_bad.append(b)
	
	if mask_metal:
		# galaxy = interpolate_metal(galaxy,noise)
		metal_mask_bad = metal_masker(lam_gal,galaxy,noise)
		for b in metal_mask_bad:
			fit_mask_bad.append(b)

	fit_mask_bad = np.sort(np.unique(fit_mask_bad))
	fit_mask_good = np.setdiff1d(np.arange(0,len(lam_gal),1,dtype=int),fit_mask_bad)

	###############################################################

	c = 299792.458				  # speed of light in km/s
	frac = lam_gal[1]/lam_gal[0]	# Constant lambda fraction per pixel
	# print(frac)
	dlam_gal = (frac - 1)*lam_gal   # Size of every pixel in Angstrom
	# print(dlam_gal)
	# # print('\n Size of every pixel: %s (A)' % dlam_gal)
	# print(fwhm/dlam_gal) # FWHM of every pixel in pixels
	# wdisp = t['wdisp'][mask]		# Intrinsic dispersion of every pixel, in pixels units
	# fwhm_gal = 2.355*wdisp*dlam_gal # Resolution FWHM of every pixel, in angstroms
	# velscale = np.log(frac)*c	   # Constant velocity scale in km/s per pixel
	fwhm_gal = np.interp(lam_gal, lam_gal[:-1], fwhm/frac*dlam_gal[:-1])

	velscale = velscale[0]

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

	#################### Correct for galactic extinction ##################

	galaxy = ccm_unred(lam_gal,galaxy,ebv)

	#######################################################################

	# Write to log
	write_log((fits_file,z,cosmology,fit_min,fit_max,velscale,ebv),'prepare_user_spec',run_dir)

	################################################################################

	if plot: 
		prepare_user_plot(lam_gal,galaxy,noise,fit_mask_bad,run_dir)
	
	if verbose:

		print('\n')
		print('-----------------------------------------------------------')
		print('{0:<30}{1:<30}'.format(' file:'			 , fits_file.name			  ))
		print('{0:<30}{1:<30}'.format(' redshift:'	   , '%0.5f' % z						  ))
		print('{0:<30}{1:<30}'.format(' fitting region:' , '(%d,%d) [A]' % (fit_reg[0],fit_reg[1])  ))
		print('{0:<30}{1:<30}'.format(' velocity scale:' , '%0.2f [km/s/pixel]' % velscale	  ))
		print('{0:<30}{1:<30}'.format(' Galactic E(B-V):', '%0.3f' % ebv						))
		print('-----------------------------------------------------------')
	################################################################################
	#
	# fit_mask_good = np.arange(0,len(lam_gal),1,dtype=int)
	#
	return lam_gal,galaxy,noise,z,ebv,velscale,fwhm_gal,fit_mask_good

##################################################################################

def prepare_user_plot(lam_gal,galaxy,noise,ibad,run_dir):
	# Plot the galaxy fitting region
	fig = plt.figure(figsize=(14,4))
	ax1 = fig.add_subplot(1,1,1)
	ax1.step(lam_gal,galaxy,label='Object Fit Region',linewidth=0.5, color='xkcd:bright aqua')
	ax1.step(lam_gal,noise,label='$1\sigma$ Uncertainty',linewidth=0.5,color='xkcd:bright orange')
	ax1.axhline(0.0,color='white',linewidth=0.5,linestyle='--')
	# Plot bad pixels
	if (len(ibad)>0):# and (len(ibad[0])>1):
		bad_wave = [(lam_gal[m],lam_gal[m+1]) for m in ibad if ((m+1)<len(lam_gal))]
		ax1.axvspan(bad_wave[0][0],bad_wave[0][0],alpha=0.25,color='xkcd:lime green',label="bad pixels")
		for i in bad_wave[1:]:
			ax1.axvspan(i[0],i[0],alpha=0.25,color='xkcd:lime green')


	fontsize = 14
	ax1.set_title(r'Fitting Region',fontsize=fontsize)
	ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\mathrm{\AA}$)',fontsize=fontsize)
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=fontsize)
	ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
	ax1.legend(loc='best')
	plt.tight_layout()
	plt.savefig(run_dir.joinpath('fitting_region.pdf'))
	ax1.clear()
	fig.clear()
	plt.close(fig)
	#
	return

##################################################################################

def prepare_ifu_spec(fits_file,fit_reg,mask_bad_pix,mask_emline,user_mask,mask_metal,cosmology,run_dir,verbose=True,plot=False):
	"""
	Adapted from example from Cappellari's pPXF (Cappellari et al. 2004,2017)
	Prepare an SDSS spectrum for pPXF, returning all necessary
	parameters.
	"""

	# Load the data
	hdu = fits.open(fits_file)
	format = hdu[0].header['FORMAT']

	specobj = hdu[2].data
	z = specobj['z'][0]
	try:
		ra  = hdu[0].header['RA']
		dec = hdu[0].header['DEC']
	except:
		ra = specobj['PLUG_RA'][0]
		dec = specobj['PLUG_DEC'][0]

	binnum = hdu[0].header['BINNUM']
	spaxelx = hdu[3].data['spaxelx']
	spaxely = hdu[3].data['spaxely']

	# t = hdu['COADD'].data
	t = hdu[1].data
	hdu.close()

	co = coordinates.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
	try:
		table = IrsaDust.get_query_table(co, section='ebv')
		ebv = table['ext SandF mean'][0]
	except:
		ebv = 0.04  # average Galactic E(B-V)
	# If E(B-V) is large, it can significantly affect normalization of the
	# spectrum, in addition to changing its shape.  Re-normalizing the spectrum
	# throws off the maximum likelihood fitting, so instead of re-normalizing,
	# we set an upper limit on the allowed ebv value for Galactic de-reddening.
	if (ebv >= 1.0):
		ebv = 0.04  # average Galactic E(B-V)

	if format != 'MANGA':
		lam_gal,galaxy,noise,z,ebv,velscale,fwhm_gal,fit_mask_good = prepare_user_spec(fits_file,t['flux']*1e-17,10**t['loglam'],np.sqrt(1.0/t['ivar'])*1e-17,t['fwhm_res'][len(t['fwhm_res'])//2],z,ebv,fit_reg,
																					   mask_emline,user_mask,mask_metal,cosmology,run_dir,verbose=verbose,plot=plot)

		return lam_gal,galaxy,noise,z,ebv,velscale,fwhm_gal,fit_mask_good,binnum,spaxelx,spaxely


	# Only use the wavelength range in common between galaxy and stellar library.
	# Determine limits of spectrum vs templates
	# mask = ( (t['loglam'] > np.log10(3540)) & (t['loglam'] < np.log10(7409)) )
	fit_min, fit_max = float(fit_reg[0]), float(fit_reg[1])

	# mask = ( ((t['loglam']) >= np.log10(fit_min*(1+z))) & ((t['loglam']) <= np.log10(fit_max*(1+z))) )

	def generate_mask(fit_min, fit_max, lam):
		"""
		This function generates a mask that includes all
		channnels *including* the user-input fit_min and fit_max.
		"""
		# Get lower limit
		low, low_idx = find_nearest(lam, fit_min)
		if (low > fit_min) & (low_idx != 0):
			low_idx -= 1
		low_val, _ = find_nearest(lam, lam[low_idx])
		# Get upper limit
		upp, upp_idx = find_nearest(lam, fit_max)
		if (upp < fit_max) & (upp_idx == len(lam)):
			upp_idx += 1
		upp_val, _ = find_nearest(lam, lam[upp_idx])

		mask = ((((10 ** t['loglam']) / (1 + z)) >= low_val) & (((10 ** t['loglam']) / (1 + z)) <= upp_val))
		return mask

	mask = generate_mask(fit_min, fit_max, (10 ** t['loglam']) / (1 + z))

	# Unpack the spectra
	galaxy = t['flux'][mask]
	# SDSS spectra are already log10-rebinned
	loglam_gal = t['loglam'][mask]  # This is the observed SDSS wavelength range, NOT the rest wavelength range of the galaxy
	lam_gal = 10 ** loglam_gal
	ivar = t['ivar'][mask]  # inverse variance
	noise = np.sqrt(1.0/ivar)  # 1-sigma spectral noise
	and_mask = t['and_mask'][mask]  # bad pixels
	bad_pix = np.where(and_mask != 0)[0]

	### Interpolating over bad pixels ############################

	# Get locations of nan or -inf pixels
	nan_gal = np.where(galaxy / galaxy != 1)[0]
	nan_noise = np.where(noise / noise != 1)[0]
	inan = np.unique(np.concatenate([nan_gal, nan_noise]))
	# Interpolate over nans and infs if in galaxy or noise
	noise[inan] = np.nan
	noise[inan] = 1.0 if all(np.isnan(noise)) else np.nanmedian(noise)

	fit_mask_bad = []
	if mask_bad_pix:
		for b in bad_pix:
			fit_mask_bad.append(b)

	if mask_emline:
		emline_mask_bad = emline_masker(lam_gal, galaxy, noise)
		for b in emline_mask_bad:
			fit_mask_bad.append(b)

	if len(user_mask) > 0:
		for i in user_mask:
			ibad = np.where((lam_gal / (1.0 + z) >= i[0]) & (lam_gal / (1.0 + z) <= i[1]))[0]
			for b in ibad:
				fit_mask_bad.append(b)

	if mask_metal:
		# galaxy = interpolate_metal(galaxy,noise)
		metal_mask_bad = metal_masker(lam_gal, galaxy, noise)
		for b in metal_mask_bad:
			fit_mask_bad.append(b)

	fit_mask_bad = np.sort(np.unique(fit_mask_bad))
	fit_mask_good = np.setdiff1d(np.arange(0, len(lam_gal), 1, dtype=int), fit_mask_bad)

	###############################################################

	c = 299792.458  # speed of light in km/s
	frac = lam_gal[1] / lam_gal[0]  # Constant lambda fraction per pixel
	# dlam_gal = (frac - 1) * lam_gal  # Size of every pixel in Angstrom
	# print('\n Size of every pixel: %s (A)' % dlam_gal)
	# wdisp = t['wdisp'][mask]  # Intrinsic dispersion of every pixel, in pixels units
	# fwhm_gal = 2.355 * wdisp * dlam_gal  # Resolution FWHM of every pixel, in angstroms
	fwhm_gal = t['fwhm_res'][mask]
	velscale = np.log(frac) * c  # Constant velocity scale in km/s per pixel

	# If the galaxy is at significant redshift, one should bring the galaxy
	# spectrum roughly to the rest-frame wavelength, before calling pPXF
	# (See Sec2.4 of Cappellari 2017). In practice there is no
	# need to modify the spectrum in any way, given that a red shift
	# corresponds to a linear shift of the log-rebinned spectrum.
	# One just needs to compute the wavelength range in the rest-frame
	# and adjust the instrumental resolution of the galaxy observations.
	# This is done with the following three commented lines:
	#
	lam_gal = lam_gal / (1.0 + z)  # Compute approximate restframe wavelength
	fwhm_gal = fwhm_gal / (1.0 + z)  # Adjust resolution in Angstrom

	# fwhm_gal = np.full_like(lam_gal,0.0)
	# We pass this interp1d class to the fit_model function to correct for
	# the instrumental resolution of emission lines in our model
	# fwhm_gal_ftn = interp1d(lam_gal,fwhm_gal,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))

	val, idx = find_nearest(lam_gal, 5175)

	################################################################################

	#################### Correct for galactic extinction ##################

	galaxy = ccm_unred(lam_gal, galaxy, ebv)

	#######################################################################

	# Write to log
	write_log((fits_file, ra, dec, z, cosmology, fit_min, fit_max, velscale, ebv), 'prepare_sdss_spec', run_dir)

	################################################################################

	if plot:
		prepare_sdss_plot(lam_gal, galaxy, noise, fit_mask_bad, run_dir)

	if verbose:
		print('\n')
		print('-----------------------------------------------------------')
		print('{0:<30}{1:<30}'.format(' file:'		, fits_file.name				  ))
		print('{0:<30}{1:<30}'.format(' SDSS redshift:'  , '%0.5f' % z						  ))
		print('{0:<30}{1:<30}'.format(' fitting region' , '(%d,%d) [A]' % (fit_reg[0 ],fit_reg[1])  ))
		print('{0:<30}{1:<30}'.format(' velocity scale' , '%0.2f [km/s/pixel]' % velscale	  ))
		print('{0:<30}{1:<30}'.format(' Galactic E(B-V):', '%0.3f' % ebv						))
		print('-----------------------------------------------------------')
	################################################################################

	return lam_gal,galaxy,noise,z,ebv,velscale,fwhm_gal,fit_mask_good,binnum,spaxelx,spaxely

##################################################################################

# Alias function
prepare_ifu_plot = prepare_sdss_plot

#### Prepare stellar templates ###################################################

def prepare_stellar_templates(galaxy, lam_gal, fit_reg, velscale, fwhm_gal,fit_mask, losvd_options, run_dir):
	"""
	Prepares stellar templates for convolution using pPXF. 
	This example is from Capellari's pPXF examples, the code 
	for which can be found here: https://www-astro.physics.ox.ac.uk/~mxc/.
	"""
	# Stellar template directory
	if (losvd_options["library"]=="IndoUS"):
		temp_dir  = "badass_data_files/IndoUS/"
		fwhm_temp = 1.35 # Indo-US Template Library FWHM in Å (linear)
	if (losvd_options["library"]=="Vazdekis2010"):
		temp_dir  = "badass_data_files/Vazdekis2010/"
		fwhm_temp = 2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A (linear)
	if (losvd_options["library"]=="eMILES"):
		temp_dir  = "badass_data_files/eMILES/"
		fwhm_temp = 2.51 # eMILES spectra have a constant resolution FWHM of 2.51A (linear)

	fit_min,fit_max = float(fit_reg[0]),float(fit_reg[1])
	#
	# Get a list of templates stored in temp_dir.  We only include 50 stellar 
	# templates of various spectral type from the Indo-US Coude Feed Library of 
	# Stellar templates (https://www.noao.edu/cflib/).  We choose this library
	# because it is (1) empirical, (2) has a broad wavelength range with 
	# minimal number of gaps, and (3) is at a sufficiently high resolution (~1.35 Å)
	# such that we can probe as high a redshift as possible with the SDSS.  It may 
	# be advantageous to use a different stellar template library (such as the MILES 
	# library) depdending on the science goals.  BADASS only uses pPXF to measure stellar
	# kinematics (i.e, stellar velocity and dispersion), and does NOT compute stellar 
	# population ages. 
	temp_list = natsort.natsorted(glob.glob(temp_dir + '/*.fits') )#
	# Extract the wavelength range and logarithmically rebin one spectrum
	# to the same velocity scale of the input galaxy spectrum, to determine
	# the size needed for the array which will contain the template spectra.
	#
	hdu = fits.open(temp_list[0])
	ssp = hdu[0].data
	h2 = hdu[0].header
	hdu.close()

	lam_temp = np.array(h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1']))
	# By cropping the templates we save some fitting time
	mask_temp = ( (lam_temp > (fit_min-100.)) & (lam_temp < (fit_max+100.)) )
	ssp = ssp[mask_temp]
	lam_temp = lam_temp[mask_temp]

	lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]

	sspNew = log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
	templates = np.empty((sspNew.size, len(temp_list)))

	# Interpolates the galaxy spectral resolution at the location of every pixel
	# of the templates. Outside the range of the galaxy spectrum the resolution
	# will be extrapolated, but this is irrelevant as those pixels cannot be
	# used in the fit anyway.
	if isinstance(fwhm_gal,(list,np.ndarray)):
		fwhm_gal_interp = np.interp(lam_temp, lam_gal, fwhm_gal)
	elif isinstance(fwhm_gal,(int,float)):
		fwhm_gal_interp = np.full_like(lam_temp,fwhm_gal)

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
	fwhm_dif = np.sqrt((fwhm_gal_interp**2 - fwhm_temp**2).clip(0))
	sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

	for j, fname in enumerate(temp_list):
		hdu = fits.open(fname)
		ssp = hdu[0].data
		ssp = ssp[mask_temp]
		ssp = gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
		sspNew,loglam_temp,velscale_temp = log_rebin(lamRange_temp, ssp, velscale=velscale)#[0]
		templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
		hdu.close()
	
	# The galaxy and the template spectra do not have the same starting wavelength.
	# For this reason an extra velocity shift DV has to be applied to the template
	# to fit the galaxy spectrum. We remove this artificial shift by using the
	# keyword VSYST in the call to PPXF below, so that all velocities are
	# measured with respect to DV. This assume the redshift is negligible.
	# In the case of a high-redshift galaxy one should de-redshift its
	# wavelength to the rest frame before using the line below (see above).
	#
	c = 299792.458 # speed of light in km/s
	vsyst = np.log(lam_temp[0]/lam_gal[0])*c	# km/s

	npix = galaxy.shape[0] # number of output pixels
	ntemp = np.shape(templates)[1]# number of templates
	
	# Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
	temp_fft,npad = template_rfft(templates) # we will use this throughout the code

	# If vel_const AND disp_const are True, there is no need to convolve during the 
	# fit, so we perform the convolution here and pass the convolved templates to fit_model.
	if (losvd_options["vel_const"]["bool"]==True) & (losvd_options["disp_const"]["bool"]==True):
		stel_vel  = losvd_options["vel_const"]["val"]
		stel_disp = losvd_options["disp_const"]["val"]

		conv_temp	= convolve_gauss_hermite(temp_fft,npad,float(velscale),\
					   [stel_vel, stel_disp],np.shape(lam_gal)[0],velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
		stel_templates = conv_temp

	# If vel_const OR disp_const is False, do not perform the convolution.
	# Package the stellar templates, vsyst, and npad (everything needed for convolution)
	# into a tuple called stel_templates, to be used in fit_model()
	elif (losvd_options["vel_const"]["bool"]==False) | (losvd_options["disp_const"]["bool"]==False):
		stel_templates = (temp_fft, npad, vsyst)

	##############################################################################

	return stel_templates

##################################################################################

#### Initialize Parameters #######################################################


def initialize_pars(lam_gal,galaxy,noise,fit_reg,fwhm_gal,fit_mask_good,velscale,
					comp_options,user_lines,user_constraints,combined_lines,losvd_options,host_options,power_options,opt_feii_options,uv_iron_options,balmer_options,
					run_dir,fit_type='init',fit_stat="RCHI2",
					fit_opt_feii=True,fit_uv_iron=True,fit_balmer=True,
					fit_losvd=False,fit_host=True,fit_power=True,
					fit_narrow=True,fit_broad=True,fit_outflow=True,fit_absorp=True,
					tie_line_fwhm=False,tie_line_voff=False,remove_lines=False,verbose=True):
	"""
	Initializes all free parameters for the fit based on user input and options.
	"""
	# Issue warnings for dumb options
	if ((fit_narrow==False) & (fit_outflow==True)): # why would you fit outflow without narrow lines?
		raise ValueError('\n Why would you fit outflows without narrow lines? Turn on narrow line component! \n')

	################################################################################
	# Initial conditions for some parameters
	max_flux	= np.max(galaxy)
	median_flux = np.median(galaxy)

	# Padding on the edges; any line(s) within this many angstroms is omitted
	# from the fit so problems do not occur with the fit
	edge_pad = 10.0

	def get_init_amp(line_center):
		line_center = float(line_center)
		try:
			return (np.max(galaxy[(lam_gal>line_center-10.) & (lam_gal<line_center+10.)]))
		except ValueError:
			return 0.0


	################################################################################
	
	par_input = {} # initialize an empty dictionary to store free parameter dicts

	#### Stellar component/Host Galaxy #############################################

	# # Fit statistic: add noise_unexp if fit_stat = "RCHI2"
	if (fit_stat=="RCHI2"):
		if verbose: 
			print('	 - Adding parameter for unexplained noise to fit reduced Chi-squared.')
		par_input["NOISE_SCALE"] = ({'init':1.0,
										   'plim':(0.0001,10.0),
									})

	# Galaxy template amplitude
	if (fit_host==True):
		if verbose:
			print('	 - Fitting a SSP host-galaxy template.')
		#
		if len(host_options["age"])==1:
			par_input['HOST_TEMP_AMP'] = ({'init':0.5*median_flux,
										   'plim':(0,max_flux),
										  })
		#
		if host_options["vel_const"]["bool"]==False:
			#
			par_input['HOST_TEMP_VEL'] = ({'init':0.0,
									   'plim':(-500.0,500),
									  })
		#
		if host_options["disp_const"]["bool"]==False:
			#
			par_input['HOST_TEMP_DISP'] = ({'init':100.0,
									   'plim':(0.001,500.0),
									  })


	# Stellar LOSVD parameters (if fit_LOSVD = True)
	if (fit_losvd==True):
		if verbose:
			print('	 - Fitting the stellar LOSVD.')
		# Stellar velocity
		if losvd_options["vel_const"]["bool"]==False:
			#
			par_input['STEL_VEL'] = ({'init':100. ,
							   		  'plim':(-500.,500.),
							   		 })
		# Stellar velocity dispersion
		if losvd_options["disp_const"]["bool"]==False:
			#
			par_input['STEL_DISP'] = ({'init':150.0,
							   			'plim':(0.001,500.),
							   		  })
		if (losvd_options['losvd_apoly']['bool']==True) and (losvd_options['losvd_apoly']['order']>=1):
			if verbose:
				print('	 		* including %d-order additive Legendre polynomial' % losvd_options['losvd_apoly']['order'])
			for n in range(int(losvd_options['losvd_apoly']['order'])+1):
				par_input["LOSVD_LCOEFF_%d" % n] = ({'init'  :0.0,
							   			  'plim'  :(-1.0e6,1.0e6),
							   			  })

	##############################################################################

	#### Simple Power-Law (AGN continuum) ########################################
	if (fit_power==True) & (power_options['type']=='simple'):
		if verbose:
			print('	 - Fitting Simple AGN power-law continuum.')
		# AGN simple power-law amplitude
		par_input['POWER_AMP'] = ({'init':(0.5*median_flux),
						   		   'plim':(0,max_flux),
						   		   })
		# AGN simple power-law slope
		par_input['POWER_SLOPE'] = ({'init':-1.0  ,
						   			 'plim':(-6.0,6.0),
						   			 })
		
	#### Smoothly-Broken Power-Law (AGN continuum) ###############################
	if (fit_power==True) & (power_options['type']=='broken'):
		if verbose:
			print('	 - Fitting Smoothly-Broken AGN power-law continuum.')
		# AGN simple power-law amplitude
		par_input['POWER_AMP'] = ({'init':(0.5*median_flux),
						   		   'plim':(0,max_flux),
						   		   })
		# AGN simple power-law break wavelength
		par_input['POWER_BREAK'] = ({'init':(np.max(lam_gal) - (0.5*(np.max(lam_gal)-np.min(lam_gal)))),
						   			 'plim':(np.min(lam_gal), np.max(lam_gal)),
						   			})
		# AGN simple power-law slope 1 (blue side)
		par_input['POWER_SLOPE_1'] = ({'init':-1.0  ,
						   			   'plim':(-6.0,6.0),
						   			 })
		# AGN simple power-law slope 2 (red side)
		par_input['POWER_SLOPE_2'] = ({'init':-1.0  ,
						   			   'plim':(-6.0,6.0),
						   			 })
		# Power-law curvature parameter (Delta)
		par_input['POWER_CURVATURE'] = ({'init':0.10,
						   				 'plim':(0.01,1.0),
						   			 })
		
	##############################################################################

	#### Optical FeII Templates ##################################################
	if (fit_opt_feii==True) & (opt_feii_options['opt_template']['type']=='VC04'):
		# Veron-Cerry et al. 2004 2-8 Parameter FeII template
		if verbose:
			print('	 - Fitting broad and narrow optical FeII using Veron-Cetty et al. (2004) optical FeII templates')
		if (opt_feii_options['opt_amp_const']['bool']==False):
			if verbose:
				print('	 		* varying optical FeII amplitudes')
			# Narrow FeII amplitude
			par_input['NA_OPT_FEII_AMP'] = ({'init'  :0.1*median_flux,
							   				 'plim'  :(0,max_flux),
							   			  })
			# Broad FeII amplitude
			par_input['BR_OPT_FEII_AMP'] = ({'init'  :0.1*median_flux,
							   				 'plim'  :(0,max_flux),
							   			  })
		if (opt_feii_options['opt_fwhm_const']['bool']==False):
			if verbose:
				print('	 		* varying optical FeII fwhm')
			# Narrow FeII FWHM
			par_input['NA_OPT_FEII_FWHM'] = ({'init'  :500.0,
							   			  	  'plim'  :(100.0,1000.0),
							   			  })
			# Broad FeII FWHM
			par_input['BR_OPT_FEII_FWHM'] = ({'init'  :3000.0,
							   			  	  'plim'  :(1000.0,10000.0),
							   			  })
		if (opt_feii_options['opt_voff_const']['bool']==False):
			if verbose:
				print('	 		* varying optical FeII voff')
			# Narrow FeII VOFF
			par_input['NA_OPT_FEII_VOFF'] = ({'init'  :0.0,
							   			  	  'plim'  :(-1000.0,1000.0),
							   			  })
			# Broad FeII VOFF
			par_input['BR_OPT_FEII_VOFF'] = ({'init'  :0.0,
							   			  	  'plim'  :(-2000.0,2000.0),
							   			  })
	elif (fit_opt_feii==True) & (opt_feii_options['opt_template']['type']=='K10'):
		if verbose:
			print('	 - Fitting optical FeII template from Kovacevic et al. (2010)')

		# Kovacevic et al. 2010 7-parameter FeII template (for NLS1s and BAL QSOs)
		# Consits of 7 free parameters
		#	- 4 amplitude parameters for S,F,G,IZw1 line families
		#	- 1 Temperature parameter determines relative intensities (5k-15k Kelvin)
		#	- 1 FWHM parameter
		#	- 1 VOFF parameter
		# 	- all lines modeled as Gaussians
		# Narrow FeII amplitude
		if (opt_feii_options['opt_amp_const']['bool']==False):
			par_input['OPT_FEII_F_AMP'] = ({'init'  :0.1*median_flux,
							   			 	'plim'  :(0,max_flux),
							   			})
			par_input['OPT_FEII_S_AMP'] = ({'init'  :0.1*median_flux,
							   			 	'plim'  :(0,max_flux),
							   			})
			par_input['OPT_FEII_G_AMP'] = ({'init'  :0.1*median_flux,
							   			 	'plim'  :(0,max_flux),
							   			})
			par_input['OPT_FEII_Z_AMP'] = ({'init'  :0.1*median_flux,
							   				'plim'  :(0,max_flux),
							   			})
		if (opt_feii_options['opt_fwhm_const']['bool']==False):
			# FeII FWHM
			par_input['OPT_FEII_FWHM'] = ({'init'  :1000.0,
							   			   'plim'  :(100.0,5000.0),
							   		   })
		if (opt_feii_options['opt_voff_const']['bool']==False):
			# Narrow FeII amplitude
			par_input['OPT_FEII_VOFF'] = ({'init'  :0.0,
							   			   'plim'  :(-1000.0,1000.0),
							   		   })
		if (opt_feii_options['opt_temp_const']['bool']==False):
			par_input['OPT_FEII_TEMP'] = ({'init'  :10000.0,
							   		 	   'plim'  :(2000.0,25000.0),
							   			})

	##############################################################################

	#### UV Iron Template ########################################################
	if (fit_uv_iron==True):
		# Veron-Cerry et al. 2004 2-8 Parameter FeII template
		if verbose:
			print('	 - Fitting UV iron emission using Vestergaard & Wilkes (2001) UV iron template')
		if (uv_iron_options['uv_amp_const']['bool']==False):
			if verbose:
				print('	 		* varying UV iron amplitudes')
			# Narrow FeII amplitude
			par_input['UV_IRON_AMP'] = ({'init'  :0.1*median_flux,
							   			 'plim'  :(0,max_flux),
							   			  })

		if (uv_iron_options['uv_fwhm_const']['bool']==False):
			if verbose:
				print('	 		* varying UV iron fwhm')
			# Narrow FeII FWHM
			par_input['UV_IRON_FWHM'] = ({'init'  :3000.0,
							   			  'plim'  :(1000.0,20000.0),
							   			  })

		if (uv_iron_options['uv_voff_const']['bool']==False):
			if verbose:
				print('	 		* varying UV iron voff')
			# Narrow FeII VOFF
			par_input['UV_IRON_VOFF'] = ({'init'  :0.0,
							   			  'plim'  :(-2000.0,2000.0),
							   			  })

		if (uv_iron_options['uv_legendre_p']['bool']==True) and (uv_iron_options['uv_legendre_p']['uv_iron_val']>=1):
			if verbose:
				print('	 		* including %d-order additive Legendre polynomial' % uv_iron_options['uv_legendre_p']['uv_iron_val'])
			for n in range(int(uv_iron_options['uv_legendre_p']['uv_iron_val'])+1):
				par_input["UV_IRON_LCOEFF_%d" % n] = ({'init'  :0.0,
							   			  'plim'  :(-1.0e6,1.0e6),
							   			  })


	##############################################################################

	#### Balmer Continuum ########################################################


	if (fit_balmer==True):
		# Balmer continuum following Kovacevic et al. (2014) and Calderone et al. (2017; QSFit)
		if verbose:
			print('	 - Fitting Balmer Continuum')

		if (balmer_options['R_const']['bool']==False):
			if verbose:
				print('	 		* varying Balmer ratio')
			# Balmer continuum ratio
			par_input['BALMER_RATIO'] = ({'init'  :10.0,
							   			  'plim'  :(0.0,100.0),
							   			  })

		if (balmer_options['balmer_amp_const']['bool']==False):
			if verbose:
				print('	 		* varying Balmer amplitude')
			# Balmer continuum amplitude
			par_input['BALMER_AMP'] = ({'init'  :0.1*median_flux,
							   			'plim'  :(0,max_flux),
							   			  })

		if (balmer_options['balmer_fwhm_const']['bool']==False):
			if verbose:
				print('	 		* varying Balmer fwhm')
			# Balmer continuum FWHM
			par_input['BALMER_FWHM'] = ({'init'  :5000.0,
							   			 'plim'  :(1000.0,25000.0),
							   			  })

		if (balmer_options['balmer_voff_const']['bool']==False):
			if verbose:
				print('	 		* varying Balmer voff')
			# Balmer continuum VOFF
			par_input['BALMER_VOFF'] = ({'init'  :0.0,
							   			 'plim'  :(-2000.0,2000.0),
							   			  })

		if (balmer_options['Teff_const']['bool']==False):
			if verbose:
				print('	 		* varying Balmer effective temperature')
			# Balmer continuum effective temperature
			par_input['BALMER_TEFF'] = ({'init'  :15000.0,
							   			 'plim'  :(1000.0,50000.0),
							   			  })

		if (balmer_options['tau_const']['bool']==False):
			if verbose:
				print('	 		* varying Balmer optical depth')
			# Balmer continuum optical depth
			par_input['BALMER_TAU'] = ({'init'  :1.0,
							   			'plim'  :(0,1.0),
							   			  })

	#### Emission Lines ##########################################################
	

	#
	if (user_lines is None) or (len(user_lines)==0):
		line_list = line_list_default()
	elif user_lines is not None:
		line_list = user_lines

	# Remove lines
	if remove_lines:
		# if len(remove_lines)==1:
		# 	line_list.pop(remove_lines,None)
		# elif len(remove_lines)>1:
		for l in remove_lines:
			line_list.pop(l,None)

	# Check line component options for 
	line_list = check_line_comp_options(lam_gal,line_list,comp_options,edge_pad=edge_pad,verbose=verbose)
	# Add the FWHM resolution and central pixel locations for each line so we don't have to 
	# find them during the fit.
	line_list = add_fwhm_res(line_list,lam_gal,fwhm_gal,velscale,verbose=verbose)
	# Generate line free parameters based on input line_list
	line_par_input = initialize_line_pars(lam_gal,galaxy,comp_options,line_list,verbose=verbose)
	# Check hard line constraints; returns updated line_list and line_par_input
	line_list, line_par_input = check_hard_cons(lam_gal,galaxy,comp_options,line_list,line_par_input,par_input,verbose=verbose)
	# Append line_par_input to par_input
	par_input = {**par_input, **line_par_input}

	##############################################################################

	# Create combined_line_list
	# A list of combined lines, and some relevant information
	if combined_lines is not None:
		combined_line_list = generate_comb_line_list(combined_lines,line_list)
	elif (combined_lines is None) or (len(combined_lines)==0):
		combined_line_list = generate_comb_line_list({},line_list)

	##############################################################################

	# Check soft-constraints
	# Default soft constraints
	# Soft constraints: If you want to vary a free parameter relative to another free parameter (such as
	# requiring that broad lines have larger widths than narrow lines), these are called "soft" constraints,
	# or "inequality" constraints. 
	# These are passed through a separate list of tuples which are used by the maximum likelihood constraints 
	# and prior constraints by emcee.  Soft constraints have a very specific format following
	# the scipy optimize SLSQP syntax: 
	#			
	#				(parameter1 - parameter2) >= 0.0 OR (parameter1 >= parameter2)
	#

	if user_constraints is not None:
		soft_cons = user_constraints
	if (user_constraints is None) or (len(user_constraints)==0):

		soft_cons = [
			("BR_H_BETA_FWHM","NA_OIII_5007_FWHM"),
			("BR_H_BETA_FWHM","OUT_OIII_5007_FWHM"),
			#
			("OUT_OIII_5007_FWHM","NA_OIII_5007_FWHM"),
			#
			("NA_OIII_5007_AMP","NA_H_BETA_AMP"),
			("NA_OIII_5007_AMP","OUT_OIII_5007_AMP"),
			#

		]


	# Append any user constraints
	# for u in user_constraints:
	# 	soft_cons.append(tuple(u))

	soft_cons = check_soft_cons(soft_cons,par_input,verbose=verbose)

	return par_input, line_list, combined_line_list, soft_cons

##################################################################################

#### Line List ###################################################################

def generate_comb_line_list(combined_lines,line_list):
	#
	combined_line_list={}
	valid_lines = [i for i in line_list]
	for comb_line in combined_lines:
		# Check to make sure lines are in line list
		if np.all([True if i in valid_lines else False for i in combined_lines[comb_line] ]):
			all_line_profiles = [line_list[i]["line_profile"] for i in combined_lines[comb_line] ]
			if ("V" in all_line_profiles) or ("L" in all_line_profiles):
				line_profile = "V"
			else:
				line_profile = "G"
			combined_line_list[comb_line] = {"lines":combined_lines[comb_line],
											 "center":line_list[combined_lines[comb_line][0]]["center"],
											 "center_pix":line_list[combined_lines[comb_line][0]]["center_pix"],
											 "fwhm_res_kms":line_list[combined_lines[comb_line][0]]["fwhm_res_kms"],
											 "line_profile":line_profile,
											}
	#
	return combined_line_list


def line_list_default():
	"""
	Below we define the "default" emission lines in BADASS.  
	
	The easiest way to disable any particular line is to simply comment out the line of interest.
		
	There are five types of line: Narrow, Broad, Outflow, Absorption, and User.  The Narrow, Broad, 
	Outflow, and Absorption lines are built into BADASS, whereas the User lines are added on the 
	front-end Jupyter interface.  
	
	Hard constraints: if you want to hold a parameter value to a constant scalar value, or to the 
	value of another parameter, this is called a "hard" constraint, because the parameter is no 
	longer free, help to a specific value.  To implement a hard constraint, BADASS parses string 
	input from the amp, fwhm, voff, h3, h4, and shape keywords for each line.  Be warned, however, 
	to tie a parameter to another paramter, requires you to know the name of the parameter in question. 
	If BADASS encounters an error in parsing hard constraint string input, it will automatically convert
	the paramter to a "free" parameter instead of raising an error.
	"""
	# Default narrow lines
	narrow_lines ={

		### Region 8 (< 2000 Å)
		"NA_LY_ALPHA"  :{"center":1215.240, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"na"},
		"NA_CIV_1549"  :{"center":1549.480, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"na"},
		"NA_CIII_1908" :{"center":1908.734, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"na"},

		##############################################################################################################################################################################################################################################

		### Region 7 (2000 Å - 3500 Å)
		"NA_MGII_2799" :{"center":2799.117, "amp":"free", "fwhm":"free"				, "voff":"free"			   , "line_type":"na","label":r"Mg II"},
		"NA_HEII_3203" :{"center":3203.100, "amp":"free", "fwhm":"free"				, "voff":"free"			   , "line_type":"na","label":r"He II"},
		"NA_NEV_3346"  :{"center":3346.783, "amp":"free", "fwhm":"free"				, "voff":"free"			   , "line_type":"na","label":r"[Ne V]"},
		"NA_NEV_3426"  :{"center":3426.863, "amp":"free", "fwhm":"NA_NEV_3346_FWHM"	, "voff":"NA_NEV_3346_VOFF", "line_type":"na","label":r"[Ne V]"},

		##############################################################################################################################################################################################################################################

		### Region 6 (3500 Å - 4400 Å):
		"NA_OII_3727"  :{"center":3727.092, "amp":"free", "fwhm":"NA_OII_3729_FWHM"   , "voff":"NA_OII_3729_VOFF"  , "line_type":"na","label":r"[O II]"},
		"NA_OII_3729"  :{"center":3729.875, "amp":"free", "fwhm":"free"				  , "voff":"free"			   , "line_type":"na"},
		"NA_NEIII_3869":{"center":3869.857, "amp":"free", "fwhm":"free"				  , "voff":"free"			   , "line_type":"na","label":r"[Ne III]"}, # Coronal Line
		"NA_HEI_3889"  :{"center":3888.647, "amp":"free", "fwhm":"free"				  , "voff":"free"			   , "line_type":"na","label":r"He I"},
		"NA_NEIII_3968":{"center":3968.593, "amp":"free", "fwhm":"NA_NEIII_3869_FWHM" , "voff":"NA_NEIII_3869_VOFF", "line_type":"na","label":r"[Ne III]"}, # Coronal Line
		"NA_H_DELTA"   :{"center":4102.900, "amp":"free", "fwhm":"NA_H_GAMMA_FWHM"	  , "voff":"NA_H_GAMMA_VOFF"   , "line_type":"na","label":r"H$\delta$"},
		"NA_H_GAMMA"   :{"center":4341.691, "amp":"free", "fwhm":"free" 			  , "voff":"free"			   , "line_type":"na","label":r"H$\gamma$"},
		"NA_OIII_4364" :{"center":4364.436, "amp":"free", "fwhm":"NA_H_GAMMA_FWHM"	  , "voff":"NA_H_GAMMA_VOFF"   , "line_type":"na","label":r"[O III]"},


		##############################################################################################################################################################################################################################################

		### Region 5 (4400 Å - 5500 Å)
		# "NA_HEI_4471"  :{"center":4471.479, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"na","label":r"He I"},
		# "NA_HEII_4687" :{"center":4687.021, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"na","label":r"He II"},

		"NA_H_BETA"	   :{"center":4862.691, "amp":"free"				   , "fwhm":"NA_OIII_5007_FWHM", "voff":"free"			   , "line_type":"na" ,"label":r"H$\beta$"},
		"NA_OIII_4960" :{"center":4960.295, "amp":"(NA_OIII_5007_AMP/2.98)", "fwhm":"NA_OIII_5007_FWHM", "voff":"NA_OIII_5007_VOFF", "line_type":"na" ,"label":r"[O III]"},
		"NA_OIII_5007" :{"center":5008.240, "amp":"free"				   , "fwhm":"free"			   , "voff":"free"			   , "line_type":"na" ,"label":r"[O III]"},

		# "NA_H_BETA"	   :{"center":4862.691, "amp":"free"				   , "fwhm":"STEL_DISP*2.355", "voff":"free"			   , "line_type":"na" ,"label":r"H$\beta$"},
		# "NA_OIII_4960" :{"center":4960.295, "amp":"(NA_OIII_5007_AMP/2.98)", "fwhm":"STEL_DISP*2.355", "voff":"NA_OIII_5007_VOFF"  , "line_type":"na" ,"label":r"[O III]"},
		# "NA_OIII_5007" :{"center":5008.240, "amp":"free"				   , "fwhm":"STEL_DISP*2.355", "voff":"free"			   , "line_type":"na" ,"label":r"[O III]"},


		##############################################################################################################################################################################################################################################

		### Region 4 (5500 Å - 6200 Å)
		"NA_FEVI_5638" :{"center":5637.600, "amp":"free", "fwhm":"NA_FEVI_5677_FWHM" , "voff":"NA_FEVI_5677_VOFF" , "line_type":"na","label":r"[Fe VI]"}, # Coronal Line
		"NA_FEVI_5677" :{"center":5677.000, "amp":"free", "fwhm":"free"				 , "voff":"free"			  , "line_type":"na","label":r"[Fe VI]"}, # Coronal Line
		"NA_FEVII_5720":{"center":5720.700, "amp":"free", "fwhm":"NA_FEVII_6087_FWHM", "voff":"NA_FEVII_6087_VOFF", "line_type":"na","label":r"[Fe VII]"}, # Coronal Line
		"NA_HEI_5876"  :{"center":5875.624, "amp":"free", "fwhm":"free"				 , "voff":"free"			  , "line_type":"na","label":r"He I"},
		"NA_FEVII_6087":{"center":6087.000, "amp":"free", "fwhm":"free"				 , "voff":"free"			  , "line_type":"na","label":r"[Fe VII]"}, # Coronal Line

		##############################################################################################################################################################################################################################################

		### Region 3 (6200 Å - 6800 Å)

		"NA_OI_6302"   :{"center":6302.046, "amp":"free"				, "fwhm":"NA_NII_6585_FWHM" , "voff":"NA_NII_6585_VOFF"	, "line_type":"na","label":r"[O I]"},
		"NA_SIII_6312" :{"center":6312.060, "amp":"free"				, "fwhm":"NA_NII_6585_FWHM" , "voff":"free" , "line_type":"na","label":r"[S III]"},
		"NA_OI_6365"   :{"center":6365.535, "amp":"NA_OI_6302_AMP/3.0"	, "fwhm":"NA_NII_6585_FWHM" , "voff":"NA_NII_6585_VOFF"	, "line_type":"na","label":r"[O I]"},
		"NA_FEX_6374"  :{"center":6374.510, "amp":"free"				, "fwhm":"free"			    , "voff":"free"				, "line_type":"na","label":r"[Fe X]"}, # Coronal Line

		#
		"NA_NII_6549"  :{"center":6549.859, "amp":"NA_NII_6585_AMP/2.93"	, "fwhm":"NA_NII_6585_FWHM", "voff":"NA_NII_6585_VOFF", "line_type":"na","label":r"[N II]"},
		# "NA_H_ALPHA"   :{"center":6564.632, "amp":"free"					, "fwhm":"NA_NII_6585_FWHM", "voff":"NA_NII_6585_VOFF", "line_type":"na","label":r"H$\alpha$"},
		"NA_NII_6585"  :{"center":6585.278, "amp":"free"					, "fwhm":"free"			   , "voff":"free"			  , "line_type":"na","label":r"[N II]"},
		"NA_SII_6718"  :{"center":6718.294, "amp":"free"					, "fwhm":"NA_NII_6585_FWHM", "voff":"NA_NII_6585_VOFF", "line_type":"na","label":r"[S II]"},
		"NA_SII_6732"  :{"center":6732.668, "amp":"free"					, "fwhm":"NA_NII_6585_FWHM", "voff":"NA_NII_6585_VOFF", "line_type":"na","label":r"[S II]"},

		##############################################################################################################################################################################################################################################

		### Region 2 (6800 Å - 8000 Å)
		"NA_HEI_7062"   :{"center":7065.196, "amp":"free", "fwhm":"free"			, "voff":"free"			   , "line_type":"na","label":r"He I"},
		"NA_ARIII_7135" :{"center":7135.790, "amp":"free", "fwhm":"free"			, "voff":"free"			   , "line_type":"na","label":r"[Ar III]"},
		"NA_OII_7319"   :{"center":7319.990, "amp":"free", "fwhm":"NA_OII_7331_FWHM", "voff":"NA_OII_7331_VOFF", "line_type":"na","label":r"[O II]"},
		"NA_OII_7331"   :{"center":7330.730, "amp":"free", "fwhm":"free"			, "voff":"free"			   , "line_type":"na","label":r"[O II]"},
		"NA_NIIII_7890" :{"center":7889.900, "amp":"free", "fwhm":"free"			, "voff":"free"			   , "line_type":"na","label":r"[Ni III]"},
		"NA_FEXI_7892"  :{"center":7891.800, "amp":"free", "fwhm":"free"			, "voff":"free"			   , "line_type":"na","label":r"[Fe XI]"},

		##############################################################################################################################################################################################################################################

		### Region 1 (> 8000 Å)
		"NA_HEII_8236"  :{"center":8236.790, "amp":"free", "fwhm":"free"			 , "voff":"free"			 , "line_type":"na","label":r"He II"},
		"NA_OI_8446"	:{"center":8446.359, "amp":"free", "fwhm":"free"			 , "voff":"free"			 , "line_type":"na","label":r"O I"},
		"NA_FEII_8616"  :{"center":8616.950, "amp":"free", "fwhm":"NA_FEII_8891_FWHM", "voff":"NA_FEII_8891_VOFF", "line_type":"na","label":r"[Fe II]"},
		"NA_FEII_8891"  :{"center":8891.910, "amp":"free", "fwhm":"free"			 , "voff":"free"			 , "line_type":"na","label":r"[Fe II]"},
		"NA_SIII_9069"  :{"center":9068.600, "amp":"free", "fwhm":"NA_SIII_9531_FWHM", "voff":"NA_SIII_9531_VOFF", "line_type":"na","label":r"[S III]"},

		##############################################################################################################################################################################################################################################

	}

	# Default Broad lines
	broad_lines = {
		### Region 8 (< 2000 Å)
		"BR_OVI_1034"  :{"center":1033.820, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"O VI"},
		"BR_LY_ALPHA"  :{"center":1215.240, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"Ly$\alpha$"},
		"BR_NV_1241"   :{"center":1240.810, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"N V"},
		"BR_OI_1305"   :{"center":1305.530, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"O I"},
		"BR_CII_1335"  :{"center":1335.310, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"C II"},
		"BR_SIIV_1398" :{"center":1397.610, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"Si IV + O IV"},
		"BR_SIIV+OIV"  :{"center":1399.800, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"Si IV + O IV"},
		"BR_CIV_1549"  :{"center":1549.480, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"C IV"},
		"BR_HEII_1640" :{"center":1640.400, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"He II"},
		"BR_CIII_1908" :{"center":1908.734, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"C III]"},

		### Region 7 (2000 Å - 3500 Å)
		"BR_CII_2326"  :{"center":2326.000, "amp":"free", "fwhm":"free", "voff":"free", "line_profile":"G", "line_type":"br","label":r"C II]"},
		"BR_FEIII_UV47":{"center":2418.000, "amp":"free", "fwhm":"free", "voff":"free", "line_profile":"G", "line_type":"br","label":r"Fe III"},
		"BR_MGII_2799" :{"center":2799.117, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br","label":r"Mg II"},

		### Region 6 (3500 Å - 4400 Å):
		"BR_H_DELTA"   :{"center":4102.900, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br"},
		"BR_H_GAMMA"   :{"center":4341.691, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br"},

		### Region 5 (4400 Å - 5500 Å)
		"BR_H_BETA"   :{"center":4862.691, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br"},

		### Region 3 (6200 Å - 6800 Å)
		"BR_H_ALPHA"  :{"center":6585.278, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"br"},
	}
	# Default Outlfow Lines
	# Outflows share a universal width and voff across all lines, but amplitudes will be different.
	# This is because outflows are (almost always) strongest in [OIII], and doesn't suffer from blending from
	# neighboring lines or broad lines as the H-alpha/[NII]/[SII] lines do.  The width and voff will be added when parameters are generated.
	# For the H-beta/[OIII] region, [OIII]5007 dicates the amplitude of the outflow component (since parameters are flux-weighted, i.e., the strongest
	# lines have more influence on the fit of a parameter than weaker ones), so the [OIII]4960 and H-beta outflow amplitudes are a fraction of the
	# narrow-to-outflow line ratio. This same reasoning applies to the H-alpha/[NII]/[SII] region, with H-alpha deciding the line amplitudes.
	outflow_lines = {
		# Ne III
		"OUT_NEIII_3869":{"center":3869.857, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"out"}, # Coronal Line
		"OUT_NEIII_3968":{"center":3968.593, "amp":"OUT_NEIII_3869_AMP/NA_NEIII_3869_AMP*NA_NEIII_3968_AMP", "fwhm":"OUT_NEIII_3869_FWHM", "voff":"OUT_NEIII_3869_VOFF", "line_type":"out"}, # Coronal Line
		# H-beta/[OIII]
		"OUT_H_BETA"    :{"center":4862.691, "amp":"OUT_OIII_5007_AMP/NA_OIII_5007_AMP*NA_H_BETA_AMP" , "fwhm":"OUT_OIII_5007_FWHM", "voff":"OUT_OIII_5007_VOFF", "line_type":"out"},
		"OUT_OIII_4960" :{"center":4960.295, "amp":"OUT_OIII_5007_AMP/2.98"							  , "fwhm":"OUT_OIII_5007_FWHM", "voff":"OUT_OIII_5007_VOFF", "line_type":"out"},
		"OUT_OIII_5007" :{"center":5008.240, "amp":"free"         									  , "fwhm":"free", "voff":"free", "line_type":"out"},
		# H-beta/[OIII] - Secondary Components
		# "OUT_H_BETA_2"    :{"center":4862.691, "amp":"OUT_OIII_5007_2_AMP/NA_OIII_5007_AMP*NA_H_BETA_AMP" , "fwhm":"OUT_OIII_5007_2_FWHM", "voff":"OUT_OIII_5007_2_VOFF", "line_type":"out"},
		# "OUT_OIII_4960_2" :{"center":4960.295, "amp":"OUT_OIII_5007_2_AMP/2.98"							  , "fwhm":"OUT_OIII_5007_2_FWHM", "voff":"OUT_OIII_5007_2_VOFF", "line_type":"out"},
		# "OUT_OIII_5007_2" :{"center":5008.240, "amp":"free"         									  , "fwhm":"free", "voff":"free", "line_type":"out"},
		# H-beta/[OIII] - Tertiary Components
		# "OUT_H_BETA_3"    :{"center":4862.691, "amp":"OUT_OIII_5007_3_AMP/NA_OIII_5007_AMP*NA_H_BETA_AMP" , "fwhm":"OUT_OIII_5007_3_FWHM", "voff":"OUT_OIII_5007_3_VOFF", "line_type":"out"},
		# "OUT_OIII_4960_3" :{"center":4960.295, "amp":"OUT_OIII_5007_3_AMP/2.98"							  , "fwhm":"OUT_OIII_5007_3_FWHM", "voff":"OUT_OIII_5007_3_VOFF", "line_type":"out"},
		# "OUT_OIII_5007_3" :{"center":5008.240, "amp":"free"         									  , "fwhm":"free", "voff":"free", "line_type":"out"},
		# [O I]
		"OUT_OI_6302"   :{"center":6302.046, "amp":"OUT_NII_6585_AMP/NA_NII_6585_AMP*NA_OI_6302_AMP", "fwhm":"OUT_NII_6585_FWHM", "voff":"OUT_NII_6585_VOFF", "line_type":"out"},
		"OUT_OI_6365"   :{"center":6365.535, "amp":"OUT_NII_6585_AMP/NA_NII_6585_AMP*NA_OI_6365_AMP", "fwhm":"OUT_NII_6585_FWHM", "voff":"OUT_NII_6585_VOFF", "line_type":"out"},
		# H-alpha/[NII]/[SiII]
		"OUT_NII_6549"  :{"center":6549.859, "amp":"OUT_NII_6585_AMP/NA_NII_6585_AMP*NA_NII_6585_AMP/2.93", "fwhm":"OUT_NII_6585_FWHM", "voff":"OUT_NII_6585_VOFF", "line_type":"out"},
		"OUT_H_ALPHA"   :{"center":6564.632, "amp":"OUT_NII_6585_AMP/NA_NII_6585_AMP*NA_H_ALPHA_AMP" 	  , "fwhm":"OUT_NII_6585_FWHM", "voff":"OUT_NII_6585_VOFF", "line_type":"out"},
		"OUT_NII_6585"  :{"center":6585.278, "amp":"free"											 	  , "fwhm":"free"			  , "voff":"free"			  , "line_type":"out"},
		"OUT_SII_6718"  :{"center":6718.294, "amp":"OUT_NII_6585_AMP/NA_NII_6585_AMP*NA_SII_6718_AMP"	  , "fwhm":"OUT_NII_6585_FWHM", "voff":"OUT_NII_6585_VOFF", "line_type":"out"},
		"OUT_SII_6732"  :{"center":6732.668, "amp":"OUT_NII_6585_AMP/NA_NII_6585_AMP*NA_SII_6732_AMP"	  , "fwhm":"OUT_NII_6585_FWHM", "voff":"OUT_NII_6585_VOFF", "line_type":"out"},
	}
	# Default Absorption Lines
	absorp_lines = {
		"ABS_NAI_5897":{"center":5897.558, "amp":"free", "fwhm":"free", "voff":"free", "line_type":"abs","label":r"Na D"},
	}
	#
	# Combine all line lists into single list
	line_list = {**narrow_lines, **broad_lines, **outflow_lines, **absorp_lines}

	return line_list

##################################################################################

#### Check Line Component Options ################################################

def check_line_comp_options(lam_gal,line_list,comp_options,edge_pad=10,verbose=True):
	"""
	Checks each entry in the complete (narrow, broad, outflow, absorption, and user) line list
	and ensures all necessary keywords are input.  It also checks every line entry against the 
	front-end component options (comp_options).  The only required keyword for a line entry is 
	the "center" wavelength of the line.  If "amp", "fwhm", "voff", "h3" and "h4" (for Gauss-Hermite)
	line profiles are missing, it assumes these are all "free" parameters in the fitting of that line. 
	If "line_type" is not defined, it is assumed to be "na" (narrow).  If "line_profile" is not defined, 
	it is assumed to be "G" (Gaussian). 
	"""
	# Input checking
	# If fit_narrow=False, set fit_outflow=False as well (doesn't make sense to fit outflows without their narrow lines)
	if (comp_options["fit_narrow"]==False) and (comp_options["fit_outflow"]==True):
		if verbose:
			raise ValueError("\n  Why would you fit outflows without narrow lines? Turn on narrow line component! \n")
	# Step 1: Check each entry to make sure "center" keyword is defined.
	for line in list(line_list):
		if ("center" not in line_list[line]) or (not isinstance(line_list[line]["center"],(int,float))):
			raise ValueError("\n Line list entry requires at least 'center' wavelength (in Angstroms) to be defined as in int or float type. \n ")
	# Step 2: Remove lines that don't fall within the fitting region.
	edge_pad = 10 # Angstroms; padding on each edge of the fitting region.  If a line is within the number of Angstroms from the edge, 
				  # it is not fit.
	for line in list(line_list):
		if ((lam_gal[0]+edge_pad)<=(line_list[line]["center"])<=(lam_gal[-1]-edge_pad)): 
			pass
		else:
			line_list.pop(line, None)
	# Step 3: Remove any line_type based on comp_options:
	# If fit_narrow=False, purge narrow lines from line_list
	for line in list(line_list):
		if (comp_options["fit_narrow"]==False) and ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="na"):
			line_list.pop(line, None)
	#
	# If fit_broad=False, purge broad lines from line_list
	for line in list(line_list):
		if (comp_options["fit_broad"]==False) and ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="br"):
			line_list.pop(line, None)
	#
	# If fit_outflow=False, purge outflow lines from line_list
	for line in list(line_list):
		if (comp_options["fit_outflow"]==False) and ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="out"):
			line_list.pop(line, None)
	#
	# If fit_absorp=False, purge outflow lines from line_list
	for line in list(line_list):
		if (comp_options["fit_absorp"]==False) and ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="abs"):
			line_list.pop(line, None)
	#
	# Step 4: Assign line_profile keyword; if line_profile is not defined, add a keyword for the line profile.  If it 
	# is defined, make sure its consisten with the comp_options and line_type:
	for line in list(line_list):
		if ("line_type" in line_list[line]) and (line_list[line]["line_type"]=='na'):
			line_list[line]["line_profile"] = comp_options["na_line_profile"]
		if ("line_type" in line_list[line]) and (line_list[line]["line_type"]=='br'):
			line_list[line]["line_profile"] = comp_options["br_line_profile"]
		if ("line_type" in line_list[line]) and (line_list[line]["line_type"]=='out'):
			line_list[line]["line_profile"] = comp_options["out_line_profile"]
		if ("line_type" in line_list[line]) and (line_list[line]["line_type"]=='abs'):
			line_list[line]["line_profile"] = comp_options["abs_line_profile"]
		if (("line_type" not in line_list[line]) and ("line_profile" not in line_list[line])) or (("line_type" in line_list[line]) and (line_list[line]["line_type"]=="user") and ("line_profile" not in line_list[line])):
			if verbose:
				print("\n Warning: %s has no defined line_type or line_profile keywords.  Assuming line_profile='G' (Gaussian).\n" % line)
			line_list[line]["line_type"] = "user" # User-defined line
			line_list[line]["line_profile"] = "G"
		if ("line_type" not in line_list[line]) and ("line_profile" in line_list[line]):
			line_list[line]["line_type"] = "user" # User-defined line
		if ("line_type" in line_list[line]) and (line_list[line]["line_type"] not in ["na","br","out","abs","user"]):
			raise ValueError("\n User-input line_type not recognized.  Available options are 'na' (narrow), 'br' (broad), 'out' (outflow), or 'abs' (absorption). If unsure, leave out this keyword.\n ")
		if ("line_profile" in line_list[line]) and (line_list[line]["line_profile"] not in ["G","L","GH","V"]):
			raise ValueError("\n User-input line_profile not recognized.  Available options are 'G' (Gaussian), 'L' (Lorentzian), 'GH' (Gauss-Hermite), or 'V' (Voigt).  Default is 'G' (Gaussian).\n ")
	#
	# Step 5: Check parameters based on the defined line profile; if line_profile is not defined, add a keyword for the line profile.  If it 
	# is defined, make sure its consisten with the comp_options and line_type:
	for line in list(line_list):
		if ("amp" not in line_list[line]): # Assume "free"
			line_list[line]["amp"]="free"
		if ("fwhm" not in line_list[line]): # Assume "free"
			line_list[line]["fwhm"]="free"
		if ("voff" not in line_list[line]): # Assume "free"
			line_list[line]["voff"]="free"
		if (line_list[line]["line_profile"]=="GH") and (comp_options["n_moments"]>2): # If Gauss-Hermite line profile
			for m in range(3,3+(comp_options["n_moments"]-2),1):
				if ("h"+str(m) not in line_list[line]): # Assume "free"
					line_list[line]["h"+str(m)]="free"
		if (line_list[line]["line_profile"]=='V'):
			if ("shape" not in line_list[line]): # Assume "free"
				line_list[line]["shape"]="free"

		# Remove unnecessary parameters
		# If the line profile is Gauss-Hermite, but the number of higher-order moments is 
		# less than or equal to 2 (for which the line profile is just Gaussian), remove any 
		# unnecessary higher-order line parameters that may be in the line dictionary.
		if (line_list[line]["line_profile"]=="GH"):
			for m in range(comp_options["n_moments"]+1,11,1):
				if ("h"+str(m) in line_list[line]):
					line_list[line].pop("h"+str(m),None) # Remove sigma key
				if ("h"+str(m)+"_init" in line_list[line]):
					line_list[line].pop("h"+str(m)+"_init",None) # Remove sigma key
				if ("h"+str(m)+"_plim" in line_list[line]):
					line_list[line].pop("h"+str(m)+"_plim",None) # Remove sigma key
		# If line profile is not Gauss-Hermite, parse all higher-order moments and parameters
		elif (line_list[line]["line_profile"]!="GH"):
			for m in range(3,11,1):
				if ("h"+str(m) in line_list[line]):
					line_list[line].pop("h"+str(m),None) # Remove sigma key
				if ("h"+str(m)+"_init" in line_list[line]):
					line_list[line].pop("h"+str(m)+"_init",None) # Remove sigma key
				if ("h"+str(m)+"_plim" in line_list[line]):
					line_list[line].pop("h"+str(m)+"_plim",None) # Remove sigma key

		# Parse unnecessary "shape" parameter is not Voigt profile
		if (line_list[line]["line_profile"]!="V") and ("shape" in line_list[line]):
			line_list[line].pop("shape",None) # Remove sigma key
		if (line_list[line]["line_profile"]!="V") and ("shape_init" in line_list[line]):
			line_list[line].pop("shape_init",None) # Remove sigma key
		if (line_list[line]["line_profile"]!="V") and ("shape_plim" in line_list[line]):
			line_list[line].pop("shape_plim",None) # Remove sigma key
	#
	# If tie_line_fwhm=True, tie line widths (narrow, broad, outflow, and absorption fwhm) are tied, respectively.
	if comp_options["tie_line_fwhm"]:
		for line in list(line_list):
			# The universal narrow, broad, and outflow widths will be added when parameters are generated
			# If h3,h4, or shape parameters are present, remove them
			for m in range(3,3+(comp_options["n_moments"]-2),1):
				if ("h"+str(m) in line_list[line]):
					line_list[line].pop("h"+str(m),None)
			if ("shape" in line_list[line]):
				line_list[line].pop("shape",None)
			# line_list[line].pop("sigma",None) # Removes the key completly
			# line_list[line].pop("fwhm",None) # Removes the key completly
			# Narrow lines
			if ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="na"): 
				# line_list[line].pop("sigma",None) # Remove sigma key
				line_list[line]["fwhm"] = "NA_FWHM" # Replace with fwhm key
				# If line profile is Gauss-Hermite, add h3 and h4
				if comp_options["na_line_profile"]=="GH":
					for m in range(3,3+(comp_options["n_moments"]-2),1):
						line_list[line]["h"+str(m)] = "NA_H"+str(m)
				if comp_options["na_line_profile"]=="V":
					line_list[line]["shape"] = "NA_SHAPE"
			# Broad lines
			elif ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="br"): 
				line_list[line]["fwhm"] = "BR_FWHM" 
				if comp_options["br_line_profile"]=="GH":
					for m in range(3,3+(comp_options["n_moments"]-2),1):
						line_list[line]["h"+str(m)] = "BR_H"+str(m)
				if comp_options["br_line_profile"]=="V":
					line_list[line]["shape"] = "BR_SHAPE"
			# Outflow lines
			elif ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="out"): 
				line_list[line]["fwhm"] = "OUT_FWHM"
				if comp_options["out_line_profile"]=="GH":
					for m in range(3,3+(comp_options["n_moments"]-2),1):
						line_list[line]["h"+str(m)] = "OUT_H"+str(m)
				if comp_options["out_line_profile"]=="V":
					line_list[line]["shape"] = "OUT_SHAPE"
			# Absorption lines
			elif ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="abs"): 
				line_list[line]["fwhm"] = "ABS_FWHM"
				if comp_options["abs_line_profile"]=="GH":
					for m in range(3,3+(comp_options["n_moments"]-2),1):
						line_list[line]["h"+str(m)] = "ABS_H"+str(m)
				if comp_options["abs_line_profile"]=="V":
					line_list[line]["shape"] = "ABS_SHAPE"
			elif ("line_type" not in line_list[line]) or (line_list[line]["line_type"]=="user"):
				if verbose:
					print("\n Warning: %s has no line_type keyword specified.  Assuming narrow line." % (line))
				line_list[line]["fwhm"] = "NA_FWHM"
				line_list[line]["line_type"] = "na"
				if comp_options["na_line_profile"]=="GH":
					for m in range(3,3+(comp_options["n_moments"]-2),1):
						line_list[line]["h"+str(m)] = "NA_H"+str(m)
				if comp_options["na_line_profile"]=="V":
					line_list[line]["shape"] = "NA_SHAPE"
	#
	# If tie_line_voff=True, tie line velocity offsets (narrow, broad, outflow, and absorption voff) are tied, respectively.
	if comp_options["tie_line_voff"]:
		for line in list(line_list):
			# The universal narrow, broad, and outflow voff will be added when parameters are generated
			# line_list[line].pop("voff",None) # Removes the key completly
			if ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="na"): line_list[line]["voff"] = "NA_VOFF"
			elif ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="br"): line_list[line]["voff"] = "BR_VOFF"
			elif ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="out"): line_list[line]["voff"] = "OUT_VOFF"
			elif ("line_type" in line_list[line]) and (line_list[line]["line_type"]=="abs"): line_list[line]["voff"] = "ABS_VOFF"
			elif ("line_type" not in line_list[line]) or (line_list[line]["line_type"]=="user"):
				if verbose:
					print("\n Warning: %s has no line_type keyword specified.  Assuming narrow line." % (line))
				line_list[line]["voff"] = "NA_VOFF"
				line_list[line]["line_type"] = "na"
	#
	# Do a final check for valid keywords. If any keywords don't belong, raise an error.
	init_hmoments = ["h"+str(m)+"_init" for m in range(3,3+(comp_options["n_moments"]-2),1)]
	plim_hmoments = ["h"+str(m)+"_plim" for m in range(3,3+(comp_options["n_moments"]-2),1)]
	hmoments	  = ["h"+str(m) for m in range(3,3+(comp_options["n_moments"]-2),1)]
	#
	for line in list(line_list):

		for key in line_list[line]:
			if key not in ["center","center_pix","fwhm_res_kms","fwhm_res_ang","amp","fwhm","voff","shape","line_type","line_profile",
						  "amp_init","amp_plim","fwhm_init","fwhm_plim","voff_init","voff_plim",
						  "shape_init","shape_plim","label"]+hmoments+init_hmoments+plim_hmoments:
				raise ValueError("\n %s not a valid keyword for the line list! \n" % key)
	#
	return line_list

##################################################################################

#### Add FWHM Resolution #########################################################

def add_fwhm_res(line_list,lam_gal,fwhm_gal,velscale,verbose=True):
	# Perform linear interpolation on the fwhm_gal array as a function of wavelength 
	# We will use this to determine the fwhm resolution as a function of wavelenth for each 
	# emission line so we can correct for the resolution at every iteration.
	fwhm_gal_ftn = interp1d(lam_gal,fwhm_gal,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
	# Interpolation function that maps x (in angstroms) to pixels so we can get the exact
	# location in pixel space of the emission line.
	x_pix = np.array(range(len(lam_gal)))
	pix_interp_ftn = interp1d(lam_gal,x_pix,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
	# iterate through the line_list and add the keywords
	for line in list(line_list):
		center = line_list[line]["center"] # line center in Angstroms
		center_pix = float(pix_interp_ftn(center)) # line center in pixels
		line_list[line]["center_pix"]   = center_pix
		fwhm_res_ang = float(fwhm_gal_ftn(center)) # instrumental FWHM resolution in angstroms
		line_list[line]["fwhm_res_ang"] = fwhm_res_ang
		c = 299792.458 # speed of light (km/s)
		fwhm_res_kms = (fwhm_res_ang/center)*c# instrumental FWHM resolution in km/s
		line_list[line]["fwhm_res_kms"] = fwhm_res_kms
	
	return line_list

##################################################################################

#### Initialize Line Parameters ##################################################

def initialize_line_pars(lam_gal,galaxy,comp_options,line_list,verbose=True):

	def get_init_amp(line_center):
		line_center = float(line_center)
		try:
			return max(np.max(galaxy[(lam_gal>(line_center-10.)) & (lam_gal<(line_center+10.))]), 0.0)
		except ValueError:
			return 0.0

	line_par_input = {}
	# Initial conditions for some parameters
	max_amp = np.max(galaxy)
	median_amp = np.median(galaxy)
	opt_feii_amp_init = (0.1*np.median(galaxy))
	uv_iron_amp_init  = (0.1*np.median(galaxy)) 
	balmer_amp_init  = (0.1*np.median(galaxy)) 
	# Defaut parameter limits for certain line types
	# Pre-defined initial values and parameter limits for different line_types.
	def amp_hyperpars(line_type,line_center): # amplitude hyperparameters
		line_center = float(line_center)
		line_window = 10.0 # sampling window for each line in Angstroms
		if (line_type in ["na","user"]):
			return get_init_amp(line_center), (0.0,max_amp)
		elif (line_type in ["br","out"]):
				return (get_init_amp(line_center))/2.0, (0.0,max_amp)
		elif (line_type=="abs"):
			return -median_amp, (-median_amp,0.0,)
	#
	def fwhm_hyperpars(line_type,line_center,line_profile): # FWHM hyperparameters
		na_fwhm_init = 100.0
		out_fwhm_init = 450.0
		br_fwhm_init = 2500.0
		abs_fwhm_init = 500.0
		na_fwhm_lim  = (0.1  , 800.0)
		out_fwhm_lim = (0.1  , 5000.0)
		br_fwhm_lim  = (500.0, 15000.0)
		abs_fwhm_lim = (0.1  , 800.0)
		if line_type in ["na","user"]:
			if (line_profile=="GH"):
				# An exception is granted to line profiles that are Gauss-Hermite, since they need to be
				# able to accomodate excess width from an outflow component.
				return 250.0, (0.1,3000.0)
			else:
				return na_fwhm_init, na_fwhm_lim
		elif line_type in ["br"]:
			return br_fwhm_init, br_fwhm_lim
		elif line_type in ["out"]:
			return out_fwhm_init, out_fwhm_lim
		elif line_type in ["abs"]:
			if (line_profile=="GH"):
				# An exception is granted to line profiles that are Gauss-Hermite, since they need to be
				# able to accomodate excess width from an outflow component.
				return 250.0, (0.1,5000.0)
			else:
				return abs_fwhm_init, abs_fwhm_lim
	#
	def voff_hyperpars(line_type, line_center):
		na_voff_init, br_voff_init = 0.001, 0.001
		na_voff_lim = (-1000,1000)
		br_voff_lim = (-1000,1000)
		if line_type in ["na","user"]:
			return na_voff_init, na_voff_lim
		elif line_type in ["br","abs","out"]:
			return br_voff_init, br_voff_lim

	def h_moment_hyperpars():
		# Higher-order moments for Gauss-Hermite line profiles
		# all start at the same initial value (0) and parameter limits [-0.5,0.5]
		# You can specify individual higher-order parameters here.
		h_init = 0.0
		h_lim  = (-0.5,0.5)
		return h_init, h_lim
	#
	def shape_hyperpars(): # shape of the Voigt profile; if line_profile="V" (Voigt)
		shape_init = 0.0
		shape_lim = (0.0,1.0)
		return shape_init, shape_lim	
	# We start with standard lines and options. These are added one-by-one.  Then we check specific line options and then override any lines that have
	# been already added.  Params are added regardless of component options as long as the parameter is set to "free"
	for line in list(line_list):
		if (("amp" in line_list[line]) and (line_list[line]["amp"]=="free")):
			amp_default = amp_hyperpars(line_list[line]["line_type"],line_list[line]["center"])
			line_par_input[line+"_AMP"] = {"init": line_list[line].get("amp_init",amp_default[0]), 
										   "plim":line_list[line].get("amp_plim",amp_default[1])}
			# Check to make sure init value is within limits of plim
			if (line_par_input[line+"_AMP"]["init"]<line_par_input[line+"_AMP"]["plim"][0]) or (line_par_input[line+"_AMP"]["init"]>line_par_input[line+"_AMP"]["plim"][1]):
				raise ValueError("\n Amplitude (amp) initial value (amp_init) for %s outside of parameter limits (amp_plim)!\n" % (line))
		if (("fwhm" in line_list[line]) and (line_list[line]["fwhm"]=="free")):
			fwhm_default = fwhm_hyperpars(line_list[line]["line_type"],line_list[line]["center"],line_list[line]["line_profile"])
			line_par_input[line+"_FWHM"] = {"init": line_list[line].get("fwhm_init",fwhm_default[0]), 
											"plim":line_list[line].get("fwhm_plim",fwhm_default[1])}
			# Check to make sure init value is within limits of plim
			if (line_par_input[line+"_FWHM"]["init"]<line_par_input[line+"_FWHM"]["plim"][0]) or (line_par_input[line+"_FWHM"]["init"]>line_par_input[line+"_FWHM"]["plim"][1]):
				raise ValueError("\n FWHM (fwhm) initial value (fwhm_init) for %s outside of parameter limits (fwhm_plim)!\n" % (line))
		if (("voff" in line_list[line]) and (line_list[line]["voff"]=="free")):
			voff_default = voff_hyperpars(line_list[line]["line_type"],line_list[line]["center"])
			line_par_input[line+"_VOFF"] = {"init": line_list[line].get("voff_init",voff_default[0]), 
											"plim":line_list[line].get("voff_plim",voff_default[1])}
			# Check to make sure init value is within limits of plim
			if (line_par_input[line+"_VOFF"]["init"]<line_par_input[line+"_VOFF"]["plim"][0]) or (line_par_input[line+"_VOFF"]["init"]>line_par_input[line+"_VOFF"]["plim"][1]):
				raise ValueError("\n Velocity offset (voff) initial value (voff_init) for %s outside of parameter limits (voff_plim)!\n" % (line))
		
		if (line_list[line]["line_profile"]=="GH") & (comp_options["n_moments"]>2):
			h_default = h_moment_hyperpars()
			for m in range(3,3+(comp_options["n_moments"]-2),1):
				if ("h"+str(m) in line_list[line]):
					if (line_list[line]["h"+str(m)]=="free"):
						line_par_input[line+"_H"+str(m)] = {"init": line_list[line].get("h"+str(m)+"_init",h_default[0]), 
											  				"plim":line_list[line].get("h"+str(m)+"_plim",h_default[1])}
						# Check to make sure init value is within limits of plim
						if (line_par_input[line+"_H"+str(m)]["init"]<line_par_input[line+"_H"+str(m)]["plim"][0]) or (line_par_input[line+"_H"+str(m)]["init"]>line_par_input[line+"_H"+str(m)]["plim"][1]):
							raise ValueError("\n Gauss-Hermite moment h%d initial value (h%d_init) for %s outside of parameter limits (h%d_plim)!\n" % (m,m,line,m))

		if (("shape" in line_list[line]) and (line_list[line]["shape"]=="free")):
			shape_default = shape_hyperpars()
			line_par_input[line+"_SHAPE"] = {"init": line_list[line].get("shape_init",shape_default[0]), 
											  "plim":line_list[line].get("shape_plim",shape_default[1])}
			# Check to make sure init value is within limits of plim
			if (line_par_input[line+"_SHAPE"]["init"]<line_par_input[line+"_SHAPE"]["plim"][0]) or (line_par_input[line+"_SHAPE"]["init"]>line_par_input[line+"_SHAPE"]["plim"][1]):
				raise ValueError("\n Voigt profile shape parameter (shape) initial value (shape_init) for %s outside of parameter limits (shape_plim)!\n" % (line))

	# If tie_line_fwhm = True, we tie all widths (including any higher order moments) by respective line groups (Na, Br, Out, Abs)
	if (comp_options["tie_line_fwhm"]==True):
		# Add the common line widths for na,br,out, and abs lines
		if (comp_options["fit_narrow"]==True) or ("na" in [line_list[line]["line_type"] for line in line_list]):
			line_par_input["NA_FWHM"] = {"init": 250.0, 
										 "plim":(0.0,1200.0)}
			if (comp_options["na_line_profile"]=="GH") and (comp_options["n_moments"]>2):
				for m in range(3,3+(comp_options["n_moments"]-2),1):
					line_par_input["NA_H"+str(m)] = {"init": 0.0, 
												 	 "plim":(-0.5,0.5)}
			if comp_options["na_line_profile"]=="V":
				line_par_input["NA_SHAPE"] = {"init": 0.0, 
											  "plim":(0.0,1.0)}
		if (comp_options["fit_broad"]==True) or ("br" in [line_list[line]["line_type"] for line in line_list]):
			line_par_input["BR_FWHM"] = {"init": 2500.0, 
										 "plim":(500.0,15000.0)}
			if (comp_options["br_line_profile"]=="GH") and (comp_options["n_moments"]>2):
				for m in range(3,3+(comp_options["n_moments"]-2),1):
					line_par_input["BR_H"+str(m)] = {"init": 0.0, 
												"plim":(-0.5,0.5)}
			if comp_options["br_line_profile"]=="V":
				line_par_input["BR_SHAPE"] = {"init": 0.0, 
											  "plim":(0.0,1.0)}

		if (comp_options["fit_outflow"]==True) or ("out" in [line_list[line]["line_type"] for line in line_list]):
			line_par_input["OUT_FWHM"] = {"init": 450.0, 
										 "plim":(0.1,2500.0)}
			if (comp_options["abs_line_profile"]=="GH") and (comp_options["n_moments"]>2):
				for m in range(3,3+(comp_options["n_moments"]-2),1):
					line_par_input["ABS_H"+str(m)] = {"init": 0.0, 
													  "plim":(-0.5,0.5)}
			if comp_options["abs_line_profile"]=="V":
				line_par_input["ABS_SHAPE"] = {"init": 0.0, 
											  "plim":(0.0,1.0)}

		if (comp_options["fit_absorp"]==True) or ("abs" in [line_list[line]["line_type"] for line in line_list]):
			line_par_input["ABS_FWHM"] = {"init": 100.0, 
										 "plim":(0.0,800.0)}
			if (comp_options["abs_line_profile"]=="GH") and (comp_options["n_moments"]>2):
				for m in range(3,3+(comp_options["n_moments"]-2),1):
					line_par_input["ABS_H"+str(m)] = {"init": 0.0, 
													  "plim":(-0.5,0.5)}
			if comp_options["abs_line_profile"]=="V":
				line_par_input["ABS_SHAPE"] = {"init": 0.0, 
											  "plim":(0.0,1.0)}
			
	# If tie_line_voff = True, we tie all velocity offsets (including any higher order moments) by respective line groups (Na, Br, Out, Abs)	
	if comp_options["tie_line_voff"]==True:
		# Add the common line voffs for na,br,out, and abs lines
		if (comp_options["fit_narrow"]==True) or ("na" in [line_list[line]["line_type"] for line in line_list]):
			line_par_input["NA_VOFF"] = {"init": 0.0, 
										 "plim":(-500.0,500.0)}
		if (comp_options["fit_broad"]==True) or ("br" in [line_list[line]["line_type"] for line in line_list]):
			line_par_input["BR_VOFF"] = {"init": 0.0, 
										 "plim":(-1000.0,1000.0)}
		if (comp_options["fit_outflow"]==True) or ("out" in [line_list[line]["line_type"] for line in line_list]):
			line_par_input["OUT_VOFF"] = {"init": 0.0, 
										 "plim":(-1000.0,1000.0)}
		if (comp_options["fit_absorp"]==True) or ("abs" in [line_list[line]["line_type"] for line in line_list]):
			line_par_input["ABS_VOFF"] = {"init": 0.0, 
										 "plim":(-500.0,500.0)}
		
		
				
	return line_par_input

##################################################################################

#### Check Line Hard Constraints #################################################

def check_hard_cons(lam_gal,galaxy,comp_options,line_list,line_par_input,par_input,verbose=True):

	# Get list of all params
	# param_dict = {par:0 for par in line_par_input}
	param_dict = {par:0 for par in {**par_input,**line_par_input}}
	for line in list(line_list):
		for hpar in line_list[line]:
			if (line_list[line][hpar]!="free") and (hpar in ["amp","fwhm","voff","h3","h4","h5","h6","h7","h8","h9","h10","shape"]):
				if (isinstance(line_list[line][hpar],(int,float))):
					line_list[line][hpar] = float(line_list[line][hpar])
					pass
				else:
					try:
						ne.evaluate(line_list[line][hpar], local_dict = param_dict).item()
					except: 
						if verbose:
							print("Hard-constraint %s not found in parameter list or could not be parsed; converting to free parameter.\n" % line_list[line][hpar])
						_line_list = {line:line_list[line]}
						_line_list[line][hpar]="free"
						_line_par_input = initialize_line_pars(lam_gal,galaxy,comp_options,_line_list)
						line_par_input = {**_line_par_input,**line_par_input}

	return line_list, line_par_input

##################################################################################

#### Check Line Soft Constraints #################################################

def check_soft_cons(soft_cons,line_par_input,verbose=True):
	# par_list = [p for p in line_par_input]
	out_cons = []
	# print(soft_cons)
	# Old method
	# for con in soft_cons:
	# 	if (np.all([c in par_list for c in con])):
	# 		out_cons.append(con)
	# 	else:
	# 		if verbose:
	# 			print("\n - %s soft constraint removed because one or more free parameters is not available." % str(con))
	
	# New method
	# Map line parameters to init
	line_par_dict = {l:line_par_input[l]["init"] for l in line_par_input}
	for con in soft_cons:
		# print(con)
		valid_cons = []
		for c in con:
			try:
				val = ne.evaluate(c,local_dict = line_par_dict).item()
				# print(c, val, "True")
				valid_cons.append(True)
			except KeyError:
				valid_cons.append(False)
				# print(c, "False")
			# print(valid_cons)
		if np.all(valid_cons):
			out_cons.append(con)
		else: 
			if verbose:
				print("\n - %s soft constraint removed because one or more free parameters is not available." % str(con))
	
	# for p in line_par_input:
	# 	print(p)
	# print(out_cons)
	# sys.exit()
	return out_cons

##################################################################################

#### Output Free Parameters ######################################################

def output_free_pars(line_list,par_input,soft_cons):
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")

	print("\n Line List:")
	nfree = 0 
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")
	for line in sorted(list(line_list)):
		print("{0:<30}{1:<30}{2:<30.2}".format(line, '',''))
		for par in sorted(list(line_list[line])):
			print("{0:<30}{1:<30}{2:<30}".format('', par,str(line_list[line][par])))
			if line_list[line][par]=="free": nfree+=1
	print("\n----------------------------------------------------------------------------------------------------------------------------------------") 
	print("\n Number of Free Line Parameters: %d" % nfree)
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")
	print("\n All Free Parameters:")
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")
	nfree = 0 
	for par in sorted(list(par_input)):
		print("{0:<30}{1:<30}{2:<30.2}".format(par, '',''))
		nfree+=1
		for hpar in sorted(list(par_input[par])):
			print("{0:<30}{1:<30}{2:<30}".format('', hpar,str(par_input[par][hpar])))   
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")
	print("\n Total number of free parameters: %d" % nfree)
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")
	print("\n Soft Constraints:\n")
	for con in soft_cons:
		print("{0:>30}{1:<0}{2:<0}".format(con[0], ' > ',con[1]))
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")
	print("\n----------------------------------------------------------------------------------------------------------------------------------------")

	return

##################################################################################

#### Outflow Tests ################################################################

def ssr_test(resid_outflow,
			 resid_no_outflow,
			 run_dir):
	"""
	Sum-of-Squares of Residuals test:
	The sum-of-squares of the residuals of the no-outflow model
	and the sum-of-squares of the residuals of outflow model for each iteration
	of the outflow test. 
	"""

	# For multiple runs
	ssr_ratio = np.empty(np.shape(resid_outflow)[0])
	ssr_outflow	= np.empty(np.shape(resid_outflow)[0])
	ssr_no_outflow = np.empty(np.shape(resid_outflow)[0])

	for i in range(np.shape(resid_outflow)[0]):
		# Compute median and std of residual standard deviations
		ssr_resid_outflow		= np.sum(resid_outflow[i,:]**2)
		ssr_resid_no_outflow	 = np.sum(resid_no_outflow[i,:]**2)
		ssr_ratio[i] = (ssr_resid_no_outflow)/(ssr_resid_outflow) # sum-of-squares ratio
		ssr_outflow[i] = ssr_resid_outflow
		ssr_no_outflow[i] = ssr_resid_no_outflow

	if (np.shape(resid_outflow)[0]>1):
		return np.median(ssr_ratio), np.std(ssr_ratio), \
		   		np.median(ssr_no_outflow), np.std(ssr_no_outflow), \
		   		np.median(ssr_outflow), np.std(ssr_outflow)
	else:
		return ssr_ratio[0], 0.0, ssr_no_outflow[0], 0.0, ssr_outflow[0], 0.0


def f_test(resid_outflow,
		   resid_no_outflow,
		   k1,
		   k2,
		   run_dir):
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

	# k1 = 3.0 # simpler model; single-Gaussian deg. of freedom
	# k2 = 6.0 # (nested) complex model; double-Gaussian model deg. of freedom

	for i in range(np.shape(resid_outflow)[0]):

		RSS1 = np.sum(resid_no_outflow[i,:]**2) # resid. sum of squares single_Gaussian
		RSS2 = np.sum(resid_outflow[i,:]**2)	# resid. sum of squares double-Gaussian

		n = float(len(resid_outflow[i,:]))
		dfn = k2 - k1 # deg. of freedom numerator
		dfd = n - k2  # deg. of freedom denominator

		f_stat[i] = ((RSS1-RSS2)/(k2-k1))/((RSS2)/(n-k2)) 
		f_pval[i] = 1 - f.cdf(f_stat[i], dfn, dfd)

	# print('f-statistic model comparison = %0.2f +/- %0.2f, p-value = %0.2e +/- %0.2f' % (np.median(f_stat), np.std(f_stat),np.median(f_pval), np.std(f_pval) ))
	# print('f-statistic model comparison = %0.2f ' % (f_stat))

	if (len(f_pval)>1):
		outflow_conf, outflow_conf_err = 1.0-np.median(f_pval),(1.0-np.median(f_pval))-(1-(np.median(f_pval)+np.std(f_pval)))
		return np.median(f_stat), np.std(f_stat), np.median(f_pval), np.std(f_pval), outflow_conf, outflow_conf_err
	else:
		outflow_conf, outflow_conf_err = 1.0-(f_pval), 0.0
		return f_stat[0], 0.0, f_pval[0], 0.0, outflow_conf[0], 0.0


def chi2_metric(eval_ind, 
				mccomps_outflow, 
				mccomps_no_outflow):
	# Outflow 
	chi2_outflow_arr = np.empty(np.shape(mccomps_outflow["DATA"])[0])
	pval_outflow_arr = np.empty(np.shape(mccomps_outflow["DATA"])[0])
	for i in range(np.shape(mccomps_no_outflow["DATA"])[0]):
		f_obs = mccomps_outflow["DATA"][i,:][eval_ind]/np.sum(mccomps_outflow["DATA"][i,:][eval_ind])
		f_exp = mccomps_outflow["MODEL"][i,:][eval_ind]/np.sum(mccomps_outflow["MODEL"][i,:][eval_ind])
		chi2_outflow_arr[i], pval_outflow_arr[i] = chisquare(f_obs=f_obs,
													 		 f_exp=f_exp
													 		 )
	chi2_outflow, chi2_outflow_err = np.median(chi2_outflow_arr), np.std(chi2_outflow_arr)

	# No outflow 
	chi2_no_outflow_arr = np.empty(np.shape(mccomps_no_outflow["DATA"])[0])
	pval_no_outflow_arr = np.empty(np.shape(mccomps_no_outflow["DATA"])[0])
	for i in range(np.shape(mccomps_no_outflow["DATA"])[0]):
		f_obs = mccomps_no_outflow["DATA"][i,:][eval_ind]/np.sum(mccomps_no_outflow["DATA"][i,:][eval_ind])
		f_exp = mccomps_no_outflow["MODEL"][i,:][eval_ind]/np.sum(mccomps_no_outflow["MODEL"][i,:][eval_ind])
		chi2_no_outflow_arr[i], pval_no_outflow_arr[i] = chisquare(f_obs=f_obs,
													 		 	   f_exp=f_exp
													 		 	   )
	chi2_no_outflow, chi2_no_outflow_err = np.median(chi2_no_outflow_arr), np.std(chi2_no_outflow_arr)
	# Calculate Ratio
	# The ratio of chi-squared values is defined as the improvement of the outflow model over the no-outflow model,
	# i.e., 1.0-(chi2_outflow/chi2_no_outflow)
	chi2_ratio = 1.0-(chi2_outflow/chi2_no_outflow)
	chi2_ratio_err = chi2_ratio*np.sqrt((chi2_no_outflow_err/chi2_no_outflow)**2+(chi2_outflow_err/chi2_outflow)**2)

	return chi2_outflow, chi2_outflow_err, chi2_no_outflow, chi2_no_outflow_err, chi2_ratio, chi2_ratio_err

def bayesian_AB_test(resid_line, resid_no_line, line, wave, noise, data, min_wave, max_wave, eval_ind, nchannel,run_dir):
	"""
	Performs a Bayesian A/B hypothesis test for the 
	likelihood distributions for two models.
	"""

	# Plot
	fig = plt.figure(figsize=(18,10)) 
	gs = gridspec.GridSpec(2, 4)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0:4])
	ax2  = plt.subplot(gs[1,0])
	ax3  = plt.subplot(gs[1,1])
	ax4  = plt.subplot(gs[1,2])
	ax5  = plt.subplot(gs[1,3])
	fontsize=16
	#
	plt.suptitle(r"BADASS A/B Likelihood Comparison Test",fontsize=fontsize)
	# ax1.plot(wave,resid_line,color="xkcd:bright aqua",linestyle="-",linewidth=0.5,label="Resid. with Line")
	# ax1.plot(wave,resid_no_line,color="xkcd:bright purple",linestyle="-",linewidth=0.5,label="Resid. without Line")
	ax1.plot(wave,resid_no_line-resid_line,color="xkcd:bright red",linestyle="-",linewidth=1.0,label=r"$\Delta~\rm{Residuals}$")
	ax1.plot(wave,noise,color="xkcd:lime green",linestyle="-",linewidth=0.5,label="Noise")
	ax1.plot(wave,-noise,color="xkcd:lime green",linestyle="-",linewidth=0.5)
	# ax1.axvline(min_wave,color="xkcd:red",linestyle="--",linewidth=1,label="Line Test Region")
	# ax1.axvline(max_wave,color="xkcd:red",linestyle="--",linewidth=1)
	ax1.axhline(0,color="xkcd:white",linestyle="--",linewidth=0.75)
	ax1.set_xlabel(r"$\lambda_{\rm{rest}}$ [$\rm{\AA}$]",fontsize=fontsize)
	ax1.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$]",fontsize=fontsize)
	ax1.set_title(r"Fitting Region Residuals",fontsize=fontsize)
	ax1.tick_params(axis='both', labelsize= fontsize)
	ax1.set_xlim(np.min(wave),np.max(wave))
	ax1.legend(fontsize=12)
	#
	# Sample the noise around the best-fit 
	nsamp = 10000
	resid_line_lnlike	= np.empty(nsamp)
	resid_no_line_lnlike = np.empty(nsamp)
	for i in range(nsamp):
		lnlike_line	= np.sum(-0.5*(np.random.normal(loc=resid_line[eval_ind],scale=noise[eval_ind],size=len(eval_ind)))**2/noise[eval_ind]**2)
		lnlike_no_line = np.sum(-0.5*(np.random.normal(loc=resid_no_line[eval_ind],scale=noise[eval_ind],size=len(eval_ind)))**2/noise[eval_ind]**2)
		resid_line_lnlike[i] = lnlike_line
		resid_no_line_lnlike[i] = lnlike_no_line
	#
	ax2.hist(resid_line_lnlike,bins="doane",histtype="step",label="Line",density=True,color="xkcd:bright aqua",linewidth=0.5)
	p_line = np.percentile(resid_line_lnlike,[16,50,84])
	ax2.axvline(p_line[1],color="xkcd:bright aqua", linestyle='--', linewidth=1,)
	ax2.axvspan(p_line[0], p_line[2], alpha=0.25, color='xkcd:bright aqua')
	#
	ax2.hist(resid_no_line_lnlike,bins="doane",histtype="step",label="No Line",density=True,color="xkcd:bright orange",linewidth=0.5)
	p_no_line = np.percentile(resid_no_line_lnlike,[16,50,84])
	ax2.axvline(p_no_line[1],color="xkcd:bright orange", linestyle='--', linewidth=1,)
	ax2.axvspan(p_no_line[0], p_no_line[2], alpha=0.25, color='xkcd:bright orange')
	ax2.set_title("Log-Likelihood",fontsize=fontsize)
	ax2.tick_params(axis='both', labelsize= fontsize)
	ax2.legend()
	# The sampled log-likelihoods should be nearly Gaussian
	x			= np.arange(np.min([resid_line_lnlike, resid_no_line_lnlike]),np.max([resid_line_lnlike, resid_no_line_lnlike]),0.1)
	norm_line	= stats.norm(loc=p_line[1],scale=np.mean([p_line[2]-p_line[1],p_line[1]-p_line[0]]))
	norm_no_line = stats.norm(loc=p_no_line[1],scale=np.mean([p_no_line[2]-p_no_line[1],p_no_line[1]-p_no_line[0]]))
	#
	ax2.plot(x,norm_line.pdf(x),color="xkcd:bright aqua",linewidth=1)
	ax2.plot(x,norm_no_line.pdf(x),color="xkcd:bright orange",linewidth=1)
	#
	# Determine which distribution has the maximum likelihood.
	# Null Hypothesis, H0: B is no different than A
	# Alternative Hypothesis, H1: B is significantly different from A
	A = resid_no_line_lnlike # no line model
	A_mean = p_no_line[1]
	B = resid_line_lnlike	# line model
	ntrials = 10000
	B_samples = norm_line.rvs(size=ntrials)
	pvalues = np.array([(norm_no_line.sf(b)) for b in B_samples])*2.0
	pvalues[pvalues>1] = 1
	pvalues[pvalues<1e-6] = 0
	conf	= (1 - pvalues)
	#
	ax3.hist(pvalues,bins="doane",histtype="step",label="Line",density=True,color="xkcd:bright aqua",linewidth=0.5)
	p_pval = np.percentile(pvalues,[16,50,84])
	ax3.axvline(p_pval[1],color="xkcd:bright aqua", linestyle='--', linewidth=1,)
	ax3.axvspan(p_pval[0], p_pval[2], alpha=0.25, color='xkcd:bright aqua')
	ax3.set_title(r"$p$-values",fontsize=fontsize)
	#	
	ax4.hist(conf,bins="doane",histtype="step",label="No Line",density=True,color="xkcd:bright aqua",linewidth=0.5)
	p_conf = np.percentile(conf,[16,50,84])
	# np.save(run_dir.joinpath("conf_arr.npy"),conf)
	ax4.axvline(p_conf[1],color="xkcd:bright aqua", linestyle='--', linewidth=1,)
	ax4.axvspan(p_conf[0], p_conf[2], alpha=0.25, color='xkcd:bright aqua')
	ax4.set_title(r"Confidence",fontsize=fontsize)
	ax3.tick_params(axis='both', labelsize= fontsize)
	#
	ax4.tick_params(axis='both', labelsize= fontsize)
	#
	# print(" p-value = %0.4f +/- (%0.4f,%0.4f)" % (p_pval[1],p_pval[2]-p_pval[1],p_pval[1]-p_pval[0]))
	# print(" Confidence = %0.4f +/- (%0.4f,%0.4f)" % (p_conf[1],p_conf[2]-p_conf[1],p_conf[1]-p_conf[0]))
	#
	d = np.abs(p_line[1] - p_no_line[1]) # statistical distance
	disp = np.sqrt((np.mean([p_line[2]-p_line[1],p_line[1]-p_line[0]]))**2+(np.mean([p_no_line[2]-p_no_line[1],p_no_line[1]-p_no_line[0]]))**2) # total dispersion
	signif = d/disp # significance
	overlap = np.min([(p_line[2]-p_no_line[0]), (p_no_line[2]-p_line[0])]).clip(0) # 1-sigma overlap
	ax5.axvline(0.0,color="black",label="\n $p$-value   = %0.4f +/- (%0.4f, %0.4f)" % (p_pval[1],p_pval[2]-p_pval[1],p_pval[1]-p_pval[0]))
	ax5.axvline(0.0,color="black",label="\n Confidence = %0.4f +/- (%0.4f, %0.4f)" % (p_conf[1],p_conf[2]-p_conf[1],p_conf[1]-p_conf[0]))
	ax5.axvline(0.0,color="black",label="\n Statistical Distance = %0.4f" % d)
	ax5.axvline(0.0,color="black",label="\n Combined Dispersion  = %0.4f" % disp)
	ax5.axvline(0.0,color="black",label="\n Significance ($\sigma$) = %0.4f" % signif)
	ax5.axvline(0.0,color="black",label="\n $1\sigma$ Overlap   = %0.4f \n" % overlap)


	ax5.legend(loc="center",fontsize=fontsize,frameon=False)
	ax5.axis('off')
	
	fig.tight_layout()
	plt.savefig(run_dir.joinpath('test_results.pdf'))
	plt.close()

	return p_pval[1],p_pval[2]-p_pval[1],p_pval[1]-p_pval[0], p_conf[1],p_conf[2]-p_conf[1],p_conf[1]-p_conf[0], d, disp, signif, overlap

##################################################################################

def line_test(param_dict,
			  line_list,
			  combined_line_list,
			  soft_cons,
			  lam_gal,
			  galaxy,
			  noise,
			  z,
			  cosmology,
			  fit_reg,
			  user_lines,
			  user_constraints,
			  combined_lines,
			  test_line,
			  comp_options,
			  losvd_options,
			  host_options,
			  power_options,
			  opt_feii_options,
			  uv_iron_options,
			  balmer_options,
			  outflow_test_options,
			  host_template,
			  opt_feii_templates,
			  uv_iron_template,
			  balmer_template,
			  stel_templates,
			  fwhm_gal,
			  fit_mask,
			  velscale,
			  run_dir,
			  fit_type='init',
			  fit_stat="RCHI2",
			  output_model=False,
			  test_outflows=False,
			  n_basinhop=5,
			  max_like_niter=10,
			  verbose=True,
			  binnum=None,
			  spaxelx=None,
			  spaxely=None):
	"""
	Performs component (or line) testing based on user input wavelength range.
	"""

	if (test_outflows==True):
		remove_lines = [line for line in line_list if line_list[line]["line_type"]=="out"]
	elif (test_outflows==False):
		if isinstance(test_line["line"],str):
			remove_lines = [test_line["line"]]
		elif isinstance(test_line["line"],list):
			remove_lines = test_line["line"]

	# # Perform fitting with line
	if verbose:
		print('\n Fitting with %s...' % remove_lines)
	mcpars_line, mccomps_line, mcLL_line = max_likelihood(param_dict,
														  line_list,
														  combined_line_list,
														  soft_cons,
														  lam_gal,
														  galaxy,
														  noise,
														  z,
														  cosmology,
														  comp_options,
														  losvd_options,
														  host_options,
														  power_options,
														  opt_feii_options,
														  uv_iron_options,
														  balmer_options,
														  outflow_test_options,
														  host_template,
														  opt_feii_templates,
														  uv_iron_template,
														  balmer_template,
														  stel_templates,
														  fwhm_gal,
														  fit_mask,
														  velscale,
														  run_dir,
														  fit_type='init',
														  fit_stat=fit_stat,
														  output_model=False,
														  test_outflows=True,
														  n_basinhop=n_basinhop,
														  max_like_niter=max_like_niter,
														  verbose=verbose)	

	# Perform fitting without line
	if verbose:
		print('\n Fitting without %s...' % test_line["line"])

	# Make copy of original line list, since initialize_pars() will override it.
	original_line_list = copy.deepcopy(line_list)

	# Generate new parameters
	param_dict_no_line, line_list_no_line, combined_line_list_no_line, soft_cons_no_line = initialize_pars(lam_gal,galaxy,noise,fit_reg,fwhm_gal,fit_mask,velscale,
								 comp_options,user_lines,user_constraints,combined_lines,losvd_options,host_options,power_options,opt_feii_options,uv_iron_options,balmer_options,
								 run_dir,fit_type='init',fit_stat=fit_stat,
								 fit_opt_feii=comp_options["fit_opt_feii"],fit_uv_iron=comp_options["fit_uv_iron"],fit_balmer=comp_options["fit_balmer"],
								 fit_losvd=comp_options["fit_losvd"],fit_host=comp_options["fit_host"],fit_power=comp_options["fit_power"],
								 fit_narrow=comp_options["fit_narrow"],fit_broad=comp_options["fit_broad"],fit_outflow=comp_options["fit_outflow"],fit_absorp=comp_options["fit_absorp"],
								 tie_line_fwhm=comp_options["tie_line_fwhm"],tie_line_voff=comp_options["tie_line_voff"],remove_lines=remove_lines,verbose=verbose)

	mcpars_no_line, mccomps_no_line, mcLL_no_line = max_likelihood(param_dict_no_line,
																   			line_list_no_line,
																   			combined_line_list_no_line,
																   			soft_cons_no_line,
																   			lam_gal,
																   			galaxy,
																   			noise,
																   			z,
																   			cosmology,
																   			comp_options,
																   			losvd_options,
																   			host_options,
																   			power_options,
																   			opt_feii_options,
																   			uv_iron_options,
																   			balmer_options,
																   			outflow_test_options,
																   			host_template,
																   			opt_feii_templates,
																   			uv_iron_template,
																   			balmer_template,
																   			stel_templates,
																   			fwhm_gal,
																   			fit_mask,
																   			velscale,
																   			run_dir,
																   			fit_type='init',
																   			fit_stat=fit_stat,
																   			output_model=False,
																   			test_outflows=True,
																   			n_basinhop=n_basinhop,
																   			max_like_niter=max_like_niter,
																   			verbose=verbose)
	# Determine wavelength bounds of F-test. For [OIII]5007, we use the full profile (core + outflow)
	# and determine the 0.1 and 99.9 percentiles of the flux of the full profile to set the bounds 
	# of the test.

	if isinstance(remove_lines,str):
		full_profile = np.median(mccomps_line[remove_lines],axis=0)
	elif isinstance(remove_lines,list):
		full_profile = np.median(np.sum([mccomps_line[l] for l in remove_lines],axis=1),axis=0)

	# min_wave, max_wave, eval_ind, nchannel = get_wavelength_range(lam_gal,noise,velscale,full_profile,line_list[test_line["line"]])
	min_wave, max_wave, eval_ind, nchannel = get_wavelength_range(lam_gal[fit_mask],noise[fit_mask],velscale,full_profile[fit_mask])#,line_list[test_line["line"]])
	# storage arrays for residuals in [OIII] test region
	resid_line	  = np.empty((max_like_niter+1,nchannel))
	resid_no_line = np.empty((max_like_niter+1,nchannel))
	resid_total	  = np.empty((max_like_niter+1,len(lam_gal[fit_mask])))
	for i in range(max_like_niter+1):
		resid_line[i,:]	   = mccomps_line['RESID'][i,:][fit_mask][eval_ind]
		resid_no_line[i,:] = mccomps_no_line['RESID'][i,:][fit_mask][eval_ind]
		resid_total[i,:]   = mccomps_line['RESID'][i,:][fit_mask]

	# Perform Bayesian A/B test
	pval, pval_upp, pval_low, conf, conf_upp, conf_low, dist, disp, signif, overlap = bayesian_AB_test(mccomps_line['RESID'][0,:][fit_mask], mccomps_no_line['RESID'][0,:][fit_mask], full_profile[fit_mask], lam_gal[fit_mask], noise[fit_mask], galaxy[fit_mask], min_wave, max_wave, eval_ind, nchannel, run_dir)

	# Calculate sum-of-square of residuals and its uncertainty
	ssr_ratio, ssr_ratio_err, ssr_no_line, ssr_no_line_err, ssr_line, ssr_line_err = ssr_test(resid_line,resid_no_line,run_dir)
	# Perform f-test model comparison(for normally distributed model residuals)
	f_stat, f_stat_err, f_pval, f_pval_err, f_conf, f_conf_err = f_test(resid_line,resid_no_line,1.0,4.0,run_dir)

	# Calculate total residual noise
	resid_noise_no_line	       = np.median([np.std(resid_no_line[i,:]) for i in range(np.shape(resid_no_line)[0])])
	resid_noise_no_line_err    = np.std([np.std(resid_no_line[i,:]) for i in range(np.shape(resid_no_line)[0])])
	resid_noise_line		   = np.median([np.std(resid_line[i,:]) for i in range(np.shape(resid_line)[0])])
	resid_noise_line_err	   = np.std([np.std(resid_line[i,:]) for i in range(np.shape(resid_line)[0])])
	total_resid_noise		   = np.median([np.std(resid_total[i,:]) for i in range(np.shape(resid_total)[0])])
	total_resid_noise_err	   = np.std([np.std(resid_total[i,:]) for i in range(np.shape(resid_total)[0])])

	# Chi2 Metrics
	# Chi-squared is evaluated in the region of the line for the two models
	# The ratio of chi squared for the outflow to the no-outflow model indicates
	# how much the model improved over the other.
	chi2_line, chi2_line_err, chi2_no_line, chi2_no_line_err, chi2_ratio, chi2_ratio_err = chi2_metric(range(len(lam_gal)),mccomps_line, mccomps_no_line)

	if verbose:
		print('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter', 'Best-fit Value', '+/- 1-sigma','Flag'))
		print('-----------------------------------------------------------------------------------------------------')
	# Sort into arrays
	pname = []
	med   = []
	std   = []
	flag  = [] 
	for key in mcpars_line:
		pname.append(key)
		med.append(mcpars_line[key]['med'])
		std.append(mcpars_line[key]['std'])
		flag.append(mcpars_line[key]['flag'])
	i_sort = np.argsort(pname)
	pname = np.array(pname)[i_sort] 
	med   = np.array(med)[i_sort]   
	std   = np.array(std)[i_sort]   
	flag  = np.array(flag)[i_sort]
	if verbose:  
		for i in range(0,len(pname),1):
			print('{0:<30}{1:<30.2f}{2:<30.2f}{3:<30}'.format(pname[i], med[i], std[i], flag[i]))
		print('-----------------------------------------------------------------------------------------------------')
		print('\n Test Statistics:')
		print('-----------------------------------------------------------------------------------------------------')
		print('{0:<30}{1:<30}{2:<30}{3:<30}'.format('','Statistic','Value','Uncertainty') )
		print('-----------------------------------------------------------------------------------------------------')
		print('{0:<30}'.format('A/B Likelihood Test::'))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30}'.format('','Confidence:',conf,"(-%0.6f,+%0.6f)" % (conf_low,conf_upp )) )
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30}'.format('','p-value:',pval,"(-%0.6f,+%0.6f)" % (pval_low,pval_upp)))
		print('{0:<30}{1:<30}{2:<30.6f}'.format('','Statistical Distance:',dist))
		print('{0:<30}{1:<30}{2:<30.6f}'.format('','Disperson:',disp))
		print('{0:<30}{1:<30}{2:<30.6f}'.format('','Significance (sigma):',signif))
		print('{0:<30}{1:<30}{2:<30.6f}'.format('','Overlap (1-sigma):',overlap))
		print('{0:<30}'.format('ANOVA (F-test):'))
		print('{0:<30}{1:<30}{2:<30.4f}{3:<30.4f}'.format('','Confidence:',f_conf, f_conf_err ) )
		print('{0:<30}{1:<30}{2:<30.4f}{3:<30.4f}'.format('','F-statistic:',f_stat,f_stat_err))
		print('{0:<30}{1:<30}{2:<30.4e}{3:<30.4e}'.format('','p-value:',f_pval,f_pval_err))
		print('{0:<30}'.format('Chi-Squared Metrics:'))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Chi-squared Ratio:',chi2_ratio, chi2_ratio_err ) )
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Chi-squared no-outflow:',chi2_no_line,chi2_no_line_err))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Chi-squared outflow:',chi2_line,chi2_line_err))
		print('{0:<30}'.format('Sum-of-Squares of Residuals (SSR):'))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','SSR ratio:',ssr_ratio,ssr_ratio_err))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','SSR no-outflow:',ssr_no_line,ssr_no_line_err))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','SSR outflow:',ssr_line,ssr_line_err))
		print('{0:<30}'.format('Residual Noise:'))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Median spec noise:',np.median(noise),np.std(noise)))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Total resid noise:',total_resid_noise,total_resid_noise_err)) 
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','No-line resid:',resid_noise_no_line,resid_noise_no_line_err))
		print('{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Line resid:',resid_noise_line,resid_noise_line_err))
		print('-----------------------------------------------------------------------------------------------------')

	# Write to log
	write_log(mcpars_no_line,'no_line_test',run_dir)
	write_log(mcpars_line,'line_test',run_dir)
	write_log((pval, pval_upp, pval_low, conf, conf_upp, conf_low, dist, disp, signif, overlap,
				  f_conf,f_conf_err,f_stat,f_stat_err,f_pval,f_pval_err,
				  chi2_ratio,chi2_ratio_err,chi2_no_line,chi2_no_line_err,chi2_line,chi2_line_err,
				  # amp_metric,fwhm_metric,voff_metric,voff_metric_err,
				  ssr_ratio,ssr_ratio_err,ssr_no_line,ssr_no_line_err,ssr_line,ssr_line_err,
				  np.median(noise), np.std(noise), 
				  total_resid_noise,total_resid_noise_err,resid_noise_no_line,resid_noise_no_line_err,resid_noise_line,resid_noise_line_err),
			  'line_test_stats',run_dir)

	# Write test statistics to FITS table
	stats_dict = {
	"PVAL":				   {"best": pval, "sigma_low": pval_low, "sigma_upp": pval_upp },
	"CONF":				   {"best": conf, "sigma_low": conf_low, "sigma_upp": conf_upp},
	"DIST":				   {"best": dist, "sigma_low": 0.0, "sigma_upp": 0.0},
	"DISP":				   {"best": disp, "sigma_low": 0.0, "sigma_upp": 0.0},
	"SIGNIF":			   {"best": signif, "sigma_low": 0.0, "sigma_upp": 0.0},
	"OVERLAP":			   {"best": overlap, "sigma_low": 0.0, "sigma_upp": 0.0},
	"F_CONF":			   {"best": f_conf, "sigma_low": f_conf_err, "sigma_upp": f_conf_err},
	"F_STAT":			   {"best": f_stat, "sigma_low": f_stat_err, "sigma_upp": f_stat_err},
	"F_PVAL":			   {"best": f_pval, "sigma_low": f_pval_err, "sigma_upp": f_pval_err},
	"CHI2_LINE":		   {"best": chi2_line, "sigma_low": chi2_line_err, "sigma_upp": chi2_line_err},
	"CHI2_NO_LINE":		   {"best": chi2_no_line, "sigma_low": chi2_no_line_err, "sigma_upp": chi2_no_line_err},
	"CHI2_RATIO":		   {"best": chi2_ratio, "sigma_low": chi2_ratio_err, "sigma_upp": chi2_ratio_err},
	"SSR_RATIO":		   {"best": ssr_ratio, "sigma_low": ssr_ratio_err, "sigma_upp": ssr_ratio_err},
	"SSR_NO_LINE":		   {"best": ssr_no_line, "sigma_low": ssr_no_line_err, "sigma_upp": ssr_no_line_err},
	"SSR_LINE":			   {"best": ssr_line, "sigma_low": ssr_line_err, "sigma_upp": ssr_line_err},
	"MEDIAN_NOISE":		   {"best": np.median(noise), "sigma_low": np.std(noise), "sigma_upp": np.std(noise)},
	"RESID_NOISE":		   {"best": total_resid_noise, "sigma_low": total_resid_noise_err, "sigma_upp": total_resid_noise_err},
	"RESID_NOISE_NO_LINE": {"best": resid_noise_no_line, "sigma_low": resid_noise_no_line_err, "sigma_upp": resid_noise_no_line_err},
	"RESID_NOISE_LINE":	   {"best": resid_noise_line, "sigma_low": resid_noise_line_err, "sigma_upp": resid_noise_line_err},
	}
	write_test_stats(stats_dict,run_dir)

	# Reinstate the original line list
	line_list = original_line_list

	# Make plot 
	# Get best fit model components for each model 
	param_names_line = [key for key in param_dict ]
	params_line		= [mcpars_line[key]['med']  for key in param_dict ]
	fit_type	 = 'line_test'
	output_model = False
	comp_dict_line = fit_model(params_line,
							  param_names_line,
							  line_list,
							  combined_line_list,
							  lam_gal,
							  galaxy,
							  noise,
							  comp_options,
							  losvd_options,
							  host_options,
							  power_options,
							  opt_feii_options,
							  uv_iron_options,
							  balmer_options,
							  outflow_test_options,
							  host_template,
							  opt_feii_templates,
							  uv_iron_template,
							  balmer_template,
							  stel_templates,
							  fwhm_gal,
							  fit_mask,
							  velscale,
							  run_dir,
							  fit_type,
							  fit_stat,
							  output_model)
	param_names_no_line = [key for key in param_dict_no_line ]
	params_no_line		= [mcpars_no_line[key]['med']  for key in param_dict_no_line ]
	fit_type	 = 'line_test'
	output_model = False
	comp_dict_no_line = fit_model(params_no_line,
							  param_names_no_line,
							  line_list_no_line,
							  combined_line_list_no_line,
							  lam_gal,
							  galaxy,
							  noise,
							  comp_options,
							  losvd_options,
							  host_options,
							  power_options,
							  opt_feii_options,
							  uv_iron_options,
							  balmer_options,
							  outflow_test_options,
							  host_template,
							  opt_feii_templates,
							  uv_iron_template,
							  balmer_template,
							  stel_templates,
							  fwhm_gal,
							  fit_mask,
							  velscale,
							  run_dir,
							  fit_type,
							  fit_stat,
							  output_model)

	# Make comparison plots of outflow and no-outflow models
	line_test_plot(lam_gal,comp_dict_line,comp_dict_no_line,line_list,line_list_no_line,
						   params_line,params_no_line,param_names_line,param_names_no_line,min_wave,max_wave,run_dir)

	# Write results to FITS
	write_line_test_results(mcpars_line,comp_dict_line,mcpars_no_line,comp_dict_no_line,fit_mask,run_dir,binnum,spaxelx,spaxely)

	return

##################################################################################

def get_wavelength_range(lam_gal, noise, velscale, full_profile):#, line_dict):

	# Get indices where we perform f-test
	eval_ind = range(len(lam_gal))
	# number of channels in the  test region 
	nchannel = len(eval_ind)
	# if the number of channels < 6 (number of degrees of freedom for double-Gaussian model), then the calculated f-statistic
	# will be zero.  To resolve this, we extend the range by one pixel on each side, i.e. nchannel = 8.
	if nchannel <= 6: 
		add_chan = 7 - nchannel# number of channels to add to each side; minimum is 7 channels since deg. of freedom  = 6
		lower_pad = np.arange(eval_ind[0]-add_chan,eval_ind[0],1)#np.arange(eval_ind[0]-add_chan,eval_ind[0],1)
		upper_pad = np.arange(eval_ind[-1]+1,eval_ind[-1]+1+add_chan,1)
		eval_ind = np.concatenate([lower_pad, eval_ind, upper_pad],axis=0)
		nchannel = len(eval_ind)

	min_wave, max_wave = lam_gal[eval_ind[0]], lam_gal[eval_ind[-1]]
		
	return min_wave, max_wave, eval_ind, nchannel

##################################################################################


def write_test_stats(stats_dict,run_dir):
	"""
	Writes statistics for outflow and line testing to a FITS table.
	"""
	#
	#
	# Write Outflow model FITS tables
	# Extract elements from dictionaries
	par_names = []
	par_best  = []
	sig_low   = []
	sig_upp   = []
	for key in stats_dict:
		par_names.append(key)
		par_best.append(stats_dict[key]['best'])
		sig_low.append(stats_dict[key]['sigma_low'])
		sig_upp.append(stats_dict[key]['sigma_upp'])
	if 0: 
		for i in range(0,len(par_names),1):
			print(par_names[i],par_best[i],sig[i])
	# Write best-fit parameters to FITS table
	col1 = fits.Column(name='parameter', format='30A', array=par_names)
	col2 = fits.Column(name='best_fit' , format='E'  , array=par_best)
	col3 = fits.Column(name='sigma_low'	, format='E'  , array=sig_low)
	col4 = fits.Column(name='sigma_upp'	, format='E'  , array=sig_upp)
	
	cols = fits.ColDefs([col1,col2,col3,col4])
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir.joinpath('log', 'test_stats.fits'),overwrite=True)
	#
	return 

##################################################################################

def line_test_plot(lam_gal,comp_dict_outflow,comp_dict_no_outflow,line_list_outflows,line_list_no_outflows,
						   params_outflows,params_no_outflows,param_names_outflows,param_names_no_outflows,min_wave,max_wave,run_dir):
	"""
	The plotting function for test_line().  It plots both the outflow
	and no_outflow results.
	"""
	def calc_new_center(center,voff):
		"""
		Calculated new center shifted 
		by some velocity offset.
		"""
		c = 299792.458 # speed of light (km/s)
		new_center = (voff*center)/c + center
		return new_center

	# Creat plot window and axes
	fig = plt.figure(figsize=(14,11)) 
	gs = gridspec.GridSpec(9,1)
	ax1  = fig.add_subplot(gs[0:3,0]) # No outflow
	ax2  = fig.add_subplot(gs[3:4,0]) # No outflow residuals
	ax3  = fig.add_subplot(gs[5:8,0]) # Outflow
	ax4  = fig.add_subplot(gs[8:9,0]) # Outflow residuals
	gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
	# No outflow model (ax1,ax2)

	# Put params in dictionary
	p = dict(zip(param_names_outflows,params_outflows))

	for key in comp_dict_outflow:
		if (key=='DATA'):
			ax1.plot(comp_dict_outflow['WAVE'],comp_dict_outflow['DATA'],linewidth=0.5,color='white',label='Data',zorder=0)
		elif (key=='MODEL'):
			ax1.plot(lam_gal,comp_dict_outflow[key], color='xkcd:bright red', linewidth=1.0, label='Model', zorder=15)
		elif (key=='HOST_GALAXY'):
			ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['HOST_GALAXY'], color='xkcd:bright green', linewidth=0.5, linestyle='-', label='Host/Stellar')

		elif (key=='POWER'):
			ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['POWER'], color='xkcd:red' , linewidth=0.5, linestyle='--', label='AGN Cont.')

		elif (key in ['NA_OPT_FEII_TEMPLATE','BR_OPT_FEII_TEMPLATE']):
			ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['NA_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='Narrow FeII')
			ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['BR_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='Broad FeII')

		elif (key in ['F_OPT_FEII_TEMPLATE','S_OPT_FEII_TEMPLATE','G_OPT_FEII_TEMPLATE','Z_OPT_FEII_TEMPLATE']):
			if key=='F_OPT_FEII_TEMPLATE':
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['F_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='F-transition FeII')
			elif key=='S_OPT_FEII_TEMPLATE':
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['S_OPT_FEII_TEMPLATE'], color='xkcd:mustard', linewidth=0.5, linestyle='-' , label='S-transition FeII')
			elif key=='G_OPT_FEII_TEMPLATE':
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['G_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='G-transition FeII')
			elif key=='Z_OPT_FEII_TEMPLATE':
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['Z_OPT_FEII_TEMPLATE'], color='xkcd:rust', linewidth=0.5, linestyle='-' , label='Z-transition FeII')
		elif (key=='UV_IRON_TEMPLATE'):
			ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['UV_IRON_TEMPLATE'], color='xkcd:bright purple', linewidth=0.5, linestyle='-' , label='UV Iron'	 )
		elif (key=='BALMER_CONT'):
			ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow['BALMER_CONT'], color='xkcd:bright green', linewidth=0.5, linestyle='--' , label='Balmer Continuum'	 )
		# Plot emission lines by cross-referencing comp_dict with line_list
		if (key in line_list_outflows):
			if (line_list_outflows[key]["line_type"]=="na"):
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow[key], color='xkcd:cerulean', linewidth=0.5, linestyle='-', label='Narrow/Core Comp.')
			if (line_list_outflows[key]["line_type"]=="br"):
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow[key], color='xkcd:bright teal', linewidth=0.5, linestyle='-', label='Broad Comp.')
			if (line_list_outflows[key]["line_type"]=="out"):
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow[key], color='xkcd:bright pink', linewidth=0.5, linestyle='-', label='Outflow Comp.')
			if (line_list_outflows[key]["line_type"]=="abs"):
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow[key], color='xkcd:pastel red', linewidth=0.5, linestyle='-', label='Absorption Comp.')
			if (line_list_outflows[key]["line_type"]=="user"):
				ax1.plot(comp_dict_outflow['WAVE'], comp_dict_outflow[key], color='xkcd:electric lime', linewidth=0.5, linestyle='-', label='Other')

	ax1.axvline(min_wave,color="xkcd:yellow",linewidth=1,linestyle="--")
	ax1.axvline(max_wave,color="xkcd:yellow",linewidth=1,linestyle="--")
	ax2.axvline(min_wave,color="xkcd:yellow",linewidth=1,linestyle="--")
	ax2.axvline(max_wave,color="xkcd:yellow",linewidth=1,linestyle="--")

	ax1.set_xticklabels([])
	ax1.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
	# ax1.set_ylim(-0.5*np.median(comp_dict['MODEL']),np.max([comp_dict['DATA'],comp_dict['MODEL']]))
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=10)
	# Residuals
	sigma_resid = np.nanstd(comp_dict_outflow['DATA']-comp_dict_outflow['MODEL'])
	sigma_noise = np.median(comp_dict_outflow['NOISE'])
	ax2.plot(lam_gal,(comp_dict_outflow['NOISE']*3.0),linewidth=0.5,color="xkcd:bright orange",label='$\sigma_{\mathrm{noise}}=%0.4f$' % (sigma_noise))
	ax2.plot(lam_gal,(comp_dict_outflow['RESID']*3.0),linewidth=0.5,color="white",label='$\sigma_{\mathrm{resid}}=%0.4f$' % (sigma_resid))
	ax2.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
	# Axes limits 
	ax_low = np.min([ax1.get_ylim()[0],ax2.get_ylim()[0]])
	ax_upp = np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
	if np.isfinite(sigma_resid):
		ax_upp += 3.0 * sigma_resid

	minimum = [np.nanmin(comp_dict_outflow[comp][np.where(np.isfinite(comp_dict_outflow[comp]))[0]]) for comp in comp_dict_outflow
			   if comp_dict_outflow[comp][np.isfinite(comp_dict_outflow[comp])[0]].size > 0]
	if len(minimum) > 0:
		minimum = np.nanmin(minimum)
	else:
		minimum = 0.0
	ax1.set_ylim(np.nanmin([0.0, minimum]), ax_upp)
	ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
	ax2.set_ylim(ax_low,ax_upp)
	ax2.set_xlim(np.min(lam_gal),np.max(lam_gal))
	# Axes labels
	ax2.set_yticklabels(np.array(ax2.get_yticks()/3.0,dtype=int))
	ax2.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
	ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
	handles, labels = ax1.get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	ax1.legend(by_label.values(), by_label.keys(),loc='upper right',fontsize=8)
	ax2.legend(loc='upper right',fontsize=8)

	# Emission line annotations
	# Gather up emission line center wavelengths and labels (if available, removing any duplicates)
	line_labels = []
	for line in line_list_outflows:
		if "label" in line_list_outflows[line]:
			line_labels.append([line,line_list_outflows[line]["label"]])
	line_labels = set(map(tuple, line_labels))   
	for label in line_labels:
		center = line_list_outflows[label[0]]["center"]
		if (line_list_outflows[label[0]]["voff"]=="free"):
			voff = p[label[0]+"_VOFF"]
		elif (line_list_outflows[label[0]]["voff"]!="free"):
			voff   =  ne.evaluate(line_list_outflows[label[0]]["voff"],local_dict = p).item()
		xloc = calc_new_center(center,voff)
		yloc = np.max([comp_dict_outflow["DATA"][find_nearest(lam_gal,xloc)[1]],comp_dict_outflow["MODEL"][find_nearest(lam_gal,xloc)[1]]])
		ax1.annotate(label[1], xy=(xloc, yloc),  xycoords='data',
		xytext=(xloc, yloc), textcoords='data',
		horizontalalignment='center', verticalalignment='bottom',
		color='xkcd:white',fontsize=6,
		)
	
	# Outlfow models (ax3,ax4)
	# Put params in dictionary
	p = dict(zip(param_names_no_outflows,params_no_outflows))

	for key in comp_dict_no_outflow:
		if (key=='DATA'):
			ax3.plot(comp_dict_no_outflow['WAVE'],comp_dict_no_outflow['DATA'],linewidth=0.5,color='white',label='Data',zorder=0)
		elif (key=='MODEL'):
			ax3.plot(lam_gal,comp_dict_no_outflow[key], color='xkcd:bright red', linewidth=1.0, label='Model', zorder=15)
		elif (key=='HOST_GALAXY'):
			ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['HOST_GALAXY'], color='xkcd:bright green', linewidth=0.5, linestyle='-', label='Host/Stellar')

		elif (key=='POWER'):
			ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['POWER'], color='xkcd:red' , linewidth=0.5, linestyle='--', label='AGN Cont.')

		elif (key in ['NA_OPT_FEII_TEMPLATE','BR_OPT_FEII_TEMPLATE']):
			ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['NA_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='Narrow FeII')
			ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['BR_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='Broad FeII')

		elif (key in ['F_OPT_FEII_TEMPLATE','S_OPT_FEII_TEMPLATE','G_OPT_FEII_TEMPLATE','Z_OPT_FEII_TEMPLATE']):
			if key=='F_OPT_FEII_TEMPLATE':
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['F_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='F-transition FeII')
			elif key=='S_OPT_FEII_TEMPLATE':
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['S_OPT_FEII_TEMPLATE'], color='xkcd:mustard', linewidth=0.5, linestyle='-' , label='S-transition FeII')
			elif key=='G_OPT_FEII_TEMPLATE':
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['G_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='G-transition FeII')
			elif key=='Z_OPT_FEII_TEMPLATE':
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['Z_OPT_FEII_TEMPLATE'], color='xkcd:rust', linewidth=0.5, linestyle='-' , label='Z-transition FeII')
		elif (key=='UV_IRON_TEMPLATE'):
			ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['UV_IRON_TEMPLATE'], color='xkcd:bright purple', linewidth=0.5, linestyle='-' , label='UV Iron'	 )
		elif (key=='BALMER_CONT'):
			ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow['BALMER_CONT'], color='xkcd:bright green', linewidth=0.5, linestyle='--' , label='Balmer Continuum'	 )
		# Plot emission lines by cross-referencing comp_dict with line_list
		if (key in line_list_no_outflows):
			if (line_list_no_outflows[key]["line_type"]=="na"):
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow[key], color='xkcd:cerulean', linewidth=0.5, linestyle='-', label='Narrow/Core Comp.')
			if (line_list_no_outflows[key]["line_type"]=="br"):
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow[key], color='xkcd:bright teal', linewidth=0.5, linestyle='-', label='Broad Comp.')
			if (line_list_no_outflows[key]["line_type"]=="out"):
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow[key], color='xkcd:bright pink', linewidth=0.5, linestyle='-', label='Outflow Comp.')
			if (line_list_no_outflows[key]["line_type"]=="abs"):
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow[key], color='xkcd:pastel red', linewidth=0.5, linestyle='-', label='Absorption Comp.')
			if (line_list_no_outflows[key]["line_type"]=="user"):
				ax3.plot(comp_dict_no_outflow['WAVE'], comp_dict_no_outflow[key], color='xkcd:electric lime', linewidth=0.5, linestyle='-', label='Other')

	ax3.axvline(min_wave,color="xkcd:yellow",linewidth=1,linestyle="--")
	ax3.axvline(max_wave,color="xkcd:yellow",linewidth=1,linestyle="--")
	ax4.axvline(min_wave,color="xkcd:yellow",linewidth=1,linestyle="--")
	ax4.axvline(max_wave,color="xkcd:yellow",linewidth=1,linestyle="--")

	ax3.set_xticklabels([])
	ax3.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
	# ax3.set_ylim(-0.5*np.median(comp_dict['MODEL']),np.max([comp_dict['DATA'],comp_dict['MODEL']]))
	ax3.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=10)
	# Residuals
	sigma_resid = np.nanstd(comp_dict_no_outflow['DATA']-comp_dict_no_outflow['MODEL'])
	sigma_noise = np.median(comp_dict_no_outflow['NOISE'])
	ax4.plot(lam_gal,(comp_dict_no_outflow['NOISE']*3.0),linewidth=0.5,color="xkcd:bright orange",label='$\sigma_{\mathrm{noise}}=%0.4f$' % (sigma_noise))
	ax4.plot(lam_gal,(comp_dict_no_outflow['RESID']*3.0),linewidth=0.5,color="white",label='$\sigma_{\mathrm{resid}}=%0.4f$' % (sigma_resid))
	ax4.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
	# Axes limits 
	ax_low = np.min([ax3.get_ylim()[0],ax4.get_ylim()[0]])
	ax_upp = np.max([ax3.get_ylim()[1], ax4.get_ylim()[1]])
	if np.isfinite(sigma_resid):
		ax_upp += 3.0 * sigma_resid

	minimum = [np.nanmin(comp_dict_no_outflow[comp][np.where(np.isfinite(comp_dict_no_outflow[comp]))[0]]) for comp in comp_dict_no_outflow
			   if comp_dict_no_outflow[comp][np.isfinite(comp_dict_no_outflow[comp])[0]].size > 0]
	if len(minimum) > 0:
		minimum = np.nanmin(minimum)
	else:
		minimum = 0.0
	ax3.set_ylim(np.nanmin([0.0, minimum]), ax_upp)
	ax3.set_xlim(np.min(lam_gal),np.max(lam_gal))
	ax4.set_ylim(ax_low,ax_upp)
	ax4.set_xlim(np.min(lam_gal),np.max(lam_gal))
	# Axes labels
	ax4.set_yticklabels(np.array(ax4.get_yticks()/3.0,dtype=int))
	ax4.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
	ax4.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
	handles, labels = ax3.get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	ax3.legend(by_label.values(), by_label.keys(),loc='upper right',fontsize=8)
	ax4.legend(loc='upper right',fontsize=8)

	# Emission line annotations
	# Gather up emission line center wavelengths and labels (if available, removing any duplicates)
	line_labels = []
	for line in line_list_no_outflows:
		if "label" in line_list_no_outflows[line]:
			line_labels.append([line,line_list_no_outflows[line]["label"]])
	line_labels = set(map(tuple, line_labels))   
	for label in line_labels:
		center = line_list_no_outflows[label[0]]["center"]
		if (line_list_no_outflows[label[0]]["voff"]=="free"):
			voff = p[label[0]+"_VOFF"]
		elif (line_list_no_outflows[label[0]]["voff"]!="free"):
			voff   =  ne.evaluate(line_list_no_outflows[label[0]]["voff"],local_dict = p).item()
		xloc = calc_new_center(center,voff)
		yloc = np.max([comp_dict_no_outflow["DATA"][find_nearest(lam_gal,xloc)[1]],comp_dict_no_outflow["MODEL"][find_nearest(lam_gal,xloc)[1]]])
		ax3.annotate(label[1], xy=(xloc, yloc),  xycoords='data',
		xytext=(xloc, yloc), textcoords='data',
		horizontalalignment='center', verticalalignment='bottom',
		color='xkcd:white',fontsize=6,
		)
	# Title
	ax1.set_title(run_dir.parent.name,fontsize=12)
	#
	fig.tight_layout()
	plt.savefig(run_dir.joinpath('line_test.pdf'),fmt='pdf')
	plt.close()
	#
	return


#### Write Outflow Test Results ##################################################

def write_line_test_results(result_dict_outflows,
							   comp_dict_outflows,
							   result_dict_no_outflows,
							   comp_dict_no_outflows,
							   fit_mask,
							   run_dir,
							   binnum=None,
							   spaxelx=None,
							   spaxely=None):
	"""
	Writes results of outflow testing.  Creates FITS tables for 
	the best-fit parameters and best-fit components for each the outflow
	and no-outflow test results.
	"""
	#
	#
	# Write Outflow model FITS tables
	# Extract elements from dictionaries
	par_names = []
	par_best  = []
	sig	   = []
	for key in result_dict_outflows:
		par_names.append(key)
		par_best.append(result_dict_outflows[key]['med'])
		sig.append(result_dict_outflows[key]['std'])
	if 0: 
		for i in range(0,len(par_names),1):
			print(par_names[i],par_best[i],sig[i])
	# Write best-fit parameters to FITS table
	col1 = fits.Column(name='parameter', format='30A', array=par_names)
	col2 = fits.Column(name='best_fit' , format='E'  , array=par_best)
	col3 = fits.Column(name='sigma'	, format='E'  , array=sig)
	
	cols = fits.ColDefs([col1,col2,col3])
	hdu = fits.BinTableHDU.from_columns(cols)

	hdr = fits.PrimaryHDU()
	hdul = fits.HDUList([hdr, hdu])
	if binnum is not None:
		hdr.header.append(('BINNUM', binnum, 'bin index of the spaxel (Voronoi)'), end=True)
	if spaxelx is not None and spaxely is not None:
		hdu2 = fits.BinTableHDU.from_columns(fits.ColDefs([
			fits.Column(name='spaxelx', array=spaxelx, format='E'),
			fits.Column(name='spaxely', array=spaxely, format='E')
		]))
		hdul.append(hdu2)

	hdul.writeto(run_dir.joinpath('log/line_par_table.fits'),overwrite=True)

	# Write best-fit components to FITS file
	cols = []
	# Construct a column for each parameter and chain
	for key in comp_dict_outflows:
		cols.append(fits.Column(name=key, format='E', array=comp_dict_outflows[key]))
	# Add fit mask to cols
	cols.append(fits.Column(name="MASK", format='E', array=fit_mask))
	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)

	hdr = fits.PrimaryHDU()
	hdul = fits.HDUList([hdr, hdu])
	if binnum is not None:
		hdr.header.append(('BINNUM', binnum, 'bin index of the spaxel (Voronoi)'), end=True)
	if spaxelx is not None and spaxely is not None:
		hdu2 = fits.BinTableHDU.from_columns(fits.ColDefs([
			fits.Column(name='spaxelx', array=spaxelx, format='E'),
			fits.Column(name='spaxely', array=spaxely, format='E')
		]))
		hdul.append(hdu2)

	hdul.writeto(run_dir.joinpath('log/line_best_model_components.fits'),overwrite=True)
	#
	#
	# Write No-outflow model FITS tables
	par_names = []
	par_best  = []
	sig	   = []
	for key in result_dict_no_outflows:
		par_names.append(key)
		par_best.append(result_dict_no_outflows[key]['med'])
		sig.append(result_dict_no_outflows[key]['std'])
	if 0: 
		for i in range(0,len(par_names),1):
			print(par_names[i],par_best[i],sig[i])
	# Write best-fit parameters to FITS table
	col1 = fits.Column(name='parameter', format='30A', array=par_names)
	col2 = fits.Column(name='best_fit' , format='E'  , array=par_best)
	col3 = fits.Column(name='sigma'	, format='E'  , array=sig)
	
	cols = fits.ColDefs([col1,col2,col3])
	hdu = fits.BinTableHDU.from_columns(cols)

	hdr = fits.PrimaryHDU()
	hdul = fits.HDUList([hdr, hdu])
	if binnum is not None:
		hdr.header.append(('BINNUM', binnum, 'bin index of the spaxel (Voronoi)'), end=True)
	if spaxelx is not None and spaxely is not None:
		hdu2 = fits.BinTableHDU.from_columns(fits.ColDefs([
			fits.Column(name='spaxelx', array=spaxelx, format='E'),
			fits.Column(name='spaxely', array=spaxely, format='E')
		]))
		hdul.append(hdu2)

	hdul.writeto(run_dir.joinpath('log/no_line_par_table.fits'),overwrite=True)

	# Write best-fit components to FITS file
	cols = []
	# Construct a column for each parameter and chain
	for key in comp_dict_no_outflows:
		cols.append(fits.Column(name=key, format='E', array=comp_dict_no_outflows[key]))
	# Add fit mask to cols
	cols.append(fits.Column(name="MASK", format='E', array=fit_mask))
	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir.joinpath('log', 'no_line_best_model_components.fits'),overwrite=True)
	#
	return 

####################################################################################

def calc_max_like_flux(comp_dict, lam_gal):
	"""
	Calculates component fluxes for maximum likelihood fitting.
	Adds fluxes to exiting parameter dictionary "pdict" in max_likelihood().

	"""
	flux_dict = {}
	for key in comp_dict: 
		if key not in ['DATA', 'WAVE', 'MODEL', 'NOISE', 'RESID', "HOST_GALAXY", "POWER", "BALMER_CONT"]:
			flux = np.log10(1.e-17*(simps(comp_dict[key],lam_gal)))
			# Add to flux_dict
			flux_dict[key+"_FLUX"]  = flux

	return flux_dict


####################################################################################

def calc_max_like_lum(flux_dict, z, H0=70.0,Om0=0.30):
	"""
	Calculates component luminosities for maximum likelihood fitting.
	Adds luminosities to exiting parameter dictionary "pdict" in max_likelihood().

	"""
	# Compute luminosity distance (in cm) using FlatLambdaCDM cosmology
	cosmo = FlatLambdaCDM(H0, Om0)
	d_mpc = cosmo.luminosity_distance(z).value
	d_cm  = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
	lum_dict = {}
	for key in flux_dict:
		flux = 10**flux_dict[key] #* 1.0E-17
		
		# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
		lum   = np.log10((flux * 4*np.pi * d_cm**2	)) #/ 1.0E+42
		# Add to lum_dict
		lum_dict[key[:-4]+'LUM']= lum


	return lum_dict

####################################################################################

def calc_max_like_eqwidth(comp_dict, line_list, lam_gal, noise, velscale):
	"""
	Calculates component fluxes for maximum likelihood fitting.
	Adds fluxes to exiting parameter dictionary "pdict" in max_likelihood().

	"""
	# Create a single continuum component based on what was fit
	cont = np.zeros(len(lam_gal))
	for key in comp_dict:
		if key in ["POWER","HOST_GALAXY","BALMER_CONT"]:
			cont+=comp_dict[key]

	# Get all spectral components, not including data, model, resid, and noise
	spec_comps= [i for i in comp_dict if i not in ["DATA","MODEL","WAVE","RESID","NOISE","POWER","HOST_GALAXY","BALMER_CONT"]]
	# Get keys of any lines that were fit for which we will compute eq. widths for
	lines = [i for i in line_list]
	if (spec_comps) and (lines) and (np.sum(cont)>0):
		eqwidth_dict = {}

		for c in spec_comps:
			if 1:#c in lines: # component is a line
				# print(c,comp_dict[c],cont)
				eqwidth = simps(comp_dict[c]/cont,lam_gal)
			#
				if ~np.isfinite(eqwidth):
					eqwidth=0.0
				# Add to eqwidth_dict
				eqwidth_dict[c+"_EW"]  = eqwidth

	else:
		eqwidth_dict = None

	return eqwidth_dict

##################################################################################

def calc_max_like_cont_lum(clum, comp_dict, lam_gal, z, H0=70.0, Om0=0.30):
	"""
	Calculate monochromatic continuum luminosities
	"""
	clum_dict  = {}
	total_cont = np.zeros(len(lam_gal))
	agn_cont   = np.zeros(len(lam_gal))
	host_cont  = np.zeros(len(lam_gal))
	for key in comp_dict:
		if key in ["POWER","HOST_GALAXY","BALMER_CONT"]:
			total_cont+=comp_dict[key]
		if key in ["POWER","BALMER_CONT"]:
			agn_cont+=comp_dict[key]
		if key in ["HOST_GALAXY"]:
			host_cont+=comp_dict[key]
	#
	# Calculate luminosity distance
	cosmo = FlatLambdaCDM(H0, Om0)
	d_mpc = cosmo.luminosity_distance(z).value
	d_cm  = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
	# Interpolation function for the continuum 
	interp_tot  = interp1d(lam_gal,total_cont,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
	interp_agn  = interp1d(lam_gal,agn_cont  ,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
	interp_host = interp1d(lam_gal,host_cont ,kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
	for c in clum:
		# Total luminosities
		if (c=="L_CONT_TOT_1350"):
			flux = interp_tot(1350.0) * 1.e-17# * 1350.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
			clum_dict["L_CONT_TOT_1350"] = lum
		if (c=="L_CONT_TOT_3000"):
			flux = interp_tot(3000.0) * 1.e-17 #* 3000.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42 
			clum_dict["L_CONT_TOT_3000"] = lum
		if (c=="L_CONT_TOT_5100"):
			flux = interp_tot(5100.0) * 1.e-17 #* 5100.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
			clum_dict["L_CONT_TOT_5100"] = lum
		# AGN luminosities
		if (c=="L_CONT_AGN_1350"):
			flux = interp_agn(1350.0) * 1.e-17# * 1350.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
			clum_dict["L_CONT_AGN_1350"] = lum
		if (c=="L_CONT_AGN_3000"):
			flux = interp_agn(3000.0) * 1.e-17 #* 3000.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42 
			clum_dict["L_CONT_AGN_3000"] = lum
		if (c=="L_CONT_AGN_5100"):
			flux = interp_agn(5100.0) * 1.e-17 #* 5100.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
			clum_dict["L_CONT_AGN_5100"] = lum
		# Host luminosities
		if (c=="L_CONT_HOST_1350"):
			flux = interp_host(1350.0) * 1.e-17# * 1350.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
			clum_dict["L_CONT_HOST_1350"] = lum
		if (c=="L_CONT_HOST_3000"):
			flux = interp_host(3000.0) * 1.e-17 #* 3000.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42 
			clum_dict["L_CONT_HOST_3000"] = lum
		if (c=="L_CONT_HOST_5100"):
			flux = interp_host(5100.0) * 1.e-17 #* 5100.0
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
			clum_dict["L_CONT_HOST_5100"] = lum
		# Host and AGN fractions
		if (c=="HOST_FRAC_4000"):
			clum_dict["HOST_FRAC_4000"] = interp_host(4000.0)/interp_tot(4000.0)
		if (c=="AGN_FRAC_4000"):
			clum_dict["AGN_FRAC_4000"] = interp_agn(4000.0)/interp_tot(4000.0)
		if (c=="HOST_FRAC_7000"):
			clum_dict["HOST_FRAC_7000"] = interp_host(7000.0)/interp_tot(7000.0)
		if (c=="AGN_FRAC_7000"):
			clum_dict["AGN_FRAC_7000"] = interp_agn(7000.0)/interp_tot(7000.0)

	return clum_dict

##################################################################################

def remove_stray_lines(line_profile):
    
    line_profile[line_profile<0] = 0
    max_idx = np.where(line_profile==np.max(line_profile))[0][0]
    # print(max_idx)
    #
    left_bad = [i for i in np.arange(max_idx,-1,-1) if line_profile[i]>0]
#     print(left_bad)
    left_bad_idx = [max_idx-np.where(np.abs(np.diff(left_bad))>1)[0]-1][0]
    # print(left_bad_idx)
    if len(left_bad_idx)>0:
        l0 = left_bad_idx[0]
        line_profile[range(0,l0+1,1)]= 0 
    #
    right_bad = [i for i in np.arange(max_idx,len(line_profile),1) if line_profile[i]>0]
    right_bad_idx = [max_idx+np.where(np.abs(np.diff(right_bad))>1)[0]+1][0]
    if len(right_bad_idx)>0:
        r0 = right_bad_idx[0]
        line_profile[range(r0,len(line_profile),1)]= 0 
    # print(right_bad_idx)
    #
    return line_profile

def calc_max_like_dispersions(comp_dict, line_list, combined_line_list, lam_gal, noise, velscale):

	# Get keys of any lines that were fit for which we will compute eq. widths for
	lines = [i for i in line_list]
	#
	disp_dict = {}
	fwhm_dict = {}
	vint_dict = {}
	#
	interp_ftn = interp1d(lam_gal,np.arange(len(lam_gal))*velscale,bounds_error=False)
	# Loop through lines
	for line in lines:
		# Calculate velocity scale centered on line
		vel = np.arange(len(lam_gal))*velscale - interp_ftn(line_list[line]["center"])
		full_profile = comp_dict[line]
		# Remove stray lines
		full_profile = remove_stray_lines(full_profile)
		#
		# Normalized line profile
		norm_profile = full_profile/np.sum(full_profile)
		# Calculate integrated velocity in pixels units
		v_int = simps(vel*norm_profile,vel)/simps(norm_profile,vel)
		# Calculate integrated dispersion and correct for instrumental dispersion
		d_int = np.sqrt(simps(vel**2*norm_profile,vel)/simps(norm_profile,vel) - (v_int**2))
		d_int = np.sqrt(d_int**2 - (line_list[line]["fwhm_res_kms"]/2.3548)**2)
		# 
		if ~np.isfinite(d_int): d_int = 0.0
		if ~np.isfinite(v_int): v_int = 0.0
		disp_dict[line+"_DISP"] = d_int
		vint_dict[line+"_VINT"] = v_int
		if line in combined_line_list:
			comb_fwhm = combined_fwhm(lam_gal,comp_dict[line],line_list[line]["fwhm_res_kms"],velscale)
			fwhm_dict[line+"_FWHM"] = comb_fwhm
	#
	return disp_dict, fwhm_dict, vint_dict

#### Maximum Likelihood Fitting ##################################################

def max_likelihood(param_dict,
				   line_list,
				   combined_line_list,
				   soft_cons,
				   lam_gal,
				   galaxy,
				   noise,
				   z,
				   cosmology,
				   comp_options,
				   losvd_options,
				   host_options,
				   power_options,
				   opt_feii_options,
				   uv_iron_options,
				   balmer_options,
				   outflow_test_options,
				   host_template,
				   opt_feii_templates,
				   uv_iron_template,
				   balmer_template,
				   stel_templates,
				   fwhm_gal,
				   fit_mask,
				   velscale,
				   run_dir,
				   fit_type='init',
				   fit_stat="RCHI2",
				   output_model=False,
				   test_outflows=False,
				   n_basinhop=5,
				   max_like_niter=10,
				   verbose=True):

	"""
	This function performs an initial maximum likelihood estimation to acquire robust
	initial parameters.  It performs the monte carlo bootstrapping for both 
	testing outflows and fit for final initial parameters for emcee.
	"""
	param_names = [key for key in param_dict ]
	params	  = [param_dict[key]['init'] for key in param_dict ]
	bounds	  = [param_dict[key]['plim'] for key in param_dict ]
	lb, ub = zip(*bounds) 
	param_bounds = op.Bounds(lb,ub,keep_feasible=True)
	# Generate constraints
	# cons = []
	# def lambda_gen(con): 
	# 	con = copy.deepcopy(con)
	# 	return lambda p: p[param_names.index(con[0])]-p[param_names.index(con[1])]
	# cons = [{"type":"ineq","fun": lambda_gen(con)} for con in soft_cons]

	def lambda_gen(con): 
		return lambda p: ne.evaluate(con[0],local_dict = {param_names[i]:p[i] for i in range(len(p))}).item()-ne.evaluate(con[1],local_dict = {param_names[i]:p[i] for i in range(len(p))}).item()
	cons = [{"type":"ineq","fun": lambda_gen(copy.deepcopy(con))} for con in soft_cons]

	#
	# Perform maximum likelihood estimation for initial guesses of MCMC fit
	if verbose:
		print('\n Performing max. likelihood fitting.')
		print('\n Using Basin-hopping algorithm to estimate parameters. niter_success = %d' % (n_basinhop))
	# Start a timer
	start_time = time.time()
	# Negative log-likelihood (to minimize the negative maximum)
	nll = lambda *args: -lnlike(*args)
	# Perform global optimization using basin-hopping algorithm (superior to minimize(), but slower)
	# We will use minimize() for the monte carlo bootstrap iterations.
	result = op.basinhopping(func = nll, 
							 x0 = params,
							 # T = 0.0,
							 stepsize=1.0,
							 niter = 100, # Max # of iterations before stopping
							 minimizer_kwargs = {'args':(
							 							param_names,
							 							line_list,
							 							combined_line_list,
							 							soft_cons,
							 							lam_gal,
							 							galaxy,
							 							noise,
							 							comp_options,
							 							losvd_options,
							 							host_options,
							 							power_options,
							 							opt_feii_options,
							 							uv_iron_options,
							 							balmer_options,
							 							outflow_test_options,
							 							host_template,
							 							opt_feii_templates,
							 							uv_iron_template,
							 							balmer_template,
							 							stel_templates,
							 							fwhm_gal,
							 							fit_mask,
							 							velscale,
							 							run_dir,
							 							fit_type,
							 							fit_stat,
							 							output_model),
		 				 	'method':'SLSQP', 'bounds':param_bounds, 'constraints':cons, "options":{"disp":False}},
		 				 	 disp=verbose,
		 				 	 niter_success=n_basinhop, # Max # of successive search iterations
		 				 	 )
	#
	# Get elapsed time
	elap_time = (time.time() - start_time)

	par_best	 = result['x']
	fit_type	 = 'init'
	output_model = True

	comp_dict = fit_model(par_best,
						  param_names,
						  line_list,
						  combined_line_list,
						  lam_gal,
						  galaxy,
						  noise,
						  comp_options,
						  losvd_options,
						  host_options,
						  power_options,
						  opt_feii_options,
						  uv_iron_options,
						  balmer_options,
						  outflow_test_options,
						  host_template,
						  opt_feii_templates,
						  uv_iron_template,
						  balmer_template,
						  stel_templates,
						  fwhm_gal,
						  fit_mask,
						  velscale,
						  run_dir,
						  fit_type,
						  fit_stat,
						  output_model)

	# if fit_stat=="RCHI2":
	# 	return {p:result['x'][i] for i,p in enumerate(param_names)},comp_dict

	#### Maximum Likelihood Bootstrapping #################################################################
	# Construct random normally-distributed noise
	# How we do the monte carlo bootstrapping (i.e., the proper way):
	# (1) The 1-sigma uncertainty (spectral "noise") from inverse variance of the SDSS spectra is 
	# 	  the pixel-to-pixel variation in the spectrum when rows of pixels are added to form the final 1-d spectrum. 
	# 	  This is always an underestimate of the true noise in the spectrum. 
	# (2) The residual noise from a fit, taken to be the median absolute deviation of the residuals from a fit.  This 
	#	 is always greater than the "noise" from (1), but closer to the actual value of the noise across the fitting 
	#	  region.  
	#  We add (1) and (2) in quadrature to simulate the noise at /every/ pixel in the fitting region.
	# Note: the SDSS noise is likely underestimated, but this is the best we can do.

	mcnoise = np.array(noise)
	# Storage dictionaries for all calculated paramters at each iteration
	mcpars = {k:np.empty(max_like_niter+1) for k in param_names}
	# flux_dict
	flux_names = [key+"_FLUX" for key in comp_dict if key not in ["DATA","WAVE","MODEL","NOISE","RESID","POWER","HOST_GALAXY","BALMER_CONT"]]
	mcflux = {k:np.empty(max_like_niter+1) for k in flux_names}
	# lum dict
	lum_names = [key+"_LUM" for key in comp_dict if key not in ["DATA","WAVE","MODEL","NOISE","RESID","POWER","HOST_GALAXY","BALMER_CONT"]]
	mclum  = {k:np.empty(max_like_niter+1) for k in lum_names}
	# eqwidth dict
	# line_names = [key+"_EW" for key in {**line_list, **combined_line_list}]
	line_names = [key+"_EW" for key in comp_dict if key not in ["DATA","WAVE","MODEL","NOISE","RESID","POWER","HOST_GALAXY","BALMER_CONT"]]
	mceqw  = {k:np.empty(max_like_niter+1) for k in line_names}
	# integrated dispersion & velocity dicts
	line_names = [key+"_DISP" for key in {**line_list, **combined_line_list}]
	mcdisp  = {k:np.empty(max_like_niter+1) for k in line_names}
	line_names = [key+"_FWHM" for key in combined_line_list]
	mcfwhm  = {k:np.empty(max_like_niter+1) for k in line_names}
	line_names = [key+"_VINT" for key in {**line_list, **combined_line_list}]
	mcvint  = {k:np.empty(max_like_niter+1) for k in line_names}
	# component dictionary
	mccomps = {k:np.empty((max_like_niter+1,len(comp_dict[k]))) for k in comp_dict}
	# log-likelihood array
	mcLL = np.empty(max_like_niter+1)
	# Monochromatic continuum luminosities array
	clum = []
	if (lam_gal[0]<1350) & (lam_gal[-1]>1350):
		clum.append("L_CONT_AGN_1350")
		clum.append("L_CONT_HOST_1350")
		clum.append("L_CONT_TOT_1350")
	if (lam_gal[0]<3000) & (lam_gal[-1]>3000):
		clum.append("L_CONT_AGN_3000")
		clum.append("L_CONT_HOST_3000")
		clum.append("L_CONT_TOT_3000")
	if (lam_gal[0]<4000) & (lam_gal[-1]>4000):
		clum.append("HOST_FRAC_4000")
		clum.append("AGN_FRAC_4000")
	if (lam_gal[0]<5100) & (lam_gal[-1]>5100):
		clum.append("L_CONT_AGN_5100")
		clum.append("L_CONT_HOST_5100")
		clum.append("L_CONT_TOT_5100")
	if (lam_gal[0]<7000) & (lam_gal[-1]>7000):
		clum.append("HOST_FRAC_7000")
		clum.append("AGN_FRAC_7000")
	mccont = {k:np.empty(max_like_niter+1) for k in clum}

	# Calculate fluxes 
	flux_dict = calc_max_like_flux(comp_dict, lam_gal)
	# Calculate luminosities
	lum_dict = calc_max_like_lum(flux_dict, z, H0=cosmology["H0"], Om0=cosmology["Om0"])
	# Calculate equivalent widths
	eqwidth_dict = calc_max_like_eqwidth(comp_dict, {**line_list, **combined_line_list}, lam_gal, noise, velscale)
	# Calculate continuum luminosities
	clum_dict = calc_max_like_cont_lum(clum, comp_dict, lam_gal, z, H0=cosmology["H0"], Om0=cosmology["Om0"])
	# Calculate integrated line dispersions
	disp_dict, fwhm_dict, vint_dict = calc_max_like_dispersions(comp_dict, {**line_list, **combined_line_list}, combined_line_list, lam_gal, noise, velscale)

	# Add first iteration to arrays
	# Add to mcpars dict
	for i,key in enumerate(param_names):
		mcpars[key][0] = result['x'][i]
	# Add to mcflux dict
	for key in flux_dict:
		mcflux[key][0] = flux_dict[key]
	# Add to mclum dict
	for key in lum_dict:
		mclum[key][0] = lum_dict[key]
	# Add to mceqw dict
	if eqwidth_dict is not None:
		# Add to mceqw dict
		for key in eqwidth_dict:
			mceqw[key][0] = eqwidth_dict[key]
	# Add to mcdisp dict
	for key in disp_dict:
		mcdisp[key][0] = disp_dict[key]
	for key in fwhm_dict:
		mcfwhm[key][0] = fwhm_dict[key]
	for key in vint_dict:
		mcvint[key][0] = vint_dict[key]
	# Add components to mccomps
	for key in comp_dict:
		mccomps[key][0,:] = comp_dict[key]
	# Add log-likelihood to mcLL
	mcLL[0] = result["fun"]
	# Add continuum luminosities
	for key in clum_dict:
		mccont[key][0] = clum_dict[key]

	if (max_like_niter>0):
		if verbose:
			print( '\n Performing Monte Carlo bootstrapping...')

		for n in range(1,max_like_niter+1,1):
			# Generate a simulated galaxy spectrum with noise added at each pixel
			mcgal  = np.random.normal(galaxy,mcnoise)
			# Get rid of any infs or nan if there are none; this will cause scipy.optimize to fail
			mcgal[~np.isfinite(mcgal)] = np.median(mcgal)
			fit_type	 = 'init'
			output_model = False

			# if (cons is not None):
			if 1:

				nll = lambda *args: -lnlike(*args)
				resultmc = op.minimize(fun = nll, 
									   x0 = result['x'],
									   args=(param_names,
											 line_list,
											 combined_line_list,
											 soft_cons,
											 lam_gal,
											 mcgal, # use mcgal
											 mcnoise, # use mcnoise
											 comp_options,
											 losvd_options,
											 host_options,
											 power_options,
											 opt_feii_options,
											 uv_iron_options,
											 balmer_options,
											 outflow_test_options,
											 host_template,
											 opt_feii_templates,
											 uv_iron_template,
											 balmer_template,
											 stel_templates,
											 fwhm_gal,
											 fit_mask,
											 velscale,
											 run_dir,
											 fit_type,
											 fit_stat,
											 output_model),
									   method='SLSQP', 
									   bounds = param_bounds, 
									   constraints=cons,
									   options={'maxiter':2500,'disp': False})
			mcLL[n] = resultmc["fun"] # add best fit function values to mcLL
			# Get best-fit model components to calculate fluxes and equivalent widths
			output_model = True
			comp_dict = fit_model(resultmc["x"],
								  param_names,
								  line_list,
								  combined_line_list,
								  lam_gal,
								  galaxy,
								  noise,
								  comp_options,
								  losvd_options,
								  host_options,
								  power_options,
								  opt_feii_options,
								  uv_iron_options,
								  balmer_options,
								  outflow_test_options,
								  host_template,
								  opt_feii_templates,
								  uv_iron_template,
								  balmer_template,
								  stel_templates,
								  fwhm_gal,
								  fit_mask,
								  velscale,
								  run_dir,
								  fit_type,
								  fit_stat,
								  output_model)
			# Calculate fluxes 
			flux_dict = calc_max_like_flux(comp_dict, lam_gal)
			# Calculate luminosities
			lum_dict = calc_max_like_lum(flux_dict, z, H0=cosmology["H0"], Om0=cosmology["Om0"])
			# Calculate equivalent widths
			eqwidth_dict = calc_max_like_eqwidth(comp_dict, {**line_list, **combined_line_list}, lam_gal, noise, velscale)
			# Calculate continuum luminosities
			clum_dict = calc_max_like_cont_lum(clum, comp_dict, lam_gal, z, H0=cosmology["H0"], Om0=cosmology["Om0"])
			# Calculate integrated line dispersions
			disp_dict, fwhm_dict, vint_dict = calc_max_like_dispersions(comp_dict, {**line_list, **combined_line_list}, combined_line_list, lam_gal, noise, velscale)

			# Add to mc storage dictionaries
			# Add to mcpars dict
			for i,key in enumerate(param_names):
				mcpars[key][n] = resultmc['x'][i]
			# Add to mcflux dict
			for key in flux_dict:
				mcflux[key][n] = flux_dict[key]
			# Add to mclum dict
			for key in lum_dict:
				mclum[key][n] = lum_dict[key]
			# Add to mceqw dict
			if eqwidth_dict is not None:
				# Add to mceqw dict
				for key in eqwidth_dict:
					mceqw[key][n] = eqwidth_dict[key]
			# Add components to mccomps
			for key in comp_dict:
				mccomps[key][n,:] = comp_dict[key]
			# Add continuum luminosities
			for key in clum_dict:
				mccont[key][n] = clum_dict[key]
			# Add to mcdisp dict
			for key in disp_dict:
				mcdisp[key][n] = disp_dict[key]
			for key in fwhm_dict:
				mcfwhm[key][n] = fwhm_dict[key]
			for key in vint_dict:
				mcvint[key][n] = vint_dict[key]

			if verbose:
				print('	   Completed %d of %d iterations.' % (n,max_like_niter) )

	# Iterate through every parameter to determine if the fit is "good" (more than 1-sigma away from bounds)
	# if not, then add 1 to that parameter flag value			
	pdict		   = {} # parameter dictionary for all fitted parameters (free parameters, fluxes, luminosities, and equivalent widths)
	best_param_dict = {} # For getting the best fit model components
	# Add parameter names to pdict
	for i,key in enumerate(param_names):
		param_flags = 0
		mc_med = np.median(mcpars[key])
		mc_std = np.std(mcpars[key])
		if ~np.isfinite(mc_med): mc_med = 0
		if ~np.isfinite(mc_std): mc_std = 0
		if (mc_med-mc_std <= bounds[i][0]):
			param_flags += 1
		if (mc_med+mc_std >= bounds[i][1]):
			param_flags += 1
		if (mc_std==0):
			param_flags += 1
		pdict[param_names[i]]		   = {'med':mc_med,'std':mc_std,'flag':param_flags}
		best_param_dict[param_names[i]] = {'med':mc_med,'std':mc_std,'flag':param_flags}
	# Add fluxes to pdict
	for key in mcflux:
		param_flags = 0
		mc_med = np.median(mcflux[key])
		mc_std = np.std(mcflux[key])
		if ~np.isfinite(mc_med): mc_med = 0
		if ~np.isfinite(mc_std): mc_std = 0
		if (key[:-5] in line_list):
			if (line_list[key[:-5]]["line_type"]=="abs") & (mc_med+mc_std >= 0.0):
				param_flags += 1
			elif (line_list[key[:-5]]["line_type"]!="abs") & (mc_med-mc_std <= 0.0):
				param_flags += 1
		elif ((key[:-5] not in line_list) & (mc_med-mc_std <= 0.0)) or (mc_std==0):
			param_flags += 1
		pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}
	# Add luminosities to pdict
	for key in mclum:
		param_flags = 0
		mc_med = np.median(mclum[key])
		mc_std = np.std(mclum[key])
		if ~np.isfinite(mc_med): mc_med = 0
		if ~np.isfinite(mc_std): mc_std = 0
		if (key[:-4] in line_list):
			if (line_list[key[:-4]]["line_type"]=="abs") & (mc_med+mc_std >= 0.0):
				param_flags += 1
			elif (line_list[key[:-4]]["line_type"]!="abs") & (mc_med-mc_std <= 0.0):
				param_flags += 1
		elif ((key[:-4] not in line_list) & (mc_med-mc_std <= 0.0)) or (mc_std==0):
			param_flags += 1
		pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}
	# Add equivalent widths to pdict
	if eqwidth_dict is not None:
		for key in mceqw:
			param_flags = 0
			mc_med = np.median(mceqw[key])
			mc_std = np.std(mceqw[key])
			if ~np.isfinite(mc_med): mc_med = 0
			if ~np.isfinite(mc_std): mc_std = 0
			if (key[:-3] in line_list):
				if (line_list[key[:-3]]["line_type"]=="abs") & (mc_med+mc_std >= 0.0):
					param_flags += 1
				elif (line_list[key[:-3]]["line_type"]!="abs") & (mc_med-mc_std <= 0.0):
					param_flags += 1
			elif ((key[:-3] not in line_list) & (mc_med-mc_std <= 0.0)) or (mc_std==0):
				param_flags += 1
			pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}
	# Add dispersions to pdict
	for key in mcdisp:
		param_flags = 0
		mc_med = np.median(mcdisp[key])
		mc_std = np.std(mcdisp[key])
		if ~np.isfinite(mc_med): mc_med = 0
		if ~np.isfinite(mc_std): mc_std = 0
		pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}
	# Add FWHMs to pdict
	for key in mcfwhm:
		param_flags = 0
		mc_med = np.median(mcfwhm[key])
		mc_std = np.std(mcfwhm[key])
		if ~np.isfinite(mc_med): mc_med = 0
		if ~np.isfinite(mc_std): mc_std = 0
		pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}
	# Add velocities to pdict
	for key in mcvint:
		param_flags = 0
		mc_med = np.median(mcvint[key])
		mc_std = np.std(mcvint[key])
		if ~np.isfinite(mc_med): mc_med = 0
		if ~np.isfinite(mc_std): mc_std = 0
		pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}
	# Add continuum luminosities to pdict
	for key in mccont:
		param_flags = 0
		mc_med = np.median(mccont[key])
		mc_std = np.std(mccont[key])
		if ~np.isfinite(mc_med): mc_med = 0
		if ~np.isfinite(mc_std): mc_std = 0
		if (mc_med-mc_std <= 0.0) or (mc_std==0):
			param_flags += 1
		pdict[key] = {'med':mc_med,'std':mc_std,'flag':param_flags}


	# Add log-likelihood function values
	mc_med = np.median(mcLL)
	mc_std = np.std(mcLL)
	pdict["LOG_LIKE"] = {'med':mc_med,'std':mc_std,'flag':0}

	#
	# Add tied parameters explicitly to final parameter dictionary
	pdict = max_like_add_tied_parameters(pdict,line_list)

	#
	# Calculate some fit quality parameters which will be added to the dictionary
	# These will be appended to result_dict and need to be in the same format {"med": , "std", "flag":}
	fit_quality_dict = fit_quality_pars(best_param_dict,line_list,combined_line_list,comp_dict,fit_type="max_like",fit_stat=fit_stat)
	pdict = {**pdict,**fit_quality_dict}

	if (test_outflows==True):
		return pdict, mccomps, mcLL

	# Get best-fit components for maximum likelihood plot
	output_model = True
	comp_dict = fit_model([best_param_dict[key]['med'] for key in best_param_dict],best_param_dict.keys(),
						  line_list,
						  combined_line_list,
						  lam_gal,
						  galaxy,
						  noise,
						  comp_options,
						  losvd_options,
						  host_options,
						  power_options,
						  opt_feii_options,
						  uv_iron_options,
						  balmer_options,
						  outflow_test_options,
						  host_template,
						  opt_feii_templates,
						  uv_iron_template,
						  balmer_template,
						  stel_templates,
						  fwhm_gal,
						  fit_mask,
						  velscale,
						  run_dir,
						  fit_type,
						  fit_stat,
						  output_model)

	
	# Plot results of maximum likelihood fit
	sigma_resid, sigma_noise = max_like_plot(lam_gal,comp_dict,line_list,
				[best_param_dict[key]['med'] for key in best_param_dict],
				 best_param_dict.keys(),fit_mask,run_dir)
	# 
	if verbose:
		print('\n Maximum Likelihood Best-fit Parameters:')
		print('--------------------------------------------------------------------------------------')
		print('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter', 'Best-fit Value', '+/- 1-sigma','Flag'))
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

	if verbose:
		for i in range(0,len(pname),1):
			print('{0:<30}{1:<30.6f}{2:<30.6f}{3:<30}'.format(pname[i], med[i], std[i], flag[i] ))

	if verbose:
		print('{0:<30}{1:<30.6f}{2:<30}{3:<30}'.format('NOISE_STD', sigma_noise, ' ',' '))
		print('{0:<30}{1:<30.6f}{2:<30}{3:<30}'.format('RESID_STD', sigma_resid, ' ',' '))
		print('--------------------------------------------------------------------------------------')

	# Write to log 
	write_log((pdict,sigma_noise,sigma_resid),'max_like_fit',run_dir)

	#
	return pdict, comp_dict


#### Add Tied Parameters Explicitly ##############################################

def max_like_add_tied_parameters(pdict,line_list):

	# for key in pdict:
	# 	print(key,pdict[key])
	# Make dictionaries for pdict
	param_names = [key for key in pdict]
	med_dict  = {key:pdict[key]["med"]  for key in pdict}
	std_dict  = {key:pdict[key]["std"]  for key in pdict}
	flag_dict = {key:pdict[key]["flag"] for key in pdict}
	# print()

	for line in line_list:
		for par in line_list[line]:
			if (line_list[line][par]!="free") & (par in ["amp","fwhm","voff","shape","h3","h4","h5","h6","h7","h8","h9","h10"]):
				expr = line_list[line][par] # expression to evaluate
				expr_vars = [i for i in param_names if i in expr]
				med  = ne.evaluate(expr,local_dict = med_dict).item()
				std  = np.sqrt(np.sum(np.array([std_dict[i] for i in expr_vars],dtype=float)**2)) 
				flag = np.sum([flag_dict[i] for i in expr_vars])
				pdict[line+"_"+par.upper()] = {"med":med, "std":std, "flag":flag}

	# for key in pdict:
	# 	print(key,pdict[key])

	return pdict


def add_tied_parameters(pdict,line_list):

	# for key in pdict:
	# 	print(key,pdict[key])
	# Make dictionaries for pdict
	param_names = [key for key in pdict]
	# init_dict  = {key:pdict[key]["init"]  for key in pdict}
	# plim_dict  = {key:pdict[key]["plim"]  for key in pdict}
	chain_dict	     = {key:pdict[key]["chain"] for key in pdict}
	par_best_dict    = {key:pdict[key]["par_best"] for key in pdict}

	ci_68_low_dict   = {key:pdict[key]["ci_68_low"] for key in pdict}
	ci_68_upp_dict   = {key:pdict[key]["ci_68_upp"] for key in pdict}
	ci_95_low_dict   = {key:pdict[key]["ci_95_low"] for key in pdict}
	ci_95_upp_dict   = {key:pdict[key]["ci_95_upp"] for key in pdict}

	mean_dict        = {key:pdict[key]["mean"] for key in pdict}
	std_dev_dict     = {key:pdict[key]["std_dev"] for key in pdict}
	median_dict      = {key:pdict[key]["median"] for key in pdict}
	med_abs_dev_dict = {key:pdict[key]["med_abs_dev"] for key in pdict}

	flat_samp_dict   = {key:pdict[key]["flat_chain"] for key in pdict}
	flag_dict	     = {key:pdict[key]["flag"] for key in pdict}
	# print()

	for line in line_list:
		for par in line_list[line]:
			if (line_list[line][par]!="free") & (par in ["amp","fwhm","voff","shape","h3","h4","h5","h6","h7","h8","h9","h10"]):
				expr = line_list[line][par] # expression to evaluate
				expr_vars = [i for i in param_names if i in expr]
				init	 = pdict[expr_vars[0]]["init"]
				plim	 = pdict[expr_vars[0]]["plim"]
				chain	 = ne.evaluate(line_list[line][par],local_dict = chain_dict)
				par_best = ne.evaluate(line_list[line][par],local_dict = par_best_dict).item()

				
				ci_68_low  = np.sqrt(np.sum(np.array([ci_68_low_dict[i] for i in expr_vars],dtype=float)**2))
				ci_68_upp  = np.sqrt(np.sum(np.array([ci_68_upp_dict[i] for i in expr_vars],dtype=float)**2))
				ci_95_low  = np.sqrt(np.sum(np.array([ci_95_low_dict[i] for i in expr_vars],dtype=float)**2))
				ci_95_upp  = np.sqrt(np.sum(np.array([ci_95_upp_dict[i] for i in expr_vars],dtype=float)**2))

				mean        = np.sqrt(np.sum(np.array([mean_dict[i] for i in expr_vars],dtype=float)**2))
				std_dev     = np.sqrt(np.sum(np.array([std_dev_dict[i] for i in expr_vars],dtype=float)**2))
				median      = np.sqrt(np.sum(np.array([median_dict[i] for i in expr_vars],dtype=float)**2))
				med_abs_dev = np.sqrt(np.sum(np.array([med_abs_dev_dict[i] for i in expr_vars],dtype=float)**2))

				flag 	 = np.sum([flag_dict[i] for i in expr_vars])
				pdict[line+"_"+par.upper()] = {"init":init, "plim":plim, "chain":chain, 
											   "par_best":par_best, "ci_68_low":ci_68_low, "ci_68_upp":ci_68_upp, 
											   "ci_95_low":ci_95_low, "ci_95_upp":ci_95_upp, 
											   "mean": mean, "std_dev":std_dev,
											   "median":median, "med_abs_dev":med_abs_dev,
											   "flag":flag}

	# for key in pdict:
	# 	print(key,pdict[key])

	return pdict

##################################################################################

#### Max Likelihood Plot #########################################################

def max_like_plot(lam_gal,comp_dict,line_list,params,param_names,fit_mask,run_dir):


		def calc_new_center(center,voff):
			"""
			Calculated new center shifted 
			by some velocity offset.
			"""
			c = 299792.458 # speed of light (km/s)
			new_center = (voff*center)/c + center
			return new_center

		# Put params in dictionary
		p = dict(zip(param_names,params))

		# Maximum Likelihood plot
		fig = plt.figure(figsize=(14,6)) 
		gs = gridspec.GridSpec(4, 1)
		gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
		ax1  = plt.subplot(gs[0:3,0])
		ax2  = plt.subplot(gs[3,0])

		for key in comp_dict:
			if (key=='DATA'):
				ax1.plot(comp_dict['WAVE'],comp_dict['DATA'],linewidth=0.5,color='white',label='Data',zorder=0)
			elif (key=='MODEL'):
				ax1.plot(lam_gal,comp_dict[key], color='xkcd:bright red', linewidth=1.0, label='Model', zorder=15)
			elif (key=='HOST_GALAXY'):
				ax1.plot(comp_dict['WAVE'], comp_dict['HOST_GALAXY'], color='xkcd:bright green', linewidth=0.5, linestyle='-', label='Host/Stellar')

			elif (key=='POWER'):
				ax1.plot(comp_dict['WAVE'], comp_dict['POWER'], color='xkcd:red' , linewidth=0.5, linestyle='--', label='AGN Cont.')

			elif (key in ['NA_OPT_FEII_TEMPLATE','BR_OPT_FEII_TEMPLATE']):
				ax1.plot(comp_dict['WAVE'], comp_dict['NA_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='Narrow FeII')
				ax1.plot(comp_dict['WAVE'], comp_dict['BR_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='Broad FeII')

			elif (key in ['F_OPT_FEII_TEMPLATE','S_OPT_FEII_TEMPLATE','G_OPT_FEII_TEMPLATE','Z_OPT_FEII_TEMPLATE']):
				if key=='F_OPT_FEII_TEMPLATE':
					ax1.plot(comp_dict['WAVE'], comp_dict['F_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='F-transition FeII')
				elif key=='S_OPT_FEII_TEMPLATE':
					ax1.plot(comp_dict['WAVE'], comp_dict['S_OPT_FEII_TEMPLATE'], color='xkcd:mustard', linewidth=0.5, linestyle='-' , label='S-transition FeII')
				elif key=='G_OPT_FEII_TEMPLATE':
					ax1.plot(comp_dict['WAVE'], comp_dict['G_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='G-transition FeII')
				elif key=='Z_OPT_FEII_TEMPLATE':
					ax1.plot(comp_dict['WAVE'], comp_dict['Z_OPT_FEII_TEMPLATE'], color='xkcd:rust', linewidth=0.5, linestyle='-' , label='Z-transition FeII')
			elif (key=='UV_IRON_TEMPLATE'):
				ax1.plot(comp_dict['WAVE'], comp_dict['UV_IRON_TEMPLATE'], color='xkcd:bright purple', linewidth=0.5, linestyle='-' , label='UV Iron'	 )
			elif (key=='BALMER_CONT'):
				ax1.plot(comp_dict['WAVE'], comp_dict['BALMER_CONT'], color='xkcd:bright green', linewidth=0.5, linestyle='--' , label='Balmer Continuum'	 )
			# Plot emission lines by cross-referencing comp_dict with line_list
			if (key in line_list):
				if (line_list[key]["line_type"]=="na"):
					ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:cerulean', linewidth=0.5, linestyle='-', label='Narrow/Core Comp.')
				if (line_list[key]["line_type"]=="br"):
					ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:bright teal', linewidth=0.5, linestyle='-', label='Broad Comp.')
				if (line_list[key]["line_type"]=="out"):
					ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:bright pink', linewidth=0.5, linestyle='-', label='Outflow Comp.')
				if (line_list[key]["line_type"]=="abs"):
					ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:pastel red', linewidth=0.5, linestyle='-', label='Absorption Comp.')
				if (line_list[key]["line_type"]=="user"):
					ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:electric lime', linewidth=0.5, linestyle='-', label='Other')

		# Plot bad pixels
		ibad = [i for i in range(len(lam_gal)) if i not in fit_mask]
		if (len(ibad)>0):# and (len(ibad[0])>1):
			bad_wave = [(lam_gal[m],lam_gal[m+1]) for m in ibad if ((m+1)<len(lam_gal))]
			ax1.axvspan(bad_wave[0][0],bad_wave[0][0],alpha=0.25,color='xkcd:lime green',label="bad pixels")
			for i in bad_wave[1:]:
				ax1.axvspan(i[0],i[0],alpha=0.25,color='xkcd:lime green')

		ax1.set_xticklabels([])
		ax1.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
		# ax1.set_ylim(-0.5*np.median(comp_dict['MODEL']),np.max([comp_dict['DATA'],comp_dict['MODEL']]))
		ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=10)
		# Residuals
		sigma_resid = np.nanstd(comp_dict['DATA']-comp_dict['MODEL'])
		sigma_noise = np.median(comp_dict['NOISE'])
		ax2.plot(lam_gal,(comp_dict['NOISE']*3.0),linewidth=0.5,color="xkcd:bright orange",label='$\sigma_{\mathrm{noise}}=%0.4f$' % (sigma_noise))
		ax2.plot(lam_gal,(comp_dict['RESID']*3.0),linewidth=0.5,color="white",label='$\sigma_{\mathrm{resid}}=%0.4f$' % (sigma_resid))
		ax1.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
		ax2.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
		# Axes limits 
		ax_low = np.min([ax1.get_ylim()[0],ax2.get_ylim()[0]])
		ax_upp = np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
		if np.isfinite(sigma_resid):
			ax_upp += 3.0 * sigma_resid

		minimum = [np.nanmin(comp_dict[comp][np.where(np.isfinite(comp_dict[comp]))[0]]) for comp in comp_dict
				   if comp_dict[comp][np.isfinite(comp_dict[comp])[0]].size > 0]
		if len(minimum) > 0:
			minimum = np.nanmin(minimum)
		else:
			minimum = 0.0
		ax1.set_ylim(np.nanmin([0.0,minimum]),ax_upp)
		ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
		ax2.set_ylim(ax_low,ax_upp)
		ax2.set_xlim(np.min(lam_gal),np.max(lam_gal))
		# Axes labels
		ax2.set_yticklabels(np.array(ax2.get_yticks()/3.0,dtype=int))
		ax2.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
		ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
		handles, labels = ax1.get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		ax1.legend(by_label.values(), by_label.keys(),loc='upper right',fontsize=8)
		ax2.legend(loc='upper right',fontsize=8)

		# Emission line annotations
		# Gather up emission line center wavelengths and labels (if available, removing any duplicates)
		line_labels = []
		for line in line_list:
			if "label" in line_list[line]:
				line_labels.append([line,line_list[line]["label"]])
		line_labels = set(map(tuple, line_labels))   
		for label in line_labels:
			center = line_list[label[0]]["center"]
			if (line_list[label[0]]["voff"]=="free"):
				voff = p[label[0]+"_VOFF"]
			elif (line_list[label[0]]["voff"]!="free"):
				voff   =  ne.evaluate(line_list[label[0]]["voff"],local_dict = p).item()
			xloc = calc_new_center(center,voff)
			offset_factor = 0.05
			yloc = np.max([comp_dict["DATA"][find_nearest(lam_gal,xloc)[1]],comp_dict["MODEL"][find_nearest(lam_gal,xloc)[1]]])+(offset_factor*np.max(comp_dict["DATA"]))
			ax1.annotate(label[1], xy=(xloc, yloc),  xycoords='data',
			xytext=(xloc, yloc), textcoords='data',
			horizontalalignment='center', verticalalignment='bottom',
			color='xkcd:white',fontsize=6,
			)
		# Title
		ax1.set_title(str(run_dir.name),fontsize=12)

		# Save figure
		plt.savefig(run_dir.joinpath('max_likelihood_fit.pdf'))
		# Close plot
		fig.clear()
		plt.close()

		return sigma_resid, sigma_noise

##################################################################################

#### Likelihood Penalization for Gauss-Hermite Line Profiles #####################

def gh_penalty_ftn(line,params,param_names):
	
	# Reconstruct a gaussian of the same amp, fwhm, and voff
	p = dict(zip(param_names, params))
	#
	gh_pnames = [i for i in param_names if i.startswith(line+"_H")]
	
	if len(gh_pnames)==0:
		return 0 # no penalty
	elif len(gh_pnames)>0:
		D = np.sum(p[i]**2 for i in gh_pnames)
		penalty = D
	#
	return penalty


#### Likelihood function #########################################################

# Maximum Likelihood (initial fitting), Prior, and log Probability functions
def lnlike(params,
		   param_names,
		   line_list,
		   combined_line_list,
		   soft_cons,
		   lam_gal,
		   galaxy,
		   noise,
		   comp_options,
		   losvd_options,
		   host_options,
		   power_options,
		   opt_feii_options,
		   uv_iron_options,
		   balmer_options,
		   outflow_test_options,
		   host_template,
		   opt_feii_templates,
		   uv_iron_template,
		   balmer_template,
		   stel_templates,
		   fwhm_gal,
		   fit_mask,
		   velscale,
		   run_dir,
		   fit_type,
		   fit_stat,
		   output_model):
	"""
	Log-likelihood function.
	"""

	

	# Create model
	if (fit_type=='final') and (output_model==False):
		model, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob = fit_model(params,
																				  param_names,
																				  line_list,
																				  combined_line_list,
																				  lam_gal,
																				  galaxy,
																				  noise,
																				  comp_options,
																				  losvd_options,
																				  host_options,
																				  power_options,
																				  opt_feii_options,
																				  uv_iron_options,
																				  balmer_options,
																				  outflow_test_options,
																				  host_template,
																				  opt_feii_templates,
																				  uv_iron_template,
																				  balmer_template,
																				  stel_templates,
																				  fwhm_gal,
																				  fit_mask,
																				  velscale,
																				  run_dir,
																				  fit_type,
																				  fit_stat,
																				  output_model)

		if fit_stat=="ML":
			# Calculate log-likelihood
			l = -0.5*(galaxy[fit_mask]-model[fit_mask])**2/(noise[fit_mask])**2
			l = np.sum(l,axis=0)
		elif fit_stat=="OLS":
			# Since emcee looks for the maximum, but Least Squares requires a minimum
			# we multiply by negative.
			l = (galaxy[fit_mask]-model[fit_mask])**2
			l = -np.sum(l,axis=0)
		elif (fit_stat=="RCHI2"):
			pdict = {p:params[i] for i,p in enumerate(param_names)}
			noise_scale = pdict["NOISE_SCALE"]

			# Calculate log-likelihood
			l = -0.5*np.sum( (galaxy[fit_mask]-model[fit_mask])**2/(noise_scale*noise[fit_mask])**2 + np.log(2*np.pi*(noise_scale*noise[fit_mask])**2),axis=0)


		# Determine if any Gauss-Hermite lines exist
		pen = 0 # accumulating penalty
		if np.isfinite(l):
			for line in line_list:
				if ((line_list[line]["line_profile"]=="GH")):
					penalty = gh_penalty_ftn(line,params,param_names)
					pen+= penalty

		return l + l*pen, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob

	else:
		# The maximum likelihood routine [by default] minimizes the negative likelihood
		# Thus for fit_stat="LS", the SSR must be multiplied by -1 to minimize it. 

		model, comp_dict = fit_model(params,
									 param_names,
									 line_list,
									 combined_line_list,
									 lam_gal,
									 galaxy,
									 noise,
									 comp_options,
									 losvd_options,
									 host_options,
									 power_options,
									 opt_feii_options,
									 uv_iron_options,
									 balmer_options,
									 outflow_test_options,
									 host_template,
									 opt_feii_templates,
									 uv_iron_template,
									 balmer_template,
									 stel_templates,
									 fwhm_gal,
									 fit_mask,
									 velscale,
									 run_dir,
									 fit_type,
									 fit_stat,
									 output_model)

		if fit_stat=="ML":
			# Calculate log-likelihood
			l = -0.5*(galaxy[fit_mask]-model[fit_mask])**2/(noise[fit_mask])**2
			l = np.sum(l,axis=0)
			# print("Log-Likelihood = %0.4f" % (l))
		elif fit_stat=="OLS":
			l = (galaxy[fit_mask]-model[fit_mask])**2
			l = -np.sum(l,axis=0)
		elif (fit_stat=="RCHI2"):
			pdict = {p:params[i] for i,p in enumerate(param_names)}
			noise_scale = pdict["NOISE_SCALE"]

			# Calculate log-likelihood
			l = -0.5*np.sum( (galaxy[fit_mask]-model[fit_mask])**2/(noise_scale*noise[fit_mask])**2 + np.log(2*np.pi*(noise_scale*noise[fit_mask])**2),axis=0)

		# Determine if any Gauss-Hermite lines exist
		pen = 0 # accumulating penalty
		if np.isfinite(l):
			for line in line_list:
				if ((line_list[line]["line_profile"]=="GH")):
					penalty = gh_penalty_ftn(line,params,param_names)
					pen+= penalty
		#
		return l + l*pen

##################################################################################

#### Priors ######################################################################
# These priors are the same constraints used for outflow testing and maximum likelihood
# fitting, simply formatted for use by emcee. 
# To relax a constraint, simply comment out the condition (*not recommended*).

def lnprior(params,param_names,bounds,soft_cons,comp_options):
	"""
	Log-prior function.
	"""

	# Create refereence dictionary for numexpr
	pdict = {}
	for k in range(0,len(param_names),1):
			pdict[param_names[k]] = params[k]


	# Loop through parameters
	lp_arr = []

	for i in range(len(params)):
		# if prior_types[i]=="gaussian":
		# 	mu, sigma = bounds[i]
		# 	lp_arr.append(-0.5 * ((params[i] - mu) / sigma)**2 - 0.5 * np.log(sigma**2 * 2 * np.pi))
		# elif prior_types[i]=="uniform":
		lower, upper = bounds[i]
		assert upper > lower
		if lower <= params[i] <= upper:
			# lp_arr.append(-1 * np.log(upper - lower))
			lp_arr.append(0.0)
		else:
			lp_arr.append(-np.inf)

	# Loop through soft constraints
	for i in range(len(soft_cons)):
		if (ne.evaluate(soft_cons[i][0],local_dict = pdict).item()-ne.evaluate(soft_cons[i][1],local_dict = pdict).item() >= 0):
			lp_arr.append(0.0)
		else:
			lp_arr.append(-np.inf)


	return np.sum(lp_arr)

##################################################################################

def lnprob(params,
		   param_names,
		   bounds,
		   line_list,
		   combined_line_list,
		   soft_cons,
		   lam_gal,
		   galaxy,
		   noise,
		   comp_options,
		   losvd_options,
		   host_options,
		   power_options,
		   opt_feii_options,
		   uv_iron_options,
		   balmer_options,
		   outflow_test_options,
		   host_template,
		   opt_feii_templates,
		   uv_iron_template,
		   balmer_template,
		   stel_templates,
		   fwhm_gal,
		   fit_mask,
		   fit_stat,
		   velscale,
		   run_dir):
	"""
	Log-probability function.
	"""
	# lnprob (params,args)

	fit_type	 = 'final'
	output_model = False
	ll, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob	= lnlike(params,
																		 param_names,
																		 line_list,
																		 combined_line_list,
																		 soft_cons,
																		 lam_gal,
																		 galaxy,
																		 noise,
																		 comp_options,
																		 losvd_options,
																		 host_options,
																		 power_options,
																		 opt_feii_options,
																		 uv_iron_options,
																		 balmer_options,
																		 outflow_test_options,
																		 host_template,
																		 opt_feii_templates,
																		 uv_iron_template,
																		 balmer_template,
																		 stel_templates,
																		 fwhm_gal,
																		 fit_mask,
																		 velscale,
																		 run_dir,
																		 fit_type,
																		 fit_stat,
																		 output_model)

	lp = lnprior(params,param_names,bounds,soft_cons,comp_options)

	if not np.isfinite(lp):
		return -np.inf, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob, ll
	elif (np.isfinite(lp)==True):
		return lp + ll, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob, ll

####################################################################################

def line_constructor(lam_gal,free_dict,comp_dict,comp_options,line,line_list,velscale,noise):
	"""
	Constructs an emission line given a line_list, and returns an updated component
	dictionary that includes the generated line.
	"""

	# Gaussian
	if (line_list[line]["line_profile"]=="G"): # Gaussian line profile
		# 
		if (isinstance(line_list[line]["amp"],(str))) and (line_list[line]["amp"]!="free"):
			amp = ne.evaluate(line_list[line]["amp"],local_dict = free_dict).item()
		else:
			amp = free_dict[line+"_AMP"]
		if (isinstance(line_list[line]["fwhm"],(str))) and (line_list[line]["fwhm"]!="free"):
			fwhm = ne.evaluate(line_list[line]["fwhm"],local_dict = free_dict).item()
		else:
			fwhm = free_dict[line+"_FWHM"]
		if (isinstance(line_list[line]["voff"],(str))) and (line_list[line]["voff"]!="free"):
			voff = ne.evaluate(line_list[line]["voff"],local_dict = free_dict).item()
		else:
			voff = free_dict[line+"_VOFF"]

		if ~np.isfinite(amp) : amp  = 0.0
		if ~np.isfinite(fwhm): fwhm = 100.0
		if ~np.isfinite(voff): voff = 0.0

		line_model = gaussian_line_profile(lam_gal,
										   line_list[line]["center"],
										   amp,
										   fwhm,
										   voff,
										   line_list[line]["center_pix"],
										   line_list[line]["fwhm_res_kms"],
										   velscale
										   )
		line_model[~np.isfinite(line_model)] = 0.0
		comp_dict[line] = line_model

	elif (line_list[line]["line_profile"]=="L"): # Lorentzian line profile
		if (isinstance(line_list[line]["amp"],(str))) and (line_list[line]["amp"]!="free"):
			amp = ne.evaluate(line_list[line]["amp"],local_dict = free_dict).item()
		else:
			amp = free_dict[line+"_AMP"]
		if (isinstance(line_list[line]["fwhm"],(str))) and (line_list[line]["fwhm"]!="free"):
			fwhm = ne.evaluate(line_list[line]["fwhm"],local_dict = free_dict).item()
		else:
			fwhm = free_dict[line+"_FWHM"]
		if (isinstance(line_list[line]["voff"],(str))) and (line_list[line]["voff"]!="free"):
			voff = ne.evaluate(line_list[line]["voff"],local_dict = free_dict).item()
		else:
			voff = free_dict[line+"_VOFF"]

		if ~np.isfinite(amp) : amp  = 0.0
		if ~np.isfinite(fwhm): fwhm = 100.0
		if ~np.isfinite(voff): voff = 0.0

		line_model = lorentzian_line_profile(lam_gal,
										   line_list[line]["center"],
										   amp,
										   fwhm,
										   voff,
										   line_list[line]["center_pix"],
										   line_list[line]["fwhm_res_kms"],
										   velscale,
										   noise
										   )
		line_model[~np.isfinite(line_model)] = 0.0
		comp_dict[line] = line_model

	elif (line_list[line]["line_profile"]=="GH"): # Gauss-Hermite line profile
		if (isinstance(line_list[line]["amp"],(str))) and (line_list[line]["amp"]!="free"):
			amp = ne.evaluate(line_list[line]["amp"],local_dict = free_dict).item()
		else:
			amp = free_dict[line+"_AMP"]
		if (isinstance(line_list[line]["fwhm"],(str))) and (line_list[line]["fwhm"]!="free"):
			fwhm = ne.evaluate(line_list[line]["fwhm"],local_dict = free_dict).item()
		else:
			fwhm = free_dict[line+"_FWHM"]
		if (isinstance(line_list[line]["voff"],(str))) and (line_list[line]["voff"]!="free"):
			voff = ne.evaluate(line_list[line]["voff"],local_dict = free_dict).item()
		else:
			voff = free_dict[line+"_VOFF"]

		hmoments = np.empty(comp_options["n_moments"]-2)
		if (comp_options["n_moments"]>2):
			for i,m in enumerate(range(3,3+(comp_options["n_moments"]-2),1)):
				if (isinstance(line_list[line]["h"+str(m)],(str))) and (line_list[line]["h"+str(m)]!="free"):
					hl = ne.evaluate(line_list[line]["h"+str(m)],local_dict = free_dict).item()
				else:
					hl = free_dict[line+"_H"+str(m)]
				hmoments[i]=hl
		else: 
			hmoments = None

		if ~np.isfinite(amp) : amp  = 0.0
		if ~np.isfinite(fwhm): fwhm = 100.0
		if ~np.isfinite(voff): voff = 0.0

		line_model = gauss_hermite_line_profile(lam_gal,
											   line_list[line]["center"],
											   amp,
											   fwhm,
											   voff,
											   hmoments,
											   line_list[line]["center_pix"],
											   line_list[line]["fwhm_res_kms"],
											   velscale,
											   noise
											   )
		line_model[~np.isfinite(line_model)] = 0.0
		comp_dict[line] = line_model

	elif (line_list[line]["line_profile"]=="V"): # Voigt line profile
		if (isinstance(line_list[line]["amp"],(str))) and (line_list[line]["amp"]!="free"):
			amp = ne.evaluate(line_list[line]["amp"],local_dict = free_dict).item()
		else:
			amp = free_dict[line+"_AMP"]
		if (isinstance(line_list[line]["fwhm"],(str))) and (line_list[line]["fwhm"]!="free"):
			fwhm = ne.evaluate(line_list[line]["fwhm"],local_dict = free_dict).item()
		else:
			fwhm = free_dict[line+"_FWHM"]
		if (isinstance(line_list[line]["voff"],(str))) and (line_list[line]["voff"]!="free"):
			voff = ne.evaluate(line_list[line]["voff"],local_dict = free_dict).item()
		else:
			voff = free_dict[line+"_VOFF"]
		if (isinstance(line_list[line]["shape"],(str))) and (line_list[line]["shape"]!="free"):
			shape = ne.evaluate(line_list[line]["shape"],local_dict = free_dict).item()
		else:
			shape = free_dict[line+"_SHAPE"]

		if ~np.isfinite(amp) : amp  = 0.0
		if ~np.isfinite(fwhm): fwhm = 100.0
		if ~np.isfinite(voff): voff = 0.0

		line_model = voigt_line_profile(lam_gal,
										line_list[line]["center"],
										amp,
										fwhm,
										voff,
										shape,
										line_list[line]["center_pix"],
										line_list[line]["fwhm_res_kms"],
										velscale,
										noise
										)
		line_model[~np.isfinite(line_model)] = 0.0
		comp_dict[line] = line_model

	return comp_dict

#### Model Function ##############################################################

def combined_fwhm(lam_gal, full_profile, fwhm_res, velscale ):
	"""
	Calculate fwhm of combined lines.
	"""
	def lin_interp(x, y, i, half):
		return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

	def half_max_x(x, y):
		half = max(y)/2.0
		signs = np.sign(np.add(y, -half))
		zero_crossings = (signs[0:-2] != signs[1:-1])
		zero_crossings_i = np.where(zero_crossings)[0]
		if len(zero_crossings_i)==2:
			return [lin_interp(x, y, zero_crossings_i[0], half),
					lin_interp(x, y, zero_crossings_i[1], half)]
		else:
			return [0.0, 0.0]

	hmx = half_max_x(range(len(lam_gal)),full_profile)
	fwhm = np.abs(hmx[1]-hmx[0])
	fwhm = np.sqrt((fwhm*velscale)**2 - fwhm_res**2)
	if ~np.isfinite(fwhm):
		fwhm = 0.0

	return fwhm


def fit_model(params,
			  param_names,
			  line_list,
			  combined_line_list,
			  lam_gal,
			  galaxy,
			  noise,
			  comp_options,
			  losvd_options,
			  host_options,
			  power_options,
			  opt_feii_options,
			  uv_iron_options,
			  balmer_options,
			  outflow_test_options,
			  host_template,
			  opt_feii_templates,
			  uv_iron_template,
			  balmer_template,
			  stel_templates,
			  fwhm_gal,
			  fit_mask,
			  velscale,
			  run_dir,
			  fit_type,
			  fit_stat,
			  output_model):
	"""
	Constructs galaxy model.
	"""

	# Construct dictionary of parameter names and their respective parameter values
	# param_names  = [param_dict[key]['name'] for key in param_dict ]
	# params	   = [param_dict[key]['init'] for key in param_dict ]
	keys = param_names
	values = params
	p = dict(zip(keys, values))
	c = 299792.458 # speed of light
	host_model = np.copy(galaxy)
	# Initialize empty dict to store model components
	comp_dict  = {}

	############################# Power-law Component ######################################################

	# if all(comp in param_names for comp in ['POWER_AMP','power_slope','power_break'])==True:
	# if all(comp in param_names for comp in ['POWER_AMP','power_slope'])==True:
	if (comp_options['fit_power']==True) & (power_options['type']=='simple'):

		# Create a template model for the power-law continuum
		# power = simple_power_law(lam_gal,p['POWER_AMP'],p['POWER_SLOPE'],p['POWER_BREAK']) # 
		power = simple_power_law(lam_gal,p['POWER_AMP'],p['POWER_SLOPE']) # 

		host_model = (host_model) - (power) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['POWER'] = power

	elif (comp_options['fit_power']==True) & (power_options['type']=='broken'):
		# Create a template model for the power-law continuum
		# power = simple_power_law(lam_gal,p['POWER_AMP'],p['POWER_SLOPE'],p['POWER_BREAK']) # 
		power = broken_power_law(lam_gal,p['POWER_AMP'],p['POWER_BREAK'],
										 p['POWER_SLOPE_1'],p['POWER_SLOPE_2'],
										 p['POWER_CURVATURE'])

		host_model = (host_model) - (power) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['POWER'] = power

	########################################################################################################

	############################# Optical FeII Component ###################################################

	if (opt_feii_templates is not None):

		if (opt_feii_options['opt_template']['type']=='VC04'):
			
			br_opt_feii_template, na_opt_feii_template = VC04_opt_feii_template(p, lam_gal, opt_feii_templates, opt_feii_options, velscale)
						 
			host_model = (host_model) - (na_opt_feii_template) - (br_opt_feii_template)
			comp_dict['NA_OPT_FEII_TEMPLATE'] = na_opt_feii_template # Add to component dictionary
			comp_dict['BR_OPT_FEII_TEMPLATE'] = br_opt_feii_template # Add to component dictionary

		elif (opt_feii_options['opt_template']['type']=='K10'):
			
			f_template, s_template, g_template, z_template = K10_opt_feii_template(p, lam_gal, opt_feii_templates, opt_feii_options, velscale)

			host_model = (host_model) - (f_template) - (s_template) - (g_template) - (z_template)
			comp_dict['F_OPT_FEII_TEMPLATE'] = f_template
			comp_dict['S_OPT_FEII_TEMPLATE'] = s_template
			comp_dict['G_OPT_FEII_TEMPLATE'] = g_template
			comp_dict['Z_OPT_FEII_TEMPLATE'] = z_template

	########################################################################################################


	############################# UV Iron Component ##########################################################

	if (uv_iron_template is not None):

		uv_iron_template = VW01_uv_iron_template(lam_gal, p, uv_iron_template, uv_iron_options, velscale, run_dir)
		host_model = (host_model) - (uv_iron_template)
		comp_dict['UV_IRON_TEMPLATE'] = uv_iron_template

	########################################################################################################

	############################# Balmer Continuum Component ###############################################

	if (balmer_template is not None):
		# Unpack Balmer template
		lam_balmer, spec_high_balmer, velscale_balmer = balmer_template
		# Parse Balmer options
		if (balmer_options['R_const']['bool']==False): 
			balmer_ratio = p['BALMER_RATIO']
		elif (balmer_options['R_const']['bool']==True): 
			balmer_ratio = balmer_options['R_const']['R_val']
		if (balmer_options['balmer_amp_const']['bool']==False): 
			balmer_amp = p['BALMER_AMP']
		elif (balmer_options['balmer_amp_const']['bool']==True): 
			balmer_amp = balmer_options['balmer_amp_const']['balmer_amp_val']
		if (balmer_options['balmer_fwhm_const']['bool']==False): 
			balmer_fwhm = p['BALMER_FWHM']
		elif (balmer_options['balmer_fwhm_const']['bool']==True): 
			balmer_fwhm = balmer_options['balmer_fwhm_const']['balmer_fwhm_val']
		if (balmer_options['balmer_voff_const']['bool']==False): 
			balmer_voff = p['BALMER_VOFF']
		elif (balmer_options['balmer_voff_const']['bool']==True): 
			balmer_voff = balmer_options['balmer_voff_const']['balmer_voff_val']
		if (balmer_options['Teff_const']['bool']==False): 
			balmer_Teff = p['BALMER_TEFF']
		elif (balmer_options['Teff_const']['bool']==True): 
			balmer_Teff = balmer_options['Teff_const']['Teff_val']
		if (balmer_options['tau_const']['bool']==False): 
			balmer_tau = p['BALMER_TAU']
		elif (balmer_options['tau_const']['bool']==True): 
			balmer_tau = balmer_options['tau_const']['tau_val']

		balmer_cont = generate_balmer_continuum(lam_gal,lam_balmer, spec_high_balmer, velscale_balmer,
					  balmer_ratio, balmer_amp, balmer_fwhm, balmer_voff, balmer_Teff, balmer_tau)

		host_model = (host_model) - (balmer_cont)
		comp_dict['BALMER_CONT'] = balmer_cont

	########################################################################################################

	############################# Emission Line Components #################################################

	# Iteratively generate lines from the line list using the line_constructor()
	for line in line_list:
		comp_dict = line_constructor(lam_gal,p,comp_dict,comp_options,line,line_list,velscale,noise)
		host_model = host_model - comp_dict[line]

	########################################################################################################

	############################# Host-galaxy Component ######################################################

	if (comp_options["fit_host"]==True):
		#
		if (host_options["vel_const"]["bool"]==True) & (host_options["disp_const"]["bool"]==True):
			# If both velocity and dispersion are constant, the host template(s) are pre-convolved
			# and the only thing left to do is to scale (or perform nnls for multiple templates)
			conv_host = host_template
			#
			if np.shape(conv_host)[1]==1:
				# conv_host = conv_host/np.median(conv_host) * p["HOST_TEMP_AMP"]
				conv_host = conv_host * p["HOST_TEMP_AMP"]
				host_galaxy = conv_host.reshape(-1)
			elif np.shape(conv_host)[1]>1:
				host_model[~np.isfinite(host_model)] = 0
				conv_host[~np.isfinite(conv_host)]	= 0
				# host_norm = np.median(host_model)
				# if (host_norm/host_norm!=1):
				# 	host_norm = 1
				weights	 = nnls(conv_host,host_model)#/host_norm) # scipy.optimize Non-negative Least Squares
				host_galaxy = (np.sum(weights*conv_host,axis=1)) #* host_norm
			#
		elif (host_options["vel_const"]["bool"]==False) | (host_options["disp_const"]["bool"]==False):
			# If templates velocity OR dispersion are not constant, we need to perform 
			# the convolution.
			ssp_fft, npad, vsyst = host_template
			if host_options["vel_const"]["bool"]==False:
				host_vel = p["HOST_TEMP_VEL"]
			elif host_options["vel_const"]["bool"]==True:
				host_vel = host_options["vel_const"]["val"]
			#
			if host_options["disp_const"]["bool"]==False:
				host_disp = p["HOST_TEMP_DISP"]
			elif host_options["disp_const"]["bool"]==True:
				host_disp = host_options["disp_const"]["val"]

			#
			conv_host	= convolve_gauss_hermite(ssp_fft,npad,float(velscale),\
						   [host_vel, host_disp],np.shape(lam_gal)[0],velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
			#
			if np.shape(conv_host)[1]==1:
				# conv_host = conv_host/np.median(conv_host) * p["HOST_TEMP_AMP"]
				conv_host = conv_host * p["HOST_TEMP_AMP"]
				host_galaxy = conv_host.reshape(-1)
			elif np.shape(conv_host)[1]>1:
				host_model[~np.isfinite(host_model)] = 0
				conv_host[~np.isfinite(conv_host)]	= 0
				# host_norm = np.median(host_model)
				# if (host_norm/host_norm!=1):
				# 	host_norm = 1
			weights	 = nnls(conv_host,host_model)#/host_norm) # scipy.optimize Non-negative Least Squares
			host_galaxy = (np.sum(weights*conv_host,axis=1))# * host_norm

		host_model = (host_model) - (host_galaxy) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['HOST_GALAXY'] = host_galaxy

	########################################################################################################   

	############################# LOSVD Component ####################################################

	if (comp_options["fit_losvd"]==True):
		#
		if (losvd_options["vel_const"]["bool"]==True) & (losvd_options["disp_const"]["bool"]==True):
			# If both velocity and dispersion are constant, the host template(s) are pre-convolved
			# and the only thing left to do is to scale (or perform nnls for multiple templates)
			conv_temp = stel_templates
			# print(np.shape(conv_temp))
			# print(np.shape(host_model))
			#
			host_model[~np.isfinite(host_model)] = 0
			conv_temp[~np.isfinite(conv_temp)]	= 0
			# host_norm = np.median(host_model)
			# if (host_norm/host_norm!=1) or (host_norm<1):
				# host_norm = 1
			weights	 = nnls(conv_temp,host_model)#/host_norm) # scipy.optimize Non-negative Least Squares
			host_galaxy = (np.sum(weights*conv_temp,axis=1)) #* host_norm
			#
			if (losvd_options['losvd_apoly']['bool']==True) and (losvd_options['losvd_apoly']['order']>=1):
				nw = np.linspace(-1,1,len(lam_gal))
				coeff = np.empty(losvd_options['losvd_apoly']['order']+1)
				for n in range(losvd_options['losvd_apoly']['order']+1):
					coeff[n] = p["LOSVD_LCOEFF_%d" % n]
				lpoly = np.polynomial.legendre.legval(nw, coeff)
				host_galaxy += lpoly

				if np.any(host_galaxy<0):
					host_galaxy+= -np.min(host_galaxy)

			# fig = plt.figure(figsize=(18,7))
			# ax1 = fig.add_subplot(1,1,1)
			# ax1.plot(lam_gal,host_galaxy)
			# plt.tight_layout()
			#
			#
		elif (losvd_options["vel_const"]["bool"]==False) | (losvd_options["disp_const"]["bool"]==False):
			# If templates velocity OR dispersion are not constant, we need to perform 
			# the convolution.
			temp_fft, npad, vsyst = stel_templates
			if losvd_options["vel_const"]["bool"]==False:
				stel_vel = p["STEL_VEL"]
			elif losvd_options["vel_const"]["bool"]==True:
				stel_vel = losvd_options["vel_const"]["val"]
			#
			if losvd_options["disp_const"]["bool"]==False:
				stel_disp = p["STEL_DISP"]
			elif losvd_options["disp_const"]["bool"]==True:
				stel_disp = losvd_options["disp_const"]["val"]

			#
			conv_temp	= convolve_gauss_hermite(temp_fft,npad,float(velscale),\
						   [stel_vel, stel_disp],np.shape(lam_gal)[0],velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
			#
			# print(np.shape(temp_fft))
			# print(len(conv_temp))
			# print(len(host_model))
			# sys.exit()

			host_model[~np.isfinite(host_model)] = 0
			conv_temp[~np.isfinite(conv_temp)]	= 0
			# host_norm = np.median(host_model)
			# if (host_norm/host_norm!=1) or (host_norm<1):
			# 	host_norm = 1
			weights	 = nnls(conv_temp,host_model)#/host_norm) # scipy.optimize Non-negative Least Squares
			host_galaxy = (np.sum(weights*conv_temp,axis=1)) #* host_norm
			#
			if (losvd_options['losvd_apoly']['bool']==True) and (losvd_options['losvd_apoly']['order']>=1):
				nw = np.linspace(-1,1,len(lam_gal))
				coeff = np.empty(losvd_options['losvd_apoly']['order']+1)
				for n in range(losvd_options['losvd_apoly']['order']+1):
					coeff[n] = p["LOSVD_LCOEFF_%d" % n]
				lpoly = np.polynomial.legendre.legval(nw, coeff)
				host_galaxy += lpoly 

				if np.any(host_galaxy<0):
					host_galaxy+= -np.min(host_galaxy)

		host_model = (host_model) - (host_galaxy) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['HOST_GALAXY'] = host_galaxy

		# # Unpack stel_templates
		# temp_fft, vsyst, npad = stel_templates
		# # Convolve the templates with a LOSVD
		# losvd_params = [p['STEL_VEL'],p['STEL_DISP']] # ind 0 = velocity*, ind 1 = sigma*
		# conv_temp	= convolve_gauss_hermite(temp_fft,npad,float(velscale),\
		# 			   losvd_params,np.shape(lam_gal)[0],velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
		
		# # Fitted weights of all templates using Non-negative Least Squares (NNLS)
		# host_model[host_model/host_model!=1] = 0 
		# conv_temp[conv_temp/conv_temp!=1]	= 0
		# host_norm = np.median(host_model) 
		# weights	 = nnls(conv_temp,host_model/host_norm) # scipy.optimize Non-negative Least Squares
		# host_galaxy = (np.sum(weights*conv_temp,axis=1)) * host_norm
		# comp_dict['HOST_GALAXY'] = host_galaxy

	 ########################################################################################################

	# The final model
	gmodel = np.sum((comp_dict[d] for d in comp_dict),axis=0)

	#########################################################################################################

	# Add combined lines to comp_dict
	for comb_line in combined_line_list:
		comp_dict[comb_line] = np.zeros(len(lam_gal))
		for indiv_line in combined_line_list[comb_line]["lines"]:
			comp_dict[comb_line]+=comp_dict[indiv_line]

	line_list = {**line_list, **combined_line_list}

	#########################################################################################################

	# Add last components to comp_dict for plotting purposes 
	# Add galaxy, sigma, model, and residuals to comp_dict
	comp_dict["DATA"]  = galaxy		  
	comp_dict["WAVE"]  = lam_gal 	  
	comp_dict["NOISE"] = noise		  
	comp_dict["MODEL"] = gmodel		  
	comp_dict["RESID"] = galaxy-gmodel

	########################## Fluxes & Equivalent Widths ###################################################
	# Equivalent widths of emission lines are stored in a dictionary and returned to emcee as metadata blob.
	# Velocity interpolation function
	interp_ftn = interp1d(lam_gal,np.arange(len(lam_gal))*velscale,bounds_error=False)
	if (fit_type=='final') and (output_model==False):
		# Create a single continuum component based on what was fit
		total_cont = np.zeros(len(lam_gal))
		agn_cont   = np.zeros(len(lam_gal))
		host_cont  = np.zeros(len(lam_gal))
		for key in comp_dict:
			if key in ["POWER","HOST_GALAXY","BALMER_CONT"]:
				total_cont+=comp_dict[key]
			if key in ["POWER","BALMER_CONT"]:
				agn_cont+=comp_dict[key]
			if key in ["HOST_GALAXY"]:
				host_cont+=comp_dict[key]
		# Get all spectral components, not including data, model, resid, and noise
		spec_comps = [i for i in comp_dict if i not in ["DATA","MODEL","WAVE","RESID","NOISE","POWER","HOST_GALAXY","BALMER_CONT"]]
		# Get keys of any lines that were fit for which we will compute eq. widths for
		lines = [line for line in line_list]
		fluxes	= {}
		eqwidths  = {}
		int_vel_disp = {}
		for key in spec_comps:
			flux = simps(comp_dict[key],lam_gal)
			# add key/value pair to dictionary
			fluxes[key+"_FLUX"] = flux
			# for line in lines:
			if (key in lines):
				comp = comp_dict[key]
				# if line_list[key]["line_profile"] in ["V","L"]:
				# 	# Truncate the component to zero for any values below the median noise level. 
				# 	# This is necessary because the wings of the Voigt and Lorentzian profiles extend to infinity,
				# 	# resulting in unrealistic line dispersions.
				# 	comp[comp < np.nanmedian(noise)] = 0
				eqwidth = simps(comp/total_cont,lam_gal)
			else: 
				eqwidth = simps(comp_dict[key]/total_cont,lam_gal)
			if ~np.isfinite(eqwidth):
				eqwidth=0.0
			# Add to eqwidth_dict
			eqwidths[key+"_EW"]  = eqwidth
			#
			if (key in lines):
				# Calculate integrated velocities and dispersions for each line
				# Calculate velocity scale centered on line
				vel = np.arange(len(lam_gal))*velscale - interp_ftn(line_list[key]["center"])
				full_profile = comp_dict[key]
				# Remove stray lines
				full_profile = remove_stray_lines(full_profile)
				# Normalized line profile
				norm_profile = full_profile/np.sum(full_profile)
				# Calculate integrated velocity in pixels units
				v_int = simps(vel*norm_profile,vel)/simps(norm_profile,vel)
				# Calculate integrated dispersion and correct for instrumental dispersion
				d_int = np.sqrt(simps(vel**2*norm_profile,vel)/simps(norm_profile,vel) - (v_int**2))
				d_int = np.sqrt(d_int**2 - (line_list[key]["fwhm_res_kms"]/2.3548)**2)
				if ~np.isfinite(d_int): d_int = 0.0
				if ~np.isfinite(v_int): v_int = 0.0
				int_vel_disp[key+"_DISP"] = d_int
				int_vel_disp[key+"_VINT"] = v_int
			# Calculate integrated FWHM for combined lines
			if (key in combined_line_list):
				comb_fwhm = combined_fwhm(lam_gal,comp_dict[key],line_list[key]["fwhm_res_kms"],velscale)
				int_vel_disp[key+"_FWHM"] = comb_fwhm

		# Continuum fluxes (to obtain continuum luminosities)
		cont_fluxes = {}
		#
		interp_tot  = interp1d(lam_gal,total_cont,kind='linear',bounds_error=False,fill_value=0.0)
		interp_agn  = interp1d(lam_gal,agn_cont  ,kind='linear',bounds_error=False,fill_value=0.0)
		interp_host = interp1d(lam_gal,host_cont ,kind='linear',bounds_error=False,fill_value=0.0)
		if (lam_gal[0]<1350) & (lam_gal[-1]>1350):
			cont_fluxes["F_CONT_TOT_1350"]  = interp_tot(1350.0) #total_cont[find_nearest(lam_gal,1350.0)[1]]#
			cont_fluxes["F_CONT_AGN_1350"]  = interp_agn(1350.0) #agn_cont[find_nearest(lam_gal,1350.0)[1]]  #
			cont_fluxes["F_CONT_HOST_1350"] = interp_host(1350.0) #host_cont[find_nearest(lam_gal,1350.0)[1]] #
		if (lam_gal[0]<3000) & (lam_gal[-1]>3000):
			cont_fluxes["F_CONT_TOT_3000"]  = interp_tot(3000.0) #total_cont[find_nearest(lam_gal,3000.0)[1]]
			cont_fluxes["F_CONT_AGN_3000"]  = interp_agn(3000.0) #agn_cont[find_nearest(lam_gal,3000.0)[1]]  
			cont_fluxes["F_CONT_HOST_3000"] = interp_host(3000.0) #host_cont[find_nearest(lam_gal,3000.0)[1]] 
		if (lam_gal[0]<5100) & (lam_gal[-1]>5100):
			cont_fluxes["F_CONT_TOT_5100"]  = interp_tot(5100.0) #total_cont[find_nearest(lam_gal,5100.0)[1]]#
			cont_fluxes["F_CONT_AGN_5100"]  = interp_agn(5100.0) #agn_cont[find_nearest(lam_gal,5100.0)[1]]  #
			cont_fluxes["F_CONT_HOST_5100"] = interp_host(5100.0) #host_cont[find_nearest(lam_gal,5100.0)[1]] #
		if (lam_gal[0]<4000) & (lam_gal[-1]>4000):
			cont_fluxes["HOST_FRAC_4000"] = interp_host(4000.0)/interp_tot(4000.0) #host_cont[find_nearest(lam_gal,4000.0)[1]]/total_cont[find_nearest(lam_gal,4000.0)[1]]#
			cont_fluxes["AGN_FRAC_4000"]  = interp_agn(4000.0)/interp_tot(4000.0) #agn_cont[find_nearest(lam_gal,4000.0)[1]]/total_cont[find_nearest(lam_gal,4000.0)[1]] #
		if (lam_gal[0]<7000) & (lam_gal[-1]>7000):
			cont_fluxes["HOST_FRAC_7000"] = interp_host(7000.0)/interp_tot(7000.0) #host_cont[find_nearest(lam_gal,7000.0)[1]]/total_cont[find_nearest(lam_gal,7000.0)[1]]#
			cont_fluxes["AGN_FRAC_7000"]  = interp_agn(7000.0)/interp_tot(7000.0) #agn_cont[find_nearest(lam_gal,7000.0)[1]]/total_cont[find_nearest(lam_gal,7000.0)[1]] #
		#		

	########################################################################################################

	if (fit_type=='init') and (output_model==False): # For max. likelihood fitting
		return gmodel, comp_dict
	if (fit_type=='init') and (output_model==True): # For max. likelihood fitting
		return comp_dict
	elif (fit_type=='line_test'):
		return comp_dict
	elif (fit_type=='final') and (output_model==False): # For emcee
		return gmodel, fluxes, eqwidths, cont_fluxes, int_vel_disp
	elif (fit_type=='final') and (output_model==True): # output all models for best-fit model
		return comp_dict

########################################################################################################


#### Host-Galaxy Template##############################################################################

def generate_host_template(lam_gal,host_options,fwhm_gal,fit_mask,velscale,verbose=True):
	"""
	
	"""

	ages = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
			9.0, 10.0, 11.0, 12.0, 13.0, 14.0],dtype=float)
	temp = ["badass_data_files/eMILES/Eku1.30Zp0.06T00.0900_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.1000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.2000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.3000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.4000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.5000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.6000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.7000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.8000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T00.9000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T01.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T02.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T03.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T04.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T05.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T06.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T07.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T08.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T09.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T10.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T11.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T12.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T13.0000_iTp0.00_baseFe_linear_FWHM_variable.fits",
			"badass_data_files/eMILES/Eku1.30Zp0.06T14.0000_iTp0.00_baseFe_linear_FWHM_variable.fits"
			]
	#
	fwhm_temp = 2.51 # FWHM resolution of eMILES in Å
	# Open a fits file
	hdu = fits.open(temp[0])
	ssp = hdu[0].data 
	h = hdu[0].header
	hdu.close()
	lam_temp = np.array(h['CRVAL1'] + h['CDELT1']*np.arange(h['NAXIS1']))
	mask = ((lam_temp>=(lam_gal[0]-100.0)) & (lam_temp<=(lam_gal[-1]+100.0)))
	# Apply mask and get lamRange
	ssp	  = ssp[mask]
	lam_temp = lam_temp[mask]
	lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
	# Create templates array
	sspNew = log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
	templates = np.empty((sspNew.size, len(host_options["age"])))
	# Variable sigma
	fwhm_gal_interp = np.interp(lam_temp, lam_gal, fwhm_gal)
	fwhm_dif  = np.sqrt((fwhm_gal_interp**2 - fwhm_temp**2).clip(0))
	sigma	 = fwhm_dif/2.355/h['CDELT1'] # Sigma difference in pixels
	#
	for j, age in enumerate(host_options["age"]):
		hdu = fits.open(temp[np.where(ages==age)[0][0]])
		ssp = hdu[0].data
		ssp = ssp[mask]
		ssp = gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
		sspNew,loglam_temp,velscale_temp = log_rebin(lamRange_temp, ssp, velscale=velscale)#[0]
		templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
		hdu.close()
	#
	# Calculate npad and vsyst
	c = 299792.458 # speed of light in km/s
	vsyst = np.log(lam_temp[0]/lam_gal[0])*c	# km/s
	ssp_fft, npad = template_rfft(templates) # we will use this throughout the code
	#
	# Pre-convolve the templates if the velocity and dispersion are to be constant during the fit; 
	# this reduces the number of convolution computations during the fit.
	if (host_options["vel_const"]["bool"]==True) & (host_options["disp_const"]["bool"]==True):
		host_vel = host_options["vel_const"]["val"]
		host_disp = host_options["disp_const"]["val"]

		conv_host	= convolve_gauss_hermite(ssp_fft,npad,float(velscale),\
					   [host_vel, host_disp],np.shape(lam_gal)[0],velscale_ratio=1,sigma_diff=0,vsyst=vsyst)
		host_template = conv_host
		#
		# fig = plt.figure(figsize=(18,7))
		# ax1 = fig.add_subplot(1,1,1)
		# ax1.plot(lam_gal,host_template.reshape(-1))
		# plt.tight_layout()
		#
	# If velocity and dispersion of the host template are free parameters, then BADASS passes
	# the fft of the host template(s) to the fit model for convolution during the fit.
	elif (host_options["vel_const"]["bool"]==False) | (host_options["disp_const"]["bool"]==False):
		host_template = (ssp_fft, npad, vsyst)
	#
	# fig = plt.figure(figsize=(18,7))
	# ax1 = fig.add_subplot(1,1,1)
	# for i in range(np.shape(templates)[1]):
	# 	ax1.plot(np.exp(loglam_temp),templates[:,i])
	# plt.tight_layout()
	#
	return host_template


##################################################################################


#### Optical FeII Templates ##############################################################

def initialize_opt_feii(lam_gal, opt_feii_options, fwhm_gal,fit_mask, velscale):
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
	if (opt_feii_options['opt_template']['type']=='VC04'):
		# Load the data into Pandas DataFrames
		df_br = pd.read_csv("badass_data_files/feii_templates/veron-cetty_2004/VC04_br_feii_template.csv")
		df_na = pd.read_csv("badass_data_files/feii_templates/veron-cetty_2004/VC04_na_feii_template.csv")
		# Generate a new grid with the original resolution, but the size of the fitting region
		dlam_feii = df_br["angstrom"].to_numpy()[1]-df_br["angstrom"].to_numpy()[0] # angstroms
		npad = 100 # anstroms
		lam_feii	 = np.arange(np.min(lam_gal)-npad, np.max(lam_gal)+npad,dlam_feii) # angstroms
		# Interpolate the original template onto the new grid
		interp_ftn_br = interp1d(df_br["angstrom"].to_numpy(),df_br["flux"].to_numpy(),kind='linear',bounds_error=False,fill_value=(0.0,0.0))
		interp_ftn_na = interp1d(df_na["angstrom"].to_numpy(),df_na["flux"].to_numpy(),kind='linear',bounds_error=False,fill_value=(0.0,0.0))
		spec_feii_br = interp_ftn_br(lam_feii)
		spec_feii_na = interp_ftn_na(lam_feii)
		# Convolve templates to the native resolution of SDSS
		fwhm_feii = 1.0 # templates were created with 1.0 FWHM resolution
		fwhm_gal_interp = np.interp(lam_feii, lam_gal, fwhm_gal)
		fwhm_diff = np.sqrt((fwhm_gal_interp**2 - fwhm_feii**2).clip(0))
		sigma = fwhm_diff/2.3548/dlam_feii # Sigma difference in pixels
		spec_feii_br = gaussian_filter1d(spec_feii_br, sigma)
		spec_feii_na = gaussian_filter1d(spec_feii_na, sigma)
		# log-rebin the spectrum to same velocity scale as the input galaxy
		lamRange_feii = [np.min(lam_feii), np.max(lam_feii)]
		spec_feii_br_new, loglam_feii, velscale_feii = log_rebin(lamRange_feii, spec_feii_br, velscale=velscale)#[0]
		spec_feii_na_new, loglam_feii, velscale_feii = log_rebin(lamRange_feii, spec_feii_na, velscale=velscale)#[0]
		#
		# fig = plt.figure(figsize=(18,7))
		# ax1 = fig.add_subplot(1,1,1)
		# ax1.plot(np.exp(loglam_feii),spec_feii_br_new, linewidth=0.5)
		# ax1.plot(np.exp(loglam_feii),spec_feii_na_new, linewidth=0.5)
		# plt.tight_layout()
		#
		# Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
		br_opt_feii_fft, npad = template_rfft(spec_feii_br_new)
		na_opt_feii_fft, npad = template_rfft(spec_feii_na_new)
		# The FeII templates are offset from the input galaxy spectrum by 100 A, so we 
		# shift the spectrum to match that of the input galaxy.
		c = 299792.458 # speed of light in km/s
		vsyst = np.log(lam_feii[0]/lam_gal[0])*c

		# If opt_fwhm_const=True AND opt_voff_const=True, we preconvolve the templates so we don't have to 
		# during the fit
		if (opt_feii_options["opt_fwhm_const"]["bool"]==True) & (opt_feii_options["opt_voff_const"]["bool"]==True):

			br_fwhm = opt_feii_options["opt_fwhm_const"]["br_opt_feii_val"]
			na_fwhm = opt_feii_options["opt_fwhm_const"]["na_opt_feii_val"]
			#
			br_voff = opt_feii_options["opt_voff_const"]["br_opt_feii_val"]
			na_voff = opt_feii_options["opt_voff_const"]["na_opt_feii_val"]
			#
			br_conv_temp = convolve_gauss_hermite(br_opt_feii_fft, npad, float(velscale),\
									 	 [br_voff, br_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
			na_conv_temp = convolve_gauss_hermite(na_opt_feii_fft, npad, float(velscale),\
									 	 [na_voff, na_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
			#
			# fig = plt.figure(figsize=(18,7))
			# ax1 = fig.add_subplot(1,1,1)
			# ax1.plot(lam_gal,br_conv_temp, linewidth=0.5)
			# ax1.plot(lam_gal,na_conv_temp, linewidth=0.5)
			# plt.tight_layout()
			#
			opt_feii_templates = (br_conv_temp, na_conv_temp)

		elif (opt_feii_options["opt_fwhm_const"]["bool"]==False) | (opt_feii_options["opt_voff_const"]["bool"]==False):

		
			# We return a tuple consisting of the FFT of the broad and narrow templates, npad, and vsyst, 
			# which are needed for the convolution.
			opt_feii_templates =(br_opt_feii_fft, na_opt_feii_fft, npad, vsyst)

		return opt_feii_templates 


	elif (opt_feii_options['opt_template']['type']=='K10'):

		# The procedure for the K10 templates is slightly difference since their relative intensities
		# are temperature dependent.  We must create a Gaussian emission line for each individual line, 
		# and store them as an array, for each of the F, S, G, and Z transitions.  We treat each transition
		# as a group of templates, which will be convolved together, but relative intensities will be calculated
		# for separately. 

		def gaussian_angstroms(x, center, amp, fwhm, voff):
			sigma = fwhm/2.3548
			x = x.reshape((len(x),1))
			g = amp*np.exp(-0.5*(x-(center))**2/(sigma)**2) # construct gaussian
			g = np.sum(g,axis=1)
			# Normalize to 1
		#	 g = g/np.max(g)
			# Make sure edges of gaussian are zero to avoid wierd things
			# g[g<1.0e-6] = 0.0
			# Replace the ends with the same value 
			g[0]  = g[1]
			g[-1] = g[-2]
			return g
		#
		# Read in template data
		F_trans_df = pd.read_csv('badass_data_files/feii_templates/kovacevic_2010/K10_F_transitions.csv')
		S_trans_df = pd.read_csv('badass_data_files/feii_templates/kovacevic_2010/K10_S_transitions.csv')
		G_trans_df = pd.read_csv('badass_data_files/feii_templates/kovacevic_2010/K10_G_transitions.csv')
		Z_trans_df = pd.read_csv('badass_data_files/feii_templates/kovacevic_2010/K10_Z_transitions.csv')
		# Generate a high-resolution wavelength scale that is universal to all transitions
		fwhm = 1.0 # Angstroms
		dlam_feii = 0.1 # linear spacing in Angstroms
		npad = 100
		lam_feii = np.arange(np.min(lam_gal)-npad, np.max(lam_gal)+npad, dlam_feii)
		lamRange_feii = [np.min(lam_feii), np.max(lam_feii)]
		# Get size of output log-rebinned spectrum 
		F = gaussian_angstroms(lam_feii, F_trans_df["wavelength"].to_numpy()[0], 1.0, fwhm, 0.0)   
		new_size, loglam_feii, velscale_feii = log_rebin(lamRange_feii, F, velscale=velscale)
		# Create storage arrays for each emission line of each transition
		F_templates = np.empty(( len(new_size), len(F_trans_df['wavelength'].to_numpy()) ))
		S_templates = np.empty(( len(new_size), len(S_trans_df['wavelength'].to_numpy()) ))
		G_templates = np.empty(( len(new_size), len(G_trans_df['wavelength'].to_numpy()) ))
		Z_templates = np.empty(( len(new_size), len(Z_trans_df['wavelength'].to_numpy()) ))
		# Generate templates with a amplitude of 1.0
		for i in range(np.shape(F_templates)[1]):
			F = gaussian_angstroms(lam_feii, F_trans_df["wavelength"].to_numpy()[i], 1.0, fwhm, 0.0)	
			new_F = log_rebin(lamRange_feii, F, velscale=velscale)[0]
			F_templates[:,i] = new_F/np.max(new_F)
		for i in range(np.shape(S_templates)[1]): 
			S = gaussian_angstroms(lam_feii, S_trans_df["wavelength"].to_numpy()[i], 1.0, fwhm, 0.0)
			new_S = log_rebin(lamRange_feii, S, velscale=velscale)[0]
			S_templates[:,i] = new_S/np.max(new_S)
		for i in range(np.shape(G_templates)[1]):
			G = gaussian_angstroms(lam_feii, G_trans_df["wavelength"].to_numpy()[i], 1.0, fwhm, 0.0)
			new_G = log_rebin(lamRange_feii, G, velscale=velscale)[0]
			G_templates[:,i] = new_G/np.max(new_G)
		for i in range(np.shape(Z_templates)[1]):
			Z = gaussian_angstroms(lam_feii, Z_trans_df["wavelength"].to_numpy()[i], 1.0, fwhm, 0.0)
			new_Z = log_rebin(lamRange_feii, Z, velscale=velscale)[0]
			Z_templates[:,i] = new_Z/np.max(new_Z)

		# Pre-compute the FFT for each transition
		F_trans_fft, F_trans_npad = template_rfft(F_templates)
		S_trans_fft, S_trans_npad = template_rfft(S_templates)
		G_trans_fft, G_trans_npad = template_rfft(G_templates)
		Z_trans_fft, Z_trans_npad = template_rfft(Z_templates)
		npad = F_trans_npad

		c = 299792.458 # speed of light in km/s
		vsyst = np.log(lam_feii[0]/lam_gal[0])*c

		# If opt_fwhm_const=True AND opt_voff_const=True, we preconvolve the templates so we don't have to 
		# during the fit
		if (opt_feii_options["opt_fwhm_const"]["bool"]==True) & (opt_feii_options["opt_voff_const"]["bool"]==True):

			feii_fwhm = opt_feii_options["opt_fwhm_const"]["opt_feii_val"]
			#
			feii_voff = opt_feii_options["opt_voff_const"]["opt_feii_val"]
			#
			f_conv_temp = convolve_gauss_hermite(F_trans_fft, F_trans_npad, float(velscale),\
									 	 [feii_voff, feii_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
			s_conv_temp = convolve_gauss_hermite(S_trans_fft, S_trans_npad, float(velscale),\
									 	 [feii_voff, feii_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
			g_conv_temp = convolve_gauss_hermite(G_trans_fft, G_trans_npad, float(velscale),\
									 	 [feii_voff, feii_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
			z_conv_temp = convolve_gauss_hermite(Z_trans_fft, Z_trans_npad, float(velscale),\
									 	 [feii_voff, feii_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
			#
			opt_feii_templates = (f_conv_temp, F_trans_df['wavelength'].to_numpy() ,F_trans_df['gf'].to_numpy(), F_trans_df['E2_J'].to_numpy(),
					s_conv_temp, S_trans_df['wavelength'].to_numpy() ,S_trans_df['gf'].to_numpy(), S_trans_df['E2_J'].to_numpy(),
					g_conv_temp, G_trans_df['wavelength'].to_numpy() ,G_trans_df['gf'].to_numpy(), G_trans_df['E2_J'].to_numpy(),
					z_conv_temp, Z_trans_df['rel_int'].to_numpy()
					)
		#
		elif (opt_feii_options["opt_fwhm_const"]["bool"]==False) | (opt_feii_options["opt_voff_const"]["bool"]==False):

			opt_feii_templates = (F_trans_fft, F_trans_df['wavelength'].to_numpy() ,F_trans_df['gf'].to_numpy(), F_trans_df['E2_J'].to_numpy(),
					S_trans_fft, S_trans_df['wavelength'].to_numpy() ,S_trans_df['gf'].to_numpy(), S_trans_df['E2_J'].to_numpy(),
					G_trans_fft, G_trans_df['wavelength'].to_numpy() ,G_trans_df['gf'].to_numpy(), G_trans_df['E2_J'].to_numpy(),
					Z_trans_fft, Z_trans_df['rel_int'].to_numpy(),
					npad, vsyst
					)

		# Return a list of arrays which will be unpacked during the fitting process
		return opt_feii_templates

#### Optical FeII Template #########################################################

def VC04_opt_feii_template(p, lam_gal, opt_feii_templates, opt_feii_options, velscale):

	# Unpack opt_feii_templates
	# Parse FeII options
	#
	if (opt_feii_options['opt_amp_const']['bool']==False): # if amp not constant
		na_opt_feii_amp = p['NA_OPT_FEII_AMP']
		br_opt_feii_amp = p['BR_OPT_FEII_AMP']
	elif (opt_feii_options['opt_amp_const']['bool']==True): # if amp constant
		na_opt_feii_amp = opt_feii_options['opt_amp_const']['na_opt_feii_val']
		br_opt_feii_amp = opt_feii_options['opt_amp_const']['br_opt_feii_val']
	#
	if (opt_feii_options['opt_fwhm_const']['bool']==False): # if amp not constant
		na_opt_feii_fwhm = p['NA_OPT_FEII_FWHM']
		br_opt_feii_fwhm = p['BR_OPT_FEII_FWHM']
	elif (opt_feii_options['opt_fwhm_const']['bool']==True): # if amp constant
		na_opt_feii_fwhm = opt_feii_options['opt_fwhm_const']['na_opt_feii_val']
		br_opt_feii_fwhm = opt_feii_options['opt_fwhm_const']['br_opt_feii_val']
	#
	if (opt_feii_options['opt_voff_const']['bool']==False): # if amp not constant
		na_opt_feii_voff = p['NA_OPT_FEII_VOFF']
		br_opt_feii_voff = p['BR_OPT_FEII_VOFF']
	elif (opt_feii_options['opt_voff_const']['bool']==True): # if amp constant
		na_opt_feii_voff = opt_feii_options['opt_voff_const']['na_opt_feii_val']
		br_opt_feii_voff = opt_feii_options['opt_voff_const']['br_opt_feii_val']
	#
	if (opt_feii_options["opt_fwhm_const"]["bool"]==True) & (opt_feii_options["opt_voff_const"]["bool"]==True):
		br_conv_temp, na_conv_temp = opt_feii_templates
		# Templates are already convolved so just normalize and multiplfy by amplitude 
		# br_opt_feii_template = br_conv_temp/np.max(br_conv_temp) * br_opt_feii_amp
		# na_opt_feii_template = na_conv_temp/np.max(na_conv_temp) * na_opt_feii_amp
		br_opt_feii_template = br_conv_temp * br_opt_feii_amp
		na_opt_feii_template = na_conv_temp * na_opt_feii_amp

		br_opt_feii_template = br_opt_feii_template.reshape(-1)
		na_opt_feii_template = na_opt_feii_template.reshape(-1)
		# Set fitting region outside of template to zero to prevent convolution loops
		br_opt_feii_template[(lam_gal < 3400) & (lam_gal > 7200)] = 0
		na_opt_feii_template[(lam_gal < 3400) & (lam_gal > 7200)] = 0
		#
		# print(br_opt_feii_amp,na_opt_feii_amp)
		# fig = plt.figure(figsize=(18,7))
		# ax1 = fig.add_subplot(1,1,1)
		# ax1.plot(lam_gal,na_opt_feii_template, linewidth=0.5)
		# ax1.plot(lam_gal,br_opt_feii_template, linewidth=0.5)
		# plt.tight_layout()
		# sys.exit()
		#
	elif (opt_feii_options["opt_fwhm_const"]["bool"]==False) | (opt_feii_options["opt_voff_const"]["bool"]==False):
		br_opt_feii_fft, na_opt_feii_fft, npad, vsyst = opt_feii_templates

		br_conv_temp = convolve_gauss_hermite(br_opt_feii_fft, npad, float(velscale),
										 	 [br_opt_feii_voff, br_opt_feii_fwhm/2.3548], lam_gal.shape[0], 
										  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
		#
		na_conv_temp = convolve_gauss_hermite(na_opt_feii_fft, npad, float(velscale),
										 	 [na_opt_feii_voff, na_opt_feii_fwhm/2.3548], lam_gal.shape[0], 
										  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
		# Re-normalize to 1
		# br_conv_temp = br_conv_temp/np.max(br_conv_temp)
		# na_conv_temp = na_conv_temp/np.max(na_conv_temp)
		# Multiplyy by amplitude
		br_opt_feii_template = br_opt_feii_amp * br_conv_temp
		na_opt_feii_template = na_opt_feii_amp * na_conv_temp
		# Reshape
		br_opt_feii_template = br_opt_feii_template.reshape(-1)
		na_opt_feii_template = na_opt_feii_template.reshape(-1)
		# Set fitting region outside of template to zero to prevent convolution loops
		br_opt_feii_template[(lam_gal < 3400) & (lam_gal > 7200)] = 0
		na_opt_feii_template[(lam_gal < 3400) & (lam_gal > 7200)] = 0

	return br_opt_feii_template, na_opt_feii_template

####################################################################################


#### UV Iron Template ##############################################################
	
def initialize_uv_iron(lam_gal, feii_options, fwhm_gal,fit_mask, velscale):
	"""
	Generate UV Iron template.
	"""

	# Load the data into Pandas DataFrames
	# df_uviron = pd.read_csv("badass_data_files/feii_templates/vestergaard-wilkes_2001/VW01_UV_B_47_191.csv") # UV B+47+191
	df_uviron = pd.read_csv("badass_data_files/feii_templates/vestergaard-wilkes_2001/VW01_UV_B.csv") # UV B only

	# Generate a new grid with the original resolution, but the size of the fitting region
	dlam_uviron = df_uviron["angstrom"].to_numpy()[1]-df_uviron["angstrom"].to_numpy()[0] # angstroms
	npad = 100 # anstroms
	lam_uviron	 = np.arange(np.min(lam_gal)-npad, np.max(lam_gal)+npad,dlam_uviron) # angstroms
	# Interpolate the original template onto the new grid
	interp_ftn_uv = interp1d(df_uviron["angstrom"].to_numpy(),df_uviron["flux"].to_numpy(),kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
	spec_uviron = interp_ftn_uv(lam_uviron)
	# log-rebin the spectrum to same velocity scale as the input galaxy
	lamRange_uviron = [np.min(lam_uviron), np.max(lam_uviron)]
	spec_uviron_new, loglam_uviron, velscale_uviron = log_rebin(lamRange_uviron, spec_uviron, velscale=velscale)#[0]
	# Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
	uv_iron_fft, npad = template_rfft(spec_uviron_new)
	# The FeII templates are offset from the input galaxy spectrum by 100 A, so we 
	# shift the spectrum to match that of the input galaxy.
	c = 299792.458 # speed of light in km/s
	vsyst = np.log(lam_uviron[0]/lam_gal[0])*c
	# We return a tuple consisting of the FFT of the broad and narrow templates, npad, and vsyst, 
	# which are needed for the convolution.

	return (uv_iron_fft, npad, vsyst)

####################################################################################

#### Balmer Template ###############################################################

def initialize_balmer(lam_gal, balmer_options, fwhm_gal,fit_mask, velscale):
	# Import the template for the higher-order balmer lines (7 <= n <= 500)
	# df = pd.read_csv("badass_data_files/balmer_template/higher_order_balmer.csv")
	df = pd.read_csv("badass_data_files/balmer_template/higher_order_balmer_n8_500.csv")
	# Generate a new grid with the original resolution, but the size of the fitting region
	dlam_balmer = df["angstrom"].to_numpy()[1]-df["angstrom"].to_numpy()[0] # angstroms
	npad = 100 # angstroms
	lam_balmer = np.arange(np.min(lam_gal)-npad, np.max(lam_gal)+npad,dlam_balmer) # angstroms
	# Interpolate the original template onto the new grid
	interp_ftn_balmer = interp1d(df["angstrom"].to_numpy(),df["flux"].to_numpy(),kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
	spec_high_balmer = interp_ftn_balmer(lam_balmer)
	# Calculate the difference in instrumental dispersion between SDSS and the template
	lamRange_balmer = [np.min(lam_balmer), np.max(lam_balmer)]
	fwhm_balmer = 1.0
	fwhm_gal_interp = np.interp(lam_balmer, lam_gal, fwhm_gal)
	fwhm_diff = np.sqrt((fwhm_gal_interp**2 - fwhm_balmer**2).clip(0))
	sigma = fwhm_diff/2.3548/dlam_balmer # Sigma difference in pixels
	# Convolve the FeII templates to the SDSS resolution
	spec_high_balmer = gaussian_filter1d(spec_high_balmer, sigma)
	# Log-rebin to same velocity scale as galaxy
	spec_high_balmer_new, loglam_balmer, velscale_balmer = log_rebin(lamRange_balmer, spec_high_balmer, velscale=velscale)#[0]
	if (np.sum(spec_high_balmer_new)>0):
		# Normalize to 1
		spec_high_balmer_new = spec_high_balmer_new/np.max(spec_high_balmer_new)
	# Package the wavelength vector and template
	balmer_template = (np.exp(loglam_balmer), spec_high_balmer_new, velscale_balmer)

	return balmer_template

####################################################################################


def get_fwhm_res(fwhm_gal_ftn,line_center,line_voff):
		c = 299792.458
		fwhm_res = (fwhm_gal_ftn(line_center + 
			   	   (line_voff*line_center/c))/(line_center + 
				   (line_voff*line_center/c))*c)
		return fwhm_res


####################################################################################

def K10_opt_feii_template(p, lam_gal, opt_feii_templates, opt_feii_options, velscale):
	"""
	Constructs an Kovacevic et al. 2010 FeII template using a series of Gaussians and ensures
	no lines are created at the edges of the fitting region.
	"""
	
	# Parse FeII options
	if (opt_feii_options['opt_amp_const']['bool']==False): # if amp not constant
		f_feii_amp  = p['OPT_FEII_F_AMP']
		s_feii_amp  = p['OPT_FEII_S_AMP']
		g_feii_amp  = p['OPT_FEII_G_AMP']
		z_feii_amp  = p['OPT_FEII_Z_AMP']
	elif (opt_feii_options['opt_amp_const']['bool']==True): # if amp constant
		f_feii_amp  = opt_feii_options['opt_amp_const']['f_feii_val']
		s_feii_amp  = opt_feii_options['opt_amp_const']['s_feii_val']
		g_feii_amp  = opt_feii_options['opt_amp_const']['g_feii_val']
		z_feii_amp  = opt_feii_options['opt_amp_const']['z_feii_val']
	#
	if (opt_feii_options['opt_fwhm_const']['bool']==False): # if fwhm not constant
		opt_feii_fwhm = p['OPT_FEII_FWHM']
	elif (opt_feii_options['opt_fwhm_const']['bool']==True): # if fwhm constant
		opt_feii_fwhm = opt_feii_options['opt_fwhm_const']['opt_feii_val']
	#
	if (opt_feii_options['opt_voff_const']['bool']==False): # if voff not constant
		opt_feii_voff = p['OPT_FEII_VOFF']
	elif (opt_feii_options['opt_voff_const']['bool']==True): # if voff constant
		opt_feii_voff = opt_feii_options['opt_voff_const']['opt_feii_val']
	#
	if (opt_feii_options['opt_temp_const']['bool']==False): # if temp not constant
		opt_feii_temp = p['OPT_FEII_TEMP']
	elif (opt_feii_options['opt_temp_const']['bool']==True): # if temp constant
		opt_feii_temp = opt_feii_options['opt_temp_const']['opt_feii_val']

	if (opt_feii_options["opt_fwhm_const"]["bool"]==True) & (opt_feii_options["opt_voff_const"]["bool"]==True):
		#
		# Unpack tables for each template
		f_conv_temp, f_feii_center, f_feii_gf, f_feii_e2  = (opt_feii_templates[0], opt_feii_templates[1], opt_feii_templates[2], opt_feii_templates[3])
		s_conv_temp, s_feii_center, s_feii_gf, s_feii_e2  = (opt_feii_templates[4], opt_feii_templates[5], opt_feii_templates[6], opt_feii_templates[7])
		g_conv_temp, g_feii_center, g_feii_gf, g_feii_e2  = (opt_feii_templates[8], opt_feii_templates[9], opt_feii_templates[10], opt_feii_templates[11])
		z_conv_temp, z_feii_rel_int					   = (opt_feii_templates[12], opt_feii_templates[13])
		# F-template
		# Normalize amplitudes to 1
		f_norm = np.array([np.max(f_conv_temp[:,i]) for i in range(np.shape(f_conv_temp)[1])])
		f_norm[f_norm<1.e-6] = 1.0
		f_conv_temp = f_conv_temp/f_norm
		# Calculate temperature dependent relative intensities 
		f_feii_rel_int = calculate_k10_rel_int("F",f_feii_center, f_feii_gf, f_feii_e2, f_feii_amp, opt_feii_temp)
		# Multiply by relative intensities
		f_conv_temp = f_conv_temp * f_feii_rel_int
		# Sum templates along rows
		f_template = np.sum(f_conv_temp, axis=1)
		f_template[(lam_gal <4472) & (lam_gal >5147)] = 0

		# S-template
		# Normalize amplitudes to 1
		s_norm = np.array([np.max(s_conv_temp[:,i]) for i in range(np.shape(s_conv_temp)[1])])
		s_norm[s_norm<1.e-6] = 1.0
		s_conv_temp = s_conv_temp/s_norm
		# Calculate temperature dependent relative intensities 
		s_feii_rel_int = calculate_k10_rel_int("S",s_feii_center, s_feii_gf, s_feii_e2, s_feii_amp, opt_feii_temp)
		# Multiply by relative intensities
		s_conv_temp = s_conv_temp * s_feii_rel_int
		# Sum templates along rows
		s_template = np.sum(s_conv_temp, axis=1)
		s_template[(lam_gal <4731) & (lam_gal >5285)] = 0

		# G-template
		# Normalize amplitudes to 1
		g_norm = np.array([np.max(g_conv_temp[:,i]) for i in range(np.shape(g_conv_temp)[1])])
		g_norm[g_norm<1.e-6] = 1.0
		g_conv_temp = g_conv_temp/g_norm
		# Calculate temperature dependent relative intensities 
		g_feii_rel_int = calculate_k10_rel_int("G",g_feii_center, g_feii_gf, g_feii_e2, g_feii_amp, opt_feii_temp)
		# Multiply by relative intensities
		g_conv_temp = g_conv_temp * g_feii_rel_int
		# Sum templates along rows
		g_template = np.sum(g_conv_temp, axis=1)
		g_template[(lam_gal <4472) & (lam_gal >5147)] = 0

		# Z template
		# Normalize amplitudes to 1
		z_norm = np.array([np.max(z_conv_temp[:,i]) for i in range(np.shape(z_conv_temp)[1])])
		z_norm[z_norm<1.e-6] = 1.0
		z_conv_temp = z_conv_temp/z_norm
		# Multiply by relative intensities
		z_conv_temp = z_conv_temp * z_feii_rel_int
		# Sum templates along rows
		z_template = np.sum(z_conv_temp, axis=1)
		# Multiply by FeII amplitude
		z_template = z_template * z_feii_amp
		z_template[(lam_gal <4418) & (lam_gal >5428)] = 0
		#
		# fig = plt.figure(figsize=(18,7))
		# ax1 = fig.add_subplot(1,1,1)
		# ax1.plot(lam_gal,f_template, linewidth=0.5)
		# ax1.plot(lam_gal,s_template, linewidth=0.5)
		# ax1.plot(lam_gal,g_template, linewidth=0.5)
		# ax1.plot(lam_gal,z_template, linewidth=0.5)
		# plt.tight_layout()
		# sys.exit()
		#

	elif (opt_feii_options["opt_fwhm_const"]["bool"]==False) | (opt_feii_options["opt_voff_const"]["bool"]==False):
		#
		# Unpack tables for each template
		f_feii_fft, f_feii_center, f_feii_gf, f_feii_e2  = (opt_feii_templates[0], opt_feii_templates[1], opt_feii_templates[2], opt_feii_templates[3])
		s_feii_fft, s_feii_center, s_feii_gf, s_feii_e2  = (opt_feii_templates[4], opt_feii_templates[5], opt_feii_templates[6], opt_feii_templates[7])
		g_feii_fft, g_feii_center, g_feii_gf, g_feii_e2  = (opt_feii_templates[8], opt_feii_templates[9], opt_feii_templates[10], opt_feii_templates[11])
		z_feii_fft, z_feii_rel_int					   = (opt_feii_templates[12], opt_feii_templates[13])
		npad											 = opt_feii_templates[14]
		vsyst											= opt_feii_templates[15]
		# F-template
		# Perform the convolution
		f_conv_temp = convolve_gauss_hermite(f_feii_fft, npad, float(velscale),\
									 	 [opt_feii_voff, opt_feii_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
		# Normalize amplitudes to 1
		f_norm = np.array([np.max(f_conv_temp[:,i]) for i in range(np.shape(f_conv_temp)[1])])
		f_norm[f_norm<1.e-6] = 1.0
		f_conv_temp = f_conv_temp/f_norm
		# Calculate temperature dependent relative intensities 
		f_feii_rel_int = calculate_k10_rel_int("F",f_feii_center, f_feii_gf, f_feii_e2, f_feii_amp, opt_feii_temp)
		# Multiply by relative intensities
		f_conv_temp = f_conv_temp * f_feii_rel_int
		# Sum templates along rows
		f_template = np.sum(f_conv_temp, axis=1)
		f_template[(lam_gal <4472) & (lam_gal >5147)] = 0

		# S-template
		# Perform the convolution
		s_conv_temp = convolve_gauss_hermite(s_feii_fft, npad, float(velscale),\
									 	 [opt_feii_voff, opt_feii_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
		# Normalize amplitudes to 1
		s_norm = np.array([np.max(s_conv_temp[:,i]) for i in range(np.shape(s_conv_temp)[1])])
		s_norm[s_norm<1.e-6] = 1.0
		s_conv_temp = s_conv_temp/s_norm
		# Calculate temperature dependent relative intensities 
		s_feii_rel_int = calculate_k10_rel_int("S",s_feii_center, s_feii_gf, s_feii_e2, s_feii_amp, opt_feii_temp)
		# Multiply by relative intensities
		s_conv_temp = s_conv_temp * s_feii_rel_int
		# Sum templates along rows
		s_template = np.sum(s_conv_temp, axis=1)
		s_template[(lam_gal <4731) & (lam_gal >5285)] = 0

		# G-template
		# Perform the convolution
		g_conv_temp = convolve_gauss_hermite(g_feii_fft, npad, float(velscale),\
									 	 [opt_feii_voff, opt_feii_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
		# Normalize amplitudes to 1
		g_norm = np.array([np.max(g_conv_temp[:,i]) for i in range(np.shape(g_conv_temp)[1])])
		g_norm[g_norm<1.e-6] = 1.0
		g_conv_temp = g_conv_temp/g_norm
		# Calculate temperature dependent relative intensities 
		g_feii_rel_int = calculate_k10_rel_int("G",g_feii_center, g_feii_gf, g_feii_e2, g_feii_amp, opt_feii_temp)
		# Multiply by relative intensities
		g_conv_temp = g_conv_temp * g_feii_rel_int
		# Sum templates along rows
		g_template = np.sum(g_conv_temp, axis=1)
		g_template[(lam_gal <4472) & (lam_gal >5147)] = 0

		# Z template
		# Perform the convolution
		z_conv_temp = convolve_gauss_hermite(z_feii_fft, npad, float(velscale),\
									 	 [opt_feii_voff, opt_feii_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
		# Normalize amplitudes to 1
		z_norm = np.array([np.max(z_conv_temp[:,i]) for i in range(np.shape(z_conv_temp)[1])])
		z_norm[z_norm<1.e-6] = 1.0
		z_conv_temp = z_conv_temp/z_norm
		# Multiply by relative intensities
		z_conv_temp = z_conv_temp * z_feii_rel_int
		# Sum templates along rows
		z_template = np.sum(z_conv_temp, axis=1)
		# Multiply by FeII amplitude
		z_template = z_template * z_feii_amp
		z_template[(lam_gal <4418) & (lam_gal >5428)] = 0


	return f_template,s_template,g_template,z_template
	
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

##################################################################################

def VW01_uv_iron_template(lam_gal, pdict, uv_iron_template, uv_iron_options, velscale, run_dir):
	"""
	Generates the UV Iron model from Vestergaard & Wilkes (2001).

	If the UV iron FWHM and/or VOFF are free to vary, perform the convolution of optical FeII template with Gauss-Hermite kernel using 
	PPXF framework.
	"""

	# UV Iron Legendre polynomial
	if (uv_iron_options['uv_legendre_p']['bool']==True) and (uv_iron_options['uv_legendre_p']['uv_iron_val']>=1):
		nw = np.linspace(-1,1,len(lam_gal))
		coeff = np.empty(uv_iron_options['uv_legendre_p']['uv_iron_val']+1)
		for n in range(uv_iron_options['uv_legendre_p']['uv_iron_val']+1):
			coeff[n] = pdict["UV_IRON_LCOEFF_%d" % n]
		lpoly = np.polynomial.legendre.legval(nw, coeff)
	
	#  Unpack opt_feii_templates (uv_iron_fft, npad, vsyst)
	uv_iron_fft, npad, vsyst = uv_iron_template

	# Parse FeII options
	if (uv_iron_options['uv_amp_const']['bool']==False): # if amp not constant
		uv_iron_amp = pdict['UV_IRON_AMP']
	elif (uv_iron_options['uv_amp_const']['bool']==True): # if amp constant
		uv_iron_amp = uv_iron_options['uv_amp_const']['uv_iron_val']
	if (uv_iron_options['uv_fwhm_const']['bool']==False): # if amp not constant
		uv_iron_fwhm = pdict['UV_IRON_FWHM']
	elif (uv_iron_options['uv_fwhm_const']['bool']==True): # if amp constant
		uv_iron_fwhm = uv_iron_options['uv_fwhm_const']['uv_iron_val']
	if (uv_iron_options['uv_voff_const']['bool']==False): # if amp not constant
		uv_iron_voff = pdict['UV_IRON_VOFF']
	elif (uv_iron_options['uv_voff_const']['bool']==True): # if amp constant
		uv_iron_voff = uv_iron_options['uv_voff_const']['uv_iron_val']

	# Convolve the UV iron FFT template and return the inverse Fourier transform.
	conv_temp = convolve_gauss_hermite(uv_iron_fft, npad, velscale,
									 	 [uv_iron_voff, uv_iron_fwhm/2.3548], lam_gal.shape[0], 
									  	 velscale_ratio=1, sigma_diff=0, vsyst=vsyst)

	# Reshape
	conv_temp = conv_temp.reshape(-1)
	# Add the additive Legendre polynomial function
	if (uv_iron_options['uv_legendre_p']['bool']==True) and (uv_iron_options['uv_legendre_p']['uv_iron_val']>=1):
		conv_temp *= lpoly
	# Re-normalize to 1
	conv_temp = conv_temp/np.max(conv_temp)
	# Multiplyy by amplitude
	template = uv_iron_amp * conv_temp
	# Reshape
	# template = template.reshape(-1)
	#
	# Set fitting region outside of template to zero to prevent convolution loops
	template[(lam_gal < 1074) & (lam_gal > 3090)] = 0
	#
	# If the summation results in 0.0, it means that features were too close 
	# to the edges of the fitting region (usua lly because the region is too 
	# small), then simply return an array of zeros.
	if (isinstance(template,int)) or (isinstance(template,float)):
		template=np.zeros(len(lam_gal))
	elif np.isnan(np.sum(template)):
		template=np.zeros(len(lam_gal))

	return template

##################################################################################

##################################################################################

def generate_balmer_continuum(lam_gal,lam_balmer, spec_high_balmer,velscale, 
							  balmer_ratio, balmer_amp, balmer_fwhm, balmer_voff, balmer_Teff, balmer_tau):
	# We need to generate a new grid for the Balmer continuum that matches
	# that we made for the higher-order lines
	def blackbody(lam, balmer_Teff):
		c = 2.99792458e+18 # speed of light [A/s]
		h = 6.626196e-11 # Planck's constant [g*A2/s2 * s]
		k = 1.380649 # Boltzmann Constant [g*A2/s2 1/K]
		Blam = ((2.0*h*c**2.0)/lam**5.0)*(1.0/(np.exp((h*c)/(lam*k*balmer_Teff))-1.0))
		return Blam
	# Construct Balmer continuum from lam_balmer
	lam_edge = 3646.0 # Balmer edge wavelength [A]
	Blam = blackbody(lam_balmer, balmer_Teff) # blackbody function [erg/s]
	cont = Blam * (1.0-1.0/np.exp(balmer_tau*(lam_balmer/lam_edge)**3.0))
	# Normalize at 3000 Å
	cont = cont / np.max(cont)
	# Set Balmer continuum to zero after Balmer edge
	cont[find_nearest(lam_balmer,lam_edge)[1]:] = 0.0
	# Normalize higher-order lines at Balmer edge
	# Unsure of how Calderone et al. (2017) (QSFit) did this normalization, so we added
	# fudge factor of 1.36 to match the QSFit implementation of the Balmer continuum.
	# spec_high_balmer = spec_high_balmer/spec_high_balmer[find_nearest(lam_balmer,lam_edge+10)[1]] * balmer_ratio #* 1.36
	if (np.sum(spec_high_balmer)>0):
		spec_high_balmer = spec_high_balmer/np.max(spec_high_balmer) * balmer_ratio #* 1.36

	# Sum the two components
	full_balmer = spec_high_balmer + cont
	# Pre-compute the FFT and vsyst
	balmer_fft, balmer_npad = template_rfft(full_balmer)
	c = 299792.458 # speed of light in km/s
	vsyst = np.log(lam_balmer[0]/lam_gal[0])*c
	# Broaden the higher-order Balmer lines
	conv_temp = convolve_gauss_hermite(balmer_fft, balmer_npad, float(velscale),\
									   [balmer_voff, balmer_fwhm/2.3548], lam_gal.shape[0], 
									   velscale_ratio=1, sigma_diff=0, vsyst=vsyst)
	conv_temp = conv_temp/conv_temp[find_nearest(lam_gal,lam_edge)[1]] * balmer_ratio
	conv_temp = conv_temp.reshape(-1)
	# Normalize the full continuum to 1
	# norm_balmer =  conv_temp[find_nearest(lam_gal,3000.0)[1]]
	# conv_temp = conv_temp/norm_balmer * balmer_amp
	conv_temp = conv_temp/np.max(conv_temp) * balmer_amp

	# Plot for testing purposes
	if 0:
		# Plot
		fig = plt.figure(figsize=(14,5))
		ax1 = fig.add_subplot(1,1,1)
		ax1.set_title('Balmer Continuum')
	#	 ax1.plot(lam_balmer, cont/np.max(cont), color='xkcd:cerulean')
	#	 ax1.plot(lam_balmer, spec_high_balmer/np.max(spec_high_balmer), color='xkcd:bright red')
		ax1.plot(lam_gal, conv_temp, color='xkcd:bright red',linewidth=0.75)
		ax1.axvline(lam_edge,linestyle='--',color='xkcd:red',linewidth=1.0)
		ax1.axvline(3000,linestyle='--',color='xkcd:black',linewidth=0.5)
	
		ax1.axhline(1.0,linestyle='--',color='xkcd:black',linewidth=0.5)
	#	 ax1.axhline(0.6,linestyle='--',color='xkcd:black',linewidth=0.5)
		ax1.set_ylim(0.0,)
	#	 ax1.set_xlim(1000,4500)
		fontsize = 16
		ax1.set_xlabel(r"Wavelength ($\lambda$)",fontsize=fontsize)

	return conv_temp

##################################################################################

#### Simple Power-Law Template ###################################################

def simple_power_law(x,amp,alpha):
	"""
	Simple power-low function to model
	the AGN continuum (Calderone et al. 2017).

	Parameters
	----------
	x	 : array_like
			wavelength vector (angstroms)
	amp   : float 
			continuum amplitude (flux density units)
	alpha : float
			power-law slope

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

#### Smoothly-Broken Power-Law Template ##########################################

def broken_power_law(x, amp, x_break, alpha_1, alpha_2, delta):
	"""
	Smoothly-broken power law continuum model; for use 
	when there is sufficient coverage in near-UV.
	(See https://docs.astropy.org/en/stable/api/astropy.modeling.
	 powerlaws.SmoothlyBrokenPowerLaw1D.html#astropy.modeling.powerlaws.
	 SmoothlyBrokenPowerLaw1D)

	Parameters
	----------
	x		: array_like
			  wavelength vector (angstroms)
	amp	 : float [0,max]
			  continuum amplitude (flux density units)
	x_break : float [x_min,x_max]
			  wavelength of the break
	alpha_1 : float [-4,2]
			  power-law slope on blue side.
	alpha_2 : float [-4,2]
			  power-law slope on red side.
	delta   : float [0.001,1.0]

	Returns
	----------
	C	 : array
			AGN continuum model the same length as x
	"""

	C = amp * (x/x_break)**(alpha_1) * (0.5*(1.0+(x/x_break)**(1.0/delta)))**((alpha_2-alpha_1)*delta)

	return C

##################################################################################



##################################################################################

def gaussian_line_profile(lam_gal,center,amp,fwhm,voff,center_pix,fwhm_res_kms,velscale):
	"""
	Produces a gaussian vector the length of
	x with the specified parameters.
	"""
	# Take into account instrumental dispersion (FWHM resolution)
	fwhm = np.sqrt(fwhm**2+fwhm_res_kms**2)
	sigma = fwhm/2.3548 # Gaussian dispersion in km/s
	sigma_pix = sigma/(velscale) # dispersion in pixels (velscale = km/s/pixel)
	voff_pix = voff/(velscale) # velocity offset in pixels
	center_pix = center_pix + voff_pix # shift the line center by voff in pixels
	#
	x_pix = np.array(range(len(lam_gal))) # pixels vector	
	x_pix = x_pix.reshape((len(x_pix),1)) # reshape into row
	g = amp*np.exp(-0.5*(x_pix-(center_pix))**2/(sigma_pix)**2) # construct gaussian
	g = np.sum(g,axis=1)
	# Make sure edges of gaussian are zero to avoid wierd things
	g[(g>-1e-6) & (g<1e-6)] = 0.0
	g[0]  = g[1]
	g[-1] = g[-2]
	#
	return g

##################################################################################

def lorentzian_line_profile(lam_gal,center,amp,fwhm,voff,center_pix,fwhm_res_kms,velscale,noise):
	"""
	Produces a lorentzian vector the length of
	x with the specified parameters.
	(See: https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Lorentz1D.html)
	"""
	
	# Take into account instrumental dispersion (FWHM resolution)
	fwhm = np.sqrt(fwhm**2+fwhm_res_kms**2)
	fwhm_pix = fwhm/velscale # fwhm in pixels (velscale = km/s/pixel)
	voff_pix = voff/velscale # velocity offset in pixels
	center_pix = center_pix + voff_pix # shift the line center by voff in pixels
	#
	x_pix = np.array(range(len(lam_gal))) # pixels vector	
	x_pix = x_pix.reshape((len(x_pix),1)) # reshape into row 

	gamma = 0.5*fwhm_pix
	l = amp*( (gamma**2) / (gamma**2+(x_pix-center_pix)**2) ) # construct lorenzian
	l= np.sum(l,axis=1)

	# Truncate wings below noise level
	l[l<=np.median(noise)] = 0.0
	l[l>np.median(noise)] -= np.median(noise)

	# Make sure edges of gaussian are zero to avoid wierd things
	l[(l>-1e-6) & (l<1e-6)] = 0.0
	l[0]  = l[1]
	l[-1] = l[-2]
	#
	return l

##################################################################################

def gauss_hermite_line_profile(lam_gal,center,amp,fwhm,voff,hmoments,center_pix,fwhm_res_kms,velscale,noise):
	"""
	Produces a Gauss-Hermite vector the length of
	x with the specified parameters.
	"""
	
	# Take into account instrumental dispersion (FWHM resolution)
	fwhm = np.sqrt(fwhm**2+fwhm_res_kms**2)
	sigma_pix = fwhm/2.3548/velscale # dispersion in pixels (velscale = km/s/pixel)
	voff_pix = voff/velscale # velocity offset in pixels
	center_pix = center_pix + voff_pix # shift the line center by voff in pixels
	#
	x_pix = np.array(range(len(lam_gal))) # pixels vector	
	x_pix = x_pix.reshape((len(x_pix),1)) #- center_pix
	
	# Taken from Riffel 2010 - profit: a new alternative for emission-line profile fitting
	w = (x_pix-center_pix)/sigma_pix
	alpha = 1.0/np.sqrt(2.0)*np.exp(-w**2/2.0)

	if hmoments is not None:
		mom = len(hmoments)+2
		n = np.arange(3, mom + 1)
		nrm = np.sqrt(special.factorial(n)*2**n)   # Normalization
		coeff = np.append([1, 0, 0],hmoments/nrm)
		h = hermite.hermval(w,coeff)
		g = (amp*alpha)/sigma_pix*h
	elif hmoments is None:
		coeff = np.array([1, 0, 0])
		h = hermite.hermval(w,coeff)
		g = (amp*alpha)/sigma_pix*h

	g = np.sum(g,axis=1)
	# We ensure any values of the line profile that are negative
	# are zeroed out (See Van der Marel 1993)
	g[g<0] = 0.0
	# Normalize to 1
	g = g/np.max(g)
	# Apply amplitude
	g = amp*g

	# Truncate wings below noise level
	g[g<=np.median(noise)] = 0.0
	g[g>np.median(noise)] -= np.median(noise)

	# Replace the ends with the same value 
	g[(g>-1e-6) & (g<1e-6)] = 0.0
	g[0]  = g[1]
	g[-1] = g[-2]
	#
	return g

##################################################################################

def voigt_line_profile(lam_gal,center,amp,fwhm,voff,shape,center_pix,fwhm_res_kms,velscale,noise):
	"""
	Pseudo-Voigt profile implementation from:
	https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html
	"""
	# Take into account instrumental dispersion (FWHM resolution)
	fwhm	   = np.sqrt(fwhm**2+fwhm_res_kms**2)
	fwhm_pix   = fwhm/velscale # fwhm in pixels (velscale = km/s/pixel)
	sigma_pix  = fwhm_pix/2.3548
	voff_pix   = voff/velscale # velocity offset in pixels
	center_pix = center_pix + voff_pix # shift the line center by voff in pixels
	#
	x_pix	  = np.array(range(len(lam_gal))) # pixels vector	
	x_pix	  = x_pix.reshape((len(x_pix),1)) # reshape into row 
	
	# Gaussian contribution
	a_G = 1.0/(sigma_pix * np.sqrt(2.0*np.pi))
	g = a_G * np.exp(-0.5*(x_pix-(center_pix))**2/(sigma_pix)**2)
	g = np.sum(g,axis=1)

	# Lorentzian contribution
	l = (1.0/np.pi) * (fwhm_pix/2.0)/((x_pix-center_pix)**2 + (fwhm_pix/2.0)**2)
	l = np.sum(l,axis=1)
	
	# Intensity
	  # I = amp/((float(shape)*a_G) + ((1.0-float(shape))*(2.0/(np.pi*fwhm_pix))))

	# Voigt profile
	pv =  (float(shape) * g) + ((1.0-float(shape))*l)

	# Normalize and multiply by amplitude
	pv = pv/np.max(pv)*amp

	# Truncate wings below noise level
	pv[pv<=np.median(noise)] = 0.0
	pv[pv>np.median(noise)] -= np.median(noise)

	# Replace the ends with the same value 
	pv[(pv>-1e-6) & (pv<1e-6)] = 0.0
	pv[0]  = pv[1]
	pv[-1] = pv[-2]
	
	return pv

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
	start = np.array(start,dtype=float)  # make copy
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
				nrm = np.sqrt(special.factorial(n)*2**n)   # Normalization
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
			  auto_stop,conv_type,min_samp,ncor_times,autocorr_tol,write_iter,write_thresh,burn_in,min_iter,max_iter,
			  verbose=True):
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
	chain_file = run_dir.joinpath('log', 'MCMC_chain.csv')
	if not chain_file.exists():
		with chain_file.open(mode='w') as f:
			param_string = ', '.join(str(e) for e in param_names)
			f.write('# iter, ' + param_string) # Write initial parameters
			best_str = ', '.join(str(e) for e in init_params)
			f.write('\n 0, '+best_str)

	# initialize the sampler
	dtype = [('fluxes',dict),('eqwidths',dict),('cont_fluxes',dict),("int_vel_disp",dict),('log_like',float)] # mcmc blobs
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args,blobs_dtype=dtype) # blobs_dtype=dtype added for Python2 -> Python3

	start_time = time.time() # start timer

	write_log((ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter),'emcee_options',run_dir)

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
			if (verbose):
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
					if (verbose):
						print('\n Only considering convergence of following parameters: ')
						for c in conv_type:	
							print('		  %s' % c)
						pass
					else:
						if (verbose):
							print('\n One of more parameters in conv_type is not a valid parameter. Defaulting to median convergence type../.\n')
						conv_type='median'

			except:
				print('\n One of more parameters in conv_type is not a valid parameter. Defaulting to median convergence type../.\n')
				conv_type='median'

	if (auto_stop==True):
		write_log((min_samp,autocorr_tol,ncor_times,conv_type),'autocorr_options',run_dir)
	# Run emcee
	for k, result in enumerate(sampler.sample(pos, iterations=max_iter)):
		
		if ((k+1) % write_iter == 0) and verbose:
			print("MCMC iteration: %d" % (k+1))
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
			with run_dir.joinpath('log', 'MCMC_chain.csv').open(mode='a') as f:
				best_str = ', '.join(str(e) for e in best)
				f.write('\n'+str(k+1)+', '+best_str)
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
					if verbose:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Not enough iterations for any autocorrelation times!')
				elif ( (par_conv.size > 0) and (k+1)>(np.mean(tau[par_conv]) * ncor_times) and (np.mean(tol[par_conv])<autocorr_tol) and (stop_iter == max_iter) ):
					if verbose:
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
					if verbose:
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
					if verbose:
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
					if verbose:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Not enough iterations for any autocorrelation times!')
				elif ( (par_conv.size > 0) and (k+1)>(np.median(tau[par_conv]) * ncor_times) and (np.median(tol[par_conv])<autocorr_tol) and (stop_iter == max_iter) ):
					if verbose:
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
					if verbose:
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
					if verbose:
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
					if verbose:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Not enough iterations for any autocorrelation times!')
				elif all( ((k+1)>(x * ncor_times)) for x in tau) and all( (x>1.0) for x in tau) and all(y<autocorr_tol for y in tol) and (stop_iter == max_iter):
					if verbose:
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
					if verbose:
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
					if verbose:
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
					if verbose:
						print('\nIteration = %d' % (k+1))
						print('-------------------------------------------------------------------------------')
						print('- Not enough iterations for any autocorrelation times!')
				elif all( ((k+1)>(x * ncor_times)) for x in tau_interest) and all( (x>1.0) for x in tau_interest) and all(y<autocorr_tol for y in tol_interest) and (stop_iter == max_iter):
					if verbose:
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
					if verbose:
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
					if verbose:
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
			if verbose:
				print('{0:<30}'.format('\nIteration = %d' % (k+1)))
				print('------------------------------------------------')
				print('{0:<30}{1:<20}'.format('Parameter','Current Value'))
				print('------------------------------------------------')
				for i in range(0,len(pnames_sorted),1):
						print('{0:<30}{1:<20.4f}'.format(pnames_sorted[i],best_sorted[i]))
				print('------------------------------------------------')

	elap_time = (time.time() - start_time)	   
	run_time = time_convert(elap_time)
	if verbose:
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
		np.save(run_dir.joinpath('log', 'autocorr_dict.npy'),autocorr_dict)

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

	# Extract metadata blobs
	blobs		   = sampler.get_blobs()
	flux_blob	   = blobs["fluxes"]
	eqwidth_blob   = blobs["eqwidths"]
	cont_flux_blob = blobs["cont_fluxes"]
	int_vel_disp_blob  = blobs["int_vel_disp"]
	log_like_blob  = blobs["log_like"]

	return a, burn_in, flux_blob, eqwidth_blob, cont_flux_blob, int_vel_disp_blob, log_like_blob


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
	
	return tau_est

##################################################################################


# Plotting Routines
##################################################################################

def gauss_kde(xs,data,h):
	"""
	Gaussian kernel density estimation.
	"""
	def gauss_kernel(x):
		return (1./np.sqrt(2.*np.pi)) * np.exp(-x**2/2)

	kde = np.sum((1./h) * gauss_kernel((xs.reshape(len(xs),1)-data)/h), axis=1)
	kde = kde/simps(kde,xs)# normalize
	return kde

def kde_bandwidth(data):
	"""
	Silverman bandwidth estimation for kernel density estimation.
	"""
	return (4./(3.*len(data)))**(1./5.) * np.std(data)

def compute_HDI(posterior_samples, credible_mass):
	"""
	Computes highest density interval from a sample of representative values,
	estimated as the shortest credible interval.
	Takes Arguments posterior_samples (samples from posterior) and credible mass (usually 0.95):
	https://www.sciencedirect.com/topics/mathematics/highest-density-interval
	BADASS uses the 0.68 interval.
	"""
	sorted_points = sorted(posterior_samples)
	ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
	nCIs = len(sorted_points) - ciIdxInc
	# If the requested credible mass is equal to the number of posterior samples than the 
	# CI is simply the extent of the data.  This is typical of the 99.7% CI case for N<1000
	if nCIs==0:
	    HDImin = np.min(posterior_samples)
	    HDImax = np.max(posterior_samples)
	else:
	    ciWidth = [0]*nCIs    
	    for i in range(0, nCIs):
	        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
	        HDImin = sorted_points[ciWidth.index(min(ciWidth))]
	        HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
	return(HDImin, HDImax)

def posterior_plots(key,flat,chain,burn_in,xs,kde,h,
					post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
					run_dir
					):
	"""
	Plot posterior distributions and chains from MCMC.
	"""
	# Initialize figures and axes
	# Make an updating plot of the chain
	fig = plt.figure(figsize=(10,8)) 
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.35, hspace=0.35) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0,0])
	ax2  = plt.subplot(gs[0,1])
	ax3  = plt.subplot(gs[1,0:2])

	# Histogram; 'Doane' binning produces the best results from tests.
	n, bins, patches = ax1.hist(flat, bins='doane', histtype="bar" , density=True, facecolor="#4200a6", alpha=1,zorder=10)
	# Plot 1: Histogram plots
	ax1.axvline(post_max	,linewidth=0.5,linestyle="-",color='xkcd:bright aqua',alpha=1.00,zorder=20,label=r'$p(\theta|x)_{\rm{max}}$')
	#
	ax1.axvline(post_max-low_68,linewidth=0.5,linestyle="--" ,color='xkcd:bright aqua',alpha=1.00,zorder=20,label=r'68% conf.')
	ax1.axvline(post_max+upp_68,linewidth=0.5,linestyle="--" ,color='xkcd:bright aqua',alpha=1.00,zorder=20)
	#
	ax1.axvline(post_max-low_95,linewidth=0.5,linestyle=":" ,color='xkcd:bright aqua',alpha=1.00,zorder=20,label=r'95% conf.')
	ax1.axvline(post_max+upp_95,linewidth=0.5,linestyle=":" ,color='xkcd:bright aqua',alpha=1.00,zorder=20)
	#
	# ax1.axvline(post_mean,linewidth=0.5,linestyle="--",color='xkcd:bright aqua',alpha=1.00,zorder=20,label=r'Mean')
	# ax1.axvline(post_mean-post_std,linewidth=0.5,linestyle=":" ,color='xkcd:bright aqua',alpha=1.00,zorder=20,label=r'Std. Dev.')
	# ax1.axvline(post_mean+post_std,linewidth=0.5,linestyle=":" ,color='xkcd:bright aqua',alpha=1.00,zorder=20)
	#
	# ax1.axvline(post_med,linewidth=0.5,linestyle="--",color='xkcd:bright yellow',alpha=1.00,zorder=20,label=r'Median')
	# ax1.axvline(post_med-post_mad,linewidth=0.5,linestyle=":" ,color='xkcd:bright yellow',alpha=1.00,zorder=20,label=r'Med. Abs. Dev.')
	# ax1.axvline(post_med+post_mad,linewidth=0.5,linestyle=":" ,color='xkcd:bright yellow',alpha=1.00,zorder=20)
	#
	ax1.plot(xs,kde	   ,linewidth=0.5,linestyle="-" ,color="xkcd:bright pink",alpha=1.00,zorder=15,label="KDE")
	ax1.plot(xs,kde	   ,linewidth=3.0,linestyle="-" ,color="xkcd:bright pink",alpha=0.50,zorder=15)
	ax1.plot(xs,kde	   ,linewidth=6.0,linestyle="-" ,color="xkcd:bright pink",alpha=0.20,zorder=15)
	ax1.grid(b=True,which="major",axis="both",alpha=0.15,color="xkcd:bright pink",linewidth=0.5,zorder=0)
	# ax1.plot(xvec,yvec,color='white')
	ax1.set_xlabel(r'%s' % key,fontsize=12)
	ax1.set_ylabel(r'$p$(%s)' % key,fontsize=12)
	ax1.legend(loc="best",fontsize=6)
	
	# Plot 2: best fit values
	values = [post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad]
	labels = [r"$p(\theta|x)_{\rm{max}}$",
          r"$\rm{CI\;68\%\;low}$",r"$\rm{CI\;68\%\;upp}$",
          r"$\rm{CI\;95\%\;low}$",r"$\rm{CI\;95\%\;upp}$",
          r"$\rm{Mean}$",r"$\rm{Std.\;Dev.}$",
          r"$\rm{Median}$",r"$\rm{Med. Abs. Dev.}$"]
	start, step = 1, 0.12
	vspace = np.linspace(start,1-len(labels)*step,len(labels),endpoint=False)
	# Plot 2: best fit values
	for i in range(len(labels)):
	    ax2.annotate('{0:>30}{1:<2}{2:<30.3f}'.format(labels[i],r"$\qquad=\qquad$",values[i]), 
	                 xy=(0.5, vspace[i]),  xycoords='axes fraction',
	                 xytext=(0.95, vspace[i]), textcoords='axes fraction',
	                 horizontalalignment='right', verticalalignment='top', 
	                 fontsize=10)
	ax2.axis('off')
	
	# Plot 3: Chain plot
	for w in range(0,np.shape(chain)[0],1):
		ax3.plot(range(np.shape(chain)[1]),chain[w,:],color='white',linewidth=0.5,alpha=0.5,zorder=0)
	# Calculate median and median absolute deviation of walkers at each iteration; we have depreciated
	# the average and standard deviation because they do not behave well for outlier walkers, which
	# also don't agree with histograms.
	c_med = np.median(chain,axis=0)
	c_madstd = mad_std(chain)
	ax3.plot(range(np.shape(chain)[1]),c_med,color='xkcd:bright pink',alpha=1.,linewidth=2.0,label='Median',zorder=10)
	ax3.fill_between(range(np.shape(chain)[1]),c_med+c_madstd,c_med-c_madstd,color='#4200a6',alpha=0.5,linewidth=1.5,label='Median Absolute Dev.',zorder=5)
	ax3.axvline(burn_in,linestyle='--',linewidth=0.5,color='xkcd:bright aqua',label='burn-in = %d' % burn_in,zorder=20)
	ax3.grid(b=True,which="major",axis="both",alpha=0.15,color="xkcd:bright pink",linewidth=0.5,zorder=0)
	ax3.set_xlim(0,np.shape(chain)[1])
	ax3.set_xlabel('$N_\mathrm{iter}$',fontsize=12)
	ax3.set_ylabel(r'%s' % key,fontsize=12)
	ax3.legend(loc='upper left')
	
	# Save the figure
	histo_dir = run_dir.joinpath('histogram_plots')
	histo_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(histo_dir.joinpath('%s_MCMC.png' % (key)), bbox_inches="tight",dpi=300,fmt='png')

	# Close plot window
	fig.clear()
	plt.close()

	return

def param_plots(param_dict,burn_in,run_dir,plot_param_hist=True,verbose=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	free parameters from MCMC sample chains.
	"""
	#
	if verbose:
		print("\n Generating model parameter distributions...\n")

	for key in param_dict:
		#
		if verbose:
			print('		  %s' % key)
		chain = param_dict[key]['chain'] # shape = (nwalkers,niter)
		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])
		# Flatten the chains
		flat = chain[:,burn_in:]
		flat = flat.flat

		# Subsample the data into a manageable size for the kde and HDI
		subsampled = np.random.choice(flat,size=10000)

		# Histogram; 'Doane' binning produces the best results from tests.
		hist, bin_edges = np.histogram(subsampled, bins='doane', density=False)

		# Generate pseudo-data on the ends of the histogram; this prevents the KDE
		# from weird edge behavior.
		n_pseudo = 3 # number of pseudo-bins 
		bin_width=bin_edges[1]-bin_edges[0]
		lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
		upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

		# Calculate bandwidth for KDE (Silverman method)
		h = kde_bandwidth(flat)

		# Create a subsampled grid for the KDE based on the subsampled data; by
		# default, we subsample by a factor of 10.
		xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

		# Calculate KDE
		kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
		p68 = compute_HDI(subsampled,0.68)
		p95 = compute_HDI(subsampled,0.95)

		post_max  = xs[kde.argmax()] # posterior max estimated from KDE
		post_mean = np.mean(flat)
		post_med  = np.median(flat)
		low_68    = post_max - p68[0]
		upp_68    = p68[1] - post_max
		low_95    = post_max - p95[0]
		upp_95    = p95[1] - post_max
		post_std  = np.std(flat)
		post_mad  = stats.median_abs_deviation(flat)

		# Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
		flag = 0  
		if ( (post_max-1.5*low_68) <= (param_dict[key]['plim'][0]) ):
			flag+=1
		if ( (post_max+1.5*upp_68) >= (param_dict[key]['plim'][1]) ):
			flag+=1
		if ~np.isfinite(post_max) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
			flag+=1

		param_dict[key]['par_best']    = post_max # maximum of posterior distribution
		param_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
		param_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
		param_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
		param_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
		param_dict[key]['mean']        = post_mean # mean of posterior distribution
		param_dict[key]['std_dev']     = post_std	# standard deviation
		param_dict[key]['median']      = post_med # median of posterior distribution
		param_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
		param_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
		param_dict[key]['flag']	       = flag 

		if (plot_param_hist==True):
			posterior_plots(key,flat,chain,burn_in,xs,kde,h,
				            post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
				            run_dir
				            )

	return param_dict


def log_like_plot(ll_blob, burn_in, nwalkers, run_dir, plot_param_hist=True,verbose=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	component fluxes from MCMC sample chains.
	"""
	
	ll = ll_blob.T

	# Burned-in + Flattened (along walker axis) chain
	# If burn_in is larger than the size of the chain, then 
	# take 50% of the chain length instead.
	if (burn_in >= np.shape(ll)[1]):
		burn_in = int(0.5*np.shape(ll)[1])
		# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

	flat = ll[:,burn_in:]
	flat = flat.flat

	# Old confidence interval stuff; replaced by np.quantile
	# p = np.percentile(flat, [16, 50, 84])
	# pdfmax = p[1]
	# low1   = p[1]-p[0]
	# upp1   = p[2]-p[1]

	# Subsample the data into a manageable size for the kde and HDI
	subsampled = np.random.choice(flat,size=10000)

	# Histogram; 'Doane' binning produces the best results from tests.
	hist, bin_edges = np.histogram(subsampled, bins='doane', density=False)

	# Generate pseudo-data on the ends of the histogram; this prevents the KDE
	# from weird edge behavior.
	n_pseudo = 3 # number of pseudo-bins 
	bin_width=bin_edges[1]-bin_edges[0]
	lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
	upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

	# Calculate bandwidth for KDE (Silverman method)
	h = kde_bandwidth(flat)

	# Create a subsampled grid for the KDE based on the subsampled data; by
	# default, we subsample by a factor of 10.
	xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

	# Calculate KDE
	kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
	p68 = compute_HDI(subsampled,0.68)
	p95 = compute_HDI(subsampled,0.95)

	post_max  = xs[kde.argmax()] # posterior max estimated from KDE
	post_mean = np.mean(flat)
	post_med  = np.median(flat)
	low_68    = post_max - p68[0]
	upp_68    = p68[1] - post_max
	low_95    = post_max - p95[0]
	upp_95    = p95[1] - post_max
	post_std  = np.std(flat)
	post_mad  = stats.median_abs_deviation(flat)

	# Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
	flag = 0
	if ~np.isfinite(post_max) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
		flag += 1

	ll_dict = {
				'par_best'    : post_max, # maximum of posterior distribution
				'ci_68_low'   : low_68,	# lower 68% confidence interval
				'ci_68_upp'   : upp_68,	# upper 68% confidence interval
				'ci_95_low'   : low_95,	# lower 95% confidence interval
				'ci_95_upp'   : upp_95,	# upper 95% confidence interval
				'mean'        : post_mean, # mean of posterior distribution
				'std_dev'     : post_std,	# standard deviation
				'median'      : post_med, # median of posterior distribution
				'med_abs_dev' : post_mad,	# median absolute deviation
				'flat_chain'  : flat,   # flattened samples used for histogram.
				'flag'        : flag, 
	}

	if (plot_param_hist==True):
			posterior_plots("LOG_LIKE",flat,ll,burn_in,xs,kde,h,
							post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
							run_dir)


	return ll_dict

def flux_plots(flux_blob, burn_in, nwalkers, run_dir, plot_flux_hist=True,verbose=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	component fluxes from MCMC sample chains.
	"""
	if verbose:
		print("\n Generating model flux distributions...\n")

	# Create a flux dictionary
	niter	= np.shape(flux_blob)[0]
	nwalkers = np.shape(flux_blob)[1]
	flux_dict = {}
	for key in flux_blob[0][0]:
		flux_dict[key] = {'chain':np.empty([nwalkers,niter])}

	# Restructure the flux_blob for the flux_dict
	for i in range(niter):
		for j in range(nwalkers):
			for key in flux_blob[0][0]:
				flux_dict[key]['chain'][j,i] = flux_blob[i][j][key]

	for key in flux_dict:
		if verbose:
			print('		  %s' % key)
		chain = np.log10(flux_dict[key]['chain']*1.e-17) # shape = (nwalkers,niter)
		chain[~np.isfinite(chain)] = 0
		flux_dict[key]['chain'] = chain
		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])

		# Remove burn_in iterations and flatten for histogram
		flat = chain[:,burn_in:]
		flat = flat.flat

		# Subsample the data into a manageable size for the kde and HDI
		subsampled = np.random.choice(flat,size=10000)

		# Histogram; 'Doane' binning produces the best results from tests.
		hist, bin_edges = np.histogram(subsampled, bins='doane', density=False)

		# Generate pseudo-data on the ends of the histogram; this prevents the KDE
		# from weird edge behavior.
		n_pseudo = 3 # number of pseudo-bins 
		bin_width=bin_edges[1]-bin_edges[0]
		lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
		upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

		# Calculate bandwidth for KDE (Silverman method)
		h = kde_bandwidth(flat)

		# Create a subsampled grid for the KDE based on the subsampled data; by
		# default, we subsample by a factor of 10.
		xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

		# Calculate KDE
		kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
		p68 = compute_HDI(subsampled,0.68)
		p95 = compute_HDI(subsampled,0.95)

		post_max  = xs[kde.argmax()] # posterior max estimated from KDE
		post_mean = np.mean(flat)
		post_med  = np.median(flat)
		low_68    = post_max - p68[0]
		upp_68    = p68[1] - post_max
		low_95    = post_max - p95[0]
		upp_95    = p95[1] - post_max
		post_std  = np.std(flat)
		post_mad  = stats.median_abs_deviation(flat)

		# Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
		flag = 0  
		if ( (post_max-1.5*low_68) <= -20 ):
			flag+=1
		if ~np.isfinite(post_max) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
			flag+=1

		flux_dict[key]['par_best']    = post_max # maximum of posterior distribution
		flux_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
		flux_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
		flux_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
		flux_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
		flux_dict[key]['mean']        = post_mean # mean of posterior distribution
		flux_dict[key]['std_dev']     = post_std	# standard deviation
		flux_dict[key]['median']      = post_med # median of posterior distribution
		flux_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
		flux_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
		flux_dict[key]['flag']	       = flag 


		if (plot_flux_hist==True):
			posterior_plots(key,flat,chain,burn_in,xs,kde,h,
							post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
							run_dir)

	return flux_dict

def lum_plots(flux_dict,burn_in,nwalkers,z,run_dir,H0=70.0,Om0=0.30,plot_lum_hist=True,verbose=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	component luminosities from MCMC sample chains.
	"""
	if verbose:
		print("\n Generating model luminosity distributions...\n")

	# Compute luminosity distance (in cm) using FlatLambdaCDM cosmology
	cosmo = FlatLambdaCDM(H0, Om0)
	d_mpc = cosmo.luminosity_distance(z).value
	d_cm  = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm

	# Create a flux dictionary
	lum_dict = {}
	for key in flux_dict:
		flux = 10**(flux_dict[key]['chain']) # * 1.e-17
		# Convert fluxes to luminosities and take log10
		lum   = np.log10((flux * 4*np.pi * d_cm**2	)) #/ 1.0E+42
		lum[~np.isfinite(lum)] = 0
		lum_dict[key[:-4]+'LUM']= {'chain':lum}

	for key in lum_dict:
		if verbose:
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

		# Subsample the data into a manageable size for the kde and HDI
		subsampled = np.random.choice(flat,size=10000)

		# Histogram; 'Doane' binning produces the best results from tests.
		hist, bin_edges = np.histogram(subsampled, bins='doane', density=False)

		# Generate pseudo-data on the ends of the histogram; this prevents the KDE
		# from weird edge behavior.
		n_pseudo = 3 # number of pseudo-bins 
		bin_width=bin_edges[1]-bin_edges[0]
		lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
		upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

		# Calculate bandwidth for KDE (Silverman method)
		h = kde_bandwidth(flat)

		# Create a subsampled grid for the KDE based on the subsampled data; by
		# default, we subsample by a factor of 10.
		xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

		# Calculate KDE
		kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
		p68 = compute_HDI(subsampled,0.68)
		p95 = compute_HDI(subsampled,0.95)

		post_max  = xs[kde.argmax()] # posterior max estimated from KDE
		post_mean = np.mean(flat)
		post_med  = np.median(flat)
		low_68    = post_max - p68[0]
		upp_68    = p68[1] - post_max
		low_95    = post_max - p95[0]
		upp_95    = p95[1] - post_max
		post_std  = np.std(flat)
		post_mad  = stats.median_abs_deviation(flat)

		# Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
		flag = 0  
		if ( (post_max-1.5*low_68) <= 30 ):
			flag+=1
		if ~np.isfinite(post_max) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
			flag+=1

		lum_dict[key]['par_best']    = post_max # maximum of posterior distribution
		lum_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
		lum_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
		lum_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
		lum_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
		lum_dict[key]['mean']        = post_mean # mean of posterior distribution
		lum_dict[key]['std_dev']     = post_std	# standard deviation
		lum_dict[key]['median']      = post_med # median of posterior distribution
		lum_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
		lum_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
		lum_dict[key]['flag']	     = flag

		if (plot_lum_hist==True):
			posterior_plots(key,flat,chain,burn_in,xs,kde,h,
							post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
							run_dir)

	return lum_dict

def eqwidth_plots(eqwidth_blob, burn_in, nwalkers, run_dir, plot_eqwidth_hist=True,verbose=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	component fluxes from MCMC sample chains.
	"""
	if verbose:
		print("\n Generating model equivalent width distributions...\n")
	# Create a flux dictionary
	niter	= np.shape(eqwidth_blob)[0]
	nwalkers = np.shape(eqwidth_blob)[1]
	eqwidth_dict = {}
	for key in eqwidth_blob[0][0]:
		eqwidth_dict[key] = {'chain':np.empty([nwalkers,niter])}

	# Restructure the flux_blob for the flux_dict
	for i in range(niter):
		for j in range(nwalkers):
			for key in eqwidth_blob[0][0]:
				eqwidth_dict[key]['chain'][j,i] = eqwidth_blob[i][j][key]

	for key in eqwidth_dict:
		if verbose:
			print('		  %s' % key)
		chain = eqwidth_dict[key]['chain'] # shape = (nwalkers,niter)

		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])

		# Remove burn_in iterations and flatten for histogram
		flat = chain[:,burn_in:]
		flat = flat.flat

		# Subsample the data into a manageable size for the kde and HDI
		subsampled = np.random.choice(flat,size=10000)

		# Histogram; 'Doane' binning produces the best results from tests.
		hist, bin_edges = np.histogram(subsampled, bins='doane', density=False)

		# Generate pseudo-data on the ends of the histogram; this prevents the KDE
		# from weird edge behavior.
		n_pseudo = 3 # number of pseudo-bins 
		bin_width=bin_edges[1]-bin_edges[0]
		lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
		upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

		# Calculate bandwidth for KDE (Silverman method)
		h = kde_bandwidth(flat)

		# Create a subsampled grid for the KDE based on the subsampled data; by
		# default, we subsample by a factor of 10.
		xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

		# Calculate KDE
		kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
		p68 = compute_HDI(subsampled,0.68)
		p95 = compute_HDI(subsampled,0.95)

		post_max  = xs[kde.argmax()] # posterior max estimated from KDE
		post_mean = np.mean(flat)
		post_med  = np.median(flat)
		low_68    = post_max - p68[0]
		upp_68    = p68[1] - post_max
		low_95    = post_max - p95[0]
		upp_95    = p95[1] - post_max
		post_std  = np.std(flat)
		post_mad  = stats.median_abs_deviation(flat)

		# Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
		flag = 0  
		if ( (post_max-1.5*low_68) <= 0 ):
			flag+=1
		if ~np.isfinite(post_max) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
			flag+=1

		eqwidth_dict[key]['par_best']    = post_max # maximum of posterior distribution
		eqwidth_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
		eqwidth_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
		eqwidth_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
		eqwidth_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
		eqwidth_dict[key]['mean']        = post_mean # mean of posterior distribution
		eqwidth_dict[key]['std_dev']     = post_std	# standard deviation
		eqwidth_dict[key]['median']      = post_med # median of posterior distribution
		eqwidth_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
		eqwidth_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
		eqwidth_dict[key]['flag']	       = flag

		if (plot_eqwidth_hist==True):
			posterior_plots(key,flat,chain,burn_in,xs,kde,h,
							post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
							run_dir)
		

	return eqwidth_dict

def cont_lum_plots(cont_flux_blob,burn_in,nwalkers,z,run_dir,H0=70.0,Om0=0.30,plot_lum_hist=True,verbose=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	component luminosities from MCMC sample chains.
	"""

	# Create a flux dictionary
	niter	= np.shape(cont_flux_blob)[0]
	nwalkers = np.shape(cont_flux_blob)[1]
	cont_flux_dict = {}
	for key in cont_flux_blob[0][0]:
		cont_flux_dict[key] = {'chain':np.empty([nwalkers,niter])}

	# Restructure the flux_blob for the flux_dict
	for i in range(niter):
		for j in range(nwalkers):
			for key in cont_flux_blob[0][0]:
				cont_flux_dict[key]['chain'][j,i] = cont_flux_blob[i][j][key]
	
	# Compute luminosity distance (in cm) using FlatLambdaCDM cosmology
	cosmo = FlatLambdaCDM(H0, Om0)
	d_mpc = cosmo.luminosity_distance(z).value
	d_cm  = d_mpc * 3.086E+24 # 1 Mpc = 3.086e+24 cm
	# Create a luminosity dictionary
	cont_lum_dict = {}
	for key in cont_flux_dict:
		# Total cont. lum.
		if (key=="F_CONT_TOT_1350"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_TOT_1350"]= {'chain':lum}
		if (key=="F_CONT_TOT_3000"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_TOT_3000"]= {'chain':lum}
		if (key=="F_CONT_TOT_5100"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_TOT_5100"]= {'chain':lum}
		# AGN cont. lum.
		if (key=="F_CONT_AGN_1350"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_AGN_1350"]= {'chain':lum}
		if (key=="F_CONT_AGN_3000"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_AGN_3000"]= {'chain':lum}
		if (key=="F_CONT_AGN_5100"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_AGN_5100"]= {'chain':lum}
		# Host cont. lum
		if (key=="F_CONT_HOST_1350"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 1350.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_HOST_1350"]= {'chain':lum}
		if (key=="F_CONT_HOST_3000"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 3000.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_HOST_3000"]= {'chain':lum}
		if (key=="F_CONT_HOST_5100"):
			flux = (cont_flux_dict[key]['chain']) * 1.0E-17
			# Convert fluxes to luminosities and normalize by 10^(+42) to avoid numerical issues 
			lum   = np.log10((flux * 4*np.pi * d_cm**2	) * 5100.0) #/ 1.0E+42
			lum[~np.isfinite(lum)] = 0
			cont_lum_dict["L_CONT_HOST_5100"]= {'chain':lum}
		# AGN fractions
		if (key=="AGN_FRAC_4000"):
			cont_lum_dict["AGN_FRAC_4000"]= {'chain':cont_flux_dict[key]['chain']}
		if (key=="AGN_FRAC_7000"):
			cont_lum_dict["AGN_FRAC_7000"]= {'chain':cont_flux_dict[key]['chain']}	
		# Host fractions
		if (key=="HOST_FRAC_4000"):
			cont_lum_dict["HOST_FRAC_4000"]= {'chain':cont_flux_dict[key]['chain']}
		if (key=="HOST_FRAC_7000"):
			cont_lum_dict["HOST_FRAC_7000"]= {'chain':cont_flux_dict[key]['chain']}	


	for key in cont_lum_dict:
		if verbose:
			print('		  %s' % key)
		chain = cont_lum_dict[key]['chain'] # shape = (nwalkers,niter)

		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])
			# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

		# Remove burn_in iterations and flatten for histogram
		flat = chain[:,burn_in:]
		flat = flat.flat

		# Subsample the data into a manageable size for the kde and HDI
		subsampled = np.random.choice(flat,size=10000)

		# Histogram; 'Doane' binning produces the best results from tests.
		hist, bin_edges = np.histogram(subsampled, bins='doane', density=False)

		# Generate pseudo-data on the ends of the histogram; this prevents the KDE
		# from weird edge behavior.
		n_pseudo = 3 # number of pseudo-bins 
		bin_width=bin_edges[1]-bin_edges[0]
		lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
		upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

		# Calculate bandwidth for KDE (Silverman method)
		h = kde_bandwidth(flat)

		# Create a subsampled grid for the KDE based on the subsampled data; by
		# default, we subsample by a factor of 10.
		xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

		# Calculate KDE
		kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
		p68 = compute_HDI(subsampled,0.68)
		p95 = compute_HDI(subsampled,0.95)

		post_max  = xs[kde.argmax()] # posterior max estimated from KDE
		post_mean = np.mean(flat)
		post_med  = np.median(flat)
		low_68    = post_max - p68[0]
		upp_68    = p68[1] - post_max
		low_95    = post_max - p95[0]
		upp_95    = p95[1] - post_max
		post_std  = np.std(flat)
		post_mad  = stats.median_abs_deviation(flat)

		# Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
		flag = 0  
		if ( (post_max-1.5*low_68) <= 0 ):
			flag+=1
		if ~np.isfinite(post_max) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
			flag+=1

		cont_lum_dict[key]['par_best']    = post_max # maximum of posterior distribution
		cont_lum_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
		cont_lum_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
		cont_lum_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
		cont_lum_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
		cont_lum_dict[key]['mean']        = post_mean # mean of posterior distribution
		cont_lum_dict[key]['std_dev']     = post_std	# standard deviation
		cont_lum_dict[key]['median']      = post_med # median of posterior distribution
		cont_lum_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
		cont_lum_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
		cont_lum_dict[key]['flag']	      = flag 

		if (plot_lum_hist==True):
			posterior_plots(key,flat,chain,burn_in,xs,kde,h,
							post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
							run_dir)

	return cont_lum_dict

def int_vel_disp_plots(int_vel_disp_blob,burn_in,nwalkers,z,run_dir,H0=70.0,Om0=0.30,plot_param_hist=True,verbose=True):
	"""
	Generates best-fit values, uncertainties, and plots for 
	component luminosities from MCMC sample chains.
	"""
	if verbose:
		print("\n Generating model integrated velocity moment distributions...\n")

	# Create a flux dictionary
	niter	= np.shape(int_vel_disp_blob)[0]
	nwalkers = np.shape(int_vel_disp_blob)[1]
	int_vel_disp_dict = {}
	for key in int_vel_disp_blob[0][0]:
		int_vel_disp_dict[key] = {'chain':np.empty([nwalkers,niter])}

	# Restructure the int_vel_disp_blob for the int_vel_disp_dict
	for i in range(niter):
		for j in range(nwalkers):
			for key in int_vel_disp_blob[0][0]:
				int_vel_disp_dict[key]['chain'][j,i] = int_vel_disp_blob[i][j][key]

	for key in int_vel_disp_dict:
		if verbose:
			print('		  %s' % key)
		chain = int_vel_disp_dict[key]['chain'] # shape = (nwalkers,niter)

		# Burned-in + Flattened (along walker axis) chain
		# If burn_in is larger than the size of the chain, then 
		# take 50% of the chain length instead.
		if (burn_in >= np.shape(chain)[1]):
			burn_in = int(0.5*np.shape(chain)[1])
			# print('\n Burn-in is larger than chain length! Using 50% of chain length for burn-in...\n')

		# Remove burn_in iterations and flatten for histogram
		flat = chain[:,burn_in:]
		flat = flat.flat

		# Subsample the data into a manageable size for the kde and HDI
		subsampled = np.random.choice(flat,size=10000)

		# Histogram; 'Doane' binning produces the best results from tests.
		hist, bin_edges = np.histogram(subsampled, bins='doane', density=False)

		# Generate pseudo-data on the ends of the histogram; this prevents the KDE
		# from weird edge behavior.
		n_pseudo = 3 # number of pseudo-bins 
		bin_width=bin_edges[1]-bin_edges[0]
		lower_pseudo_data = np.random.uniform(low=bin_edges[0]-bin_width*n_pseudo, high=bin_edges[0], size=hist[0]*n_pseudo)
		upper_pseudo_data = np.random.uniform(low=bin_edges[-1], high=bin_edges[-1]+bin_width*n_pseudo, size=hist[-1]*n_pseudo)

		# Calculate bandwidth for KDE (Silverman method)
		h = kde_bandwidth(flat)

		# Create a subsampled grid for the KDE based on the subsampled data; by
		# default, we subsample by a factor of 10.
		xs = np.linspace(np.min(subsampled),np.max(subsampled),10*len(hist))

		# Calculate KDE
		kde = gauss_kde(xs,np.concatenate([subsampled,lower_pseudo_data,upper_pseudo_data]),h)
		p68 = compute_HDI(subsampled,0.68)
		p95 = compute_HDI(subsampled,0.95)

		post_max  = xs[kde.argmax()] # posterior max estimated from KDE
		post_mean = np.mean(flat)
		post_med  = np.median(flat)
		low_68    = post_max - p68[0]
		upp_68    = p68[1] - post_max
		low_95    = post_max - p95[0]
		upp_95    = p95[1] - post_max
		post_std  = np.std(flat)
		post_mad  = stats.median_abs_deviation(flat)

		# Quality flags; flag any parameter that violates parameter limits by 1.5 sigma
		flag = 0  
		if ( (post_max-1.5*low_68) <= 0 ):
			flag+=1
		if ~np.isfinite(post_max) or ~np.isfinite(low_68) or ~np.isfinite(upp_68):
			flag+=1

		int_vel_disp_dict[key]['par_best']    = post_max # maximum of posterior distribution
		int_vel_disp_dict[key]['ci_68_low']   = low_68	# lower 68% confidence interval
		int_vel_disp_dict[key]['ci_68_upp']   = upp_68	# upper 68% confidence interval
		int_vel_disp_dict[key]['ci_95_low']   = low_95	# lower 95% confidence interval
		int_vel_disp_dict[key]['ci_95_upp']   = upp_95	# upper 95% confidence interval
		int_vel_disp_dict[key]['mean']        = post_mean # mean of posterior distribution
		int_vel_disp_dict[key]['std_dev']     = post_std	# standard deviation
		int_vel_disp_dict[key]['median']      = post_med # median of posterior distribution
		int_vel_disp_dict[key]['med_abs_dev'] = post_mad	# median absolute deviation
		int_vel_disp_dict[key]['flat_chain']  = flat   # flattened samples used for histogram.
		int_vel_disp_dict[key]['flag']	      = flag 

		if (plot_param_hist==True):
			posterior_plots(key,flat,chain,burn_in,xs,kde,h,
							post_max,low_68,upp_68,low_95,upp_95,post_mean,post_std,post_med,post_mad,
							run_dir)

	return int_vel_disp_dict


# def write_params(param_dict,flux_dict,lum_dict,eqwidth_dict,cont_lum_dict,int_vel_disp_dict,extra_dict,header_dict,bounds,run_dir,
# 				binnum=None,spaxelx=None,spaxely=None):
def write_params(param_dict,header_dict,bounds,run_dir,binnum=None,spaxelx=None,spaxely=None):
	"""
	Writes all measured parameters, fluxes, luminosities, and extra stuff 
	(black hole mass, systemic redshifts) and all flags to a FITS table.
	"""
	# Extract elements from dictionaries
	par_names   = []
	par_best    = []
	ci_68_low   = []
	ci_68_upp   = []
	ci_95_low   = []
	ci_95_upp   = []
	mean        = []
	std_dev     = []
	median      = []
	med_abs_dev = []
	flags 	    = []

	# Param dict
	for key in param_dict:
		par_names.append(key)
		par_best.append(param_dict[key]['par_best'])
		ci_68_low.append(param_dict[key]['ci_68_low'])
		ci_68_upp.append(param_dict[key]['ci_68_upp'])
		ci_95_low.append(param_dict[key]['ci_95_low'])
		ci_95_upp.append(param_dict[key]['ci_95_upp'])
		mean.append(param_dict[key]['mean'])
		std_dev.append(param_dict[key]['std_dev'])
		median.append(param_dict[key]['median'])
		med_abs_dev.append(param_dict[key]['med_abs_dev'])
		flags.append(param_dict[key]['flag'])

	# Sort param_names alphabetically
	i_sort	    = np.argsort(par_names)
	par_names   = np.array(par_names)[i_sort] 
	par_best    = np.array(par_best)[i_sort]  
	ci_68_low   = np.array(ci_68_low)[i_sort]   
	ci_68_upp   = np.array(ci_68_upp)[i_sort]
	ci_95_low   = np.array(ci_95_low)[i_sort]   
	ci_95_upp   = np.array(ci_95_upp)[i_sort]  
	mean        = np.array(mean)[i_sort]   
	std_dev     = np.array(std_dev)[i_sort]
	median      = np.array(median)[i_sort]   
	med_abs_dev = np.array(med_abs_dev)[i_sort] 
	flags	    = np.array(flags)[i_sort]	 

	# Write best-fit parameters to FITS table
	col1  = fits.Column(name='parameter', format='30A', array=par_names)
	col2  = fits.Column(name='best_fit', format='E', array=par_best)
	col3  = fits.Column(name='ci_68_low', format='E', array=ci_68_low)
	col4  = fits.Column(name='ci_68_upp', format='E', array=ci_68_upp)
	col5  = fits.Column(name='ci_95_low', format='E', array=ci_95_low)
	col6  = fits.Column(name='ci_95_upp', format='E', array=ci_95_upp)
	col7  = fits.Column(name='mean', format='E', array=mean)
	col8  = fits.Column(name='std_dev', format='E', array=std_dev)
	col9  = fits.Column(name='median', format='E', array=median)
	col10 = fits.Column(name='med_abs_dev', format='E', array=med_abs_dev)
	col11 = fits.Column(name='flag', format='E', array=flags)
	cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11])
	table_hdu  = fits.BinTableHDU.from_columns(cols)

	if binnum is not None:
		header_dict['binnum'] = binnum
	# Header information
	hdr = fits.Header()
	for key in header_dict:
		hdr[key] = header_dict[key]
	empty_primary = fits.PrimaryHDU(header=hdr)

	hdu = fits.HDUList([empty_primary,table_hdu])
	if spaxelx is not None and spaxely is not None:
		hdu2 = fits.BinTableHDU.from_columns(fits.ColDefs([
			fits.Column(name='spaxelx', array=spaxelx, format='E'),
			fits.Column(name='spaxely', array=spaxely, format='E')
		]))
		hdu.append(hdu2)

	hdu.writeto(run_dir.joinpath('log', 'par_table.fits'), overwrite=True)

	del hdu
	# Write full param dict to log file
	write_log((par_names,par_best,ci_68_low,ci_68_upp,ci_95_low,ci_95_upp,mean,std_dev,median,med_abs_dev,flags),'emcee_results',run_dir)
	return 

def write_chains(param_dict,run_dir):
	"""
	Writes all MCMC chains to a FITS Image HDU.  Each FITS 
	extension corresponds to 
	"""

	# for key in param_dict:
	# 	print(key,np.shape(param_dict[key]["chain"]))

	cols = []
	# Construct a column for each parameter and chain
	for key in param_dict:
		# cols.append(fits.Column(name=key, format='D',array=param_dict[key]['chain']))
		values = param_dict[key]['chain']
		cols.append(fits.Column(name=key, format="%dD" % (values.shape[0]*values.shape[1]), dim="(%d,%d)" % (values.shape[1],values.shape[0]), array=[values]))
	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir.joinpath('log', 'MCMC_chains.fits'), overwrite=True)

	return 
	
def plot_best_model(param_dict,
					line_list,
					combined_line_list,
					lam_gal,
					galaxy,
					noise,
					comp_options,
					losvd_options,
					host_options,
					power_options,
					opt_feii_options,
					uv_iron_options,
					balmer_options,
					outflow_test_options,
					host_template,
					opt_feii_templates,
					uv_iron_template,
					balmer_template,
					stel_templates,
					fwhm_gal,
					fit_mask,
					fit_stat,
					velscale,
					run_dir):
	"""
	Plots the best fig model and outputs the components to a FITS file for reproduction.
	"""
	def calc_new_center(center,voff):
		"""
		Calculated new center shifted 
		by some velocity offset.
		"""
		c = 299792.458 # speed of light (km/s)
		new_center = (voff*center)/c + center
		return new_center

	param_names  = [key for key in param_dict ]
	par_best	 = [param_dict[key]['par_best'] for key in param_dict ]


	output_model = True
	fit_type	 = 'final'
	comp_dict = fit_model(par_best,
						  param_names,
						  line_list,
						  combined_line_list,
						  lam_gal,
						  galaxy,
						  noise,
						  comp_options,
						  losvd_options,
						  host_options,
						  power_options,
						  opt_feii_options,
						  uv_iron_options,
						  balmer_options,
						  outflow_test_options,
						  host_template,
						  opt_feii_templates,
						  uv_iron_template,
						  balmer_template,
						  stel_templates,
						  fwhm_gal,
						  fit_mask,
						  velscale,
						  run_dir,
						  fit_type,
						  fit_stat,
						  output_model)

	

	# Put params in dictionary
	p = dict(zip(param_names,par_best))

	# Maximum Likelihood plot
	fig = plt.figure(figsize=(14,6)) 
	gs = gridspec.GridSpec(4, 1)
	gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
	ax1  = plt.subplot(gs[0:3,0])
	ax2  = plt.subplot(gs[3,0])

	for key in comp_dict:
		if (key=='DATA'):
			ax1.plot(comp_dict['WAVE'],comp_dict['DATA'],linewidth=0.5,color='white',label='Data',zorder=0)
		elif (key=='MODEL'):
			ax1.plot(lam_gal,comp_dict[key], color='xkcd:bright red', linewidth=1.0, label='Model', zorder=15)
		elif (key=='HOST_GALAXY'):
			ax1.plot(comp_dict['WAVE'], comp_dict['HOST_GALAXY'], color='xkcd:bright green', linewidth=0.5, linestyle='-', label='Host/Stellar')

		elif (key=='POWER'):
			ax1.plot(comp_dict['WAVE'], comp_dict['POWER'], color='xkcd:red' , linewidth=0.5, linestyle='--', label='AGN Cont.')

		elif (key in ['NA_OPT_FEII_TEMPLATE','BR_OPT_FEII_TEMPLATE']):
			ax1.plot(comp_dict['WAVE'], comp_dict['NA_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='Narrow FeII')
			ax1.plot(comp_dict['WAVE'], comp_dict['BR_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='Broad FeII')

		elif (key in ['F_OPT_FEII_TEMPLATE','S_OPT_FEII_TEMPLATE','G_OPT_FEII_TEMPLATE','Z_OPT_FEII_TEMPLATE']):
			if key=='F_OPT_FEII_TEMPLATE':
				ax1.plot(comp_dict['WAVE'], comp_dict['F_OPT_FEII_TEMPLATE'], color='xkcd:yellow', linewidth=0.5, linestyle='-' , label='F-transition FeII')
			elif key=='S_OPT_FEII_TEMPLATE':
				ax1.plot(comp_dict['WAVE'], comp_dict['S_OPT_FEII_TEMPLATE'], color='xkcd:mustard', linewidth=0.5, linestyle='-' , label='S-transition FeII')
			elif key=='G_OPT_FEII_TEMPLATE':
				ax1.plot(comp_dict['WAVE'], comp_dict['G_OPT_FEII_TEMPLATE'], color='xkcd:orange', linewidth=0.5, linestyle='-' , label='G-transition FeII')
			elif key=='Z_OPT_FEII_TEMPLATE':
				ax1.plot(comp_dict['WAVE'], comp_dict['Z_OPT_FEII_TEMPLATE'], color='xkcd:rust', linewidth=0.5, linestyle='-' , label='Z-transition FeII')
		elif (key=='UV_IRON_TEMPLATE'):
			ax1.plot(comp_dict['WAVE'], comp_dict['UV_IRON_TEMPLATE'], color='xkcd:bright purple', linewidth=0.5, linestyle='-' , label='UV Iron'	 )
		elif (key=='BALMER_CONT'):
			ax1.plot(comp_dict['WAVE'], comp_dict['BALMER_CONT'], color='xkcd:bright green', linewidth=0.5, linestyle='--' , label='Balmer Continuum'	 )
		# Plot emission lines by cross-referencing comp_dict with line_list
		if (key in line_list):
			if (line_list[key]["line_type"]=="na"):
				ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:cerulean', linewidth=0.5, linestyle='-', label='Narrow/Core Comp.')
			if (line_list[key]["line_type"]=="br"):
				ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:bright teal', linewidth=0.5, linestyle='-', label='Broad Comp.')
			if (line_list[key]["line_type"]=="out"):
				ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:bright pink', linewidth=0.5, linestyle='-', label='Outflow Comp.')
			if (line_list[key]["line_type"]=="abs"):
				ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:pastel red', linewidth=0.5, linestyle='-', label='Absorption Comp.')
			if (line_list[key]["line_type"]=="user"):
				ax1.plot(comp_dict['WAVE'], comp_dict[key], color='xkcd:electric lime', linewidth=0.5, linestyle='-', label='Other')

	# Plot bad pixels
	ibad = [i for i in range(len(lam_gal)) if i not in fit_mask]
	if (len(ibad)>0):# and (len(ibad[0])>1):
		bad_wave = [(lam_gal[m],lam_gal[m+1]) for m in ibad if ((m+1)<len(lam_gal))]
		ax1.axvspan(bad_wave[0][0],bad_wave[0][0],alpha=0.25,color='xkcd:lime green',label="bad pixels")
		for i in bad_wave[1:]:
			ax1.axvspan(i[0],i[0],alpha=0.25,color='xkcd:lime green')

	ax1.set_xticklabels([])
	ax1.set_xlim(np.min(lam_gal)-10,np.max(lam_gal)+10)
	# ax1.set_ylim(-0.5*np.median(comp_dict['MODEL']),np.max([comp_dict['DATA'],comp_dict['MODEL']]))
	ax1.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)',fontsize=10)
	# Residuals
	sigma_resid = np.nanstd(comp_dict['DATA']-comp_dict['MODEL'])
	sigma_noise = np.median(comp_dict['NOISE'])
	ax2.plot(lam_gal,(comp_dict['NOISE']*3.0),linewidth=0.5,color="xkcd:bright orange",label='$\sigma_{\mathrm{noise}}=%0.4f$' % (sigma_noise))
	ax2.plot(lam_gal,(comp_dict['RESID']*3.0),linewidth=0.5,color="white",label='$\sigma_{\mathrm{resid}}=%0.4f$' % (sigma_resid))
	ax1.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
	ax2.axhline(0.0,linewidth=1.0,color='white',linestyle='--')
	# Axes limits 
	ax_low = np.min([ax1.get_ylim()[0],ax2.get_ylim()[0]])
	ax_upp = np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
	if np.isfinite(sigma_resid):
		ax_upp += 3.0 * sigma_resid

	minimum = [np.nanmin(comp_dict[comp][np.where(np.isfinite(comp_dict[comp]))[0]]) for comp in comp_dict
			   if comp_dict[comp][np.isfinite(comp_dict[comp])[0]].size > 0]
	if len(minimum) > 0:
		minimum = np.nanmin(minimum)
	else:
		minimum = 0.0
	ax1.set_ylim(np.nanmin([0.0, minimum]), ax_upp)
	ax1.set_xlim(np.min(lam_gal),np.max(lam_gal))
	ax2.set_ylim(ax_low,ax_upp)
	ax2.set_xlim(np.min(lam_gal),np.max(lam_gal))
	# Axes labels
	ax2.set_yticklabels(np.array(ax2.get_yticks()/3.0,dtype=int))
	ax2.set_ylabel(r'$\Delta f_\lambda$',fontsize=12)
	ax2.set_xlabel(r'Wavelength, $\lambda\;(\mathrm{\AA})$',fontsize=12)
	handles, labels = ax1.get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	ax1.legend(by_label.values(), by_label.keys(),loc='upper right',fontsize=8)
	ax2.legend(loc='upper right',fontsize=8)

	# Emission line annotations
	# Gather up emission line center wavelengths and labels (if available, removing any duplicates)
	line_labels = []
	for line in line_list:
		if "label" in line_list[line]:
			line_labels.append([line,line_list[line]["label"]])
	line_labels = set(map(tuple, line_labels))   
	for label in line_labels:
		center = line_list[label[0]]["center"]
		if (line_list[label[0]]["voff"]=="free"):
			voff = p[label[0]+"_VOFF"]
		elif (line_list[label[0]]["voff"]!="free"):
			voff   =  ne.evaluate(line_list[label[0]]["voff"],local_dict = p).item()
		xloc = calc_new_center(center,voff)
		yloc = np.max([comp_dict["DATA"][find_nearest(lam_gal,xloc)[1]],comp_dict["MODEL"][find_nearest(lam_gal,xloc)[1]]])
		ax1.annotate(label[1], xy=(xloc, yloc),  xycoords='data',
		xytext=(xloc, yloc), textcoords='data',
		horizontalalignment='center', verticalalignment='bottom',
		color='xkcd:white',fontsize=6,
		)

	# Save figure
	plt.savefig(run_dir.joinpath('best_fit_model.pdf'))
	# Close plot
	fig.clear()
	plt.close()
	


	# Store best-fit components in a FITS file
	# Construct a column for each parameter and chain
	cols = []
	for key in comp_dict:
		cols.append(fits.Column(name=key, format='E', array=comp_dict[key]))

	# Add fit mask to cols
	cols.append(fits.Column(name="MASK", format='E', array=fit_mask))

	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir.joinpath('log', 'best_model_components.fits'), overwrite=True)
	
	return comp_dict

def fit_quality_pars(param_dict,line_list,combined_line_list,comp_dict,fit_type,fit_stat):

	fit_quality_dict = {}
	if fit_stat=="RCHI2":
		if fit_type=="max_like":
			noise = comp_dict["NOISE"]*param_dict["NOISE_SCALE"]["med"]
		elif fit_type=="mcmc":
			noise = comp_dict["NOISE"]*param_dict["NOISE_SCALE"]["par_best"]
	elif fit_stat!="RHIC2":
		noise = comp_dict["NOISE"]
	# compute NPIX for each line in the line list
	for l in line_list:
		npix = len(np.where(comp_dict[l]>noise)[0])
		if fit_type=="max_like":
			fit_quality_dict[l+"_NPIX"] = {"med":npix,"std":0,"flag":0}
		elif fit_type=="mcmc":
			fit_quality_dict[l+"_NPIX"] = {"par_best":npix,
										   "ci_68_low":0,"ci_68_upp":0,
										   "ci_95_low":0,"ci_95_upp":0,
										   "mean":0,"std_dev":0,
										   "median":0,"med_abs_dev":0,
										   "flag":0
										  }
	# compute NPIX for any combined lines
	if len(combined_line_list)>0:
		for c in combined_line_list:
			comb_line = np.zeros(len(noise))
			for l in combined_line_list[c]["lines"]:
				comb_line+=comp_dict[l]
			npix = len(np.where(comb_line>noise)[0])
			if fit_type=="max_like":
				fit_quality_dict[c+"_NPIX"] = {"med":npix,"std":0,"flag":0}
			elif fit_type=="mcmc":
				fit_quality_dict[c+"_NPIX"] = {"par_best":npix,
											   "ci_68_low":0,"ci_68_upp":0,
											   "ci_95_low":0,"ci_95_upp":0,
											   "mean":0,"std_dev":0,
											   "median":0,"med_abs_dev":0,
											   "flag":0
											   }

	# compute a total chi-squared and r-squared
	r_sqaured = 1-(np.sum((comp_dict["DATA"]-comp_dict["MODEL"])**2/np.sum(comp_dict["DATA"]**2)))
	nu = len(comp_dict["DATA"])-len(param_dict)
	rchi_squared = (np.sum((comp_dict["DATA"]-comp_dict["MODEL"])**2/(noise)**2,axis=0))/nu
	if fit_type=="max_like":
		fit_quality_dict["R_SQUARED"] = {"med":r_sqaured,"std":0,"flag":0}
		fit_quality_dict["RCHI_SQUARED"] = {"med":rchi_squared,"std":0,"flag":0}
	elif fit_type=="mcmc": 
		fit_quality_dict["R_SQUARED"] =  {"par_best":r_sqaured,
											   "ci_68_low":0,"ci_68_upp":0,
											   "ci_95_low":0,"ci_95_upp":0,
											   "mean":0,"std_dev":0,
											   "median":0,"med_abs_dev":0,
											   "flag":0
											   }
		fit_quality_dict["RCHI_SQUARED"] =  {"par_best":rchi_squared,
											   "ci_68_low":0,"ci_68_upp":0,
											   "ci_95_low":0,"ci_95_upp":0,
											   "mean":0,"std_dev":0,
											   "median":0,"med_abs_dev":0,
											   "flag":0
											   }

	return fit_quality_dict

def write_max_like_results(result_dict,comp_dict,header_dict,fit_mask,run_dir,
						   binnum=None,spaxelx=None,spaxely=None):
	"""
	Write maximum likelihood fit results to FITS table
	if MCMC is not performed. 
	"""
	# for key in result_dict:
	# 	print(key, result_dict[key])
	# Extract elements from dictionaries

	par_names = []
	par_best  = []
	sig	      = []
	for key in result_dict:
		par_names.append(key)
		par_best.append(result_dict[key]['med'])
		if "std" in result_dict[key]:
			sig.append(result_dict[key]['std'])

	# Sort the fit results
	i_sort	= np.argsort(par_names)
	par_names = np.array(par_names)[i_sort] 
	par_best  = np.array(par_best)[i_sort]  
	sig   = np.array(sig)[i_sort]   

	# Write best-fit parameters to FITS table
	col1 = fits.Column(name='parameter', format='30A', array=par_names)
	col2 = fits.Column(name='best_fit' , format='E'  , array=par_best)
	if "std" in result_dict[par_names[0]]:
		col3 = fits.Column(name='sigma'	, format='E'  , array=sig)
	
	if "std" in result_dict[par_names[0]]:
		cols = fits.ColDefs([col1,col2,col3])
	else: 
		cols = fits.ColDefs([col1,col2])
	table_hdu = fits.BinTableHDU.from_columns(cols)
	# Header information
	hdr = fits.Header()
	if binnum is not None:
		header_dict['binnum'] = binnum
	for key in header_dict:
		hdr[key] = header_dict[key]
	empty_primary = fits.PrimaryHDU(header=hdr)
	hdu = fits.HDUList([empty_primary, table_hdu])

	if spaxelx is not None and spaxely is not None:
		hdu2 = fits.BinTableHDU.from_columns(fits.ColDefs([
			fits.Column(name='spaxelx', array=spaxelx, format='E'),
			fits.Column(name='spaxely', array=spaxely, format='E')
		]))
		hdu.append(hdu2)

	hdu.writeto(run_dir.joinpath('log', 'par_table.fits'), overwrite=True)
	del hdu
	# Write best-fit components to FITS file
	cols = []
	# Construct a column for each parameter and chain
	for key in comp_dict:
		cols.append(fits.Column(name=key, format='E', array=comp_dict[key]))
	# Add fit mask to cols
	mask = np.zeros(len(comp_dict["WAVE"]),dtype=bool)
	mask[fit_mask] = True
	cols.append(fits.Column(name="MASK", format='E', array=mask))
	# Write to fits
	cols = fits.ColDefs(cols)
	hdu = fits.BinTableHDU.from_columns(cols)
	hdu.writeto(run_dir.joinpath('log', 'best_model_components.fits'), overwrite=True)
	#
	return 

def plotly_best_fit(objname,line_list,fit_mask,run_dir):
	"""
	Generates an interactive HTML plot of the best fit model
	using plotly.
	"""
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
	# Open the best_fit_components file
	hdu = fits.open(run_dir.joinpath("log", "best_model_components.fits") )
	tbdata = hdu[1].data	 # FITS table data is stored on FITS extension 1
	cols = [i.name for i in tbdata.columns]
	hdu.close()

	# Create a figure with subplots
	fig = make_subplots(rows=2, cols=1, row_heights=(3,1) )
	# tracenames = []
	# Plot
	for comp in cols:
		if comp=="DATA":
			tracename = "Data"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["DATA"] , mode="lines", line=go.scatter.Line(color="white", width=1), name=tracename, legendrank=1, showlegend=True), row=1, col=1)
		if comp=="MODEL":
			tracename="Model"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["MODEL"], mode="lines", line=go.scatter.Line(color="red"  , width=1), name=tracename, legendrank=2, showlegend=True), row=1, col=1)
		if comp=="NOISE":
			tracename="Noise"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["NOISE"], mode="lines", line=go.scatter.Line(color="#FE00CE"  , width=1), name=tracename, legendrank=3, showlegend=True), row=1, col=1)
		# Continuum components
		if comp=="HOST_GALAXY":
			tracename="Host Galaxy"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["HOST_GALAXY"], mode="lines", line=go.scatter.Line(color="lime", width=1), name=tracename, legendrank=4, showlegend=True), row=1, col=1)
		if comp=="POWER":
			tracename="Power-law"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["POWER"], mode="lines", line=go.scatter.Line(color="red", width=1, dash="dash"), name=tracename, legendrank=5, showlegend=True), row=1, col=1)
		if comp=="BALMER_CONT":
			tracename="Balmer cont."
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["BALMER_CONT"], mode="lines", line=go.scatter.Line(color="lime", width=1, dash="dash"), name=tracename, legendrank=6, showlegend=True), row=1, col=1)
		# FeII componentes
		if comp=="UV_IRON_TEMPLATE":
			tracename="UV Iron"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["UV_IRON_TEMPLATE"], mode="lines", line=go.scatter.Line(color="#AB63FA", width=1), name=tracename, legendrank=7, showlegend=True), row=1, col=1)
		if comp=="NA_OPT_FEII_TEMPLATE":
			tracename="Narrow FeII"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["NA_OPT_FEII_TEMPLATE"], mode="lines", line=go.scatter.Line(color="rgb(255,255,51)", width=1), name=tracename, legendrank=7, showlegend=True), row=1, col=1)
		if comp=="BR_OPT_FEII_TEMPLATE":
			tracename="Broad FeII"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["BR_OPT_FEII_TEMPLATE"], mode="lines", line=go.scatter.Line(color="#FF7F0E", width=1), name=tracename, legendrank=8, showlegend=True), row=1, col=1)
		if comp=='F_OPT_FEII_TEMPLATE':
			tracename="F-transition FeII"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["F_OPT_FEII_TEMPLATE"], mode="lines", line=go.scatter.Line(color="rgb(255,255,51)", width=1), name=tracename, legendrank=7, showlegend=True), row=1, col=1)
		if comp=='S_OPT_FEII_TEMPLATE':
			tracename="S-transition FeII"
			fig.add_trace(go.Scatter( x = tbdata["waVe"], y = tbdata["S_OPT_FEII_TEMPLATE"], mode="lines", line=go.scatter.Line(color="rgb(230,171,2)", width=1), name=tracename, legendrank=8, showlegend=True), row=1, col=1)
		if comp=='G_OPT_FEII_TEMPLATE':
			tracename="G-transition FeII"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["G_OPT_FEII_TEMPLATE"], mode="lines", line=go.scatter.Line(color="#FF7F0E", width=1), name=tracename, legendrank=9, showlegend=True), row=1, col=1)
		if comp=='Z_OPT_FEII_TEMPLATE':
			tracename="Z-transition FeII"
			fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["Z_OPT_FEII_TEMPLATE"], mode="lines", line=go.scatter.Line(color="rgb(217,95,2)", width=1), name=tracename, legendrank=10, showlegend=True), row=1, col=1)
		# Line components
		if comp in line_list:
			if line_list[comp]["line_type"]=="na":
				  # tracename="narrow line"
				fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="#00B5F7", width=1), name=comp, legendgroup="narrow lines",legendgrouptitle_text="narrow lines", legendrank=11,), row=1, col=1)
				  # tracenames.append(tracename)
			if line_list[comp]["line_type"]=="br":
				  # tracename="broad line"
				fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="#22FFA7", width=1), name=comp, legendgroup="broad lines",legendgrouptitle_text="broad lines", legendrank=13,), row=1, col=1)
				  # tracenames.append(tracename)
			if line_list[comp]["line_type"]=="out":
				  # tracename="outflow line"
				fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="#FC0080", width=1), name=comp, legendgroup="outflow lines",legendgrouptitle_text="outflow lines", legendrank=14,), row=1, col=1)
				  # tracenames.append(tracename)
			if line_list[comp]["line_type"]=="abs":
				  # tracename="absorption line"
				fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="#DA16FF", width=1), name=comp, legendgroup="absorption lines",legendgrouptitle_text="absorption lines", legendrank=15,), row=1, col=1)
				  # tracenames.append(tracename)
			if line_list[comp]["line_type"]=="user":
				  # tracename="absorption line"
				fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata[comp], mode="lines", line=go.scatter.Line(color="rgb(153,201,59)", width=1), name=comp, legendgroup="user lines",legendgrouptitle_text="user lines", legendrank=16,), row=1, col=1)
				  # tracenames.append(tracename)
		
	fig.add_hline(y=0.0, line=dict(color="gray", width=2), row=1, col=1)  
	
	# Plot bad pixels
	# lam_gal = tbdata["WAVE"]
	# ibad = [i for i in range(len(lam_gal)) if i not in fit_mask]
	# if (len(ibad)>0):# and (len(ibad[0])>1):
	# 	bad_wave = [(lam_gal[m],lam_gal[m+1]) for m in ibad if ((m+1)<len(lam_gal))]
	# 	# ax1.axvspan(bad_wave[0][0],bad_wave[0][0],alpha=0.25,color='xkcd:lime green',label="bad pixels")
	# 	fig.add_vrect(
	# 					x0=bad_wave[0][0], x1=bad_wave[0][0],
	# 					fillcolor="rgb(179,222,105)", opacity=0.25,
	# 					layer="below", line_width=0,name="bad pixels",
	# 					),
	# 	for i in bad_wave[1:]:
	# 		# ax1.axvspan(i[0],i[0],alpha=0.25,color='xkcd:lime green')
	# 		fig.add_vrect(
	# 						x0=i[0], x1=i[1],
	# 						fillcolor="rgb(179,222,105)", opacity=0.25,
	# 						layer="below", line_width=0,name="bad pixels",
	# 					),
		
		
	# Residuals
	fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["RESID"], mode="lines", line=go.scatter.Line(color="white"  , width=1), name="Residuals", showlegend=False), row=2, col=1)
	fig.add_trace(go.Scatter( x = tbdata["WAVE"], y = tbdata["NOISE"], mode="lines", line=go.scatter.Line(color="#FE00CE"  , width=1), name="Noise", showlegend=False, legendrank=3,), row=2, col=1)
	# Figure layout, size, margins
	fig.update_layout(
		autosize=False,
		width=1700,
		height=800,
		margin=dict(
			l=100,
			r=100,
			b=100,
			t=100,
			pad=1
		),
		title= objname,
		font_family="Times New Roman",
		font_size=16,
		font_color="white",
		legend_title_text="Components",
		legend_bgcolor="black",
		paper_bgcolor="black",
		plot_bgcolor="black",
	)
	# Update x-axis properties
	fig.update_xaxes(title=r"$\Large\lambda_{\rm{rest}}\;\left[Å\right]$", linewidth=0.5, linecolor="gray", mirror=True, 
					 gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
					 row=1, col=1)
	fig.update_xaxes(title=r"$\Large\lambda_{\rm{rest}}\;\left[Å\right]$", linewidth=0.5, linecolor="gray", mirror=True,
					 gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
					 row=2, col=1)
	# Update y-axis properties
	fig.update_yaxes(title=r"$\Large f_\lambda\;\left[\rm{erg}\;\rm{cm}^{-2}\;\rm{s}^{-1}\;Å^{-1}\right]$", linewidth=0.5, linecolor="gray",  mirror=True,
					 gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
					 row=1, col=1)
	fig.update_yaxes(title=r"$\Large\Delta f_\lambda$", linewidth=0.5, linecolor="gray", mirror=True,
					 gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
					 row=2, col=1)
		
	fig.update_xaxes(matches='x')
	# fig.update_yaxes(matches='y')
	# fig.show()
	
	# Write to HTML
	fig.write_html(run_dir.joinpath("%s_bestfit.html" % objname),include_mathjax="cdn")
	# Write to PDF
	# fig.write_image(run_dir.joinpath("%s_bestfit.pdf" % objname))

	return

# Clean-up Routine
##################################################################################

def cleanup(run_dir):
	"""
	Cleans up the run directory.
	"""
	# Remove param_plots folder if empty
	histo_dir = run_dir.joinpath('histogram_plots')
	if histo_dir.is_dir() and not any(histo_dir.iterdir()):
		histo_dir.rmdir()

	# If run_dir is empty because there aren't enough good pixels, remove it
	if run_dir.is_dir() and not any(run_dir.iterdir()):
		run_dir.rmdir()

	return None

##################################################################################

def write_log(output_val,output_type,run_dir):
	"""
	This function writes values to a log file as the code runs.
	"""

	log_file_path = run_dir.joinpath('log', 'log_file.txt')
	log_file_path.parent.mkdir(parents=True, exist_ok=True)
	if not log_file_path.is_file():
		with log_file_path.open(mode='w') as logfile:
			logfile.write('\n############################### BADASS v9.1.1 LOGFILE ####################################\n')

	# sdss_prepare
	# output_val=(file,ra,dec,z,fit_min,fit_max,velscale,ebv), output_type=0
	if (output_type=='prepare_sdss_spec'):
		fits_file,ra,dec,z,cosmology,fit_min,fit_max,velscale,ebv = output_val
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}'.format('file:'		   , fits_file.name			))
			logfile.write('\n{0:<30}{1:<30}'.format('(RA, DEC):'	  , '(%0.6f,%0.6f)' % (ra,dec)	 ))
			logfile.write('\n{0:<30}{1:<30}'.format('SDSS redshift:'  , '%0.5f' % z					))
			logfile.write('\n{0:<30}{1:<30}'.format('fitting region:' , '(%d,%d) [A]' % (fit_min,fit_max)  ))
			logfile.write('\n{0:<30}{1:<30}'.format('velocity scale:' , '%0.2f [km/s/pixel]' % velscale))
			logfile.write('\n{0:<30}{1:<30}'.format('Galactic E(B-V):', '%0.3f' % ebv))
			logfile.write('\n')
			logfile.write('\n{0:<30}'.format('Units:'))
			logfile.write('\n{0:<30}'.format('	- Note: SDSS Spectra are in units of [1.e-17 erg/s/cm2/Å]'))
			logfile.write('\n{0:<30}'.format('	- Velocity, dispersion, and FWHM have units of [km/s]'))
			logfile.write('\n{0:<30}'.format('	- Fluxes and Luminosities are in log-10'))
			logfile.write('\n')
			logfile.write('\n{0:<30}'.format('Cosmology:'))
			logfile.write('\n{0:<30}'.format('	H0 = %0.1f' % cosmology["H0"]))
			logfile.write('\n{0:<30}'.format('	Om0 = %0.2f' % cosmology["Om0"]))
			logfile.write('\n')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	if (output_type=='prepare_user_spec'):
		fits_file,z,cosmology,fit_min,fit_max,velscale,ebv = output_val
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}'.format('file:'		   , fits_file.name			))
			# logfile.write('\n{0:<30}{1:<30}'.format('(RA, DEC):'	  , '(%0.6f,%0.6f)' % (ra,dec)	 ))
			logfile.write('\n{0:<30}{1:<30}'.format('SDSS redshift:'  , '%0.5f' % z					))
			logfile.write('\n{0:<30}{1:<30}'.format('fitting region:' , '(%d,%d) [A]' % (fit_min,fit_max)  ))
			logfile.write('\n{0:<30}{1:<30}'.format('velocity scale:' , '%0.2f [km/s/pixel]' % velscale))
			logfile.write('\n{0:<30}{1:<30}'.format('Galactic E(B-V):', '%0.3f' % ebv))
			logfile.write('\n')
			logfile.write('\n{0:<30}'.format('Units:'))
			logfile.write('\n{0:<30}'.format('	- Note: SDSS Spectra are in units of [1.e-17 erg/s/cm2/Å]'))
			logfile.write('\n{0:<30}'.format('	- Velocity, dispersion, and FWHM have units of [km/s]'))
			logfile.write('\n{0:<30}'.format('	- Fluxes and Luminosities are in log-10'))
			logfile.write('\n')
			logfile.write('\n{0:<30}'.format('Cosmology:'))
			logfile.write('\n{0:<30}'.format('	H0 = %0.1f' % cosmology["H0"]))
			logfile.write('\n{0:<30}'.format('	Om0 = %0.2f' % cosmology["Om0"]))
			logfile.write('\n')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	if (output_type=='fit_information'):
		fit_options,mcmc_options,comp_options,losvd_options,host_options,power_options,opt_feii_options,uv_iron_options,balmer_options,\
		plot_options,output_options = output_val
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n### User-Input Fitting Paramters & Options ###')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n')
			# General fit options
			logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   fit_options:','',''))
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_reg',':',str(fit_options['fit_reg']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('good_thresh',':',str(fit_options['good_thresh']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('mask_bad_pix',':',str(fit_options['mask_bad_pix']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('n_basinhop',':',str(fit_options['n_basinhop']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('test_outflows',':',str(fit_options['test_outflows']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('test_line',':',str(fit_options['test_line']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('max_like_niter',':',str(fit_options['max_like_niter']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('output_pars',':',str(fit_options['output_pars']) )) 
			logfile.write('\n')
			# MCMC options
			if mcmc_options['mcmc_fit']==False:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   mcmc_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('','','MCMC fitting is turned off.' )) 
				logfile.write('\n')
			elif mcmc_options['mcmc_fit']==True:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   mcmc_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('mcmc_fit',':',str(mcmc_options['mcmc_fit']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('nwalkers',':',str(mcmc_options['nwalkers']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('auto_stop',':',str(mcmc_options['auto_stop']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('conv_type',':',str(mcmc_options['conv_type']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('min_samp',':',str(mcmc_options['min_samp']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('ncor_times',':',str(mcmc_options['ncor_times']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('autocorr_tol',':',str(mcmc_options['autocorr_tol']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('write_iter',':',str(mcmc_options['write_iter']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('write_thresh',':',str(mcmc_options['write_thresh']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('burn_in',':',str(mcmc_options['burn_in']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('min_iter',':',str(mcmc_options['min_iter']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('max_iter',':',str(mcmc_options['max_iter']) )) 
				logfile.write('\n')
			# Fit Component options
			logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   comp_options:','',''))
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_opt_feii',':',str(comp_options['fit_opt_feii']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_uv_iron',':',str(comp_options['fit_uv_iron']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_balmer',':',str(comp_options['fit_balmer']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_losvd',':',str(comp_options['fit_losvd']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_host',':',str(comp_options['fit_host']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_power',':',str(comp_options['fit_power']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_narrow',':',str(comp_options['fit_narrow']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_broad',':',str(comp_options['fit_broad']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_outflow',':',str(comp_options['fit_outflow']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('fit_absorp',':',str(comp_options['fit_absorp']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('tie_line_fwhm',':',str(comp_options['tie_line_fwhm']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('tie_line_voff',':',str(comp_options['tie_line_voff']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('na_line_profile',':',str(comp_options['na_line_profile']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('br_line_profile',':',str(comp_options['br_line_profile']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('out_line_profile',':',str(comp_options['out_line_profile']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('abs_line_profile',':',str(comp_options['abs_line_profile']) )) 

			logfile.write('\n')
			# LOSVD options
			if comp_options["fit_losvd"]==True:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   losvd_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('library',':',str(losvd_options['library']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('vel_const',':',str(losvd_options['vel_const']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('disp_const',':',str(losvd_options['disp_const']) )) 
				logfile.write('\n')
			elif comp_options["fit_losvd"]==False:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   losvd_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('','','LOSVD fitting is turned off.' )) 
				logfile.write('\n')
			# Host Options
			if comp_options["fit_host"]==True:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   host_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('age',':',str(host_options['age']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('vel_const',':',str(host_options['vel_const']) )) 
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('disp_const',':',str(host_options['disp_const']) )) 
				logfile.write('\n')
			elif comp_options["fit_host"]==False:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   host_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('','','Host-galaxy template fitting is turned off.' )) 
				logfile.write('\n')
			# Power-law continuum options
			if comp_options['fit_power']==True:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   power_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('type',':',str(power_options['type']) )) 
				logfile.write('\n')
			elif comp_options["fit_power"]==False:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   power_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('','','Power Law fitting is turned off.' )) 
				logfile.write('\n')
			# Optical FeII fitting options
			if (comp_options['fit_opt_feii']==True):
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   opt_feii_options:','',''))
			if (comp_options['fit_opt_feii']==True) and (opt_feii_options['opt_template']['type']=='VC04'):
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_template:',':','type: %s' % str(opt_feii_options['opt_template']['type']) ))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_amp_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_amp_const']['bool']),str(opt_feii_options['opt_amp_const']['br_opt_feii_val']),str(opt_feii_options['opt_amp_const']['na_opt_feii_val']))))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_fwhm_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_fwhm_const']['bool']),str(opt_feii_options['opt_fwhm_const']['br_opt_feii_val']),str(opt_feii_options['opt_fwhm_const']['na_opt_feii_val']))))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_voff_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_voff_const']['bool']),str(opt_feii_options['opt_voff_const']['br_opt_feii_val']),str(opt_feii_options['opt_voff_const']['na_opt_feii_val']))))
			if (comp_options['fit_opt_feii']==True) and (opt_feii_options['opt_template']['type']=='K10'):
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_template:',':','type: %s' % str(opt_feii_options['opt_template']['type']) ))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_amp_const',':','bool: %s, f_feii_val: %s, s_feii_val: %s, g_feii_val: %s, z_feii_val: %s' % (str(opt_feii_options['opt_amp_const']['bool']),str(opt_feii_options['opt_amp_const']['f_feii_val']),str(opt_feii_options['opt_amp_const']['s_feii_val']),str(opt_feii_options['opt_amp_const']['g_feii_val']),str(opt_feii_options['opt_amp_const']['z_feii_val']))))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_fwhm_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_fwhm_const']['bool']),str(opt_feii_options['opt_fwhm_const']['opt_feii_val']),)))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_voff_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_voff_const']['bool']),str(opt_feii_options['opt_voff_const']['opt_feii_val']),)))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('opt_temp_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_temp_const']['bool']),str(opt_feii_options['opt_temp_const']['opt_feii_val']),)))
			elif comp_options["fit_opt_feii"]==False:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   opt_feii_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('','','Optical FeII fitting is turned off.' )) 
				logfile.write('\n')
			# UV Iron options
			if (comp_options['fit_uv_iron']==True):
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   uv_iron_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('uv_amp_const',':','bool: %s, uv_iron_val: %s' % (str(uv_iron_options['uv_amp_const']['bool']),str(uv_iron_options['uv_amp_const']['uv_iron_val']) )))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('uv_fwhm_const',':','bool: %s, uv_iron_val: %s' % (str(uv_iron_options['uv_fwhm_const']['bool']),str(uv_iron_options['uv_fwhm_const']['uv_iron_val']),)))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('uv_voff_const',':','bool: %s, uv_iron_val: %s' % (str(uv_iron_options['uv_voff_const']['bool']),str(uv_iron_options['uv_voff_const']['uv_iron_val']),)))
			elif comp_options["fit_uv_iron"]==False:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   uv_iron_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('','','UV Iron fitting is turned off.' )) 
				logfile.write('\n')	
			# Balmer options
			if (comp_options['fit_balmer']==True):
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   balmer_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('R_const',':','bool: %s, R_val: %s' % (str(balmer_options['R_const']['bool']),str(balmer_options['R_const']['R_val']) )))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('balmer_amp_const',':','bool: %s, balmer_amp_val: %s' % (str(balmer_options['balmer_amp_const']['bool']),str(balmer_options['balmer_amp_const']['balmer_amp_val']),)))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('balmer_fwhm_const',':','bool: %s, balmer_fwhm_val: %s' % (str(balmer_options['balmer_fwhm_const']['bool']),str(balmer_options['balmer_fwhm_const']['balmer_fwhm_val']),)))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('balmer_voff_const',':','bool: %s, balmer_voff_val: %s' % (str(balmer_options['balmer_voff_const']['bool']),str(balmer_options['balmer_voff_const']['balmer_voff_val']),)))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('Teff_const',':','bool: %s, Teff_val: %s' % (str(balmer_options['Teff_const']['bool']),str(balmer_options['Teff_const']['Teff_val']),)))
				logfile.write('\n{0:>30}{1:<2}{2:<100}'.format('tau_const',':','bool: %s, tau_val: %s' % (str(balmer_options['tau_const']['bool']),str(balmer_options['tau_const']['tau_val']),)))
			elif comp_options["fit_balmer"]==False:
				logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   balmer_options:','',''))
				logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('','','Balmer pseudo-continuum fitting is turned off.' )) 
				logfile.write('\n')

			# Plotting options
			logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   plot_options:','',''))
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('plot_param_hist',':',str(plot_options['plot_param_hist']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('plot_flux_hist',':',str(plot_options['plot_flux_hist']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('plot_lum_hist',':',str(plot_options['plot_lum_hist']) ))
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('plot_eqwidth_hist',':',str(plot_options['plot_eqwidth_hist']) ))
			# Output options
			logfile.write('\n{0:<30}{1:<30}{2:<30}'.format('   output_options:','',''))
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('write_chain',':',str(output_options['write_chain']) )) 
			logfile.write('\n{0:>30}{1:<2}{2:<30}'.format('verbose',':',str(output_options['verbose']) )) 
			#
			logfile.write('\n')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None
	
	if (output_type=='update_opt_feii'):
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n	* optical FeII templates outside of fitting region and disabled.')
		return None

	if (output_type=='update_uv_iron'): 
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n	* UV iron template outside of fitting region and disabled.')
		return None

	if (output_type=='update_balmer'):
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n	* Balmer continuum template outside of fitting region and disabled.')
		return None

	if (output_type=='output_line_list'):
		line_list, param_dict, soft_cons = output_val 
		with log_file_path.open(mode='a') as logfile:
			logfile.write("\n----------------------------------------------------------------------------------------------------------------------------------------")
			logfile.write("\n Line List:")
			nfree = 0 
			logfile.write("\n----------------------------------------------------------------------------------------------------------------------------------------")
			for line in sorted(list(line_list)):
				logfile.write("\n{0:<30}{1:<30}{2:<30.2}".format(line, '',''))
				for par in sorted(list(line_list[line])):
					logfile.write("\n{0:<30}{1:<30}{2:<30}".format('', par,str(line_list[line][par])))
					if line_list[line][par]=="free": nfree+=1
			logfile.write("\n----------------------------------------------------------------------------------------------------------------------------------------")
			logfile.write("\n Soft Constraints:\n")
			for con in soft_cons:
				logfile.write("\n{0:>30}{1:<0}{2:<0}".format(con[0], ' > ',con[1]))
			logfile.write("\n----------------------------------------------------------------------------------------------------------------------------------------")
		return None

	if (output_type=='no_line_test'):
		rdict = output_val
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n### No-Line Model Fitting Results ###')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Best-fit Value','+/- 1-sigma','Flag'))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
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
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	if (output_type=='line_test'):
		rdict = output_val
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n### Line Model Fitting Results ###')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Best-fit Value','+/- 1-sigma','Flag'))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
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
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None


	if (output_type=='line_test_stats'):
		(pval, pval_upp, pval_low, conf, conf_upp, conf_low, dist, disp, signif, overlap,
		f_conf,f_conf_err,f_stat,f_stat_err,f_pval,f_pval_err,
		chi2_ratio,chi2_ratio_err,chi2_no_line,chi2_no_line_err,chi2_line,chi2_line_err,
		# amp_metric,fwhm_metric,voff_metric,voff_metric_err,
		ssr_ratio,ssr_ratio_err,ssr_no_line,ssr_no_line_err,ssr_line,ssr_line_err,
		median_noise, median_noise_err, 
		total_resid_noise,total_resid_noise_err,resid_noise_no_line,resid_noise_no_line_err,resid_noise_line,resid_noise_line_err) = output_val
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			# logfile.write('-----------------------------------------------------------------------------------------------------')
			logfile.write('\n Line Test Statistics:')
			logfile.write('\n-----------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('','Statistic','Value','Uncertainty') )
			logfile.write('\n-----------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}'.format('A/B Likelihood Test::'))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30}'.format('','Confidence:',conf,"(-%0.6f,+%0.6f)" % (conf_low,conf_upp )) )
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30}'.format('','p-value:',pval,"(-%0.6f,+%0.6f)" % (pval_low,pval_upp)))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}'.format('','Statistical Distance:',dist))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}'.format('','Disperson:',disp))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}'.format('','Significance (sigma):',signif))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}'.format('','Overlap (1-sigma):',overlap))
			logfile.write('\n{0:<30}'.format('ANOVA (F-test):'))
			logfile.write('\n{0:<30}{1:<30}{2:<30.4f}{3:<30.4f}'.format('','Confidence:',f_conf, f_conf_err ) )
			logfile.write('\n{0:<30}{1:<30}{2:<30.4f}{3:<30.4f}'.format('','F-statistic:',f_stat,f_stat_err))
			logfile.write('\n{0:<30}{1:<30}{2:<30.4e}{3:<30.4e}'.format('','p-value:',f_pval,f_pval_err))
			logfile.write('\n{0:<30}'.format('Chi-Squared Metrics:'))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Chi-squared Ratio:',chi2_ratio, chi2_ratio_err ) )
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Chi-squared no-line:',chi2_no_line,chi2_no_line_err))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Chi-squared line:',chi2_line,chi2_line_err))
			logfile.write('\n{0:<30}'.format('Sum-of-Squares of Residuals (SSR):'))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','SSR ratio:',ssr_ratio,ssr_ratio_err))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','SSR no-line:',ssr_no_line,ssr_no_line_err))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','SSR line:',ssr_line,ssr_line_err))
			logfile.write('\n{0:<30}'.format('Residual Noise:'))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Median spec noise:',median_noise, median_noise_err))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','Total resid noise:',total_resid_noise,total_resid_noise_err)) 
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','no-line resid:',resid_noise_no_line,resid_noise_no_line_err))
			logfile.write('\n{0:<30}{1:<30}{2:<30.6f}{3:<30.6f}'.format('','line resid:',resid_noise_line,resid_noise_line_err))
			logfile.write('\n-----------------------------------------------------------------------------------------------------')
		return None


	# Maximum likelihood/Initial parameters
	if (output_type=='max_like_fit'):
		pdict,noise_std,resid_std = output_val
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n### Maximum Likelihood Fitting Results ###')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			if "std" in pdict[list(pdict.keys())[0]]:
				logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}'.format('Parameter','Max. Like. Value','+/- 1-sigma', 'Flag') )
			else:
				logfile.write('\n{0:<30}{1:<30}'.format('Parameter','Max. Like. Value') )
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			# Sort into arrays
			pname = []
			med   = []
			std   = []
			flag  = [] 
			for key in pdict:
				pname.append(key)
				med.append(pdict[key]['med'])
				if "std" in pdict[list(pdict.keys())[0]]:
					std.append(pdict[key]['std'])
					flag.append(pdict[key]['flag'])
			i_sort = np.argsort(pname)
			pname = np.array(pname)[i_sort] 
			med   = np.array(med)[i_sort] 
			if "std" in pdict[list(pdict.keys())[0]]:
				std   = np.array(std)[i_sort]   
				flag  = np.array(flag)[i_sort]  
			for i in range(0,len(pname),1):
				if "std" in pdict[list(pdict.keys())[0]]:
					logfile.write('\n{0:<30}{1:<30.4f}{2:<30.4f}{3:<30}'.format(pname[i], med[i], std[i], flag[i]))
				else:
					logfile.write('\n{0:<30}{1:<30.4f}'.format(pname[i], med[i]))
			logfile.write('\n{0:<30}{1:<30.4f}'.format('NOISE_STD.', noise_std ))
			logfile.write('\n{0:<30}{1:<30.4f}'.format('RESID_STD', resid_std ))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	# run_emcee
	if (output_type=='emcee_options'): # write user input emcee options
		ndim,nwalkers,auto_stop,conv_type,burn_in,write_iter,write_thresh,min_iter,max_iter = output_val
		# write_log((ndim,nwalkers,auto_stop,burn_in,write_iter,write_thresh,min_iter,max_iter),40)
		a = str(datetime.datetime.now())
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n### Emcee Options ###')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}'.format('ndim'		, ndim ))
			logfile.write('\n{0:<30}{1:<30}'.format('nwalkers'	, nwalkers ))
			logfile.write('\n{0:<30}{1:<30}'.format('auto_stop'   , str(auto_stop) ))
			logfile.write('\n{0:<30}{1:<30}'.format('user burn_in', burn_in ))
			logfile.write('\n{0:<30}{1:<30}'.format('write_iter'  , write_iter ))
			logfile.write('\n{0:<30}{1:<30}'.format('write_thresh', write_thresh ))
			logfile.write('\n{0:<30}{1:<30}'.format('min_iter'	, min_iter ))
			logfile.write('\n{0:<30}{1:<30}'.format('max_iter'	, max_iter ))
			logfile.write('\n{0:<30}{1:<30}'.format('start_time'  , a ))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	if (output_type=='autocorr_options'): # write user input auto_stop options
		min_samp,autocorr_tol,ncor_times,conv_type = output_val
		with log_file_path.open(mode='a') as logfile:
			# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
			logfile.write('\n')
			logfile.write('\n### Autocorrelation Options ###')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}'.format('min_samp'  , min_samp	 ))
			logfile.write('\n{0:<30}{1:<30}'.format('tolerance%', autocorr_tol ))
			logfile.write('\n{0:<30}{1:<30}'.format('ncor_times', ncor_times   ))
			logfile.write('\n{0:<30}{1:<30}'.format('conv_type' , str(conv_type)	))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	if (output_type=='autocorr_results'): # write autocorrelation results to log
		# write_log((k+1,burn_in,stop_iter,param_names,tau),42,run_dir)
		burn_in,stop_iter,param_names,tau,autocorr_tol,tol,ncor_times = output_val
		with log_file_path.open(mode='a') as logfile:
			# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
			i_sort = np.argsort(param_names)
			param_names = np.array(param_names)[i_sort]
			tau = np.array(tau)[i_sort]
			tol = np.array(tol)[i_sort]
			logfile.write('\n')
			logfile.write('\n### Autocorrelation Results ###')
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}'.format('conv iteration', burn_in   ))
			logfile.write('\n{0:<30}{1:<30}'.format('stop iteration', stop_iter ))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}'.format('Parameter','Autocorr. Time','Target Autocorr. Time','Tolerance','Converged?'))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
			for i in range(0,len(param_names),1):
				if (burn_in > (tau[i]*ncor_times)) and (0 < tol[i] < autocorr_tol):
					c = 'True'
				elif (burn_in < (tau[i]*ncor_times)) or (tol[i]>= 0.0):
					c = 'False'
				else: 
					c = 'False'
				logfile.write('\n{0:<30}{1:<30.5f}{2:<30.5f}{3:<30.5f}{4:<30}'.format(param_names[i],tau[i],(tau[i]*ncor_times),tol[i],c))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	if (output_type=='emcee_time'): # write autocorrelation results to log
		# write_log(run_time,43,run_dir)
		run_time = output_val
		a = str(datetime.datetime.now())
		with log_file_path.open(mode='a') as logfile:
			# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
			logfile.write('\n{0:<30}{1:<30}'.format('end_time',  a ))
			logfile.write('\n{0:<30}{1:<30}'.format('emcee_runtime',run_time ))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	if (output_type=='emcee_results'): # write best fit parameters results to log
		par_names,par_best,ci_68_low,ci_68_upp,ci_95_low,ci_95_upp,mean,std_dev,median,med_abs_dev,flags = output_val 
		# write_log((par_names,par_best,sig_low,sig_upp),50,run_dir)
		with log_file_path.open(mode='a') as logfile:
			logfile.write('\n')
			logfile.write('\n### Best-fit Parameters & Uncertainties ###')
			logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
			logfile.write('\n{0:<30}{1:<16}{2:<16}{3:<16}{4:<16}{5:<16}{6:<16}{7:<16}{8:<16}{9:<16}{10:<16}'.format('Parameter','Best-fit Value','68% CI low','68% CI upp','95% CI low','95% CI upp','Mean','Std. Dev.','Median','Med. Abs. Dev.','Flag'))
			logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
			for par in range(0,len(par_names),1):
				logfile.write('\n{0:<30}{1:<16.5f}{2:<16.5f}{3:<16.5f}{4:<16.5f}{5:<16.5f}{6:<16.5f}{7:<16.5f}{8:<16.5f}{9:<16.5f}{10:<16.5f}'.format(par_names[par],par_best[par],ci_68_low[par],ci_68_upp[par],ci_95_low[par],ci_95_upp[par],mean[par],std_dev[par],median[par],med_abs_dev[par],flags[par]))
			logfile.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
		return None

	# Total runtime
	if (output_type=='total_time'): # write total time to log
		# write_log(run_time,43,run_dir)
		tot_time = output_val
		a = str(datetime.datetime.now())
		with log_file_path.open(mode='a') as logfile:
			# write_log((min_samp,tol,ntol,atol,ncor_times,conv_type),41,run_dir)
			logfile.write('\n{0:<30}{1:<30}'.format('total_runtime',time_convert(tot_time) ))
			logfile.write('\n{0:<30}{1:<30}'.format('end_time',a ))
			logfile.write('\n-----------------------------------------------------------------------------------------------------------------')
		return None

	return None

##################################################################################
