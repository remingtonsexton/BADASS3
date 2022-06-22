#!/usr/bin/env python

# check_badass_args.py
"""
A set of functions that checks user input for BADASS and issues warnings 
if incorrect values are used.
"""
import sys
import glob
import numpy as np
from astropy.io import fits
from astroquery.irsa_dust import IrsaDust
import astropy.units as u
from astropy import coordinates

def check_dict(input,keyword_dict):
	
	output={} # output dictionary
	# Check each input keyword, and provide a default if not present
	for key in keyword_dict:
		if key in input:
			if all([f(input[key]) for f in keyword_dict[key]["conds"]]):
				output[key] = input[key]
			else:
				raise ValueError(keyword_dict[key]["error_message"])
	
		elif key not in input: # default value
			output[key] = keyword_dict[key]["default"]
	
	return output

#### Check Fit Options ###########################################################

def check_fit_options(input,comp_options,verbose=False):
	"""
	Checks the inputs of the fit_options dictionary and ensures that 
	all keywords have valid values. 

	fit_options={
		"fit_reg"	: (4400,5500),#(2000,3500), # Fitting region; Note: Indo-US Library=(3460,9464)
		"good_thresh": 0.0, # percentage of "good" pixels required in fig_reg for fit.
		"mask_bad_pix" : False, # mask pixels SDSS flagged as 'bad' (careful!)
		"mask_emline" : False, # mask emission lines for continuum fitting.
		"mask_metal": True, # interpolate over metal absorption lines for high-z spectra
		"n_basinhop": 5, # Number of consecutive basinhopping thresholds before solution achieved
		"test_outflows": False, # only test for outflows; stops after test
		"test_line": {"bool":False,
					  "line":"br_H_beta"},
		"max_like_niter": 1, # number of maximum likelihood iterations
		"output_pars": False, # only output free parameters of fit and stop code (diagnostic)
		# Rerun fitting on detected lines only and in final fit.
		}
	}

	"""
	# "" : {"conds":,
	# 	  "default":,
	# 	  "error_message"},

	output={} # output dictionary

	if not input:
		output={
				"fit_reg"    : (4400,5500),# Fitting region; Note: Indo-US Library=(3460,9464)
				"good_thresh": 0.0, # percentage of "good" pixels required in fig_reg for fit.
				"mask_bad_pix": False, # mask pixels SDSS flagged as 'bad' (careful!)
				"mask_emline" : False, # automatically mask lines for continuum fitting.
				"mask_metal": False, # interpolate over metal absorption lines for high-z spectra
				"fit_stat": "RMSE", # fit statistic; ML = Max. Like. , OLS = Ordinary Least Squares
				"n_basinhop": 10, # Number of consecutive basinhopping thresholds before solution achieved
				"test_outflows": False, # only test for outflows; "fit_outflows" must be set to True!
				"test_line": {"bool":False,
				              "line":"NA_OIII_5007"},
				"max_like_niter": 10, # number of maximum likelihood iterations
				"output_pars": False, # only output free parameters of fit and stop code (diagnostic)
				"cosmology": {"H0":70.0, "Om0": 0.30}, # Flat Lam-CDM Cosmology
				}
		return output
	
	keyword_dict ={
	"fit_reg" : {
				"conds": [
						   lambda x: isinstance(x,(tuple,list)),
						   lambda x: len(x)==2,
						   lambda x: isinstance(x[0],(int,float)),
						   lambda x: isinstance(x[1],(int,float)),
						   lambda x: x[1]>x[0],
						  ],
				"default": (4400.0,5500.0),
				"error_message": "\n Fitting region (fit_reg) must be an ordered tuple or list of the lower and upper wavelength limits (in Å) of the spectral region to be fit.\n"
				},
	"good_thresh" : {
					 "conds" : [
								 lambda x: isinstance(x,(int,float)),
								 lambda x: (x>=0 and x<=1)
							],
					 "default" : 0.0,
					 "error_message" : "\n Good pixel threshold (good_thresh) must be an int or float between 0 and 1.\n",
					},
	"mask_bad_pix" : {
					 "conds" : [
								 lambda x: isinstance(x,(bool))
								 ],
					 "default" : False,
					 "error_message" : "\n Mask SDSS-flagged bad pixels (mask_bad_pix) must be either True or False.\n",
					},
	"mask_emline" : {
					 "conds" : [
								 lambda x: isinstance(x,(bool))
								 ],
					 "default" : False,
					 "error_message" : "\n Masking of emission lines (mask_emline) must be either True or False.\n",
					},
	"mask_metal" : {
					 "conds" : [
								 lambda x: isinstance(x,(bool))
								 ],
					 "default" : False,
					 "error_message" : "\n Interpolation over metal absorption lines (mask_metal) must be either True or False.\n",
					},
	"fit_stat" : {
					 "conds" : [
								lambda x: isinstance(x,(str)),
								lambda x: x in ["ML","OLS","RCHI2","RMSE"]
							   ],
					 "default" : "ML",
					 "error_message" : "\n Fit statistic can be either ML (Maximum Likelihood), OLS (Ordinary Least Squares) , RCHI2 (Reduced Chi-Squared), or RMSE (Root-Mean Squared Error).\n",
					},
	"n_basinhop" : {
					 "conds" : [
								lambda x: isinstance(x,(int)),
								lambda x: (x>=0)
							   ],
					 "default" : 5,
					 "error_message" : "\n Number of consecutive successful basin-hopping iterations before maximum-likelihood fit is acheived must be an integer.\n",
					},
	"test_outflows" : {
					 "conds" : [
								 lambda x: isinstance(x,(bool))
								 ],
					 "default" : False,
					 "error_message" : "\n Testing for outflows (test_outflows) must be either True or False.\n",
					},
	"test_line" : {
					 "conds" : [
								 lambda x: isinstance(x,(dict))
								 ],
					 "default" : {"bool":False, "line":"br_H_beta"},
					 "error_message" : "\n Line test (test_line) must be a dict.\n",
					},
	"max_like_niter" : {
					 "conds" : [
								 lambda x: isinstance(x,(int)),
								 lambda x: (x>=0)
								 ],
					 "default" : 10,
					 "error_message" : "\n Number of bootstrap iterations used for initial maximum-likelihood fit must be an integer.\n",
					},
	"output_pars" : {
					 "conds" : [
								 lambda x: isinstance(x,(bool)),
								 ],
					 "default" : False,
					 "error_message" : "\n Only output parameters of fit (output_pars) must be True or False.\n",
					},
	"cosmology" : {
					 "conds" : [
								 lambda x: isinstance(x,(dict)),
								 ],
					 "default" : {"H0":70.0, "Om0": 0.30},
					 "error_message" : "\n Flat Lambda-CDM Cosmology must be a dict with H0 and Om0 specified.\n",
					},

	}
	output = check_dict(input,keyword_dict)

	# test_line_dict
	test_line_dict = {
	"bool" : 	{"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n test_line bool must be True or False.\n",},
	"line" : 	{"conds":[	lambda x: isinstance(x,(str,tuple,list))],
				"default": "br_H_beta",
				"error_message": "\n test_line line must be string and a valid line, or list of lines, from the line list.\n",},
		}
	output["test_line"] = check_dict(output["test_line"],test_line_dict)

	# cosmology_dict 
	cosmology_dict = {
	"H0" : 	{"conds":[	lambda x: isinstance(x,(int,float))],
				"default": 70.0,
				"error_message": "\n Hubble parameter must be an integer or float.\n",},
	"Om0" : 	{"conds":[	lambda x: isinstance(x,(float)),
								],
				"default": 0.30,
				"error_message": "\n Density parameter must be a float.\n",},
		}
	output["cosmology"] = check_dict(output["cosmology"],cosmology_dict)

			
	return output
	
################################################################################## 

#### Check MCMC options ##########################################################

def check_mcmc_options(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	mcmc_options={
	"mcmc_fit"	: True, # Perform robust fitting using emcee
	"nwalkers"	: 100,  # Number of emcee walkers; min = 2 x N_parameters
	"auto_stop"   : False, # Automatic stop using autocorrelation analysis
	"conv_type"   : "median", # "median", "mean", "all", or (tuple) of parameters
	"min_samp"	: 1500,  # min number of iterations for sampling post-convergence
	"ncor_times"  : 5.0,  # number of autocorrelation times for convergence
	"autocorr_tol": 10.0,  # percent tolerance between checking autocorr. times
	"write_iter"  : 100,   # write/check autocorrelation times interval
	"write_thresh": 100,   # when to start writing/checking parameters
	"burn_in"	 : 7500, # burn-in if max_iter is reached
	"min_iter"	: 1500, # min number of iterations before stopping
	"max_iter"	: 1500, # max number of MCMC iterations
	}
	"""
	output={} # output dictionary

	if not input:
		output={
		"mcmc_fit"	: False, # Perform robust fitting using emcee
		"nwalkers"	: 100,  # Number of emcee walkers; min = 2 x N_parameters
		"auto_stop"   : False, # Automatic stop using autocorrelation analysis
		"conv_type"   : "median", # "median", "mean", "all", or (tuple) of parameters
		"min_samp"	: 1500,  # min number of iterations for sampling post-convergence
		"ncor_times"  : 5.0,  # number of autocorrelation times for convergence
		"autocorr_tol": 10.0,  # percent tolerance between checking autocorr. times
		"write_iter"  : 100,   # write/check autocorrelation times interval
		"write_thresh": 100,   # when to start writing/checking parameters
		"burn_in"	 : 7500, # burn-in if max_iter is reached
		"min_iter"	: 1500, # min number of iterations before stopping
		"max_iter"	: 1500, # max number of MCMC iterations
		}
		return output

	# "" : {"conds":,
	# 	  "default":,
	# 	  "error_message"},

	keyword_dict ={
	"mcmc_fit" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": False,
				  "error_message": "\nToggle MCMC fitting (mcmc_fit) must be a bool.\n", 
				  },
	"nwalkers" : {"conds":[
							lambda x: isinstance(x,(int)),
							lambda x: x>0,
							],
				  "default": 100,
				  "error_message": "\n Number of MCMC walkers (nwalkers) must be an integer.\n",
				  },
	"auto_stop" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				   "default": False,
				   "error_message": "\n Automatic stop using autocorrelation (auto_stop) must be a bool.\n",
				   },
	"conv_type" : {"conds":[
							# lambda x: isinstance(x,(str,tuple)),
							lambda x: x in ["mean","median","all"] or isinstance(x,tuple),
							],
		  		   "default": "median",
		  		   "error_message": "\n Type of autocorrelation convergence (conv_type) must be a string.  Options are .'mean', 'median', 'all', or a tuple of valid free parameters.\n",
		  		   },
	"min_samp" : {"conds":[
							lambda x: isinstance(x,(int)),
							lambda x: x>0
							],
		  		  "default": 100,
		  		  "error_message": "\n Minimum number of sampling iterations (min_samp) before autocorrelation convergence must be an integer.\n",
		  		  },
	"ncor_times" : {"conds":[
							lambda x: isinstance(x,(int,float)),
							lambda x: x>=0
							],
					"default": 5.0,
					"error_message": "\n Number of autocorrelation times (ncor_times) required before convergence is reached must be an integer or float.\n",
					},
	"autocorr_tol" : {"conds":[
							lambda x: isinstance(x,(int,float)),
							lambda x: x>=0
							],
					  "default": 10.0,
					  "error_message": "\n Autocorrelation tolerance (autocorr_tol) required before convergence is reached must be an integer or float.\n",
					  },
	"write_iter" : {"conds":[
							lambda x: isinstance(x,(int)),
							lambda x: x>0,
							lambda x: x<=input["max_iter"]
							],
					"default": 100,
					"error_message": "\n Iteration frequency to write to chain (write_iter) must be an integer at least equal to max_iter.\n",
					},
	"write_thresh" : {"conds":[
							lambda x: isinstance(x,(int)),
							lambda x: x>0,
							lambda x: x<=input["max_iter"]
							],
					  "default": 100,
					  "error_message": "\n Iteration at which to begin writing (write_thresh) to chain must be an integer.\n",
					  },
	"burn_in" : {"conds":[ 
							lambda x: isinstance(x,(int)),
							lambda x: x>0,
							],
				 "default": 1000,
				 "error_message": "\n Burn-in iteration (burn_in) to use if maximum number of iterations (max_iter) is reached must be an integer.\n",
				 },
	"min_iter" : {"conds":[
							lambda x: isinstance(x,(int)),
							lambda x: x>0
							],
				  "default": 100,
				  "error_message": "\n Minimum number of iterations (min_iter) to perform before autocorrelation checking is performed.\n",
				  },
	"max_iter" : {"conds":[
							lambda x: isinstance(x,(int)),
							lambda x: x>0,
							lambda x: x>=input["min_iter"]
							],
				  "default": 1500,
				  "error_message": "\n Maximum number of iterations (max_iter) to perform before stopping.\n",
				  },
	}

	output = check_dict(input,keyword_dict)
			
	return output

##################################################################################

#### Check Component options #####################################################

def check_comp_options(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	comp_options={
			"fit_opt_feii"	 : True, # optical FeII
			"fit_uv_iron"	  : True, # UV Iron 
			"fit_balmer"	   : True, # Balmer continuum (<4000 A)
			"fit_losvd"		: False, # stellar LOSVD
			"fit_host"		 : True, # host template
			"fit_power"		: True, # AGN power-law
			"fit_narrow"	   : False, # narrow lines
			"fit_broad"		: False, # broad lines
			"fit_outflow"	  : False, # outflow lines
			"fit_absorp"	   : False, # absorption lines
			"tie_line_disp"	: False, # tie line widths
			"tie_line_voff"	: False, # tie line velocity offsets
			"na_line_profile"  : "G",	 # narrow line profile
			"br_line_profile"  : "G",	 # broad line profile
			"out_line_profile" : "G",	 # outflow line profile
			"abs_line_profile" : "G",	 # absorption line profile
	}

	Note: if fit_losvd and fit_host are both true, fit_losvd will override fit_host
	and set fit_host to false.
	"""

	if not input:
		output={
			"fit_opt_feii"	 : True, # optical FeII
			"fit_uv_iron"	  : True, # UV Iron 
			"fit_balmer"	   : True, # Balmer continuum (<4000 A)
			"fit_losvd"		: False, # stellar LOSVD
			"fit_host"		 : True, # host template
			"fit_power"		: True, # AGN power-law
			"fit_narrow"	   : False, # narrow lines
			"fit_broad"		: False, # broad lines
			"fit_outflow"	  : False, # outflow lines
			"fit_absorp"	   : False, # absorption lines
			"tie_line_disp"	: False, # tie line widths
			"tie_line_voff"	: False, # tie line velocity offsets
			"na_line_profile"  : "G",	 # narrow line profile
			"br_line_profile"  : "G",	 # broad line profile
			"out_line_profile" : "G",	 # outflow line profile
			"abs_line_profile" : "G",	 # absorption line profile
			"n_moments"		   : 4, # number of higher-order moments for GH line profiles
		}
		return output

	keyword_dict ={
	"fit_opt_feii" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  		"default": True,
				  		"error_message": "\n fit_opt_feii must be a bool.\n",
				  },
	"fit_uv_iron" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  	"default": False,
				  	"error_message": "\n fit_uv_iron must be a bool.\n",
				  },
	"fit_balmer" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  	"default": False,
				  	"error_message": "\n fit_balmer must be a bool.\n",
				  },
	"fit_losvd" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  	"default": False,
				  	"error_message": "\n fit_losvd must be a bool.\n",
				  },
	"fit_host" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": True,
				  "error_message": "\n fit_host must be a bool.\n",
				  },
	"fit_power" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": True,
				  "error_message": "\n fit_power must be a bool.\n",
				  },
	"fit_poly" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": False,
				  "error_message": "\n fit_poly must be a bool.\n",
				  },
	"fit_narrow" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": True,
				  "error_message": "\n fit_narrow must be a bool.\n",
				  },
	"fit_broad" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": True,
				  "error_message": "\n fit_broad must be a bool.\n",
				  },
	"fit_outflow" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": True,
				  "error_message": "\n fit_outflow must be a bool.\n",
				  },
	"fit_absorp" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": True,
				  "error_message": "\n fit_absorp must be a bool.\n",
				  },
	"tie_line_disp" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": False,
				  "error_message": "\n tie_line_disp must be a bool.\n",
				  },
	"tie_line_voff" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  "default": False,
				  "error_message": "\n tie_line_voff must be a bool.\n",
				  },
	"na_line_profile" : {"conds":[
							lambda x: isinstance(x,(str)),
							lambda x: x in ["G","L","GH","V"]
							],
				  "default": "G",
				  "error_message": "\n na_line_profile must be a bool.\n",
				  },
	"br_line_profile" : {"conds":[
							lambda x: isinstance(x,(str)),
							lambda x: x in ["G","L","GH","V"]
							],
				  "default": "G",
				  "error_message": "\n br_line_profile must be a bool.\n",
				  },
	"out_line_profile" : {"conds":[
							lambda x: isinstance(x,(str)),
							lambda x: x in ["G","L","GH","V"]
							],
				  "default": "G",
				  "error_message": "\n out_line_profile must be a bool.\n",
				  },
	"abs_line_profile" : {"conds":[
							lambda x: isinstance(x,(str)),
							lambda x: x in ["G","L","GH","V"]
							],
				  "default": "G",
				  "error_message": "\n abs_line_profile must be a bool.\n",
				  },
	"n_moments" : {"conds" : [
								 lambda x: isinstance(x,(int)),
								 lambda x: (x>=2) & (x<=10)
								 ],
					 "default" : 4,
					 "error_message" : "\n Higher-order moments for Gauss-Hermite line profiles must be >=2 or <= 10.\n",
					},
	}
	
	output = check_dict(input,keyword_dict)

	# Check to see if fit_losvd and fit_host are both True.  If both True,
	# issue a warning, then fit_losvd will override fit_host and set fit_host=False.
	if (output["fit_losvd"]==True) & (output["fit_host"]==True):
		output["fit_host"]=False
		if (verbose==True):
			print("\n Warning: fit_losvd and fit_host both True.  fit_losvd will override fit_host and set fit_host=False.\n")
		

	return output

##################################################################################

#### User Emission Lines #######################################################

def check_user_lines(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	power_options={
		"type" : "simple" # alternatively, "broken" for smoothly-broken power-law
	}
	"""
	if not input:
		output={}
		return output

	output = {}
	# Check user_lines dict to make sure each entry is a dictionary
	for line in input:
		if not isinstance(input[line],(dict)):
			raise ValueError("\n Each user-input emission line must be a dictionary that includes the central wavelength (in Å) of the line to be fit.\n")
		else: 
			output[line] = input[line]
			
	return output

##################################################################################

#### User Constraints ############################################################

def check_user_constraints(input,verbose=False):
	"""
	Checks user-defined soft constraints by ensuring that input is a list
	of strings
	"""
	output = []

	if not input:
		ouput = []
		return output
	if (not isinstance(input,(list))):
		raise ValueError("\n User constraints must be in the form of a list of tuples, each of length 2.  The format for soft constraints (paramter1, parameter2) defined as (parameter1 - parameter2) >= 0.0 OR (parameter1 >= parameter2). \n")

	for con in input:
		if (not isinstance(con,(list,tuple))) or (len(con)!=2) or (not isinstance(con[0],(str))) or (not isinstance(con[1],(str))):
			raise ValueError("\n User constraints must be in the form of a list of tuples, each of length 2.  The format for soft constraints (paramter1, parameter2) defined as (parameter1 - parameter2) >= 0.0 OR (parameter1 >= parameter2). \n")
		else:
			output.append(con)

	return output

##################################################################################

#### User Constraints ############################################################

def check_user_mask(input,verbose=False):
	"""
	Checks user-defined soft constraints by ensuring that input is a list
	of strings
	"""
	output = []

	if not input:
		ouput = []
		return output
	if (not isinstance(input,(list,tuple))):
		raise ValueError("\n User mask must be in the form of a list of tuples, each of length 2. \n")

	for con in input:
		if (not isinstance(con,(list,tuple))) or (len(con)!=2) or (not isinstance(con[0],(int,float))) or (not isinstance(con[1],(int,float))) or (con[1]<con[0]):
			raise ValueError("\n User mask must be in the form of a list of tuples, each of length 2. \n")
		else:
			output.append(con)

	return output

##################################################################################

#### AGN power-law options #######################################################

def check_power_options(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	power_options={
		"type" : "simple" # alternatively, "broken" for smoothly-broken power-law
	}
	"""
	if not input:
		output={
			"type" : "simple" # alternatively, "broken" for smoothly-broken power-law
		}
		return output

	keyword_dict ={
	"type" : {"conds":[
							lambda x: isinstance(x,(str)),
							lambda x: x in ["simple","broken"]
							],
				  		"default": "simple",
				  		"error_message": "\n Power-law type must be a string and options are 'simple' or 'broken'.\n",
				  },
	}
	output = check_dict(input,keyword_dict)

	return output

##########################################################################################

#### Check LOSVD Fitting (pPXF) Options ##################################################

def check_losvd_options(input,verbose=False):
	"""
	Checks the inputs of the losvd_options dictionary and ensures that 
	all keywords have valid values. 

	losvd_options = {
					"library"	   : "IndoUS", # Options: IndoUS, Vazdekis2010, eMILES
					"vel_const" : {"bool":True, "val":0.0},
					"disp_const": {"bool":True, "val":100.0}
					}
	"""
	output={} # output dictionary

	# If feii_options not specified
	if not input:
		output = {
					"library"	   : "IndoUS", # Options: IndoUS, Vazdekis2010, eMILES
					"vel_const" : {"bool":False, "val":0.0},
					"disp_const": {"bool":False, "val":100.0},
					}
		return output

	# for dictionaries within dictionaries, we check the outer-most level first to ensure
	# they are correct dictionaries.
	keyword_dict={
	"library" 			: {"conds":[
								lambda x: isinstance(x,(str)),
								lambda x: x in ["IndoUS","Vazdekis2010","eMILES"]
								],
		  				"default": "IndoUS",
		  				"error_message": "\n Stellar template library must be a string.  Available options: 'IndoUS', 'Vazdekis2010', 'eMILES'. \n"},
	"vel_const" 	: {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False, "val":0.0},
		  				"error_message": "\n vel_const must be a dictionary.\n"},
	"disp_const" 	: {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False, "val":100.0},
		  				"error_message": "\n disp_const must be a dictionary.\n"},
	}
	output = check_dict(input,keyword_dict)
	# now we check to ensure the inner dictionaries are correct.  We do this by defining a sub-keyword dictionary for each 
	# main keyword. 
	#
	# vel_const_dict
	vel_const_dict = {
	"bool" : 	{"conds":[	lambda x: isinstance(x,(bool))],
				"default": True,
				"error_message": "\n .vel_const bool must be True or False.\n",},
	"val" : 	{"conds":[	lambda x: isinstance(x,(int,float)),
							lambda x: x>=0
								],
				"default": 0.0,
				"error_message": "\n vel_const val must be an integer or float.\n",},
		}
	output["vel_const"] = check_dict(output["vel_const"],vel_const_dict)
	# disp_const_dict
	disp_const_dict = {
	"bool" : 	{"conds":[	lambda x: isinstance(x,(bool))],
				"default": True,
				"error_message": "\n .disp_const bool must be True or False.\n",},
	"val" :		{"conds":[	lambda x: isinstance(x,(int,float)),
							lambda x: x>=0
								],
				"default": 100.0,
				"error_message": "\n disp_const val must be an integer or float.\n",},
		}
	output["disp_const"] = check_dict(output["disp_const"],disp_const_dict)

	if output["disp_const"]["val"]<=0: # convolution error if dispersion equal to zero
			output["disp_const"]["val"]=1.e-3

	return output


##########################################################################################

#### Check Polynomial Continuum Options ##################################################

def check_poly_options(input,verbose=False):
	"""
	Checks the inputs of the poly_options dictionary and ensures that 
	all keywords have valid values. 

	poly_options = {
					"ppoly" : {"bool": True,  "order": 3}, # positive definite additive polynomial 
					"apoly" : {"bool": False, "order": 3}, # Legendre additive polynomial 
					"mpoly" : {"bool": False, "order": 3}, # Legendre multiplicative polynomial 
					}
	"""
	output={} # output dictionary

	# If feii_options not specified
	if not input:
		output = {
				  "ppoly" : {"bool": True,  "order": 3}, # positive definite additive polynomial 
				  "apoly" : {"bool": False, "order": 3}, # Legendre additive polynomial 
				  "mpoly" : {"bool": False, "order": 3}, # Legendre multiplicative polynomial 
				  }
		return output

	# for dictionaries within dictionaries, we check the outer-most level first to ensure
	# they are correct dictionaries.
	keyword_dict={
	"ppoly" 		 : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False, "order":3.0},
		  				"error_message": "\n ppoly must be a dictionary.\n"},
	"apoly" 		 : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False, "order":3.0},
		  				"error_message": "\n apoly must be a dictionary.\n"},
	"mpoly" 		 : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False, "order":3.0},
		  				"error_message": "\n mpoly must be a dictionary.\n"},
	}
	output = check_dict(input,keyword_dict)
	# now we check to ensure the inner dictionaries are correct.  We do this by defining a sub-keyword dictionary for each 
	# main keyword. 
	#
	# ppoly_dict
	ppoly_dict = {
	"bool" : 	{"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n ppoly bool must be True or False.\n",},
	"order" :		{"conds":[	lambda x: isinstance(x,(int)),
								lambda x: (x>=0) & (x<=99)
								],
				"default": 3.0,
				"error_message": "\n ppoly order must be an integer of 0<=x<=99.\n",},
		}
	output["ppoly"] = check_dict(output["ppoly"],ppoly_dict)
	# apoly_dict
	apoly_dict = {
	"bool" : 	{"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n apoly bool must be True or False.\n",},
	"order" :		{"conds":[	lambda x: isinstance(x,(int)),
								lambda x: (x>=0) & (x<=99)
								],
				"default": 3.0,
				"error_message": "\n apoly order must be an integer of 0<=x<=99.\n",},
		}
	output["apoly"] = check_dict(output["apoly"],apoly_dict)

	# mpoly_dict
	mpoly_dict = {
	"bool" : 	{"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n mpoly bool must be True or False.\n",},
	"order" :		{"conds":[	lambda x: isinstance(x,(int)),
								lambda x: (x>=0) & (x<=99)
								],
				"default": 3.0,
				"error_message": "\n mpoly order must be an integer of 0<=x<=99.\n",},
		}
	output["mpoly"] = check_dict(output["mpoly"],mpoly_dict)

	return output


##########################################################################################

#### Check Host Galaxy Template Options ##################################################

def check_host_options(input,verbose=False):
	"""
	Checks the inputs of the host_options dictionary and ensures that 
	all keywords have valid values. 

	host_options = {
					"age"	   : 10, # Gyr; [0.09 Gyr - 14 Gyr] 
					"vel_const" : {"bool":True, "val":0.0},
					"disp_const": {"bool":True, "val":100.0}
					}
	"""
	output={} # output dictionary

	# If feii_options not specified
	if not input:
		output = {
					"age"	   : [0.1,1.0,10.0], # Gyr; [0.09 Gyr - 14 Gyr] 
					"vel_const" : {"bool":True, "val":0.0},
					"disp_const": {"bool":True, "val":100.0}
					}
		return output

	# for dictionaries within dictionaries, we check the outer-most level first to ensure
	# they are correct dictionaries.
	keyword_dict={
	"age" 			: {"conds":[
								lambda x: isinstance(x,(list,tuple)),
								lambda x: np.all([x in [0.9,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0] for x in x])
								],
		  				"default": [0.1,1.0,10.0],
		  				"error_message": "\n SSP host template age must by an integer or float in the range 0.9-14.0.  Available options are [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]. \n"},
	"vel_const" 	: {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":True, "val":0.0},
		  				"error_message": "\n vel_const must be a dictionary.\n"},
	"disp_const" 	: {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":True, "val":100.0},
		  				"error_message": "\n disp_const must be a dictionary.\n"},
	}
	output = check_dict(input,keyword_dict)
	# now we check to ensure the inner dictionaries are correct.  We do this by defining a sub-keyword dictionary for each 
	# main keyword. 
	#
	# vel_const_dict
	vel_const_dict = {
	"bool" : 	{"conds":[	lambda x: isinstance(x,(bool))],
				"default": True,
				"error_message": "\n .vel_const bool must be True or False.\n",},
	"val" : 	{"conds":[	lambda x: isinstance(x,(int,float)),
							lambda x: x>=0
								],
				"default": 0.0,
				"error_message": "\n vel_const val must be an integer or float.\n",},
		}
	output["vel_const"] = check_dict(output["vel_const"],vel_const_dict)
	# disp_const_dict
	disp_const_dict = {
	"bool" : 	{"conds":[	lambda x: isinstance(x,(bool))],
				"default": True,
				"error_message": "\n .disp_const bool must be True or False.\n",},
	"val" :		{"conds":[	lambda x: isinstance(x,(int,float)),
							lambda x: x>=0
								],
				"default": 100.0,
				"error_message": "\n disp_const val must be an integer or float.\n",},
		}
	output["disp_const"] = check_dict(output["disp_const"],disp_const_dict)

	if output["disp_const"]["val"]<=0: # convolution error if dispersion equal to zero
			output["disp_const"]["val"]=1.e-3

	return output


##########################################################################################

#### Check Optical FeII options ##########################################################

def check_opt_feii_options(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	opt_feii_options={
			"opt_template"  :{"type":"VC04"}, 
			"opt_amp_const" :{"bool":False,"br_opt_feii_val":1.0   ,"na_opt_feii_val":1.0},
			"opt_disp_const":{"bool":True ,"br_opt_feii_val":3000.0,"na_opt_feii_val":500.0},
			"opt_voff_const":{"bool":True ,"br_opt_feii_val":0.0   ,"na_opt_feii_val":0.0},
	}
	
	"""
	output={} # output dictionary

	if not input:
		output={
				"opt_template"  :{"type":"VC04"}, 
				"opt_amp_const" :{"bool":False,"br_opt_feii_val":1.0   ,"na_opt_feii_val":1.0},
				"opt_disp_const":{"bool":True ,"br_opt_feii_val":3000.0,"na_opt_feii_val":500.0},
				"opt_voff_const":{"bool":True ,"br_opt_feii_val":0.0   ,"na_opt_feii_val":0.0},
				}
		return output


	if input["opt_template"]["type"]=="VC04":
		# If feii_options not specified
		if not input:
			output={
				"opt_template"  :{"type":"VC04"}, 
				"opt_amp_const" :{"bool":False,"br_opt_feii_val":1.0   ,"na_opt_feii_val":1.0},
				"opt_disp_const":{"bool":True ,"br_opt_feii_val":3000.0,"na_opt_feii_val":500.0},
				"opt_voff_const":{"bool":True ,"br_opt_feii_val":0.0   ,"na_opt_feii_val":0.0},
					}
			return output

		# for dictionaries within dictionaries, we check the outer-most level first to ensure
		# they are correct dictionaries.
		keyword_dict={
		"opt_template" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"type":"VC04"},
			  				"error_message": "\n opt_template must be a dictionary.\n"},
		"opt_amp_const" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"bool":False,"br_opt_feii_val":1.0   ,"na_opt_feii_val":1.0},
			  				"error_message": "\n opt_amp_const must be a dictionary.\n"},
		"opt_disp_const" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"bool":True ,"br_opt_feii_val":3000.0,"na_opt_feii_val":500.0},
			  				"error_message": "\n opt_disp_const must be a dictionary.\n"},
		"opt_voff_const" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"bool":True ,"br_opt_feii_val":0.0   ,"na_opt_feii_val":0.0},
			  				"error_message": "\n opt_voff_const must be a dictionary.\n"},
		}
		output = check_dict(input,keyword_dict)
		# now we check to ensure the inner dictionaries are correct.  We do this by defining a sub-keyword dictionary for each 
		# main keyword. 
		#
		# opt_template
		opt_template_dict = {
		"type" : {"conds":[	lambda x: isinstance(x,(str)),
							lambda x: x in ["VC04","K10"]],
					"default": "VC04",
					"error_message": "\n Optical FeII template type (opt_template type) must be a string and either 'VC04' or 'K10'.\n",}
			}
		output["opt_template"] = check_dict(output["opt_template"],opt_template_dict)
		# opt_amp_const
		opt_amp_const_dict = {
		"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
					"default": False,
					"error_message": "\n .opt_amp_const bool must be True or False.\n",},
		"br_opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
										lambda x: x>0
									],
					"default": 1.0,
					"error_message": "\n opt_amp_const br_opt_feii_val must be an integer or float.\n",},
		"na_opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
										lambda x: x>0
									],
					"default": 1.0,
					"error_message": "\n opt_amp_const na_opt_feii_val must be an integer or float.\n",}
			}
		output["opt_amp_const"] = check_dict(output["opt_amp_const"],opt_amp_const_dict)

		# opt_disp_const
		opt_disp_const_dict = {
		"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
					"default": True,
					"error_message": "\n .opt_disp_const bool must be True or False.\n",},
		"br_opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
										lambda x: x>0
									],
					"default": 3000.0,
					"error_message": "\n opt_disp_const br_opt_feii_val must be an integer or float.\n",},
		"na_opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
										lambda x: x>0
									],
					"default": 500.0,
					"error_message": "\n opt_disp_const na_opt_feii_val must be an integer or float.\n",}
			}
		output["opt_disp_const"] = check_dict(output["opt_disp_const"],opt_disp_const_dict)


		# opt_voff_const
		opt_voff_const_dict = {
		"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
					"default": True,
					"error_message": "\n .opt_voff_const bool must be True or False.\n",},
		"br_opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float))],
					"default": 0.0,
					"error_message": "\n opt_voff_const br_opt_feii_val must be an integer or float.\n",},
		"na_opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float))],
					"default": 0.0,
					"error_message": "\n opt_voff_const na_opt_feii_val must be an integer or float.\n",}
			}
		output["opt_voff_const"] = check_dict(output["opt_voff_const"],opt_voff_const_dict)

		return output

	if input["opt_template"]["type"]=="K10":
		# If feii_options not specified
		if not input:
			output={
					"opt_template"  :{"type":"K10"},
					"opt_amp_const" :{"bool":False,"f_feii_val":1.0,"s_feii_val":1.0,"g_feii_val":1.0,"z_feii_val":1.0},
					"opt_disp_const":{"bool":False,"opt_feii_val":1500.0},
					"opt_voff_const":{"bool":False,"opt_feii_val":0.0},
					"opt_temp_const":{"bool":True,"opt_feii_val":10000.0},
					}

			return output

		# for dictionaries within dictionaries, we check the outer-most level first to ensure
		# they are correct dictionaries.
		keyword_dict={
		"opt_template" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"type":"K10"},
			  				"error_message": "\n opt_template must be a dictionary.\n"},
		"opt_amp_const" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"bool":False,"f_feii_val":1.0,"s_feii_val":1.0,"g_feii_val":1.0,"z_feii_val":1.0},
			  				"error_message": "\n opt_amp_const must be a dictionary.\n"},
		"opt_disp_const" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"bool":False,"opt_feii_val":1500.0},
			  				"error_message": "\n opt_disp_const must be a dictionary.\n"},
		"opt_voff_const" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"bool":False,"opt_feii_val":0.0},
			  				"error_message": "\n opt_voff_const must be a dictionary.\n"},
		"opt_temp_const" : {"conds":[
									lambda x: isinstance(x,(dict))
									],
			  				"default": {"bool":True,"opt_feii_val":10000.0},
			  				"error_message": "\n opt_temp_const must be a dictionary.\n"},
		}
		output = check_dict(input,keyword_dict)
		# now we check to ensure the inner dictionaries are correct.  We do this by defining a sub-keyword dictionary for each 
		# main keyword. 
		#
		# opt_template
		opt_template_dict = {
		"type" : {"conds":[	lambda x: isinstance(x,(str)),
							lambda x: x in ["VC04","K10"]],
					"default": "K10",
					"error_message": "\n Optical FeII template type (opt_template type) must be a string and either 'VC04' or 'K10'.\n",}
			}
		output["opt_template"] = check_dict(output["opt_template"],opt_template_dict)
		# opt_amp_const
		opt_amp_const_dict = {
		"bool" : 		{"conds":[	lambda x: isinstance(x,(bool))],
					"default": False,
					"error_message": "\n .opt_amp_const bool must be True or False.\n",},
		"f_feii_val" : 	{"conds":[	lambda x: isinstance(x,(int,float)),
									lambda x: x>=0
								 ],
					"default": 1.0,
					"error_message": "\n opt_amp_const f_feii_val must be an integer or float.\n",},
		"s_feii_val" : 	{"conds":[	lambda x: isinstance(x,(int,float)),
									lambda x: x>=0
								 ],
					"default": 1.0,
					"error_message": "\n opt_amp_const s_feii_val must be an integer or float.\n",},
		"g_feii_val" : 	{"conds":[	lambda x: isinstance(x,(int,float)),
									lambda x: x>=0
								 ],
					"default": 1.0,
					"error_message": "\n opt_amp_const g_feii_val must be an integer or float.\n",},
		"z_feii_val" : 	{"conds":[	lambda x: isinstance(x,(int,float)),
									lambda x: x>=0
								 ],
					"default": 1.0,
					"error_message": "\n opt_amp_const z_feii_val must be an integer or float.\n",},
			}
		output["opt_amp_const"] = check_dict(output["opt_amp_const"],opt_amp_const_dict)

		# opt_disp_const
		opt_disp_const_dict = {
		"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
					"default": False,
					"error_message": "\n opt_disp_const bool must be True or False.\n",},
		"opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
										lambda x: x>0
									],
					"default": 1500.0,
					"error_message": "\n opt_disp_const opt_feii_val must be an integer or float.\n",},
			}
		output["opt_disp_const"] = check_dict(output["opt_disp_const"],opt_disp_const_dict)

		# opt_voff_const
		opt_voff_const_dict = {
		"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
					"default": False,
					"error_message": "\n .opt_voff_const bool must be True or False.\n",},
		"opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float))],
					"default": 0.0,
					"error_message": "\n opt_voff_const opt_feii_val must be an integer or float.\n",},
			}
		output["opt_voff_const"] = check_dict(output["opt_voff_const"],opt_voff_const_dict)

		# opt_temp_const
		opt_temp_const_dict = {
		"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
					"default": True,
					"error_message": "\n .opt_temp_const bool must be True or False.\n",},
		"opt_feii_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
									lambda x: x>=0
									],
					"default": 10000.0,
					"error_message": "\n opt_temp_const opt_feii_val must be an integer or float.\n",},
			}
		output["opt_temp_const"] = check_dict(output["opt_temp_const"],opt_temp_const_dict)

		return output



####################################################################################

#### Check UV Iron options ##########################################################

def check_uv_iron_options(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	uv_iron_options={
			"uv_amp_const"  :{"bool":False,"uv_iron_val":1.0},
			"uv_disp_const" :{"bool":False ,"uv_iron_val":3000.0},
			"uv_voff_const" :{"bool":True ,"uv_iron_val":0.0},
			"uv_legendre_m" :{"bool":False , "uv_iron_val":3},
	}
	
	"""
	output={} # output dictionary

	# If feii_options not specified
	if not input:
		output={
			"uv_amp_const"  :{"bool":False,"uv_iron_val":1.0},
			"uv_disp_const" :{"bool":False ,"uv_iron_val":3000.0},
			"uv_voff_const" :{"bool":True ,"uv_iron_val":0.0},
				}
		return output

	# for dictionaries within dictionaries, we check the outer-most level first to ensure
	# they are correct dictionaries.
	keyword_dict={
	"uv_amp_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False,"uv_iron_val":1.0},
		  				"error_message": "\n uv_amp_const must be a dictionary.\n"},
	"uv_disp_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False ,"uv_iron_val":3000.0},
		  				"error_message": "\n uv_disp_const must be a dictionary.\n"},
	"uv_voff_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":True ,"uv_iron_val":0.0},
		  				"error_message": "\n uv_voff_const must be a dictionary.\n"},
	
	}
	output = check_dict(input,keyword_dict)
	# now we check to ensure the inner dictionaries are correct.  We do this by defining a sub-keyword dictionary for each 
	# main keyword. 
	#
	# uv_amp_const
	uv_amp_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n .uv_amp_const bool must be True or False.\n",},
	"uv_iron_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
								lambda x: x>=0
							],
				"default": 1.0,
				"error_message": "\n uv_amp_const uv_iron_val must be an integer or float.\n",}
		}
	output["uv_amp_const"] = check_dict(output["uv_amp_const"],uv_amp_const_dict)


	# uv_disp_const
	uv_disp_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n uv_disp_const bool must be True or False.\n",},
	"uv_iron_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
								lambda x: x>0
								],
				"default": 3000.0,
				"error_message": "\n uv_disp_const uv_iron_val must be an integer or float.\n",}
		}
	output["uv_disp_const"] = check_dict(output["uv_disp_const"],uv_disp_const_dict)


	# uv_voff_const
	uv_voff_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n uv_voff_const bool must be True or False.\n",},
	"uv_iron_val" : {"conds":[	lambda x: isinstance(x,(int,float))],
				"default": 0.0,
				"error_message": "\n uv_voff_const uv_iron_val must be an integer or float.\n",}
		}
	output["uv_voff_const"] = check_dict(output["uv_voff_const"],uv_voff_const_dict)

	return output

####################################################################################


#### Check Balmer options ##########################################################

def check_balmer_options(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	balmer_options = {
			"R_const"		  :{"bool":False, "R_val":0.5}, # ratio between balmer continuum and higher-order balmer lines
			"balmer_amp_const" :{"bool":False, "balmer_amp_val":1.0}, # amplitude of overall balmer model (continuum + higher-order lines)
			"balmer_disp_const":{"bool":True,  "balmer_disp_val":5000.0}, # broadening of higher-order Balmer lines
			"balmer_voff_const":{"bool":True,  "balmer_voff_val":0.0}, # velocity offset of higher-order Balmer lines
			"Teff_const"	   :{"bool":True,  "Teff_val":15000.0}, # effective temperature
			"tau_const"		:{"bool":True,  "tau_val":1.0}, # optical depth
	}

	"""
	# If feii_options not specified
	if not input:
		output = {
			"R_const"		  :{"bool":False, "R_val":0.5}, # ratio between balmer continuum and higher-order balmer lines
			"balmer_amp_const" :{"bool":False, "balmer_amp_val":1.0}, # amplitude of overall balmer model (continuum + higher-order lines)
			"balmer_disp_const":{"bool":True,  "balmer_disp_val":5000.0}, # broadening of higher-order Balmer lines
			"balmer_voff_const":{"bool":True,  "balmer_voff_val":0.0}, # velocity offset of higher-order Balmer lines
			"Teff_const"	   :{"bool":True,  "Teff_val":15000.0}, # effective temperature
			"tau_const"		:{"bool":True,  "tau_val":1.0}, # optical depth
		}
		return output

	keyword_dict={
	"R_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False, "R_val":0.5},
		  				"error_message": "\n R_const must be a dictionary.\n"},
	"balmer_amp_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":False, "balmer_amp_val":1.0},
		  				"error_message": "\n balmer_amp_const must be a dictionary.\n"},
	"balmer_disp_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":True,  "balmer_disp_val":5000.0},
		  				"error_message": "\n balmer_disp_const must be a dictionary.\n"},
	"balmer_voff_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":True,  "balmer_voff_val":0.0},
		  				"error_message": "\n balmer_voff_const must be a dictionary.\n"},
	"Teff_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":True,  "Teff_val":15000.0}, 
		  				"error_message": "\n Teff_const must be a dictionary.\n"},
	"tau_const" : {"conds":[
								lambda x: isinstance(x,(dict))
								],
		  				"default": {"bool":True,  "tau_val":1.0},
		  				"error_message": "\n tau_const must be a dictionary.\n"},
	
	}
	output = check_dict(input,keyword_dict)
	
	# R_const
	R_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n R_const bool must be True or False.\n",},
	"R_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
							lambda x: x>0
						],
				"default": 0.5,
				"error_message": "\n R_const R_val must be an integer or float.\n",},
		}
	output["R_const"] = check_dict(output["R_const"],R_const_dict)


	# balmer_amp_const
	balmer_amp_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": False,
				"error_message": "\n balmer_amp_const bool must be True or False.\n",},
	"balmer_amp_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
									lambda x: x>0
						],
				"default": 1.0,
				"error_message": "\n balmer_amp_const balmer_amp_val must be an integer or float.\n",},
		}
	output["balmer_amp_const"] = check_dict(output["balmer_amp_const"],balmer_amp_const_dict)

	# balmer_disp_const
	balmer_disp_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": True,
				"error_message": "\n balmer_disp_const bool must be True or False.\n",},
	"balmer_disp_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
									lambda x: x>0
						],
				"default": 5000.0,
				"error_message": "\n balmer_disp_const balmer_disp_val must be an integer or float.\n",},
		}
	output["balmer_disp_const"] = check_dict(output["balmer_disp_const"],balmer_disp_const_dict)

	# balmer_voff_const
	balmer_voff_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": True,
				"error_message": "\n balmer_voff_const bool must be True or False.\n",},
	"balmer_voff_val" : {"conds":[	lambda x: isinstance(x,(int,float))],
				"default": 0.0,
				"error_message": "\n balmer_voff_const balmer_voff_val must be an integer or float.\n",},
		}
	output["balmer_voff_const"] = check_dict(output["balmer_voff_const"],balmer_voff_const_dict)

	# Teff_const
	Teff_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": True,
				"error_message": "\n Teff_const bool must be True or False.\n",},
	"Teff_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
							lambda x: x>0
							],
				"default": 15000.0,
				"error_message": "\n Teff_const Teff_val must be an integer or float.\n",},
		}
	output["Teff_const"] = check_dict(output["Teff_const"],Teff_const_dict)

	# tau_const
	tau_const_dict = {
	"bool" : {"conds":[	lambda x: isinstance(x,(bool))],
				"default": True,
				"error_message": "\n tau_const bool must be True or False.\n",},
	"tau_val" : {"conds":[	lambda x: isinstance(x,(int,float)),
							lambda x: x>=0 and x<=1
							],
				"default": 1.0,
				"error_message": "\n tau_const tau_val must be an integer or float.\n",},
		}
	output["tau_const"] = check_dict(output["tau_const"],tau_const_dict)

	return output

##################################################################################

#### Check Plot options ##########################################################

def check_plot_options(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	plot_options={
			"plot_param_hist"	: True,# Plot MCMC histograms and chains for each parameter
			"plot_flux_hist"	 : True,# Plot MCMC hist. and chains for component fluxes
			"plot_lum_hist"	  : True,# Plot MCMC hist. and chains for component luminosities
			"plot_eqwidth_hist"  : True, # Plot MCMC hist. and chains for equivalent widths 
	}
	"""
	output={} # output dictionary

	if not input:
		output={
			"plot_param_hist"	: True,# Plot MCMC histograms and chains for each parameter
			"plot_flux_hist"	 : False,# Plot MCMC hist. and chains for component fluxes
			"plot_lum_hist"	  : False,# Plot MCMC hist. and chains for component luminosities
			"plot_eqwidth_hist"  : False, # Plot MCMC hist. and chains for equivalent widths 
			"plot_HTML"          : False,# make interactive plotly HTML best-fit plot
		}

		return output

	keyword_dict ={
	"plot_param_hist" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  		"default": True,
				  		"error_message": "\n plot_param_hist must be a bool.\n",
				  },
	"plot_flux_hist" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  		"default": False,
				  		"error_message": "\n plot_flux_hist must be a bool.\n",
				  },
	"plot_lum_hist" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  		"default": False,
				  		"error_message": "\n plot_lum_hist must be a bool.\n",
				  },
	"plot_eqwidth_hist" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  		"default": False,
				  		"error_message": "\n plot_eqwidth_hist must be a bool.\n",
				  },
	"plot_HTML" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  		"default": False,
				  		"error_message": "\n plot_HTML must be a bool.\n",
				  },
	
	}
	
	output = check_dict(input,keyword_dict)
			
	return output

##################################################################################

#### Check Output options ########################################################

def check_output_options(input,verbose=False):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	output_options={
			"write_chain"   : False, # Write MCMC chains for all paramters, fluxes, and
									 # luminosities to a FITS table We set this to false 
									 # because MCMC_chains.FITS file can become very large, 
									 # especially  if you are running multiple objects.  
									 # You only need this if you want to reconstruct chains 
									 # and histograms. 
			"verbose"  : True,  # prints steps of fitting process in Notebook
	}
	"""
	output={} # output dictionary

	if not input:
		output={
			"write_chain"   : False, # Write MCMC chains for all paramters, fluxes, and
									 # luminosities to a FITS table We set this to false 
									 # because MCMC_chains.FITS file can become very large, 
									 # especially  if you are running multiple objects.  
									 # You only need this if you want to reconstruct chains 
									 # and histograms. 
			"verbose"  : True,  # prints steps of fitting process in Notebook
		}
		return output

	keyword_dict ={
	"write_chain" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  		"default": True,
				  		"error_message": "\n write_chain must be a bool.\n",
				  },
	"verbose" : {"conds":[
							lambda x: isinstance(x,(bool))
							],
				  		"default": False,
				  		"error_message": "\n verbose must be a bool.\n",
				  },
	
	}
	
	output = check_dict(input,keyword_dict)
			
	return output

##################################################################################

#### Check User Input Spectrum ###################################################

def check_user_input_spec(spec,wave,err,fwhm_res,z,ebv,verbose=False):

	if (spec is not None) and (isinstance(spec,(list,np.ndarray))):
		pass
	else: 
		raise TypeError("\n Required user-input spectrum must be a list or array.\n")

	if (wave is not None) and (isinstance(wave,(list,np.ndarray))) and (len(wave)==len(spec)):
		pass
	else: 
		raise TypeError("\n Required user-input wavelength vector must be a list or array the same size as the input spectrum.\n")

	if (err is not None) and (isinstance(err,(list,np.ndarray))) and (len(err)==len(spec)):
		pass
	else: 
		raise TypeError("\n Required user-input error spectrum must be a list or array the same size as the input spectrum.\n")

	if (fwhm_res is not None) and (isinstance(fwhm_res,(list,np.ndarray,int,float))) and (fwhm_res>=0):
		pass
	elif (fwhm_res is None):
		fwhm_res = 0.0 # Assume the user does not want to correct for instrumental dispersion
		if (verbose==True):
			print("\n Warning: no specified instrumental FWHM resolution in Angstroms!  BADASS cannot correct for instrumental dispersion or accurately fit the stellar LOSVD without an input FWHM value.  Assuming FWHM = 0.0 Å.\n")
	else: 
		raise TypeError("\n User-input disp must be a list/array the same size as the input spectrum, or integer/float value.\n")

	if (z is not None) and (isinstance(z,(int,float))) and (z>=0):
		pass
	else: 
		raise TypeError("\n Required user-input redshift must be an integer or float value.\n")

	if (ebv is not None) and (isinstance(ebv,(int,float))) and (ebv>=0):
		pass
	elif (ebv is None):
		ebv = 0.0 # Assume the user does not want to correct for galactic extinction
		if (verbose==True):
			print("\n Warning: no specified E(B-V).  BADASS cannot correct for Galactic extinction without a user-input E(B-V) value.  Assuming ebv = 0.0.\n")
	else: 
		raise TypeError("\n User-input ebv must be an integer/float value.\n")


	return spec,wave,err,fwhm_res,z,ebv

##################################################################################


def fetch_IRSA_dust(spec_fold):
	"""
	Fetches IRSA E(B-V) for Galactic extinction correction based
	on a spectrum's RA and DEC.  The RA and DEC header keywords
	must be in the file header, otherwise an average Galactic 
	E(B-V) = 0.04 is assumed by BADASS and used for extinction
	correction.
	"""
	if isinstance(spec_fold,(str)):
		spec_fold = [spec_fold]

	print("\n Fetching IRSA Dust for spectra...")
	for i in range(len(spec_fold)):
		# if i%100==0:
			# print(i)
		# Get path of of spectra file
		spec_file = glob.glob(spec_fold[i]+'/*.fits')[0]
		# Get object name 
		# obj_name = spec_fold.split('/')[-1]
		
		# Load the data
		hdu = fits.open(spec_file)
		header_cols = [i.keyword for i in hdu[0].header.cards]
		if ("RA" in header_cols) and ("DEC" in header_cols):
			ra  = hdu[0].header['RA']
			dec = hdu[0].header['DEC']

			co = coordinates.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
			table = IrsaDust.get_query_table(co,section='ebv')
			ebv = table['ext SandF mean'][0]
			hdu.close()

	print("\n Done fetching IRSA dust!	\n")
	return 

##################################################################################