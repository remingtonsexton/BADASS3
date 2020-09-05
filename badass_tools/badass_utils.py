#!/usr/bin/env python

# check_badass_args.py
"""
A set of functions that checks user input for BADASS and issues warnings 
if incorrect values are used.
"""

#### Check Fit Options ###########################################################

def check_fit_options(input,comp_options):
	"""
	Checks the inputs of the fit_options dictionary and ensures that 
	all keywords have valid values. 

	fit_options={
	'fit_reg'       : (4400,5800), # Fitting region; Indo-US Library=(3460,9464)
	'good_thresh'   : 0.0, # percentage of "good" pixels required in fig_reg for fit.
	'interp_bad'    : False, # interpolate over pixels SDSS flagged as 'bad' (careful!)
	'n_basinhop'	: 10, # Number of consecutive basinhopping thresholds before solution achieved
	'test_outflows' : False, # test for outflows?
	'outflow_test_niter': 10, # number of monte carlo iterations for outflows
	'max_like_niter': 10, # number of maximum likelihood iterations
	'min_sn_losvd'  : 20,  # minimum S/N threshold for fitting the LOSVD
	}

	"""
	output={} # output dictionary

	if not input:
		output={
		'fit_reg'    : (4400,5500), # Fitting region; Indo-US Library=(3460,9464)
		'good_thresh': 0.0, # percentage of "good" pixels required in fig_reg for fit.
		'interp_bad' : False, # interpolate over pixels SDSS flagged as 'bad' (careful!)
		# Number of consecutive basinhopping thresholds before solution achieved
		'n_basinhop': 10,
		# Outflow Testing Parameters
		'test_outflows': False, 
		'outflow_test_niter': 10, # number of monte carlo iterations for outflows
		# Maximum Likelihood Fitting for Final Model Parameters
		'max_like_niter': 10, # number of maximum likelihood iterations
		# LOSVD parameters
		'min_sn_losvd': 20,  # minimum S/N threshold for fitting the LOSVD
		# Emission line profile parameters
		'line_profile':'G' # Gaussian (G) or Lorentzian (L)
		}
		return output
	# check fit_reg
	if 'fit_reg' in input:
		fit_min = input['fit_reg'][0]
		fit_max = input['fit_reg'][1]
		fit_losvd = comp_options['fit_losvd']
		# Check size of fitting range (ideally more than 100 Angstroms)
		if (fit_max-fit_min<100):
			raise ValueError('\n Fitting region must be larger than 100 Angstroms!\n')
		elif (fit_min>=fit_max):
			raise ValueError('\n Fit region minimum must be less than the maximum!\n')
			# fit_reg = (min,max), min<max, min>3460 , max< 9464(Indo-Us Coude Feed Stellar Spectra Library)
		elif ((fit_min<3460) or (fit_max>9464)) & (fit_losvd==True):
			raise ValueError('\n Fitting region exceeds that of template stars!\n Indo-Us Coude Feed Stellar Spectra Library = (3460,9464)\n')
		else: 
			output['fit_reg']=(fit_min,fit_max)
	else:
		print('\n No fitting region set.  Setting fit region to [4400,5800].\n')
		output['fit_reg']=(4400.,5800.)
	# Check good pixel threshold
	if 'good_thresh' in input:
		good_thresh = input['good_thresh']
		if (good_thresh<0.0) or (good_thresh>1.0):
			raise ValueError('\n Good pixel threshold must be a float in the range [0.0,1.0]\n')
		else:
			output['good_thresh']=good_thresh
	else:
		output['good_thresh']=0.0
	# Check bad pixel interpolation
	if 'interp_bad' in input:
		interp_bad = input['interp_bad']
		if (not isinstance(interp_bad,(bool,int))):
				raise TypeError('\n interp_bad must be set to "True" or "False" \n')
		else: 
			output['interp_bad']=interp_bad
	else: 
		output['interp_bad']= False
	# Check n_basinhop
	if 'n_basinhop' in input:
		n_basinhop = input['n_basinhop']
		if (not isinstance(n_basinhop,int)) or (n_basinhop<1):
			raise ValueError('\n n_basinhop must be an integer and must be >=1! Suggested = 5-10 \n')
		else: 
			output['n_basinhop']=n_basinhop
	else: 
		output['n_basinhop']= 10
	# Check test_outflows
	if 'test_outflows' in input:
		test_outflows = input['test_outflows']
		if (not isinstance(test_outflows,(bool,int))):
				raise TypeError('\n test_outflows must be set to "True" or "False" \n')
		else: 
			output['test_outflows']=test_outflows
	else: 
		output['test_outflows']= True
	# Check outflow_test_niter
	if 'outflow_test_niter' in input:
		outflow_test_niter = input['outflow_test_niter']
		if (not isinstance(outflow_test_niter,int)) or (outflow_test_niter<0):
			raise ValueError('\n outflow_test_niter must be an integer and must be >=0 ! \n')
		else: 
			output['outflow_test_niter']=outflow_test_niter
	else: 
		output['outflow_test_niter']= 10
	# Check max_like_niter
	if 'max_like_niter' in input:
		max_like_niter = input['max_like_niter']
		if (not isinstance(max_like_niter,int)) or (max_like_niter<1):
			raise ValueError('\n max_like_niter must be an integer and must be >=1 ! \n')
		else: 
			output['max_like_niter']=max_like_niter
	else: 
		output['max_like_niter']= 10
	# Check max_like_niter
	if 'min_sn_losvd' in input:
		min_sn_losvd = input['min_sn_losvd']
		if (not isinstance(min_sn_losvd,(int,float))) or (min_sn_losvd<0):
			raise ValueError('\n min_sn_losvd must be an integer or float and must be >=0 ! \n')
		else: 
			output['min_sn_losvd']=min_sn_losvd
	else: 
		output['min_sn_losvd']= 20

	# Check line_profile
	if 'line_profile' in input:
		line_profile = input['line_profile']
		if line_profile=='Gaussian' or  line_profile=='Lorentzian':
			output['line_profile']=line_profile
		elif line_profile=='G':
			output['line_profile']='Gaussian'
		elif line_profile=='L':
			output['line_profile']='Lorentzian'	
		else: 
			raise TypeError("\n Options for line_profile are 'Gaussian' or 'Lorentzian'. \n")
	else: 
		output['conv_type']= 'Gaussian'

	return output
	
################################################################################## 

#### Check MCMC options ##########################################################

def check_mcmc_options(input,):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	mcmc_options={
	'mcmc_fit'    : False, # Perform robust fitting using emcee
	'nwalkers'    : 100,  # Number of emcee walkers; min = 2 x N_parameters
	'auto_stop'   : True, # Automatic stop using autocorrelation analysis
	'conv_type'   : 'median', # 'median', 'mean', 'all', or (tuple) of parameters
	'min_samp'    : 2500,  # min number of iterations for sampling post-convergence
	'ncor_times'  : 10.0,  # number of autocorrelation times for convergence
	'autocorr_tol': 10.0,  # percent tolerance between checking autocorr. times
	'write_iter'  : 100,   # write/check autocorrelation times interval
	'write_thresh': 100,   # when to start writing/checking parameters
	'burn_in'     : 23500, # burn-in if max_iter is reached
	'min_iter'    : 100,   # min number of iterations before stopping
	'max_iter'    : 25000, # max number of MCMC iterations
	}
	"""
	output={} # output dictionary

	if not input:
		output={
		'mcmc_fit'    : True, # Perform robust fitting using emcee
		'nwalkers'    : 100,  # Number of emcee walkers; min = 2 x N_parameters
		'auto_stop'   : True, # Automatic stop using autocorrelation analysis
		'conv_type'   : 'all', # 'median', 'mean', 'all', or (tuple) of parameters
		'min_samp'    : 2500,  # min number of iterations for sampling post-convergence
		'ncor_times'  : 10.0,  # number of autocorrelation times for convergence
		'autocorr_tol': 10.0,  # percent tolerance between checking autocorr. times
		'write_iter'  : 100,   # write/check autocorrelation times interval
		'write_thresh': 100,   # when to start writing/checking parameters
		'burn_in'     : 47500, # burn-in if max_iter is reached
		'min_iter'    : 2500,   # min number of iterations before stopping
		'max_iter'    : 50000, # max number of MCMC iterations
		}
		return output
	# Check mcmc_fit
	if 'mcmc_fit' in input:
		mcmc_fit = input['mcmc_fit']
		if (not isinstance(mcmc_fit,(bool,int))):
				raise TypeError('\n mcmc_fit must be set to "True" or "False" \n')
		else: 
			output['mcmc_fit']=mcmc_fit
	else: 
		output['mcmc_fit']= True

	# Check nwalkers
	if 'nwalkers' in input:
		nwalkers = input['nwalkers']
		if (not isinstance(nwalkers,int)) or (nwalkers<1):
			raise ValueError('\n nwalkers must be an integer and must be at least 2 x number of free parameters! \n')
		else: 
			output['nwalkers']=nwalkers
	else: 
		output['max_like_niter']= 100

	# Check auto_stop
	if 'auto_stop' in input:
		auto_stop = input['auto_stop']
		if (not isinstance(auto_stop,(bool,int))):
				raise TypeError('\n auto_stop must be set to "True" or "False" \n')
		else: 
			output['auto_stop']=auto_stop
	else: 
		output['auto_stop']= True
	# Check conv_type
	if 'conv_type' in input:
		conv_type = input['conv_type']
		if conv_type=='median' or conv_type=='mean' or conv_type=='all' or isinstance(conv_type,(list,tuple)):
			output['conv_type']=conv_type
				
		else: 
			raise TypeError("\n Options for conv_type are 'median','mean','mode', or a list of valid parameters. \n")
	else: 
		output['conv_type']= 'median'
	# Check min_samp
	if 'min_samp' in input:
		min_samp = input['min_samp']
		if (not isinstance(min_samp,int)) or (min_samp<1):
			raise ValueError('\n min_samp must be an integer in range [1,max_iter]! \n')
		else: 
			output['min_samp']=min_samp
	else: 
		output['min_samp']= 2500

	# Check ncor_times
	if 'ncor_times' in input:
		ncor_times = input['ncor_times']
		if (not isinstance(ncor_times,(int,float))) or (ncor_times<1):
			raise ValueError('\n ncor_times must be an int or float >0! \n')
		else: 
			output['ncor_times']=ncor_times
	else: 
		output['ncor_times']= 10.0
	# Check autocorr_tol
	if 'autocorr_tol' in input:
		autocorr_tol = input['autocorr_tol']
		if (not isinstance(autocorr_tol,(int,float))) or (autocorr_tol<1):
			raise ValueError('\n autocorr_tol must be an int or float >0! \n')
		else: 
			output['autocorr_tol']=autocorr_tol
	else: 
		output['autocorr_tol']= 10.0
	# Check write_iter
	if 'write_iter' in input:
		write_iter = input['write_iter']
		if (not isinstance(write_iter,int)) or (write_iter<1):
			raise ValueError('\n write_iter must be an integer >1! \n')
		else: 
			output['write_iter']=write_iter
	else: 
		output['write_iter']= 100
	# Check write_thresh
	if 'write_thresh' in input:
		write_thresh = input['write_thresh']
		if (not isinstance(write_thresh,int)) or (write_thresh<1):
			raise ValueError('\n write_thresh must be an integer >1! \n')
		else: 
			output['write_thresh']=write_thresh
	else: 
		output['write_thresh']= 100
	# Check burn_in
	if 'burn_in' in input:
		burn_in = input['burn_in']
		if (not isinstance(burn_in,int)) or (burn_in<1):
			raise ValueError('\n burn_in must be an integer >1! \n')
		else: 
			output['burn_in']=burn_in
	else: 
		output['burn_in']= 23500
	# Check min_iter
	# Min_iter must be an integer multiple of write_iter
	if 'min_iter' in input:
		min_iter = input['min_iter']
		if (not isinstance(min_iter,int)) or (min_iter<1):
			raise ValueError('\n min_iter must be an integer >1! \n')
		if (min_iter%write_iter > 0):
			raise ValueError('\n min_iter must be an integer multiple of write_iter! \n')
		else: 
			output['min_iter']=min_iter
	else: 
		output['min_iter']= 2500
	# Check max_iter
	# max_iter must be an integer multiple of write_iter
	if 'max_iter' in input:
		max_iter = input['max_iter']
		if (not isinstance(max_iter,int)) or (max_iter<1):
			raise ValueError('\n max_iter must be an integer >1! \n')
		if (max_iter%write_iter > 0):
			raise ValueError('\n max_iter must be an integer multiple of write_iter! \n')
		else: 
			output['max_iter']=max_iter
	else: 
		output['max_iter']= 25000
	# print output

	return output

##################################################################################

#### Check Component options #####################################################

def check_comp_options(input,):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	comp_options={
	'fit_feii'    : True, # fit broad and narrow FeII emission
	'fit_losvd'   : True, # fit LOSVD (stellar kinematics) in final model
	'fit_host'    : True, # fit host-galaxy using template (if fit_LOSVD turned off)
	'fit_power'   : True, # fit AGN power-law continuum
	'fit_broad'   : True, # fit broad lines (Type 1 AGN)
	'fit_narrow'  : True, # fit narrow lines
	'fit_outflows': True, # fit outflows;
	'tie_narrow'  : False,  # tie narrow widths (don't do this)
	}
	"""
	output={} # output dictionary

	if not input:
		output={
		'fit_feii'    : True, # fit broad and narrow FeII emission
		'fit_losvd'   : False, # fit LOSVD (stellar kinematics) in final model
		'fit_host'    : True, # fit host-galaxy using template (if fit_LOSVD turned off)
		'fit_power'   : True, # fit AGN power-law continuum
		'fit_broad'   : True, # fit broad lines (Type 1 AGN)
		'fit_narrow'  : True, # fit narrow lines
		'fit_outflows': True, # fit outflows;
		'tie_narrow'  : False,  # tie narrow widths (don't do this)
		}
		return output

	# Check fit_feii
	if 'fit_feii' in input:
		fit_feii = input['fit_feii']
		if (not isinstance(fit_feii,(bool,int))):
				raise TypeError('\n fit_feii must be set to "True" or "False" \n')
		else: 
			output['fit_feii']=fit_feii
	else: 
		output['fit_feii']= True
	# Check fit_losvd
	if 'fit_losvd' in input:
		fit_losvd = input['fit_losvd']
		if (not isinstance(fit_losvd,(bool,int))):
				raise TypeError('\n fit_losvd must be set to "True" or "False" \n')
		else: 
			output['fit_losvd']=fit_losvd
	else: 
		output['fit_losvd']= True
	# Check fit_host
	if 'fit_host' in input:
		fit_host = input['fit_host']
		if (not isinstance(fit_host,(bool,int))):
				raise TypeError('\n fit_host must be set to "True" or "False" \n')
		else: 
			output['fit_host']=fit_host
	else: 
		output['fit_host']= True
	# Check fit_power
	if 'fit_power' in input:
		fit_power = input['fit_power']
		if (not isinstance(fit_power,(bool,int))):
				raise TypeError('\n fit_power must be set to "True" or "False" \n')
		else: 
			output['fit_power']=fit_power
	else: 
		output['fit_power']= True
	# Check fit_broad
	if 'fit_broad' in input:
		fit_broad = input['fit_broad']
		if (not isinstance(fit_broad,(bool,int))):
				raise TypeError('\n fit_broad must be set to "True" or "False" \n')
		else: 
			output['fit_broad']=fit_broad
	else: 
		output['fit_broad']= True
	# Check fit_narrow
	if 'fit_narrow' in input:
		fit_narrow = input['fit_narrow']
		if (not isinstance(fit_narrow,(bool,int))):
				raise TypeError('\n fit_narrow must be set to "True" or "False" \n')
		else: 
			output['fit_narrow']=fit_narrow
	else: 
		output['fit_narrow']= True
	# Check fit_outflows
	if 'fit_outflows' in input:
		fit_outflows = input['fit_outflows']
		if (not isinstance(fit_outflows,(bool,int))):
				raise TypeError('\n fit_outflows must be set to "True" or "False" \n')
		else: 
			output['fit_outflows']=fit_outflows
	else: 
		output['fit_outflows']= True
	# Check tie_narrow
	if 'tie_narrow' in input:
		tie_narrow = input['tie_narrow']
		if (not isinstance(tie_narrow,(bool,int))):
				raise TypeError('\n tie_narrow must be set to "True" or "False" \n')
		else: 
			output['tie_narrow']=tie_narrow
	else: 
		output['tie_narrow']= False


	# print output
	return output

##################################################################################


#### Check FeII options ##########################################################

def check_feii_options(input,):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	feii_options={
	'amp_const':{'bool':False,'br_feii_val':1.0,'na_feii_val':1.0},
	'fwhm_const':{'bool':True,'br_feii_val':3000.0,'na_feii_val':500.0},
	'voff_const':{'bool':True,'br_feii_val':0.0,'na_feii_val':0.0}
	}

	"""
	output={} # output dictionary

	# If feii_options not specified
	if not input:
		output={
					'template'  :{'type':'VC04'}, 
					'amp_const' :{'bool':False,'br_feii_val':1.0,'na_feii_val':1.0},
					'fwhm_const':{'bool':False,'br_feii_val':3000.0,'na_feii_val':500.0},
					'voff_const':{'bool':False,'br_feii_val':0.0,'na_feii_val':0.0},
					}
		return output


	# check template type
	if 'template' in input:
		temp_type = input['template']['type']
		if (temp_type=='VC04') or (temp_type=='K10'):
			output = {'template':{}}
			output['template'].update({'type':temp_type})
		else: 
			raise TypeError('\n Template type options are Veron-Cetty 2004 (VC04) or Kovacevic 2010 (K10). \n')
	else:
		output['template']={'type':'VC04'} # <- VC04 is BADASS default

	# If VC04, check options or set defaults
	if (input['template']['type']=='VC04'):
		# check amp_const
		if 'amp_const' in input:
			if 'bool' in input['amp_const']:
				amp_bool = input['amp_const']['bool']
				if (not isinstance(amp_bool,(bool,int))):
					raise TypeError('\n amp bool must be set to "True" or "False" \n')
				else:
					output.update({'amp_const':{}})
					output['amp_const'].update({'bool':amp_bool})
			else: 
				output['amp_const']={'bool': False}
			#
			if 'br_feii_val' in input['amp_const']:
				br_feii_val = input['amp_const']['br_feii_val']
				if (not isinstance(br_feii_val,(float,int))):
					raise TypeError('\n amp br_feii_val must be an integer or float \n')
				else:
					output['amp_const'].update({'br_feii_val':br_feii_val})
			else: 
				output['amp_const']={'br_feii_val': 1.0}
			#
			if 'na_feii_val' in input['amp_const']:
				na_feii_val = input['amp_const']['na_feii_val']
				if (not isinstance(na_feii_val,(float,int))):
					raise TypeError('\n amp na_feii_val must be an integer or float \n')
				else:
					output['amp_const'].update({'na_feii_val':na_feii_val})
			else: 
				output['amp_const']={'na_feii_val':1.0}
			#
		else: 
			output['amp_const']={'bool':False,'br_feii_val':1.0,'na_feii_val':1.0} # <- BADASS default

		# check fwhm_const
		if 'fwhm_const' in input:
			if 'bool' in input['fwhm_const']:
				fwhm_bool = input['fwhm_const']['bool']
				if (not isinstance(fwhm_bool,(bool,int))):
					raise TypeError('\n fwhm bool must be set to "True" or "False" \n')
				else:
					output.update({'fwhm_const':{}})
					output['fwhm_const'].update({'bool':fwhm_bool})
			else: 
				output['fwhm_const']={'bool': True}
			#
			if 'br_feii_val' in input['fwhm_const']:
				br_feii_val = input['fwhm_const']['br_feii_val']
				if (not isinstance(br_feii_val,(float,int))):
					raise TypeError('\n fwhm br_feii_val must be an integer or float \n')
				else:
					output['fwhm_const'].update({'br_feii_val':br_feii_val})
			else: 
				output['fwhm_const']={'br_feii_val': 3000.0}
			#
			if 'na_feii_val' in input['fwhm_const']:
				na_feii_val = input['fwhm_const']['na_feii_val']
				if (not isinstance(na_feii_val,(float,int))):
					raise TypeError('\n fwhm na_feii_val must be an integer or float \n')
				else:
					output['fwhm_const'].update({'na_feii_val':na_feii_val})
			else: 
				output['fwhm_const']={'na_feii_val':500.0}
			#
		else: 
			output['fwhm_const']={'bool':True,'br_feii_val':3000.0,'na_feii_val':500.0}# <- BADASS default

		# check voff_const
		if 'voff_const' in input:
			if 'bool' in input['voff_const']:
				voff_const = input['voff_const']['bool']
				if (not isinstance(voff_const,(bool,int))):
					raise TypeError('\n voff bool must be set to "True" or "False" \n')
				else:
					output.update({'voff_const':{}})
					output['voff_const'].update({'bool':voff_const})
			else: 
				output['voff_const']={'bool': True}
			#
			if 'br_feii_val' in input['voff_const']:
				br_feii_val = input['voff_const']['br_feii_val']
				if (not isinstance(br_feii_val,(float,int))):
					raise TypeError('\n voff br_feii_val must be an integer or float \n')
				else:
					output['voff_const'].update({'br_feii_val':br_feii_val})
			else: 
				output['voff_const']={'br_feii_val': 0.0}
			#
			if 'na_feii_val' in input['voff_const']:
				na_feii_val = input['voff_const']['na_feii_val']
				if (not isinstance(na_feii_val,(float,int))):
					raise TypeError('\n voff na_feii_val must be an integer or float \n')
				else:
					output['voff_const'].update({'na_feii_val':na_feii_val})
			else: 
				output['voff_const']={'na_feii_val':0.0}
			#
		else: 
			output['voff_const']={'bool':True,'br_feii_val':0.0,'na_feii_val':0.0}# <- BADASS default

	# If K10, check options or set defaults
	if (input['template']['type']=='K10'):
		# check amp_const
		if 'amp_const' in input:
			if 'bool' in input['amp_const']:
				amp_bool = input['amp_const']['bool']
				if (not isinstance(amp_bool,(bool,int))):
					raise TypeError('\n amp bool must be set to "True" or "False" \n')
				else:
					output.update({'amp_const':{}})
					output['amp_const'].update({'bool':amp_bool})
			else: 
				output['amp_const']={'bool': False}
			#
			if 'f_feii_val' in input['amp_const']:
				f_feii_val = input['amp_const']['f_feii_val']
				if (not isinstance(f_feii_val,(float,int))):
					raise TypeError('\n amp f_feii_val must be an integer or float \n')
				else:
					output['amp_const'].update({'f_feii_val':f_feii_val})
			else: 
				output['amp_const']={'f_feii_val': 1.0}
			#
			if 's_feii_val' in input['amp_const']:
				s_feii_val = input['amp_const']['s_feii_val']
				if (not isinstance(s_feii_val,(float,int))):
					raise TypeError('\n amp s_feii_val must be an integer or float \n')
				else:
					output['amp_const'].update({'s_feii_val':s_feii_val})
			else: 
				output['amp_const']={'s_feii_val':1.0}
			#
			if 'g_feii_val' in input['amp_const']:
				g_feii_val = input['amp_const']['g_feii_val']
				if (not isinstance(g_feii_val,(float,int))):
					raise TypeError('\n amp g_feii_val must be an integer or float \n')
				else:
					output['amp_const'].update({'g_feii_val':g_feii_val})
			else: 
				output['amp_const']={'g_feii_val':1.0}
			#
			if 'z_feii_val' in input['amp_const']:
				z_feii_val = input['amp_const']['z_feii_val']
				if (not isinstance(z_feii_val,(float,int))):
					raise TypeError('\n amp z_feii_val must be an integer or float \n')
				else:
					output['amp_const'].update({'z_feii_val':z_feii_val})
			else: 
				output['amp_const']={'z_feii_val':1.0}
			#
		else: 
			output['amp_const']={'bool':False,'f_feii_val':1.0,'s_feii_val':1.0,'g_feii_val':1.0,'z_feii_val':1.0}

		# check fwhm_const
		if 'fwhm_const' in input:
			if 'bool' in input['fwhm_const']:
				fwhm_bool = input['fwhm_const']['bool']
				if (not isinstance(fwhm_bool,(bool,int))):
					raise TypeError('\n fwhm bool must be set to "True" or "False" \n')
				else:
					output.update({'fwhm_const':{}})
					output['fwhm_const'].update({'bool':fwhm_bool})
			else: 
				output['fwhm_const']={'bool': True}
			#
			if 'val' in input['fwhm_const']:
				val = input['fwhm_const']['val']
				if (not isinstance(val,(float,int))):
					raise TypeError('\n fwhm val must be an integer or float \n')
				else:
					output['fwhm_const'].update({'val':val})
			else: 
				output['fwhm_const']={'val': 3000.0}
			#
		else: 
			output['fwhm_const']={'bool':False,'val':1500.0}

		# check voff_const
		if 'voff_const' in input:
			if 'bool' in input['voff_const']:
				voff_const = input['voff_const']['bool']
				if (not isinstance(voff_const,(bool,int))):
					raise TypeError('\n voff bool must be set to "True" or "False" \n')
				else:
					output.update({'voff_const':{}})
					output['voff_const'].update({'bool':voff_const})
			else: 
				output['voff_const']={'bool': True}
			#
			if 'val' in input['voff_const']:
				val = input['voff_const']['val']
				if (not isinstance(val,(float,int))):
					raise TypeError('\n voff val must be an integer or float \n')
				else:
					output['voff_const'].update({'val':val})
			else: 
				output['voff_const']={'val': 0.0}
			#
		else: 
			output['voff_const']={'bool':True,'val':0.0}

		# check temp_const
		if 'temp_const' in input:
			if 'bool' in input['temp_const']:
				temp_const = input['temp_const']['bool']
				if (not isinstance(temp_const,(bool,int))):
					raise TypeError('\n temp bool must be set to "True" or "False" \n')
				else:
					output.update({'temp_const':{}})
					output['temp_const'].update({'bool':temp_const})
			else: 
				output['temp_const']={'bool': True}
			if 'val' in input['temp_const']:
				val = input['temp_const']['val']
				if (not isinstance(val,(float,int))):
					raise TypeError('\n temp val must be an integer or float \n')
				else:
					output['temp_const'].update({'val':val})
			else: 
				output['temp_const']={'val': 10000.0}
		else: 
			output['temp_const']={'bool':False,'val':10000.0}

	return output


##################################################################################



#### Check Outflow Test options ##################################################

def check_outflow_test_options(input,):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	outflow_test_options={
	'amp_test':{'test':True,'nsigma':3.0}, # Amplitude-over-noise by n-sigma
	'fwhm_test':{'test':True,'nsigma':1.0}, # FWHM difference by n-sigma
	'voff_test':{'test':True,'nsigma':1.0}, # blueshift voff from core by n-sigma
	'outflow_confidence':{'test':True,'conf':0.95}, # outflow confidence acceptance
	'bounds_test':{'test':True,'nsigma':1.0} # within bounds by n-sigma
	}

	"""
	output={} # output dictionary

	if not input:
		output={
		'amp_test':{'test':True,'nsigma':3.0}, # Amplitude-over-noise by n-sigma
		'fwhm_test':{'test':True,'nsigma':1.0}, # FWHM difference by n-sigma
		'voff_test':{'test':True,'nsigma':1.0}, # blueshift voff from core by n-sigma
		'outflow_confidence':{'test':True,'conf':0.95}, # outflow confidence acceptance
		'bounds_test':{'test':True,'nsigma':1.0} # within bounds by n-sigma
		}
		return output

	# check amp_test
	if 'amp_test' in input:
		if 'test' in input['amp_test']:
			amp_test = input['amp_test']['test']
			if (not isinstance(amp_test,(bool,int))):
				raise TypeError('\n amp test must be set to "True" or "False" \n')
			else:
				output = {'amp_test':{}}
				output['amp_test'].update({'test':amp_test})
		else: 
			output['amp_test']={'test': True}
		if 'nsigma' in input['amp_test']:
			nsigma = input['amp_test']['nsigma']
			if (not isinstance(nsigma,(float,int))):
				raise TypeError('\n amp_test nsigma must be an integer or float \n')
			else:
				output['amp_test'].update({'nsigma':nsigma})
		else: 
			output['amp_test']={'nsigma': 3.0}
	else: 
		output['amp_test']={'test':True,'nsigma':3.0}

	# check fwhm_test
	if 'fwhm_test' in input:
		if 'test' in input['fwhm_test']:
			fwhm_test = input['fwhm_test']['test']
			if (not isinstance(fwhm_test,(bool,int))):
				raise TypeError('\n fwhm test must be set to "True" or "False" \n')
			else:
				output.update({'fwhm_test':{}})
				output['fwhm_test'].update({'test':fwhm_test})
		else: 
			output['fwhm_test']={'test': True}
		if 'nsigma' in input['fwhm_test']:
			nsigma = input['fwhm_test']['nsigma']
			if (not isinstance(nsigma,(float,int))):
				raise TypeError('\n fwhm_test nsigma must be an integer or float \n')
			else:
				output['fwhm_test'].update({'nsigma':nsigma})
		else: 
			output['fwhm_test']={'nsigma': 1.0}
	else: 
		output['fwhm_test']={'test':True,'nsigma':1.0}

	# check voff_test
	if 'voff_test' in input:
		if 'test' in input['voff_test']:
			voff_test = input['voff_test']['test']
			if (not isinstance(voff_test,(bool,int))):
				raise TypeError('\n voff test must be set to "True" or "False" \n')
			else:
				output.update({'voff_test':{}})
				output['voff_test'].update({'test':voff_test})
		else: 
			output['voff_test']={'test': True}
		if 'nsigma' in input['voff_test']:
			nsigma = input['voff_test']['nsigma']
			if (not isinstance(nsigma,(float,int))):
				raise TypeError('\n voff_test nsigma must be an integer or float \n')
			else:
				output['voff_test'].update({'nsigma':nsigma})
		else: 
			output['voff_test']={'nsigma': 1.0}
	else: 
		output['voff_test']={'test':True,'nsigma':1.0}

	# check outflow_confidence
	if 'outflow_confidence' in input:
		if 'test' in input['outflow_confidence']:
			outflow_confidence_test = input['outflow_confidence']['test']
			if (not isinstance(outflow_confidence_test,(bool,int))):
				raise TypeError('\n outflow_confidence test must be set to "True" or "False" \n')
			else:
				output.update({'outflow_confidence':{}})
				output['outflow_confidence'].update({'test':outflow_confidence_test})
		else: 
			output['outflow_confidence']={'test': True}
		if 'conf' in input['outflow_confidence']:
			conf = input['outflow_confidence']['conf']
			if (not isinstance(conf,(float,int))):
				raise TypeError('\n outflow_confidence conf must be an integer or float \n')
			else:
				output['outflow_confidence'].update({'conf':conf})
		else: 
			output['outflow_confidence']={'conf': 0.95}
	else: 
		output['outflow_confidence']={'test':True,'conf':0.95}

	# check bounds_test
	if 'bounds_test' in input:
		if 'test' in input['bounds_test']:
			bounds_test = input['bounds_test']['test']
			if (not isinstance(bounds_test,(bool,int))):
				raise TypeError('\n bounds test must be set to "True" or "False" \n')
			else:
				output.update({'bounds_test':{}})
				output['bounds_test'].update({'test':bounds_test})
		else: 
			output['bounds_test']={'test': True}
		if 'nsigma' in input['bounds_test']:
			nsigma = input['bounds_test']['nsigma']
			if (not isinstance(nsigma,(float,int))):
				raise TypeError('\n bounds_test nsigma must be an integer or float \n')
			else:
				output['bounds_test'].update({'nsigma':nsigma})
		else: 
			output['bounds_test']={'nsigma': 1.0}
	else: 
		output['bounds_test']={'test':True,'nsigma':1.0}

	return output

##################################################################################



#### Check Plot options ##########################################################

def check_plot_options(input,):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	plot_options={
	'plot_param_hist': True, # Plot MCMC histograms and chains for each parameter
	'plot_flux_hist' : True, # Plot MCMC hist. and chains for component fluxes
	'plot_lum_hist'  : True, # Plot MCMC hist. and chains for component luminosities
	'plot_mbh_hist'  : True, # Plot MCMC hist. for estimated AGN lum. and BH masses
	'plot_corner'    : False,# Plot corner plot of relevant parameters; Corner plots 
	                         # of free paramters can be quite large require a PDF 
	                         # output, and have significant time and space overhead, 
	                         # so we set this to False by default. 
	'plot_bpt'       : True, # Plot BPT diagram 
	}
	"""
	output={} # output dictionary

	if not input:
		output={
		'plot_param_hist': True,# Plot MCMC histograms and chains for each parameter
		'plot_flux_hist' : True,# Plot MCMC hist. and chains for component fluxes
		'plot_lum_hist'  : True,# Plot MCMC hist. and chains for component luminosities
		'plot_mbh_hist'  : True,# Plot MCMC hist. for estimated AGN lum. and BH masses
		'plot_corner'    : False,# Plot corner plot of relevant parameters; Corner plots 
		                         # of free paramters can be quite large require a PDF 
		                         # output, and have significant time and space overhead, 
		                         # so we set this to False by default. 
		'plot_bpt'      : True,  # Plot BPT diagram 
		}

		return output

	# Check plot_param_hist
	if 'plot_param_hist' in input:
		plot_param_hist = input['plot_param_hist']
		if (not isinstance(plot_param_hist,(bool,int))):
				raise TypeError('\n plot_param_hist must be set to "True" or "False" \n')
		else: 
			output['plot_param_hist']=plot_param_hist
	else: 
		output['plot_param_hist']= True

	# Check plot_flux_hist
	if 'plot_flux_hist' in input:
		plot_flux_hist = input['plot_flux_hist']
		if (not isinstance(plot_flux_hist,(bool,int))):
				raise TypeError('\n plot_flux_hist must be set to "True" or "False" \n')
		else: 
			output['plot_flux_hist']=plot_flux_hist
	else: 
		output['plot_flux_hist']= True

	# Check plot_lum_hist
	if 'plot_lum_hist' in input:
		plot_lum_hist = input['plot_lum_hist']
		if (not isinstance(plot_lum_hist,(bool,int))):
				raise TypeError('\n plot_lum_hist must be set to "True" or "False" \n')
		else: 
			output['plot_lum_hist']=plot_lum_hist
	else: 
		output['plot_lum_hist']= True

	# Check plot_mbh_hist
	if 'plot_mbh_hist' in input:
		plot_mbh_hist = input['plot_mbh_hist']
		if (not isinstance(plot_mbh_hist,(bool,int))):
				raise TypeError('\n plot_mbh_hist must be set to "True" or "False" \n')
		else: 
			output['plot_mbh_hist']=plot_mbh_hist
	else: 
		output['plot_mbh_hist']= True

	# Check plot_corner
	if 'plot_corner' in input:
		plot_corner = input['plot_corner']
		if (not isinstance(plot_corner,(bool,int))):
				raise TypeError('\n plot_corner must be set to "True" or "False" \n')
		else: 
			output['plot_corner']=plot_corner
	else: 
		output['plot_corner']= False

	# Check plot_bpt
	if 'plot_bpt' in input:
		plot_bpt = input['plot_bpt']
		if (not isinstance(plot_bpt,(bool,int))):
				raise TypeError('\n plot_bpt must be set to "True" or "False" \n')
		else: 
			output['plot_bpt']=plot_bpt
	else: 
		output['plot_bpt']= True

	# print output
	return output

##################################################################################


#### Check Output options ########################################################

def check_output_options(input,):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	output_options={
	'write_chain'    : False,# Write MCMC chains for all paramters, fluxes, and
	                         # luminosities to a FITS table We set this to false 
	                         # because MCMC_chains.FITS file can become very large, 
	                         # especially  if you are running multiple objects.  
	                         # You only need this if you want to reconstruct chains 
	                         # and histograms. 
	'print_output'   : True,  # prints steps of fitting process in Jupyter output
	}
	"""
	output={} # output dictionary

	if not input:
		output={
		'write_chain'   : False, # Write MCMC chains for all paramters, fluxes, and
		                         # luminosities to a FITS table We set this to false 
		                         # because MCMC_chains.FITS file can become very large, 
		                         # especially  if you are running multiple objects.  
		                         # You only need this if you want to reconstruct chains 
		                         # and histograms. 
		'print_output'  : True,  # prints steps of fitting process in Jupyter output
		}
		return output

	# Check write_chain
	if 'write_chain' in input:
		write_chain = input['write_chain']
		if (not isinstance(write_chain,(bool,int))):
				raise TypeError('\n write_chain must be set to "True" or "False" \n')
		else: 
			output['write_chain']=write_chain
	else: 
		output['write_chain']= False

	# Check print_output
	if 'print_output' in input:
		print_output = input['print_output']
		if (not isinstance(print_output,(bool,int))):
				raise TypeError('\n print_output must be set to "True" or "False" \n')
		else: 
			output['print_output']=print_output
	else: 
		output['print_output']= True

	# print output
	return output

##################################################################################


#### Check Multiprocessing options ###############################################

def check_mp_options(input,):
	"""
	Checks the inputs of the mcmc_options dictionary and ensures that 
	all keywords have valid values. 

	mp_options={
	'threads' : 4# number of processes per object
	}

	"""
	output={} # output dictionary

	if not input:
		output={
		'threads' : 4 # number of processes per object
		}
		return output

	# Check threads
	if 'threads' in input:
		threads = input['threads']
		if (not isinstance(threads,(int))):
				raise TypeError('\n threads must be an integer! \n')
		else: 
			output['threads']=threads
	else: 
		output['threads']= 4

	# print output
	return output

##################################################################################
