import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import utils.constants as consts
from utils.templates.common import BadassTemplate, convolve_gauss_hermite, gaussian_filter1d, template_rfft
from utils.utils import log_rebin


class OpticalFeIITemplate(BadassTemplate):
	TEMP_LAM_RANGE = [0.0, -1.0]

	@classmethod
	def initialize_template(cls, ctx):
		if not ctx.options.comp_options.fit_opt_feii:
			return None

		temp_type = ctx.options.opt_feii_options.opt_template.type
		class_name = '%s_OpticalFeIITemplate' % (temp_type)
		if not class_name in globals():
			ctx.error('Optical FeII template unsupported: %s' % temp_type)
			return None

		temp_class = globals()[class_name]

		if (temp_class.TEMP_LAM_RANGE[1] > 0.0 and ctx.wave[0] > temp_class.TEMP_LAM_RANGE[1]) or (ctx.wave[-1] < temp_class.TEMP_LAM_RANGE[0]):
			ctx.log.warn('Optical FeII template disabled because template is outside of fitting region.')
			ctx.log.update_opt_feii()
			ctx.options.comp_options.fit_opt_feii = False
			return None

		return temp_class.initialize_template(ctx)


	def __init__(self, ctx):
		self.ctx = ctx


	def convolve(self, fft, feii_voff, feii_disp, npad=None):
		if npad is None:
			npad = self.npad
		return convolve_gauss_hermite(fft, npad, float(self.ctx.velscale),\
									[feii_voff, feii_disp/2.3548], self.ctx.wave.shape[0], 
									velscale_ratio=1, sigma_diff=0, vsyst=self.vsyst)



class VC04_OpticalFeIITemplate(OpticalFeIITemplate):
	"""
	'VC04' : Veron-Cetty et al. (2004) template, which utilizes a single broad 
			 and single narrow line template with fixed relative intensities. 
			 One can choose to fix FWHM and VOFF for each, and only vary 
			 amplitudes (2 free parameters), or vary amplitude, FWHM, and VOFF
			 for each template (6 free parameters)
	"""

	TEMP_LAM_RANGE = [3400.0, 7200.0] # Angstrom

	vc04_data_dir = consts.BADASS_DATA_DIR.joinpath('feii_templates', 'veron-cetty_2004')
	br_path = vc04_data_dir.joinpath('VC04_br_feii_template.csv')
	na_path = vc04_data_dir.joinpath('VC04_na_feii_template.csv')


	@classmethod
	def initialize_template(cls, ctx):
		if (not VC04_OpticalFeIITemplate.br_path.exists()) or (not VC04_OpticalFeIITemplate.na_path.exists()):
			ctx.log.error('VC04 data directory not found: %s' % str(VC04_OpticalFeIITemplate.vc04_data_dir))
			return None

		return cls(ctx)


	def __init__(self, ctx):
		super().__init__(ctx)

		df_br = pd.read_csv(self.br_path)
		df_na = pd.read_csv(self.na_path)

		# Generate a new grid with the original resolution, but the size of the fitting region
		dlam_feii = df_br['angstrom'].to_numpy()[1]-df_br['angstrom'].to_numpy()[0] # angstroms
		npad = 100 # anstroms
		lam_feii = np.arange(np.min(self.ctx.wave)-npad, np.max(self.ctx.wave)+npad, dlam_feii) # angstroms

		# Interpolate the original template onto the new grid
		interp_ftn_br = interp1d(df_br['angstrom'].to_numpy(),df_br['flux'].to_numpy(),kind='linear',bounds_error=False,fill_value=(0.0,0.0))
		interp_ftn_na = interp1d(df_na['angstrom'].to_numpy(),df_na['flux'].to_numpy(),kind='linear',bounds_error=False,fill_value=(0.0,0.0))
		spec_feii_br = interp_ftn_br(lam_feii)
		spec_feii_na = interp_ftn_na(lam_feii)

		# Convolve templates to the native resolution of SDSS
		fwhm_feii = 1.0 # templates were created with 1.0 FWHM resolution
		disp_feii = fwhm_feii/2.3548
		disp_res_interp = np.interp(lam_feii, self.ctx.wave, self.ctx.disp_res)
		disp_diff = np.sqrt((disp_res_interp**2 - disp_feii**2).clip(0))
		sigma = disp_diff/dlam_feii # Sigma difference in pixels
		spec_feii_br = gaussian_filter1d(spec_feii_br, sigma)
		spec_feii_na = gaussian_filter1d(spec_feii_na, sigma)

		# log-rebin the spectrum to same velocity scale as the input galaxy
		lamRange_feii = [np.min(lam_feii), np.max(lam_feii)]
		spec_feii_br_new, loglam_feii, velscale_feii = log_rebin(lamRange_feii, spec_feii_br, velscale=self.ctx.velscale)
		spec_feii_na_new, loglam_feii, velscale_feii = log_rebin(lamRange_feii, spec_feii_na, velscale=self.ctx.velscale)

		# Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
		self.br_opt_feii_fft, self.npad = template_rfft(spec_feii_br_new)
		self.na_opt_feii_fft, self.npad = template_rfft(spec_feii_na_new)

		# The FeII templates are offset from the input galaxy spectrum by 100 A, so we 
		# shift the spectrum to match that of the input galaxy.
		self.vsyst = np.log(lam_feii[0]/self.ctx.wave[0]) * consts.c

		# If opt_disp_const AND opt_voff_const, we preconvolve the templates so we don't have to during the fit
		opt_feii_options = self.ctx.options.opt_feii_options
		self.pre_convolve = (opt_feii_options.opt_disp_const.bool) and (opt_feii_options.opt_voff_const.bool)
		if self.pre_convolve:

			br_voff = opt_feii_options.opt_voff_const.br_opt_feii_val
			br_disp = opt_feii_options.opt_disp_const.br_opt_feii_val
			self.br_conv_temp = self.convolve(self.br_opt_feii_fft, br_voff, br_disp)

			na_voff = opt_feii_options.opt_voff_const.na_opt_feii_val
			na_disp = opt_feii_options.opt_disp_const.na_opt_feii_val
			self.na_conv_temp = self.convolve(self.na_opt_feii_fft, na_voff, na_disp)


	def initialize_parameters(self, params):
		# TODO: implement
		return


	def add_components(self, params, comp_dict, host_model):

		opt_feii_options = self.ctx.options.opt_feii_options
		val = lambda ok, ov, pk : opt_feii_options[ok][ov] if opt_feii_options[ok].bool else params[pk]

		# TODO: would this option ever change? ie. if amp, etc. are const, just set in init
		br_opt_feii_amp = val('opt_amp_const', 'br_opt_feii_val', 'BR_OPT_FEII_AMP')
		na_opt_feii_amp = val('opt_amp_const', 'na_opt_feii_val', 'NA_OPT_FEII_AMP')
		br_opt_feii_disp = val('opt_disp_const', 'br_opt_feii_val', 'BR_OPT_FEII_DISP')
		na_opt_feii_disp = val('opt_disp_const', 'na_opt_feii_val', 'NA_OPT_FEII_DISP')
		br_opt_feii_voff = val('opt_voff_const', 'br_opt_feii_val', 'BR_OPT_FEII_VOFF')
		na_opt_feii_voff = val('opt_voff_const', 'na_opt_feii_val', 'NA_OPT_FEII_VOFF')

		if not self.pre_convolve:
			self.br_conv_temp = self.convolve(self.br_opt_feii_fft, br_opt_feii_voff, br_opt_feii_disp)
			self.na_conv_temp = self.convolve(self.na_opt_feii_fft, na_opt_feii_voff, na_opt_feii_disp)

		br_opt_feii_template = br_opt_feii_amp * self.br_conv_temp
		na_opt_feii_template = na_opt_feii_amp * self.na_conv_temp

		br_opt_feii_template = br_opt_feii_template.reshape(-1)
		na_opt_feii_template = na_opt_feii_template.reshape(-1)

		# Set fitting region outside of template to zero to prevent convolution loops
		br_opt_feii_template[(self.ctx.wave < self.TEMP_LAM_RANGE[0]) & (self.ctx.wave > self.TEMP_LAM_RANGE[1])] = 0
		na_opt_feii_template[(self.ctx.wave < self.TEMP_LAM_RANGE[0]) & (self.ctx.wave > self.TEMP_LAM_RANGE[1])] = 0

		# Update the component dict with the templates
		comp_dict['BR_OPT_FEII_TEMPLATE'] = br_opt_feii_template
		comp_dict['NA_OPT_FEII_TEMPLATE'] = na_opt_feii_template

		# Subtract the br and na templates from the host model and return
		host_model -= na_opt_feii_template
		host_model -= br_opt_feii_template
		return comp_dict, host_model


class K10_OpticalFeIITemplate(OpticalFeIITemplate):
	"""
	'K10'  : Kovacevic et al. (2010) template, which treats the F, S, and G line 
			 groups as independent templates (each amplitude is a free parameter)
			 and whose relative intensities are temperature dependent (1 free 
			 parameter).  There are additonal lines from IZe1 that only vary in 
			 amplitude.  All 4 line groups share the same FWHM and VOFF, for a 
			 total of 7 free parameters.  This template is only recommended 
			 for objects with very strong FeII emission, for which the LOSVD
			 cannot be determined at all.
	"""

	TEMP_LAM_RANGE = [4400.0, 5500.0]

	class Transition:

		# Values from Kovacevic et al. 2010
		TRANSITION_DICT = {
			'F': {
					'range_min': 4472,
					'range_max': 5147,
					'lam2': 4549.474,
					'gf2': 1.10e-02,
					'e1': 8.896255e-19,
				 },
			'S': {
					'range_min': 4731,
					'range_max': 5285,
					'lam2': 5018.440,
					'gf2': 3.98e-02,
					'e1': 8.589111e-19,
				 },
			'G': {
					'range_min': 4472,
					'range_max': 5147,
					'lam2': 5316.615,
					'gf2': 1.17e-02,
					'e1': 8.786549e-19,
				 },
			'Z': {
					'range_min': 4418,
					'range_max': 5428,
				 },
		}

		def __init__(self, name):
			self.name = name
			self.__dict__.update(self.TRANSITION_DICT[self.name])

			self.data_path = None
			self.df = None
			self.fft = None
			self.npad = None
			self.conv_temp = None

			self.wavelength = None
			self.gf = None
			self.e2 = None
			self.rel_int = None

			self.feii_amp = None
			self.template = None


		def read_data(self, data_path):
			self.data_path = data_path
			self.df = pd.read_csv(self.data_path)

			self.wavelength = self.df['wavelength'].to_numpy()

			if self.name == 'Z':
				self.rel_int = self.df['rel_int'].to_numpy()
			else:
				self.gf = self.df['gf'].to_numpy()
				self.e2 = self.df['E2'].to_numpy()


		def calc_rel_int(self, temp):
			"""
			Calculate relative intensities for the S, F, and G FeII line groups
			from Kovacevic et al. 2010 template as a function a temperature.
			"""
			self.rel_int = self.feii_amp*(self.lam2/self.wavelength)**3 * (self.gf/self.gf2) \
							* np.exp(-1.0/(consts.k*temp) * (self.e2 - self.e1))


	@classmethod
	def initialize_template(cls, ctx):
		return cls(ctx)


	def __init__(self, ctx):
		super().__init__(ctx)

		# The procedure for the K10 templates is slightly difference since their relative intensities
		# are temperature dependent.  We must create a Gaussian emission line for each individual line, 
		# and store them as an array, for each of the F, S, G, and Z transitions.  We treat each transition
		# as a group of templates, which will be convolved together, but relative intensities will be calculated
		# for separately. 

		def gaussian_angstroms(x, center, amp, disp, voff):
			x = x.reshape((len(x),1))
			g = amp*np.exp(-0.5*(x-(center))**2/(disp)**2) # construct gaussian
			g = np.sum(g,axis=1)
			# Replace the ends with the same value 
			g[0]  = g[1]
			g[-1] = g[-2]
			return g

		k10_data_dir = consts.BADASS_DATA_DIR.joinpath('feii_templates', 'kovacevic_2010')

		self.transitions = {name:self.Transition(name) for name in self.Transition.TRANSITION_DICT.keys()}
		for trans in self.transitions.values():
			trans.read_data(k10_data_dir.joinpath('K10_%s_transitions.csv' % trans.name))

		# Generate a high-resolution wavelength scale that is universal to all transitions
		fwhm = 1.0 # Angstroms
		disp = fwhm/2.3548
		dlam_feii = 0.1 # linear spacing in Angstroms
		npad = 100
		lam_feii = np.arange(np.min(self.ctx.wave)-npad, np.max(self.ctx.wave)+npad, dlam_feii)
		lamRange_feii = [np.min(lam_feii), np.max(lam_feii)]
		# Get size of output log-rebinned spectrum 
		ga = gaussian_angstroms(lam_feii, self.transitions[0].wavelength[0], 1.0, disp, 0.0)   
		new_size, loglam_feii, velscale_feii = log_rebin(lamRange_feii, ga, velscale=velscale)

		for trans in self.transitions.values():
			# Create storage arrays for each emission line of each transition
			templates = np.empty((len(new_size), len(trans.wavelength)))

			# Generate templates with an amplitude of 1.0
			for i in range(np.shape(trans.templates)[1]):
				ga = gaussian_angstroms(lam_feii, trans.wavelength[i], 1.0, disp, 0.0)	
				new_temp = log_rebin(lamRange_feii, ga, velscale=self.ctx.velscale)[0]
				templates[:,i] = new_temp/np.max(new_temp)

			# Pre-compute the FFT for each transition
			trans.fft, trans.npad = template_rfft(templates)

		self.npad = self.transitions[0].npad
		self.vsyst = np.log(lam_feii[0]/self.ctx.wave[0]) * consts.c

		# If opt_disp_const AND opt_voff_const, we preconvolve the templates so we don't have to during the fit
		opt_feii_options = self.ctx.options.opt_feii_options
		self.pre_convolve = (opt_feii_options.opt_disp_const.bool) and (opt_feii_options.opt_voff_const.bool)
		if self.pre_convolve:

			feii_voff = opt_feii_options.opt_voff_const.opt_feii_val
			feii_disp = opt_feii_options.opt_disp_const.opt_feii_val

			for trans in self.transitions.values():
				trans.conv_temp = convolve(trans.fft, feii_voff, feii_disp, npad=trans.npad)


	def initialize_parameters(self, params):
		# TODO: implement
		return params


	def add_components(self, params, comp_dict, host_model):
		# f_template, s_template, g_template, z_template = K10_opt_feii_template(p, lam_gal, opt_feii_templates, opt_feii_options, velscale)

		opt_feii_options = self.ctx.options.opt_feii_options
		# TODO: would this option ever change? ie. if amp is const, just set in init
		amp_const = opt_feii_options.opt_amp_const.bool
		for trans in self.transitions.values():
			trans.feii_amp = opt_feii_options.opt_amp_const[trans.name.lower()+'_feii_val'] if amp_const else params['OPT_FEII_%s_AMP'%trans.name]

		val = lambda ok, ov, pk : opt_feii_options[ok][ov] if opt_feii_options[ok].bool else params[pk]

		opt_feii_disp = val('opt_disp_const', 'opt_feii_val', 'OPT_FEII_DISP')
		opt_feii_voff = val('opt_voff_const', 'opt_feii_val', 'OPT_FEII_VOFF')
		opt_feii_temp = val('opt_temp_const', 'opt_feii_val', 'OPT_FEII_TEMP')

		for trans in self.transitions.values():
			if not self.pre_convolve:
				# Perform the convolution
				# TODO: set npad for each transition?
				trans.conv_temp = self.convolve(trans.fft, opt_feii_voff, opt_feii_disp)

			# TODO: if we do pre-convolve do we need to do this here? Or can we do this once in init?
			# Normalize amplitudes to 1
			norm = np.array([np.max(trans.conv_temp[:,i]) for i in range(np.shape(trans.conv_temp)[1])])
			norm[norm<1.e-6] = 1.0
			trans.conv_temp = trans.conv_temp/norm

			# Calculate temperature dependent relative intensities
			# TODO: if temp is constant, do this in init?
			if trans.name != 'Z': # relative intensity set for Z in initialization
				trans.calc_rel_int(opt_feii_temp)

			# Multiply by relative intensities
			trans.conv_temp *= trans.rel_int

			# Sum templates along rows
			trans.template = np.sum(trans.conv_temp, axis=1)
			# TODO: should this be an '|'?
			trans.template[(self.ctx.wave < trans.range_min) & (self.ctx.wave > trans.range_max)] = 0

			comp_dict[trans.name+'_OPT_FEII_TEMPLATE'] = trans.template
			host_model -= trans.template

		return comp_dict, host_model

