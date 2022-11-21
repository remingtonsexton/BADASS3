from astropy.io import fits
import numpy as np

import utils.constants as consts
from utils.templates.common import BadassTemplate, convolve_gauss_hermite, gaussian_filter1d, nnls, template_rfft
from utils.utils import log_rebin

HOST_GAL_TEMP_WAVE_MIN = 1680.2
HOST_GAL_TEMP_AGE_MIN = 0.09

class HostTemplate(BadassTemplate):

	temp_file = None
	def get_host_template_file(age):
		temp_file_fmt = 'Eku1.30Zp0.06T{:0>7.4f}_iTp0.00_baseFe_linear_FWHM_variable.fits'
		return consts.BADASS_DATA_DIR.joinpath('eMILES', temp_file_fmt.format(age))


	@classmethod
	def initialize_template(cls, ctx):
		if not ctx.options.comp_options.fit_host:
			return None

		if ctx.wave[0] < HOST_GAL_TEMP_WAVE_MIN:
			ctx.options.comp_options.fit_host = False
			ctx.warn('Host galaxy SSP template disabled because template is outside of fitting region.')
			return None

		HostTemplate.temp_file = HostTemplate.get_host_template_file(HOST_GAL_TEMP_AGE_MIN)
		if not HostTemplate.temp_file.exists():
			ctx.log.error('Could not find host galaxy template file: %s' % str(HostTemplate.temp_file))
			return None

		return cls(ctx)


	def __init__(self, ctx):

		self.ctx = ctx

		self.ssp_fft = None
		self.npad = None
		self.vsyst = None
		self.conv_host = None

		host_options = self.ctx.options.host_options
		fwhm_temp = consts.LOSVD_LIBRARIES.eMILES.fwhm_temp # FWHM resolution of eMILES in Å
		disp_temp = fwhm_temp/2.3548

		hdu = fits.open(self.temp_file)
		ssp = hdu[0].data 
		h = hdu[0].header
		hdu.close()

		lam_temp = np.array(h['CRVAL1'] + h['CDELT1']*np.arange(h['NAXIS1']))
		mask = ((lam_temp>=(self.ctx.wave[0]-100.0)) & (lam_temp<=(self.ctx.wave[-1]+100.0)))
		# Apply mask and get lamRange
		ssp = ssp[mask]
		lam_temp = lam_temp[mask]
		lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]

		# Variable sigma
		disp_res_interp = np.interp(lam_temp, self.ctx.wave, self.ctx.disp_res)
		disp_dif = np.sqrt((disp_res_interp**2 - disp_temp**2).clip(0))
		sigma = disp_dif/2.355/h['CDELT1'] # Sigma difference in pixels

		sspNew = log_rebin(lamRange_temp, ssp, velscale=self.ctx.velscale)[0]
		templates = np.empty((sspNew.size, len(host_options.age)))
		for j, age in enumerate(host_options.age):
			atemp = HostTemplate.get_host_template_file(age)
			if not atemp.exists():
				self.ctx.log.error('Could not find host galaxy template file: %s' % str(atemp))
				continue

			hdu = fits.open(atemp)
			ssp = hdu[0].data
			ssp = ssp[mask]
			ssp = gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
			sspNew,loglam_temp,velscale_temp = log_rebin(lamRange_temp, ssp, velscale=self.ctx.velscale)
			templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
			hdu.close()

		self.vsyst = np.log(lam_temp[0]/self.ctx.wave[0]) * consts.c
		self.ssp_fft, self.npad = template_rfft(templates)

		self.pre_convolve = (host_options.vel_const.bool) and (host_options.disp_const.bool)
		if self.pre_convolve:
			host_vel = host_options.vel_const.val
			host_disp = host_options.disp_const.val

			self.conv_host = convolve_gauss_hermite(self.ssp_fft, self.npad, float(self.ctx.velscale), \
						   [host_vel, host_disp], np.shape(self.ctx.wave)[0], vsyst=self.vsyst)


	def initialize_parameters(self, params):
		# TODO: implement
		return params


	def add_components(self, params, comp_dict, host_model):
		if not self.pre_convolve:
			host_options = self.ctx.options.host_options
			val = lambda ok, ov, pk : host_options[ok][ov] if host_options[ok].bool else params[pk]

			host_vel = val('vel_const', 'val', 'HOST_TEMP_VEL')
			host_disp = val('disp_const', 'val', 'HOST_TEMP_DISP')

			self.conv_host = convolve_gauss_hermite(self.ssp_fft, self.npad, float(self.ctx.velscale), \
						   [host_vel, host_disp], np.shape(self.ctx.wave)[0], vsyst=self.vsyst)


		if np.shape(self.conv_host)[1] == 1 :
			host_galaxy = (self.conv_host * params['HOST_TEMP_AMP']).reshape(-1)
		elif np.shape(self.conv_host)[1] > 1:
			host_model[~np.isfinite(host_model)] = 0
			self.conv_host[~np.isfinite(self.conv_host)] = 0
			# scipy.optimize Non-negative Least Squares
			weights	= nnls(self.conv_host, host_model)
			host_galaxy = (np.sum(weights*self.conv_host, axis=1))


		comp_dict['HOST_GALAXY'] = host_galaxy
		# Subtract off continuum from galaxy, since we only want template weights to be fit
		host_model -= host_galaxy
		return comp_dict, host_model

