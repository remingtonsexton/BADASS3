from astropy.io import fits
import natsort
import numpy as np

from utils.templates.common import BadassTemplate, convolve_gauss_hermite, gaussian_filter1d, nnls, template_rfft
import utils.constants as consts
from utils.utils import log_rebin


class StellarTemplate(BadassTemplate):

	temp_dir = None

	@classmethod
	def initialize_template(cls, ctx):
		if not ctx.options.comp_options.fit_losvd:
			return None

		StellarTemplate.temp_dir = consts.BADASS_DATA_DIR.joinpath(ctx.options.losvd_options.library)
		if not StellarTemplate.temp_dir.exists():
			ctx.log.error('Unable to find directory for stellar templates: %s' % str(StellarTemplate.temp_dir))
			return None

		return cls(ctx)


	def __init__(self, ctx):
		"""
		Prepares stellar templates for convolution using pPXF. 
		This example is from Capellari's pPXF examples, the code 
		for which can be found here: https://www-astro.physics.ox.ac.uk/~mxc/.
		"""

		self.ctx = ctx

		self.temp_fft = None
		self.npad = None
		self.vsyst = None
		self.conv_temp = None

		losvd_options = self.ctx.options.losvd_options
		fwhm_temp = consts.LOSVD_LIBRARIES[losvd_options.library].fwhm_temp

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
		temp_list = natsort.natsorted(list(self.temp_dir.glob('*.fits')), key=str)

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
		mask_temp = ( (lam_temp > (ctx.fit_reg.min-100.)) & (lam_temp < (ctx.fit_reg.max+100.)) )
		ssp = ssp[mask_temp]
		lam_temp = lam_temp[mask_temp]

		lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]

		# Interpolates the galaxy spectral resolution at the location of every pixel
		# of the templates. Outside the range of the galaxy spectrum the resolution
		# will be extrapolated, but this is irrelevant as those pixels cannot be
		# used in the fit anyway.
		if isinstance(ctx.fwhm_res, (list,np.ndarray)):
			fwhm_gal_interp = np.interp(lam_temp, ctx.wave, ctx.fwhm_res)
		elif isinstance(ctx.fwhm_res, (int,float)):
			fwhm_gal_interp = np.full_like(lam_temp, ctx.fwhm_res)

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
		fwhm_dif = np.sqrt((fwhm_gal_interp**2 - fwhm_temp**2).clip(0))
		sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

		sspNew = log_rebin(lamRange_temp, ssp, velscale=ctx.velscale)[0]
		templates = np.empty((sspNew.size, len(temp_list)))
		for j, fname in enumerate(temp_list):
			hdu = fits.open(fname)
			ssp = hdu[0].data
			ssp = ssp[mask_temp]
			ssp = gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
			sspNew,loglam_temp,velscale_temp = log_rebin(lamRange_temp, ssp, velscale=ctx.velscale)#[0]
			templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
			hdu.close()
		
		# The galaxy and the template spectra do not have the same starting wavelength.
		# For this reason an extra velocity shift DV has to be applied to the template
		# to fit the galaxy spectrum. We remove this artificial shift by using the
		# keyword VSYST in the call to PPXF below, so that all velocities are
		# measured with respect to DV. This assume the redshift is negligible.
		# In the case of a high-redshift galaxy one should de-redshift its
		# wavelength to the rest frame before using the line below (see above).
		self.vsyst = np.log(lam_temp[0]/ctx.wave[0]) * consts.c

		npix = ctx.spec.shape[0] # number of output pixels
		ntemp = np.shape(templates)[1] # number of templates
		
		# Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
		self.temp_fft, self.npad = template_rfft(templates)

		# If vel_const AND disp_const are True, there is no need to convolve during the 
		# fit, so we perform the convolution here and pass the convolved templates to fit_model.
		self.pre_convolve = (losvd_options.vel_const.bool) and (losvd_options.disp_const.bool)
		if self.pre_convolve:
			stel_vel = losvd_options.vel_const.val
			stel_disp = losvd_options.disp_const.val

			self.conv_temp = convolve_gauss_hermite(self.temp_fft, self.npad, float(self.ctx.velscale), \
						   [stel_vel, stel_disp], np.shape(self.ctx.wave)[0], vsyst=self.vsyst)


	def initialize_parameters(self, params):
		# TODO: implement
		return params


	def add_components(self, params, comp_dict, host_model):
		losvd_options = self.ctx.options.losvd_options
		if not self.pre_convolve:
			stel_vel = losvd_options.vel_const.val if losvd_options.vel_const.bool \
					   else params['STEL_VEL']
			stel_disp = losvd_options.disp_const.val if losvd_options.disp_const.bool \
						else params['STEL_DISP']

			self.conv_temp = convolve_gauss_hermite(self.temp_fft, self.npad, float(self.ctx.velscale), \
						   [stel_vel, stel_disp], np.shape(self.ctx.wave)[0], vsyst=self.vsyst)


		host_model[~np.isfinite(host_model)] = 0
		self.conv_temp[~np.isfinite(self.conv_temp)] = 0
		# scipy.optimize Non-negative Least Squares
		weights	 = nnls(self.conv_temp, host_model)
		host_galaxy = (np.sum(weights*self.conv_temp, axis=1))

		if (losvd_options.losvd_apoly.bool) and (losvd_options.losvd_apoly.order >= 1):
			nw = np.linspace(-1,1,len(self.ctx.wave))
			coeff = np.empty(losvd_options.losvd_apoly.order+1)
			for n in range(losvd_options.losvd_apoly.order+1):
				coeff[n] = params['LOSVD_LCOEFF_%d' % n]
			lpoly = np.polynomial.legendre.legval(nw, coeff)
			host_galaxy += lpoly 

			if np.any(host_galaxy < 0):
				host_galaxy += -np.min(host_galaxy)

		host_model = (host_model) - (host_galaxy) # Subtract off continuum from galaxy, since we only want template weights to be fit
		comp_dict['HOST_GALAXY'] = host_galaxy
		return comp_dict, host_model

