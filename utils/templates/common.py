from astropy.io import fits
import natsort
import numpy as np
import pandas as pd
from scipy import fftpack, optimize

import utils.constants as consts
from utils.utils import log_rebin, rebin


#### Common template utils ####

def template_rfft(templates):
	npix_temp = templates.shape[0]
	templates = templates.reshape(npix_temp, -1)
	npad = fftpack.next_fast_len(npix_temp)
	templates_rfft = np.fft.rfft(templates, npad, axis=0)
	
	return templates_rfft,npad


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


# TODO: make BadassTemplate instance function
# 		see OpticalFeIITemplate.convolve
def convolve_gauss_hermite(templates_rfft, npad, velscale, start, npix,
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
	start = np.array(start,dtype=float)  # make copy
	start[:2] /= velscale
	vsyst /= velscale

	lvd_rfft = losvd_rfft(start, 1, start.shape, templates_rfft.shape[0],
						  1, vsyst, velscale_ratio, sigma_diff)

	conv_temp = np.fft.irfft(templates_rfft*lvd_rfft[:, 0], npad, axis=0)
	conv_temp = rebin(conv_temp[:npix*velscale_ratio, :], velscale_ratio)

	return conv_temp


# MODIFICATION HISTORY:
#   V1.0.0: Written as a replacement for the Scipy routine with the same name,
#	   to be used with variable sigma per pixel. MC, Oxford, 10 October 2015
# (from Cappellari 2017)
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

	if isinstance(sig,(int,float)):
        sig = np.full_like(spec,float(sig))

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


def nnls(A, b, npoly=0):
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



class BadassTemplate:
	def __init__(self, ctx):
		self.ctx = ctx


	@classmethod
	def initialize_template(cls, ctx):
		return None


	def initialize_parameters(self, params):
		return params


	def add_components(self, params, comp_dict, host_model):
		return comp_dict, host_model



def initialize_templates(ctx):
	# TODO: eventually make templates a list that BADASS
	# 		can iterate over as needed
	#		Is there a reason it *really* needs to know
	# 		which template is which?
	templates = {}

	from utils.templates.host import HostTemplate
	from utils.templates.stellar import StellarTemplate
	from utils.templates.optical_feii import OpticalFeIITemplate
	from utils.templates.uv_iron import UVIronTemplate
    from utils.templates.balmer import BalmerTemplate

    for temp_class in [HostTemplate, StellarTemplate, OpticalFeIITemplate, UVIronTemplate, BalmerTemplate]:
		if temp:
			templates[temp_class.__name__] = temp

	return templates

