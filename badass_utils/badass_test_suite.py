#!/usr/bin/env python

import numpy as np
import pandas as pd
# Plotting Libraries
import matplotlib.pyplot as plt
plt.style.use('dark_background') # For cool tron-style dark plots
plt.rcParams['text.usetex'] = True
import matplotlib.gridspec as gridspec
# Basic Libraries
import sys
import os
import natsort
import glob
import copy
# Astropy
from astropy.io import fits
import scipy.optimize as op
from scipy import stats
from scipy.stats import f, chisquare
from scipy.ndimage import generic_filter,gaussian_filter1d

##################################################################################

def r_squared(data,model):
	"""
	Simple calculation of R-squared
	statistic for a single fit.
	"""
	# Calculate residual sum-of-squares (RSS)
	rss = np.nansum((data-model)**2)
	# Calculate total sum-of-squares (TSS)
	tss = np.nansum((data)**2)
	return 1-rss/tss

##################################################################################

def r_chi_squared(data,model,noise,npar):
	"""
	Simple calculation of reduced Chi-squared
	statistic for a single fit.
	"""
	# Degrees of freedon (number of data minus free fitted parameters0)
	nu = len(data)-npar
	rchi2 = np.nansum((data-model)**2/noise**2)/nu
	return rchi2

##################################################################################

def root_mean_squared_error(data,model):
	"""
	Simple calculation of root mean squared error (RMSE)
	statistic for a single fit.
	"""
	# Normalize by subtracting by the median of the data
	data_med = np.nanmedian(data)
	data  /= data_med
	model /= data_med
	return np.sqrt(len(data) * np.nansum((data-model)**2))

##################################################################################

def mean_abs_error(data,model):
	"""
	Simple calculation of mean absolute error (MAE)
	statistic for a single fit.
	"""
	# Normalize by subtracting by the median of the data
	data_med = np.nanmedian(data)
	data  /= data_med
	model /= data_med
	return len(data) * np.nansum(np.abs(data-model))

##################################################################################




def ssr_test(resid_B,
			 resid_A,
			 run_dir):
	"""
	Sum-of-Squares of Residuals test:
	The sum-of-squares of the residuals of the simple model (A)
	and the sum-of-squares of the residuals of complex model (B) for each iteration
	of the test. 
	"""

	# Compute median and std of residual standard deviations
	ssr_resid_outflow		= np.sum(resid_B**2)
	ssr_resid_no_outflow	 = np.sum(resid_A**2)
	ssr_ratio = (ssr_resid_no_outflow)/(ssr_resid_outflow) # sum-of-squares ratio
	ssr_outflow = ssr_resid_outflow
	ssr_no_outflow = ssr_resid_no_outflow
	return ssr_ratio, ssr_no_outflow, ssr_outflow


##################################################################################


def anova_test(resid_B,
		   resid_A,
		   k_A,
		   k_B,
		   run_dir):
	"""
	f-test:
	Perform an f-statistic for model comparison between a single and double-component
	model for the [OIII] line.  The f_oneway test is only accurate for normally-distributed 
	values and should be compared against the Kruskal-Wallis test (non-normal distributions),
	as well as the Bartlett and Levene variance tests.  We use the sum-of-squares of residuals
	for each model for the test. 
	"""
	# k_A = 3.0 # simpler model; single-Gaussian deg. of freedom
	# k_B = 6.0 # (nested) complex model; double-Gaussian model deg. of freedom

	RSS1 = np.sum(resid_A**2) # resid. sum of squares single_Gaussian
	RSS2 = np.sum(resid_B**2)	# resid. sum of squares double-Gaussian

	n = float(len(resid_B))
	dfn = k_B - k_A # deg. of freedom numerator
	dfd = n - k_B  # deg. of freedom denominator

	f_stat = ((RSS1-RSS2)/(k_B-k_A))/((RSS2)/(n-k_B)) 
	f_pval = 1 - f.cdf(f_stat, dfn, dfd)

	# print('f-statistic model comparison = %0.2f +/- %0.2f, p-value = %0.2e +/- %0.2f' % (np.median(f_stat), np.std(f_stat),np.median(f_pval), np.std(f_pval) ))
	# print('f-statistic model comparison = %0.2f ' % (f_stat))

	outflow_conf = 1.0-(f_pval)
	return f_stat, f_pval, outflow_conf


##################################################################################

def f_ratio(resid_B, resid_A):
	"""
	The F-ratio is defined as the ratio in variances
	between two sets of data (residuals)
	"""
	return np.nanstd(resid_A)/np.nanstd(resid_B)


##################################################################################

def chi2_metric(eval_ind, 
				mccomps_B, 
				mccomps_A):
	# B 
	f_obs = mccomps_B["DATA"][0,:][eval_ind]/np.sum(mccomps_B["DATA"][0,:][eval_ind])
	f_exp = mccomps_B["MODEL"][0,:][eval_ind]/np.sum(mccomps_B["MODEL"][0,:][eval_ind])
	chi2_B, pval_B = chisquare(f_obs=f_obs,f_exp=f_exp)

	# A 
	f_obs = mccomps_A["DATA"][0,:][eval_ind]/np.sum(mccomps_A["DATA"][0,:][eval_ind])
	f_exp = mccomps_A["MODEL"][0,:][eval_ind]/np.sum(mccomps_A["MODEL"][0,:][eval_ind])
	chi2_A, pval_A = chisquare(f_obs=f_obs,f_exp=f_exp)

	# Calculate Ratio
	# The ratio of chi-squared values is defined as the improvement of the outflow model over the no-outflow model,
	# i.e., 1.0-(chi2_B/chi2_no_outflow)
	chi2_ratio = 1.0-(chi2_B/chi2_A)

	return chi2_B, chi2_A, chi2_ratio


##################################################################################


def normal_log_likelihood(data,model,sigma):
	"""
	A simple normal log-likelihood for data, model, 
	and noise.
	"""
	ll = -0.5*np.sum((data-model)**2/sigma**2 + np.log(2*np.pi*sigma**2))
	return ll


##################################################################################


def calculate_BIC(mccomps_A, mccomps_B, k_A, k_B):
	"""
	Calculates the Bayesian information criterion (BIC)
	for two models, and outputs the ratio of the two.
	"""

	# Unpack the likelihood parameters
	data_A, model_A, noise_A = mccomps_A["DATA"], mccomps_A["MODEL"], mccomps_A["NOISE"] 
	data_B, model_B, noise_B = mccomps_B["DATA"], mccomps_B["MODEL"], mccomps_B["NOISE"] 
	ll_A = normal_log_likelihood(data_A, model_A, noise_A)
	ll_B = normal_log_likelihood(data_B, model_B, noise_B)
	# print(ll_A,ll_B)

	bic_A = -2*ll_A+k_A*np.log(len(data_A))
	bic_B = -2*ll_B+k_B*np.log(len(data_B))
	bic_ratio = bic_B/bic_A
	# delta_bic = bic_A - bic_B

	return bic_A, bic_B, bic_ratio


##################################################################################


def calculate_AIC(mccomps_A, mccomps_B, k_A, k_B):
	"""
	Calculates the Akaike information criterion (BIC)
	for two models, and outputs the ratio of the two.
	"""
	# Unpack the likelihood parameters
	data_A, model_A, noise_A = mccomps_A["DATA"], mccomps_A["MODEL"], mccomps_A["NOISE"] 
	data_B, model_B, noise_B = mccomps_B["DATA"], mccomps_B["MODEL"], mccomps_B["NOISE"] 
	ll_A = normal_log_likelihood(data_A, model_A, noise_A)
	ll_B = normal_log_likelihood(data_B, model_B, noise_B)
	# print(ll_A,ll_B)

	aic_A = -2*ll_A+2*(k_A)
	aic_B = -2*ll_B+2*(k_B)
	aic_ratio = aic_B/aic_A
	# delta_aic = aic_A - aic_B

	return aic_A, aic_B, aic_ratio

##################################################################################


def calculate_rsquared_ratio(mccomps_A, mccomps_B):
	"""

	"""

	# Unpack the likelihood parameters
	data_A, model_A = mccomps_A["DATA"][0], mccomps_A["MODEL"][0]
	data_B, model_B = mccomps_B["DATA"][0], mccomps_B["MODEL"][0]
	# Since R-squared takes into account lines+continuum, we only want 
	# to be sensitive to flux that comes from lines, so we subtract
	# any contribution to the continuum from both before the calculation.
	# NOTE: this assumes that the continuum subtraction is generally good
	# for both models.
	cont_comps = ["HOST_GALAXY","POWER","APOLY","PPOLY","MPOLY","NA_OPT_FEII_TEMPLATE","BR_OPT_FEII_TEMPLATE",
				  'F_OPT_FEII_TEMPLATE','S_OPT_FEII_TEMPLATE','G_OPT_FEII_TEMPLATE','Z_OPT_FEII_TEMPLATE',
				  "UV_IRON_TEMPLATE","BALMER_CONT",
					]
	cont_model_A = np.zeros(len(data_A))
	for p in mccomps_A:
		if p in cont_comps:
			cont_model_A += mccomps_A[p][0]
	
	cont_model_B = np.zeros(len(data_B))
	for p in mccomps_B:
		if p in cont_comps:
			cont_model_B += mccomps_B[p][0]

	data_A = data_A - cont_model_A
	model_A = model_A - cont_model_A

	data_B = data_B - cont_model_B
	model_B = model_B - cont_model_B

	rsquared_A = 1 - (np.sum((data_A-model_A)**2))/(np.sum(data_A**2))
	rsquared_B = 1 - (np.sum((data_B-model_B)**2))/(np.sum(data_B**2))

	rsquared_ratio = rsquared_B/rsquared_A
	if rsquared_ratio/rsquared_ratio!=1: rsquared_ratio = 0.0
	return rsquared_A, rsquared_B, rsquared_ratio


##################################################################################


def bayesian_AB_test(resid_B, resid_A, wave, noise, data, eval_ind, ddof, run_dir, plot=False):
	"""
	Performs a Bayesian A/B hypothesis test for the 
	likelihood distributions for two models.
	"""

	# Smooth the noise using a 3-pixel Gaussian kernel
	noise = gaussian_filter1d(noise,2.0,mode="nearest")
	#
	# Sample the noise around the best-fit 
	nsamp = 10000
	resid_B_lnlike	= np.empty(nsamp)
	resid_A_lnlike = np.empty(nsamp)
	for i in range(nsamp):
		lnlike_B	= np.sum(-0.5*(np.random.normal(loc=resid_B,scale=noise[eval_ind],size=len(eval_ind)))**2/noise[eval_ind]**2)
		lnlike_A = np.sum(-0.5*(np.random.normal(loc=resid_A,scale=noise[eval_ind],size=len(eval_ind)))**2/noise[eval_ind]**2)
		resid_B_lnlike[i] = lnlike_B
		resid_A_lnlike[i] = lnlike_A

	# Penalize by degrees of freedom
	resid_B_lnlike	/= (len(data)-ddof)
	resid_A_lnlike /= (len(data))
	#
	p_B = np.percentile(resid_B_lnlike,[16,50,84])
	p_A = np.percentile(resid_A_lnlike,[16,50,84])
	#
	# The sampled log-likelihoods should be nearly Gaussian
	x			 = np.linspace(np.min([resid_B_lnlike, resid_A_lnlike]),np.max([resid_B_lnlike, resid_A_lnlike]),1000)
	norm_line	 = stats.norm(loc=p_B[1],scale=np.mean([p_B[2]-p_B[1],p_B[1]-p_B[0]]))
	norm_no_line = stats.norm(loc=p_A[1],scale=np.mean([p_A[2]-p_A[1],p_A[1]-p_A[0]]))
	#
	# Determine which distribution has the maximum likelihood.
	# Null Hypothesis, H0: B is no different than A
	# Alternative Hypothesis, H1: B is significantly different from A
	A = resid_A_lnlike # no line model
	A_mean = p_A[1]
	B = resid_B_lnlike	# line model
	ntrials = 10000
	B_samples = norm_line.rvs(size=ntrials)
	pvalues = np.array([(norm_no_line.sf(b)) for b in B_samples])*2.0
	pvalues[pvalues>1] = 1
	pvalues[pvalues<1e-6] = 0
	conf	= (1 - pvalues)
	#
	p_pval = np.percentile(pvalues,[16,50,84])
	p_conf = np.percentile(conf,[16,50,84])
	#
	d = np.abs(p_B[1] - p_A[1]) # statistical distance
	disp = np.sqrt((np.mean([p_B[2]-p_B[1],p_B[1]-p_B[0]]))**2+(np.mean([p_A[2]-p_A[1],p_A[1]-p_A[0]]))**2) # total dispersion
	signif = d/disp # significance
	overlap = np.min([(p_B[2]-p_A[0]), (p_A[2]-p_B[0])]).clip(0) # 1-sigma overlap

	if plot:


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
		# ax1.plot(wave,resid_B,color="xkcd:bright aqua",linestyle="-",linewidth=0.5,label="Resid. with Line")
		# ax1.plot(wave,resid_A,color="xkcd:bright purple",linestyle="-",linewidth=0.5,label="Resid. without Line")
		ax1.plot(wave[eval_ind],resid_A-resid_B,color="xkcd:bright red",linestyle="-",linewidth=1.0,label=r"$\Delta~\rm{Residuals}$")
		ax1.plot(wave[eval_ind],noise[eval_ind],color="xkcd:lime green",linestyle="-",linewidth=0.5,label="Noise")
		ax1.plot(wave[eval_ind],-noise[eval_ind],color="xkcd:lime green",linestyle="-",linewidth=0.5)
		ax1.axhline(0,color="xkcd:white",linestyle="--",linewidth=0.75)
		ax1.set_xlabel(r"$\lambda_{\rm{rest}}$ [$\rm{\AA}$]",fontsize=fontsize)
		ax1.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$]",fontsize=fontsize)
		ax1.set_title(r"Fitting Region Residuals",fontsize=fontsize)
		ax1.tick_params(axis='both', labelsize= fontsize)
		ax1.set_xlim(np.min(wave[eval_ind]),np.max(wave[eval_ind]))
		ax1.legend(fontsize=12)
		#
		ax2.hist(resid_B_lnlike,bins="doane",histtype="step",label="Line",density=True,color="xkcd:bright aqua",linewidth=0.5)
		ax2.axvline(p_B[1],color="xkcd:bright aqua", linestyle='--', linewidth=1,)
		ax2.axvspan(p_B[0], p_B[2], alpha=0.25, color='xkcd:bright aqua')
		ax2.plot(x,norm_line.pdf(x),color="xkcd:bright aqua",linewidth=1)
		ax2.plot(x,norm_no_line.pdf(x),color="xkcd:bright orange",linewidth=1)
		#
		ax2.hist(resid_A_lnlike,bins="doane",histtype="step",label="No Line",density=True,color="xkcd:bright orange",linewidth=0.5)
		ax2.axvline(p_A[1],color="xkcd:bright orange", linestyle='--', linewidth=1,)
		ax2.axvspan(p_A[0], p_A[2], alpha=0.25, color='xkcd:bright orange')
		ax2.set_title("Log-Likelihood",fontsize=fontsize)
		ax2.tick_params(axis='both', labelsize= fontsize)
		ax2.legend()
		#
		ax3.hist(pvalues,bins="doane",histtype="step",label="Line",density=True,color="xkcd:bright aqua",linewidth=0.5)
		ax3.axvline(p_pval[1],color="xkcd:bright aqua", linestyle='--', linewidth=1,)
		ax3.axvspan(p_pval[0], p_pval[2], alpha=0.25, color='xkcd:bright aqua')
		ax3.set_title(r"$p$-values",fontsize=fontsize)
		#	
		ax4.hist(conf,bins="doane",histtype="step",label="No Line",density=True,color="xkcd:bright aqua",linewidth=0.5)
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


def calculate_aon(test,line_list,mccomps):
	"""
	Calculates the amplitude-over-noise for the maximum of 
	all lines being tested for a given test.
	"""

	full_profile = np.zeros(len(mccomps["WAVE"][0]))
	for l in line_list:
		if (l in test) or (("parent" in line_list[l]) and (line_list[l]["parent"] in test)):
			full_profile+=mccomps[l][0]

	avg_noise = np.nanmean(mccomps["NOISE"][0])

	aon = np.nanmax(full_profile)/avg_noise

	return aon


##################################################################################


def check_test_stats(target,current,verbose=False):
	"""
	Function for checking thresholds of line
	tests.  Note, this omits the AON (amplitude-over-noise)
	test, since it is not a test between models.
	"""

	target = {t:target[t] for t in target if t in ["ANOVA","BADASS","CHI2_RATIO","F_RATIO","SSR_RATIO"]}
	current = {c:current[c] for c in current if c in ["ANOVA","BADASS","CHI2_RATIO","F_RATIO","SSR_RATIO"]}

	checked = []
	for stat in target:
		# Confidence based matrics; metric must remain above a current threshold to 
		# remain False, and becomes True once it drops below that confidence threshold.
		if current[stat]<=target[stat]:
			checked.append(True)
		else:
			checked.append(False)

	return checked


##################################################################################












