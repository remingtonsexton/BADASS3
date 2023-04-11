#!/usr/bin/env python

import numpy as np
import pandas as pd
# Plotting Libraries
import matplotlib.pyplot as plt
plt.style.use('dark_background') # For cool tron-style dark plots
plt.rcParams['text.usetex'] = True
# Basic Libraries
import sys
import os
import natsort
import glob
import copy
# Astropy
from astropy.io import fits
import scipy.optimize as op
from scipy.ndimage import generic_filter


def continuum_subtract(wave,flux,noise,sigma_clip=3.0,clip_iter=25,filter_size=[25,50,100,150,200,250,500],
                      noise_scale=1.0,opt_rchi2=True,plot=True,fig_scale=8,fontsize=16,verbose=True,
                      ):
    """
    This function performs a first-order continuum subtraction of the spectrum using 
    a series of median filters ranging from narrow to broad bandwidths, while also 
    using sigma clipping for a default threshold of 3.0.  It works well with both 
    small and large fitting regions, and with a number of types of objects.  It 
    does poorly when large fraction of the fitting region is occupied with strong
    metal absorption features.
    """
    #

    ############################################################################################################################################

    def rchi2_optimize(flux,model,noise,verbose=False):
        """
        Performs optimization to achieve 
        a reduced chi-squared of 1.0.
        """
        # Optimization function for reduced chi-squared
        def f(n,flux,model,noise,nu):
    #         flux,model,noise,nu = theta
            rchi2 = np.nansum((flux-model)**2/(n*noise)**2)/nu
            return np.abs(rchi2-1)
        # Run the optimizer
        nu = len(flux)# deg of freedom
        init = np.nansum((flux-model)**2/noise**2)/nu
        noise_scale = op.fmin(f,1.0,args=(flux,model,noise,nu,),disp=verbose)
        #
        if verbose:
            print(" Best fit noise scaling factor %0.2f" % noise_scale)
        #
        return noise_scale[0]

    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.
    
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

    def plot_cont_sub():
        # Determine 
        masked_vals = np.where(mask/mask!=1)[0]
        x_ = np.arange(len(flux))
                
        # Plot
        fig = plt.figure(figsize=(fig_scale*2,fig_scale*1))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        #
        ax1.step(wave,flux,linestyle="-",linewidth=0.5,label=r"$\textrm{Data}$")
        ax1.step(wave,masked_flux,linestyle="-",linewidth=0.5,label=r"$\textrm{Masked}$")
        ax1.step(wave,noise,linestyle="-",linewidth=0.5,label=r"$\textrm{Noise}$")
        ax1.step(wave,smoothed,linestyle="-",linewidth=1,color="xkcd:bright orange",label=r"$\textrm{Median Filter}$")
        ax2.step(wave,resid,linestyle="-",linewidth=0.5,color="xkcd:white",label=r"$\textrm{Residuals}$")
        # Noise intervals
#         ax2.fill_between(wave,-sigma_clip*np.nanmedian(noise)*noise_scale,sigma_clip*np.nanmedian(noise)*noise_scale,color="xkcd:bright red",alpha=0.5)
        # Masked pixels
        for m in masked_vals:
            try:
                lower, upper = wave[m], wave[m+1]
            except:
                lower, upper = wave[m], wave[-1]+1
            ax1.axvspan(lower,upper,color="xkcd:bright green",alpha=0.15)
        #
        ax1.axhline(0.0,color="xkcd:white",linestyle="--",linewidth=1)
        ax2.axhline(0.0,color="xkcd:white",linestyle="--",linewidth=1)
        ax1.set_xlim(wave.min(),wave.max())
        ax2.set_xlim(wave.min(),wave.max())
        plt.suptitle(r"$\textrm{sigma clip iteration %d}$" % (i),fontsize=fontsize+4)
        ax1.set_xlabel(r"$\lambda_\textrm{rest}~(\textrm{\AA})$",fontsize=fontsize)
        ax1.set_ylabel(r"$f_\lambda~(10^{-17}~\textrm{erg}~\textrm{cm}^{-2}~\textrm{s}^{-1}~\textrm{\AA}^{-1})$",fontsize=fontsize)
        ax1.tick_params(axis='both', labelsize=fontsize-4)
        ax1.legend(loc="best",fontsize=fontsize-4)
        ax2.set_xlabel(r"$\lambda_\textrm{rest}~(\textrm{\AA})$",fontsize=fontsize)
        ax2.set_ylabel(r"$f_\lambda~(10^{-17}~\textrm{erg}~\textrm{cm}^{-2}~\textrm{s}^{-1}~\textrm{\AA}^{-1})$",fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize-4)
        ax2.legend(loc="best",fontsize=fontsize-4)
        plt.tight_layout()
        return
        
    ############################################################################################################################################

    mask = np.ones(len(flux))
    clip_sum = None
    # sigma clipping iterations
    for i in range(clip_iter):
        # Apply mask
        masked_flux = flux*mask
        # Perform median smoothing
        # scipy's median filter doesn't respect masked arrays so we use a generic filter and pass it numpy's nanmedian()
        if isinstance(filter_size,(int,float)):
            smoothed = generic_filter(masked_flux,function=np.nanmedian,size=filter_size,mode="mirror")
            # Interpolate over nans
            nans, x= nan_helper(smoothed)
            smoothed[nans]= np.interp(x(nans), x(~nans), smoothed[~nans])
        if isinstance(filter_size,(list,tuple)):
            # Storage array for all 
            smoothed_arr = np.empty((len(filter_size),len(flux)))
            for j,f in enumerate(filter_size):
    #             print(j,f)
                smoothedf = generic_filter(masked_flux,function=np.nanmedian,size=f,mode="mirror")
                # Interpolate over nans
                nans, x= nan_helper(smoothedf)
                smoothedf[nans]= np.interp(x(nans), x(~nans), smoothedf[~nans])
                smoothed_arr[j,:] = smoothedf
    #         smoothed = np.nanmin(smoothed_arr,axis=0)
    #         smoothed = np.nanmean(smoothed_arr,axis=0)
            smoothed = np.nanmedian(smoothed_arr,axis=0)
        
    
    
        # Calculate residuals
        resid    = flux-smoothed 
    
        # Perform optimization on the noise scaling factor to acheive
        # reduced chi-squared of 1.0 on 
        if opt_rchi2:
            noise_scale = rchi2_optimize(flux,smoothed,noise*sigma_clip,verbose=False)
        else: 
            noise_scale =1
    
        # mask to be iteratively updated
        mask = np.ones(len(flux))
    #     print(noise_scale,np.nanmedian(noise),np.nanmedian(noise)*noise_scale)
        mask[np.where((resid <= -sigma_clip*np.nanmedian(noise)*noise_scale) | (resid >= sigma_clip*np.nanmedian(noise)*noise_scale))[0]] = np.nan
    #     mask[np.where((resid <= -sigma_clip*noise) | (resid >= sigma_clip*noise))[0]] = np.nan
        
        # Check to see if any new 
        if len(np.where(mask/mask!=1)[0])==clip_sum:
            if verbose:
                print("\t sigma clipping successfully stopped at %d iterations" % (i))
            if plot:
                plot_cont_sub()
            break    
        if i+1==clip_iter:
            if plot:
                plot_cont_sub()
    
        # Update clip_sum
        clip_sum = len(np.where(mask/mask!=1)[0])
        if verbose:
            print(" sigma clip iteration %d out of %d (%s clipped pixels)" % (i+1, clip_iter, clip_sum))
        #
    return resid