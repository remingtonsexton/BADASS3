![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_logo.gif)

Ridiculous acronyms are a long-running joke in astronomy, but here, spectral fitting ain't no joke!

BADASS is an open-source spectral analysis tool designed for detailed decomposition of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical.  The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte Carlo sampler [emcee](https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract) for robust parameter and uncertainty estimation, as well as autocorrelation analysis to access parameter chain convergence.  BADASS can fit the following spectral features:
- Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å.
- Broad and Narrow FeII emission features using the FeII templates from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract).
- Broad permitted and narrow forbidden emission line features. 
- AGN power-law continuum. 
- "Blue-wing" outflow emission components found in narrow-line emission. 

All spectral components can be turned off and on via the [Jupyter Notebook](https://jupyter.org/) interface, from which all fitting options can be easily changed to fit non-AGN-host galaxies (or even stars!).  BADASS uses multiprocessing to fit multiple spectra simultaneously depending on your hardware configuration.  The code was originally written in Python 2.7 to fit Keck Low-Resolution Imaging Spectrometer (LRIS) data ([Sexton et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)), but because BADASS is open-source and *not* written in an expensive proprietary language, one can easily contribute to or modify the code to fit data from other instruments.

Before getting started you should [read the wiki](https://github.com/remingtonsexton/BADASS3/wiki) or the readme below.

<b>  
If you use BADASS for any of your fits, I'd be interested to know what you're doing and what version of Python you are using, please let me know via email at remington.sexton-at-email.ucr.edu.
</b>

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  * [Fitting Options](#fitting-options)
- [MCMC & Autocorrelation/Convergence Options](#mcmc---autocorrelation-convergence-options)
  * [Model Options](#model-options)
  * [Outflow Testing Options](#outflow-testing-options)
  * [Plotting & Output Options](#plotting---output-options)
  * [Output Options](#output-options)
  * [Multiprocessing Options](#multiprocessing-options)
  * [The Main Function](#the-main-function)
- [Output](#output)
  * [Best-fit Model](#best-fit-model)
  * [Parameter Chains, Histograms, Best-fit Values & Uncertainties](#parameter-chains--histograms--best-fit-values---uncertainties)
  * [Log File](#log-file)
  * [Best-fit Model Components](#best-fit-model-components)
  * [Best-fit Parameters & Uncertainties](#best-fit-parameters---uncertainties)
  * [Autocorrelation Time & Tolerance History](#autocorrelation-time---tolerance-history)
- [How to...](#how-to)
- [Known Issues](#known-issues)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)



# Installation

The easiest way to get started is to simply clone the repository. 

As of BADASS v7.7.1, the following packages are required (Python 3.6.10):
- `astropy 4.0.1`
- `astroquery 0.4`
- `corner 2.0.1`
- `emcee 0.0.0`
- `ipython 7.14.0`
- `jupyter-client 6.1.3`
- `matplotlib 3.1.3`
- `natsort 7.0.1`
- `numpy 1.18.1`
- `pandas 1.0.3`
- `psutil 5.7.0`
- `scipy 1.4.1` 

The code is run entirely through the Jupyter Notebook interface, and is set up to run on the included SDSS spectrum file in the ".../examples/" folder.  If one wants to fit multiple spectra consecutively, simply add folders for each spectrum to the folder.  This is the recommended directory structure:

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_example_spectra.png)

Simply create a folder containing the SDSS FITS format spectrum, and BADASS will generate a working folder for each fit titled "MCMC_output_#" inside each spectrum folder.  BADASS automatically generates a new output folder if the same object is fit again (it does not delete previous fits).

In the Notebook, one need only specify the location of the spectra, the location of the BADASS support files, and the location of the templates for pPXF, as shown below:

```python
########################## Directory Structure #################################
spec_dir = 'examples/' # folder with spectra in it
ppxf_dir = 'badass_data_files/' # support files
temp_dir = ppxf_dir+'indo_us_library' # stellar templates
# Get full list of spectrum folders; these will be the working directories
spec_loc = natsort.natsorted( glob.glob(spec_dir+'*') )
################################################################################
```


# Usage

## Fitting Options
 
```python 
################################## Fit Options #################################
# Fitting Parameters
fit_options={
'fit_reg'    : (4400,5500), # Fitting region; Indo-US Library=(3460,9464)
'good_thresh': 0.0, # percentage of "good" pixels required in fig_reg for fit.
'interp_bad' : False, # interpolate over pixels SDSS flagged as 'bad' (careful!)
# Number of consecutive basinhopping thresholds before solution achieved
'n_basinhop': 5,
# Outflow Testing Parameters
'test_outflows': True, 
'outflow_test_niter': 10, # number of monte carlo iterations for outflows
# Maximum Likelihood Fitting for Final Model Parameters
'max_like_niter': 10, # number of maximum likelihood iterations
# LOSVD parameters
'min_sn_losvd': 5,  # minimum S/N threshold for fitting the LOSVD
# Emission line profile parameters
'line_profile':'G' # Gaussian (G) or Lorentzian (L)
}
################################################################################
```

**`fit_reg`**: *tuple/list of length (2,)*; *Default: (4400,5800) # Hb/[OIII]/Mg1b/FeII region*  
the minimum and maximum desired fitting wavelength in angstroms, for example (4400,7000).  This is passed to the `determine_fit_reg()` function to check if this region is valid and and which emission lines to fit.

**`good_thresh`**: *float [0.0,1.0]*; *Default: 0.0*  
the cutoff for minimum fraction of "good" pixels (determined by SDSS) within the fitting range to allow for fitting of a given spectrum.  If the spectrum has fewer good pixels than this value, BADASS skips over it and moves onto the next spectrum.

**`interp_bad`**: *bool*; *Default: False*
Interpolate over pixels which SDSS flagged as bad due to sky line subtraction or cosmic rays.  Warning: if large portions of the fitting region are marked as bad pixels, this can cause BADASS to crash.  One should only use this if only a few pixels are affected by contamination.  

**`n_basinhop`**: *int*; *Default: 5*
Number of successive `niter_success` times the basinhopping alogirhtm needs to achieve a solution.  The fit becomes much better with more success times, however this can increase the time to a solution by a lot.  Recommended 3-5. 

**`test_outflows`**: *bool*; *Default: True*  
if *False*, BADASS does not test for outflows and instead does whatever you tell it to. If *True*, BADASS performs maximum likelihood fitting of outflows, using monte carlo bootstrap resampling to determine uncertainties, and uses the BADASS prescription for determining the presence of outflows.  Testing for outflows requires the region from 4400 Å - 5800 Å included in the fitting region to accurately account for possible FeII emission.  This region is also required since [OIII] is used to constrain outflow parameters of the H-alpha/[NII]/[SII] outflows.  If all of the BADASS outflow criteria are satisfied, the final model includes outflow components.  The BADASS outflow criteria to justify the inclusion of outflow components are the following:

1. Amplitude metric: ![\cfrac{A_{\rm{outflow}}}{\left(\sigma^2_{\rm{noise}} + \delta A^2_{\rm{outflow}}\right)^{1/2}} > 3.0](https://render.githubusercontent.com/render/math?math=%5Ccfrac%7BA_%7B%5Crm%7Boutflow%7D%7D%7D%7B%5Cleft(%5Csigma%5E2_%7B%5Crm%7Bnoise%7D%7D%20%2B%20%5Cdelta%20A%5E2_%7B%5Crm%7Boutflow%7D%7D%5Cright)%5E%7B1%2F2%7D%7D%20%3E%203.0)
2. Width metric: ![\cfrac{\sigma_{\rm{outflow}}- \sigma_{\rm{core}}}{\left(\delta \sigma^2_{\rm{outflow}}+\delta \sigma^2_{\rm{core}}\right)^{1/2}} > 1.0](https://render.githubusercontent.com/render/math?math=%5Ccfrac%7B%5Csigma_%7B%5Crm%7Boutflow%7D%7D-%20%5Csigma_%7B%5Crm%7Bcore%7D%7D%7D%7B%5Cleft(%5Cdelta%20%5Csigma%5E2_%7B%5Crm%7Boutflow%7D%7D%2B%5Cdelta%20%5Csigma%5E2_%7B%5Crm%7Bcore%7D%7D%5Cright)%5E%7B1%2F2%7D%7D%20%3E%201.0)
3. Velocity offset metric: ![\cfrac{v_{\rm{core}}- v_{\rm{outflow}}}{\left(\delta v^2_{\rm{core}}+\delta v^2_{\rm{outflow}}\right)^{1/2}} > 1.0](https://render.githubusercontent.com/render/math?math=%5Ccfrac%7Bv_%7B%5Crm%7Bcore%7D%7D-%20v_%7B%5Crm%7Boutflow%7D%7D%7D%7B%5Cleft(%5Cdelta%20v%5E2_%7B%5Crm%7Bcore%7D%7D%2B%5Cdelta%20v%5E2_%7B%5Crm%7Boutflow%7D%7D%5Cright)%5E%7B1%2F2%7D%7D%20%3E%201.0)
4. F-statistic (model comparison): ![F-\textrm{statistic:}\quad\cfrac{\left(\cfrac{\textrm{RSS}_{\textrm{no}\;\textrm{outflow}}-\textrm{RSS}_{\textrm{outflow}}}{k_2-k_1}\right)}{\left(\cfrac{\textrm{RSS}_{\textrm{outflow}}}{N-k_2}\right)}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+F-%5Ctextrm%7Bstatistic%3A%7D%5Cquad%5Ccfrac%7B%5Cleft%28%5Ccfrac%7B%5Ctextrm%7BRSS%7D_%7B%5Ctextrm%7Bno%7D%5C%3B%5Ctextrm%7Boutflow%7D%7D-%5Ctextrm%7BRSS%7D_%7B%5Ctextrm%7Boutflow%7D%7D%7D%7Bk_2-k_1%7D%5Cright%29%7D%7B%5Cleft%28%5Ccfrac%7B%5Ctextrm%7BRSS%7D_%7B%5Ctextrm%7Boutflow%7D%7D%7D%7BN-k_2%7D%5Cright%29%7D)


5. Bounds metric: parameters are within their allowed parameter limits.

**`outflow_test_niter`**: *int*; *Default: 10*  
the number of monte carlo bootstrap simulations for outflow testing.  If set to 0, BADASS will not test for outflows.

**`max_like_niter`**: *int*; *Default: 10*  
Maximum likelihood fitting of the region defined by `fit_reg`, which can be larger than the region used for outflow testing.  Only one iteration is required, however, more iterations can be performed to obtain better initial parameter values for emcee.  If one elects to only use maximum likelihood fitting (`mcmc_fit=False`), one can perform as many `max_like_niter` iterations to obtain parameter uncertainties in the similar bootstrap method used to test for outflows.

**`min_sn_losvd`**: *int*; *Default: 10*  
minimum S/N threshold for fitting the LOSVD.  Below this threshold, BADASS does not perform template fitting with pPXF and instead uses a 5.0 Gyr SSP galaxy template as a stand-in for the stellar continuum.

**`line_profile`**: *str*; *Default: G*  
Broad line profile shape.  Narrow lines are unaffected and still fit with a Gaussian profile.  Choose 'G' for Gaussian, or 'L' for Lorentzian profile (used for NLS1 type AGNs).

# MCMC & Autocorrelation/Convergence Options

```python
########################### MCMC algorithm parameters ##########################
mcmc_options={
'mcmc_fit'    : True, # Perform robust fitting using emcee
'nwalkers'    : 100,  # Number of emcee walkers; min = 2 x N_parameters
'auto_stop'   : True, # Automatic stop using autocorrelation analysis
'conv_type'   : 'median', # 'median', 'mean', 'all', or (tuple) of parameters
'min_samp'    : 2500,  # min number of iterations for sampling post-convergence
'ncor_times'  : 5.0,  # number of autocorrelation times for convergence
'autocorr_tol': 10.0,  # percent tolerance between checking autocorr. times
'write_iter'  : 100,   # write/check autocorrelation times interval
'write_thresh': 100,   # when to start writing/checking parameters
'burn_in'     : 17500, # burn-in if max_iter is reached
'min_iter'    : 2500, # min number of iterations before stopping
'max_iter'    : 20000, # max number of MCMC iterations
}
################################################################################
```

**`mcmc_fit`**: *bool*; *Default: True*  
while it is *highly recommended* one uses MCMC for parameter estimation, we leave it as an option to turn off for faster (but less accurate) maximum likelihood estimation.  If `mcmc_fit=False`, then BADASS performs `max_like_niter` bootstrap iterations to estimate parameter values and uncertainties.  It will then output the best fit values and spectral components is FITS files.

**`nwalkers`**: *int*; *Default: 100*  
number of "walkers" per parameter used by emcee to explore each parameter space.  The minimum number of walkers is 2 x ( # of parameters), set by emcee.

**`auto_stop`**: *bool*; *Default: True*  
if set to *True*, autocorrelation is used to automatically stop the fitting process when a convergence criteria (`conv_type`) is achieved. 

**`conv_type`**: *str*; *Default: "median"*; *options: "all", "median", "mean", list of parameters*  
mode of convergence.  Convergence of 'all' ensures all fitting parameters have achieved the desired `ncor_times` and `autocorr_tol` criteria, while "median" and "mean" only ensure that `ncor_times` and `autocorr_tol` criteria have been met for the median or mean of all parameters, respectively.  A list of valid parameters is also acceptable to ensure specific parameters have achieved convergence even if others have not.  In general "median" requires the fewest number of iterations and is not sensitive to poorly-constrained parameters, and "all" and "mean" require the most number of iterations and are much more sensitive to fluctuations in calculated autocorrelation times and tolerances.  A list of parameters is suitable in cases where one is only interested in certain spectral features.

**`min_samp`**: *int*; *Default: 2500*  
if `auto_stop=True`, then the `burn_in` is the iteration at which convergence is achieved, and `min_samp` is the number of iterations *past convergence* used for posterior sampling (the samples used for histograms and estimating best-fit parameters and uncertainties.  If for some reason the parameters "jump out" of convergence, the `burn_in` will reset and BADASS will continue to sample until convergence is met again.  If emcee completes `min_samp` iterations after convergence is achieved without jumping out of convergence, this concludes the MCMC sampling.

**`ncor_times`**: *int* or *float*; *Default=10*  
The number of integrated autocorrelation times (iterations) needed for convergence.  We recommend a minimum of `ncor_times=2.0`.  In general, it will require more than 2.0 autocorrelation times to calculate the autocorrelation time for a parameter chain.  Increasing `ncor_times` ensures that the parameter chain has stopped exploring the parameter space and is ready to begin sampling for the posterior distribution. 

**`autocorr_tol`**: *int* or *float*; *Default=10*; the percent change in the current integrated autocorrelation time and the previously calculated integrated autocorrelation time.  The `write_iter` determines how often BADASS checks a parameter's integrated autocorrelation time.  In general, we find that `autocorr_tol=5` (a 5% change) is acceptable for a converged parameter chain.  A parameter chain that diverges more than 10% in 100 iterations could still be exploring the parameter space for a stable solution.  A `autocorr_tol=1` (a 1% change) typically requires many more iterations than necessary for convergence. 

**`write_iter`**: *int*; *Default=100*  
the frequency at which BADASS writes the current parameter values (median walker positions).  If `auto_stop=True`, then BADASS checks for convergence every `write_iter` iteration for convergence.

**`write_thresh`**: *int*; *Default=100*  
the iteration at which writing (and checking for convergence if `auto_stop=True`) begins.  BADASS does not check for convergence before this value.

**`burn_in`**: *int*; *Default=47500*  
if `auto_stop=False` then this serves as the burn-in for a maximum number of iterations.  If `auto_stop=True`, this value is ignored.

**`min_iter`**: *int*; *Default=100*  
the minimum number of iterations BADASS performs before it is allowed to stop.  This is true regardless of the value of `auto_stop`.

**`max_iter`**: *int*; *Default=50000*  
the maximum number of iterations BADASS performs before stopping.  This value is adhered to regardless of the value of `auto_stop` to set a limit on the number of iterations before BADASS should "give up."


## Model Options

```python
############################ Fit component options #############################
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
################################################################################
```

These options are more-or-less self explanatory.  One can fit components appropriate (or not appropriate) for the types of objects they are fitting.  We summarize each component below:

**`fit_feii`**: *Default=True*  
Broad and narrow FeII templates are taken from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) with each line modeled using a Gaussian.  FeII emission can be very strong in some Type 1 (broad line) AGN, but is almost undetectable in Type 2 (narrow line) AGN.

**`fit_losvd`**: *Default=True*  
Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å.  This is used to obtain stellar kinematics in spectra with resolvable absorption features, such as stellar velocity and dispersion.  If the S/N of the continuum (determined by the initial maximum likelihood fit) is less than 2.0, then `fit_losvd` is set to `False` by BADASS, since most of the time, trying to fit stellar features with S/N<5.0 produces non-sensical uncertainties.

**`fit_host`**: *Default=True*  
this fits a 5.0 Gyr SSP galaxy template for the maximum likelihood fit.  If it is determined that the S/N of the spectra is too low to fit the LOSVD (S/N<2), then this simple host galaxy template takes the place of the spectral template fitting.

**`fit_power`**: *Default=True*  
this fits a power-law component (steepness decreasing with increasing wavelength) to simulate the effect of the AGN continuum. 

**`fit_broad`**: *Default=True*  
broad permitted emission lines commonly seen in Type 1 AGN.  

**``*fit_narrow*: *Default=True*  
narrow forbidden emission lines seen in both Type 1 and Type 2 AGN, and some starforming galaxies. 

**`fit_outflows`**: *Default=True*  
fitting of blueshifted (or "blue wing") emission in narrow-line features, indicative of outflowing NLR gas.  If `test_outflows=True`, and if BADASS determines that the inclusion of an "outflow" component does not satisfy the outflow criterion (read above), then `fit_outflows` is overridden to `False`.

**`tie_narrow`**: *Default=False*  
tying all narrow-line widths (FWHM) together across the entire wavelength range is a common option in many fitting pipelines.  This is not recommended since different atomic families of lines can have different kinematics (and this is measurable!), however it is included as an option.

Examples of the aforementioned spectral components can be seen in the example fit below:

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_model_options.png)

## FeII Options

There are two FeII templates built into BADASS.  The default is the broad and narrow templates from Veron-Cetty et al. (2004) (VC04).  This model allows the user to have amplitude, FWHM, and velocity offset as free-parameters, with options to constrain them to constant values during the fit.  BADASS can also use the temperature-dependent template from Kovacevic et al. (2010) (K10), which allows for the fitting of indidual F, S, G, and IZw1 atomic transitions, as well as temperature.  The K10 template is best suited for modelling FeII in NLS1 objects with strong FeII emission.

```python
############################### FeII Fit options ###############################
# Below are options for fitting FeII.  For most objects, you don't need to 
# perform detailed fitting on FeII (only fit for amplitudes) use the 
# Veron-Cetty 2004 template ('VC04') (2-6 free parameters)
# However in NLS1 objects, FeII is much stronger, and sometimes more detailed 
# fitting is necessary, use the Kovacevic 2010 template 
# ('K10'; 7 free parameters).

# The options are:
# template   : VC04 (Veron-Cetty 2004) or K10 (Kovacevic 2010)
# amp_const  : constant amplitude (default False)
# fwhm_const : constant fwhm (default True)
# voff_const : constant velocity offset (default True)
# temp_const : constant temp ('K10' only)

feii_options={
'template'  :{'type':'VC04'}, 
'amp_const' :{'bool':False,'br_feii_val':1.0,'na_feii_val':1.0},
'fwhm_const':{'bool':True,'br_feii_val':3000.0,'na_feii_val':500.0},
'voff_const':{'bool':True,'br_feii_val':0.0,'na_feii_val':0.0},
}
# or
# feii_options={
# 'template'  :{'type':'K10'},
# 'amp_const' :{'bool':False,'f_feii_val':1.0,'s_feii_val':1.0,'g_feii_val':1.0,'z_feii_val':1.0},
# 'fwhm_const':{'bool':False,'val':1500.0},
# 'voff_const':{'bool':False,'val':0.0},
# 'temp_const':{'bool':False,'val':10000.0} 
# }
################################################################################
```

## Outflow Testing Options

```python
############################# Outflow Test options #############################
# Here one can choose how outflows are fit and tested for 
# Amp. test   : outflow amp. must be N-sigma greater than noise
# FWHM test   : outflow must have greater FWHM than core comp by N-sigma
# VOFF test   : outflow must have a larger offset than core relative to rest;
#               picks out only blueshifted outflows by N-sigma
# Resid. test : there must be a measurable difference in residuals by N-sigma
# Bounds. test: if paramters of fit reach bounds by N-sigma, 
#               consider it a bad fit.
outflow_test_options={
'amp_test':{'test':True,'nsigma':3.0}, # Amplitude-over-noise by n-sigma
'fwhm_test':{'test':True,'nsigma':1.0}, # FWHM difference by n-sigma
'voff_test':{'test':True,'nsigma':1.0}, # blueshift voff from core by n-sigma
'outflow_confidence':{'test':True,'conf':0.95}, # outflow confidence acceptance
'bounds_test':{'test':True,'nsigma':1.0} # within bounds by n-sigma
}
################################################################################
```

*Note*: we turn the `voff_test` option off by default to allow for a range of blueshifted to redshifted outflow components.  If turned on, this test will only select blueshifted outflow components.

## Plotting & Output Options

```python
############################### Plotting options ###############################
plot_options={
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
################################################################################
```

**`plot_param_hist`**: *Default: True*  
For each free parameter fit by emcee, BADASS outputs a figure that contains a histogram of the MCMC samples, the best-fit values and uncertainties, and a plot of the MCMC chain, as shown below:

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_chain.png)

**`plot_flux_hist`**: *Default: True*  
For each spectral component (i.e., emission lines, stellar continuum, power-law continuum, etc.), the integrated flux is calculated within the fitting region.  These fluxes are returned by emcee as metadata 'blobs' at each iteration, and have corresponding MCMC chains and sample histograms as shown above.

**`plot_lum_hist`**: *Default: True*  
For each spectral component (i.e., emission lines, stellar continuum, power-law continuum, etc.), the integrated luminosity is calculated within the fitting region.  These luminosities calculated from flux chains output by emcee.  The cosmology used to calculate luminosities from redshifts is a flat ![$\Lambda$](https://render.githubusercontent.com/render/math?math=%24%5CLambda%24)CDM model with ![$H_0=71$](https://render.githubusercontent.com/render/math?math=%24H_0%3D71%24) km/s/Mpc and ![$\Omega_M=0.27$](https://render.githubusercontent.com/render/math?math=%24%5COmega_M%3D0.27%24). 

**`plot_mbh_hist`**: *Default: True*  
If broad H![$\alpha$](https://render.githubusercontent.com/render/math?math=%24%5Calpha%24) and/or H![$\beta$](https://render.githubusercontent.com/render/math?math=%24%5Cbeta%24) are included in the fit, BADASS will estimate the BH mass from the H![$\alpha$](https://render.githubusercontent.com/render/math?math=%24%5Calpha%24) width and luminosity using the equation from [Woo et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...801...38W/abstract), and/or the H![$\beta$](https://render.githubusercontent.com/render/math?math=%24%5Cbeta%24) beta width and luminosity using the equation from [Sexton et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract).  Black hole masses are written to the `par_table.fits` file. 

**`plot_corner`**: *Default: False*  
Do you like corner plots? Well here you go.  BADASS will make a full corner plot of every free parameter, no matter how many there are.  Keep in mind that rendering such a large image requires a PDF format and some computational overhead.  Therefore, we set this to *False* by default.  This plot should only be used to assess *how emcee performs in fitting free parameters* and nothing else. 

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_corner.png)

**`plot_bpt`**: *Default: True*  
If marrow H![$\alpha$](https://render.githubusercontent.com/render/math?math=%24%5Calpha%24) and H![$\beta$](https://render.githubusercontent.com/render/math?math=%24%5Cbeta%24)emission line components are included in the fit, then BADASS will output a [Baldwin, Phillips & Terlevich (BPT)](https://ned.ipac.caltech.edu/level5/Glossary/Essay_bpt.html) diagnostic figure.  You'll be surprised to see where your Type 1 AGN may end up on this dogmatic approach to classifying AGNs. 

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_BPT.png)

**`write_chain`**: *Default: False* 
Write the full flattened MCMC chain (# walkers x # iterations) to a FITS file.  We set this to *False*, because the file can get quite large, and takes up a lot of space if one is fitting many spectra.  One should only need this file if one wants to reconstruct chains and re-compute histograms. 

## Output Options

```python
################################ Output options ################################
output_options={
'write_chain'   : False, # Write MCMC chains for all paramters, fluxes, and
                         # luminosities to a FITS table We set this to false 
                         # because MCMC_chains.FITS file can become very large, 
                         # especially  if you are running multiple objects.  
                         # You only need this if you want to reconstruct chains 
                         # and histograms. 
'print_output'  : True,  # prints steps of fitting process in Jupyter output
}
################################################################################

```

## Multiprocessing Options

```python
############################ Multiprocessing options ###########################
# If fitting single object at a time (no for loops!) then one can set threads>1
# If one wants to fit objects sequentially (one after another), it must be set 
# to threads=1, and must use multiprocessing to spawn subprocesses without 
# significant memory leaks. 
mp_options={
'threads' : 4 # number of processes per object
}
################################################################################

```

**`threads`**: *Default: 4*  
emcee is capable of multiprocessing however performance is system dependent.  For a 2017 multi-core MacBook Pro, 4 simultaneous threads is optimal.  You may see better or worse performance depending on how many threads you choose and your system's hardware, so use with caution! 


## The Main Function

All of the above options are fed into the `run_BADASS()` function as such:

```python
# Call the main function in BADASS
badass.run_BADASS(file,run_dir,temp_dir,
                  fit_options,
                  mcmc_options,
                  comp_options,
                  feii_options,
                  outflow_test_options,
                  plot_options,
                  output_options,
                  mp_options,
                 )
    #
```


# Output

BADASS produces a number of different outputs for the user at the end of the fitting process.  We summarize these below.

## Best-fit Model

This is simply a figure that shows the data, model, residuals, and best-fit components, to visually ascertain the quality of the fit. 

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_bestfit.png)

## Parameter Chains, Histograms, Best-fit Values & Uncertainties

For every parameter that is fit, BADASS outputs a figure to visualize the full parameter chain and all walkers, the burn-in, and the final posterior histogram with the best-fit values and uncertainties.  The purpose of these is for visual inspection of the parameter chain to observe its behavior during the fitting process, to get a sense of how well initial parameters were fit, and how walkers behave as the fit progresses.

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_chain.png)

## Log File

The entire fitting process, including selected options, is recorded in a text file for reference purposes.  It provides:
- File location
- FITS Header information, such as (RA,DEC) coordinates, SDSS redshift, velocity scale (km/s/pixel), and *E(B-V)*, which is retrieved from NED online during the fitting process and used to correct for Galactic extinction.
- Outflow fitting results (if `test_outflows=True`)
- Initial fitting parameters
- Autocorrelation times and tolerances for each parameter (if `auto_stop=True`)
- Final MCMC parameter best-fit values and uncertainties 
- Systemic redshift measured from stellar velocity (if `fit_losvd=True`)
- AGN luminosities at 5100 Å inferred from broad-line luminosity relation ([Greene et al. 2005](https://ui.adsabs.harvard.edu/abs/2005ApJ...630..122G/abstract))
- Black hole mass estimates using broad H-beta ([Sexton et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)) and broad H-alpha ([Woo et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...801...38W/abstract)) measurements.

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_logfile.png)

## Best-fit Model Components

Best-fit model components are stored as arrays in `best_model_components.fits` files to reproduce the best-fit model figures as shown above.  This can be accessed using the [`astropy.io.fits`](https://docs.astropy.org/en/stable/io/fits/) module, for example:
```
from astropy.io import fits

hdu = fits.open(best_model_components.fits)
tbdata = hdu[1].data     # FITS table data is stored on FITS extension 1
data   = tbdata['data']  # the SDSS spectrum 
wave   = tbdata['wave']  # the rest-frame wavelength vector
model  = tbdata['model'] # the best-fit model
hdu.close()
```

Below we show an example data model for the FITS-HDU of `best_model_components.fits`.  You can print out the columns using 
```
print(tbdata.columns)
```
which shows

```
ColDefs(
    name = 'na_oiii4959_core'; format = 'E'
    name = 'br_Hb'; format = 'E'
    name = 'power'; format = 'E'
    name = 'na_Hb_outflow'; format = 'E'
    name = 'na_oiii5007_core'; format = 'E'
    name = 'na_Hb_core'; format = 'E'
    name = 'wave'; format = 'E'
    name = 'na_feii_template'; format = 'E'
    name = 'na_oiii4959_outflow'; format = 'E'
    name = 'noise'; format = 'E'
    name = 'resid'; format = 'E'
    name = 'host_galaxy'; format = 'E'
    name = 'na_oiii5007_outflow'; format = 'E'
    name = 'data'; format = 'E'
    name = 'model'; format = 'E'
)
```

## Best-fit Parameters & Uncertainties 

All best-fit parameter values and their upper and lower 1-sigma uncertainties are stored in `par_table.fits` files so they can be more quickly accessed than from a text file.  These are most easily accessed using the (`astropy.table`](https://docs.astropy.org/en/stable/table/pandas.html) module, which can convert a FITS table into a Pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html):

```
from astropy.table import Table  

table = Table.read('par_table.fits')
pandas_df = table.to_pandas()
print pandas_df
```
which shows

|    |             parameter |     best_fit  |   sigma_low  |   sigma_upp  | flag  |
| ---| --------------------- | -------------:| -----------: | -----------: | ----: |
| 0  |      host_galaxy_flux |  43179.789062 |  2146.635986 |  2523.700439 |   0.0 |
| 1  |       host_galaxy_lum |      2.993121 |     0.148800 |     0.174937 |   0.0 |
| 2  |        na_Ha_core_amp |    458.180511 |     2.664992 |     2.122619 |   0.0 |
| 3  |       na_Ha_core_flux |   2304.035889 |    13.468710 |    19.964769 |   0.0 |
| .  |            .          |  	 .	     |		 .	    |		 .     |	.  |
| .  |            .          |  	 .	     |		 .	    |		 .     |	.  |
| .  |            .          |  	 .	     |		 .	    |		 .     |	.  |
| 57 |             power_amp |     13.716220 |     0.670133 |     0.736425 |   0.0 |
| 58 |            power_flux |  47760.011719 |  2512.798340 |  2167.185303 |   0.0 |
| 59 |             power_lum |      3.310611 |     0.174181 |     0.150224 |   0.0 |
| 60 |           power_slope |     -0.705488 |     0.294422 |     0.352110 |   0.0 |
| 61 |             stel_disp |     99.216248 |     1.282358 |     0.785158 |   0.0 |
| 62 |              stel_vel |     97.768555 |     3.329233 |     2.168582 |   0.0 |
| 63 |                z_best |      0.055001 |     0.000012 |     0.000008 |   0.0 | 

## Autocorrelation Time & Tolerance History

BADASS will output the full history of parameter autocorrelation times and tolerances for every `write_iter` iterations.  This is done for post-fit analysis to assess how individual parameters behave as MCMC walkers converge on a solution.   Parameter autocorrelation times and tolerances are stored as arrays in a dictionary, which is saved as a numpy `.npy` file named `autocorr_dict.npy', which can be accessed using the `numpy.load()' function: 

```
autocorr_dict = np.load('autocorr_dict.npy')

# Display parameters in dictionary
for key in autocorr_dict.item():
        print key

# Print the autocorrelation times and tolerances for the 
# 'na_oiii5007_core_voff' parameter and store them as 
# "tau" and "tol", respectively:
tau = autocorr_dict.item().get('na_oiii5007_core_voff').get('tau')
tol = autocorr_dict.item().get('na_oiii5007_core_voff').get('tol')

```

# How to
## How to add an emission line to BADASS

This may look complicated, but its really just a lot of copy, pasting, and changing a few variable names.  

If one wishes to add (or remove) an emission line for BADASS to fit, one must modify the source code (`.py` file) in two places:
1. **`initialize_mcmc()`** - to define the parameters to be fit, for example, amplitude, width, velocity offset, etc.  For example, if we want to add a broad H-beta emission line, 
```python
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
```
The dictionary values for each parameter defined above are described below:
`name`: a unique identifiable name for the parameter.

**`label`**: a LaTeX label used for plot labelling on chain/histogram plots.

**`init`**: initial value of parameter, which is used by maximum likelihood routine for finding a better initial position (units of km/s)

**`plim`**: parameter limits; the lower and upper bounds of allowed parameter values. (units of km/s)

**`pcolor`**: plot color for histogram.

**`min_width`**: minimum valid width for emission lines.  This already takes into account the instrumental dispersion, so we instead set the minimum to the typical uncertainty in width measurements ~15 km/s (units of km/s)

**Note**: the only parameter dictionary value that needs to be unique is `name`.  The others do not need to be unique but there still needs to be a valid value (for example, a valid `matplotlib` color) for color.  Also, make sure that the `init` value is between the lower and upper bounds of `plim`.

2. **`fit_model()`** - add the component to the model; this is where we specify if component parameters are tied to other components, are constant, etc.

```python 
if all(comp in param_names for comp in ['br_Hb_amp','br_Hb_fwhm','br_Hb_voff'])==True:
     br_hb_center       = 4862.68 # Angstroms
     br_hb_amp	        = p['br_Hb_amp'] # flux units
     br_hb_fwhm_res     = get_fwhm_res(fwhm_gal_ftn,br_hb_center,p['br_Hb_voff'])
     br_hb_fwhm	        = np.sqrt(p['br_Hb_fwhm']**2+(br_hb_fwhm_res)**2) # km/s
     br_hb_voff	        = p['br_Hb_voff']  # km/s
     br_Hb	        = gaussian(lam_gal,br_hb_center,br_hb_amp,br_hb_fwhm,br_hb_voff,velscale)
     host_model	        = host_model - br_Hb
     comp_dict['br_Hb'] = {'comp':br_Hb,'pcolor':'xkcd:turquoise','linewidth':1.0}
	
```

The first `if` statement check to see if the parameters needed to create the emission line component were added to the list of parameters to be fit using `initialize_mcmc()` (because without them you can't create the line model).  The following lines define components of the line.  These lines in the source code were written to be readable - not pythonic - so they could be understood by anybody seeing the code for the first time.

**`br_hb_center`**: the rest wavelength of the line in units of Angstroms.

**`br_hb_amp`**: the amplitude parameter in units of flux density.

**`br_hb_fwhm_res`**: the SDSS instrumental resolution interpolated at the rest wavelength of the emission line.

**`br_hb_fwhm`**: the width (FWHM) parameter in units of km/s. We add this in quadrature to the instrumental resolution of the line so that BADASS returns the intrinsic width of the line.

**`br_hb_voff`**: the velocity offset in units of km/s from rest wavelength.

**`br_Hb`**: the above parameters are sent to the `gaussian()` function to return a model for the line.

**`host_model`**: the line model is then subtracted from the data.  At the end the only thing should be left is stellar continuum (if any), which is then fit by pPXF.  All components are then added back to subtract from the data in the likelihood function.

**`comp_dict['br_Hb']`**: stores the line model in the component dictionary so it can be plot and stored later as output. Again, this require a valid `matplotlib` color.

And thats it!

If you wanted to tie the width of your line to another line, simply replace the with with the dictionary parameter width of the other line.  All of this can be done similarly for other components, such as a continuum model.  

# Known Issues

### "BADASS only fits one spectra using Multiprocessing and hangs up on the others"

When running BADASS for multiple spectra using the multiprocessing notebook, BADASS will hangup when trying to fit spectra it has not previously obtained E(B-V) values for via Astroquery's [IRSA Dust Extinction Service Query](https://astroquery.readthedocs.io/en/latest/irsa/irsa_dust.html).  This is a known issue (see [this](https://github.com/astropy/astroquery/issues/684).  The problem stems from the fact that `IrsaDust. get_query_table()` treats multiple Python subprocesses as a single-process. For example, if you are running 4 subprocesses (fitting 4 spectra simultaneously), it will only query the last process of the four, and leave the first three hanging.  

Luckily there is a workaround.  `IrsaDust. get_query_table()` stores previous queries on your local machine so they can be accessed without looking them up every single time.  The solution is to simply query E(B-V) values for all of your objects before fitting, which seems dumb but it's the only workaround and its quick.   In the [badass_tools](https://github.com/remingtonsexton/BADASS3/tree/master/badass_tools) directory there is a notebook called `Fetch IRSA Dust E(B-V).ipynb`.  Simply run this notebook before your fitting run with BADASS to pre-fetch all E(B-V) valuees.


# Contributing

Please let us know if you wish to contribute to this project by reporting any bugs, issuing pull requests, or requesting any additional features to help make BADASS the most detailed spectroscopic analysis tool in astronomy.


# Credits

- [Remington Oliver Sexton](https://astro.ucr.edu/members/graduate-students/#Remington) (UC Riverside, Physics & Astronomy)
- William Matzko (George Mason University, Physics and Astronomy)
- Nicholas Darden (UC Riverside, Physics & Astronomy)


# License

MIT License

Copyright (c) 2020 Remington Oliver Sexton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


