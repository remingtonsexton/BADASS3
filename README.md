
![BADASS logo](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS2_logo.gif)

## New in Version 10.2.0
* New line and configuration testing framework. See [Line Testing and Options](#line-testing-and-options).
* Line component options.  See [Line Component Options](#line-component-options).
* Improvements in autocorrelation calculations.
* W80 now a standard output parameter for all lines.
* Outputs line widths that are both corrected and uncorrected for input resolution
* BADASS now normalizes the spectrum internally for fitting purposes.
* Various improvements to global optimizer used in initial fit.
* **Note**: The algorithm used for scaling the noise to achieve a $\Chi_\nu^2=1$ was found to be numerically unstable and users should use `fit_stat='ML'` instead.
<hr>

Ridiculous acronyms are a [long-running joke in astronomy](https://lweb.cfa.harvard.edu/~gpetitpas/Links/Astroacro.html), but here, spectral fitting ain't no joke!

[BADASS](https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.2871S/abstract) is an open-source spectral analysis tool designed for detailed decomposition of Sloan Digital Sky Survey (SDSS) spectra, and specifically designed for the fitting of Type 1 ("broad line") Active Galactic Nuclei (AGN) in the optical.  The fitting process utilizes the Bayesian affine-invariant Markov-Chain Monte Carlo sampler [emcee](https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract) for robust parameter and uncertainty estimation, as well as autocorrelation analysis to access parameter chain convergence.  BADASS can fit the following spectral features:
- Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å.
- Broad and Narrow FeII emission features using the FeII templates from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) or [Kovačević et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MSAIS..15..176K/abstract).
- UV iron template from [Vestergaard and Wilkes (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJS..134....1V/abstract)
- Individual narrow, broad, and/or absorption line features.
- AGN power-law continuum and Balmer pseudo-continuum.
- "Blue-wing" outflow emission components found in narrow-line emission. 

A more-detailed summary of BADASS, as well as a case-study of ionized gas outflows, is given in [Sexton et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.2871S/abstract). 

All spectral components can be turned off and on via the [Jupyter Notebook](https://jupyter.org/) interface, from which all fitting options can be easily changed to fit non-AGN-host galaxies (or even stars!).  BADASS uses multiprocessing to fit multiple spectra simultaneously depending on your hardware configuration.  The code was originally written in Python 2.7 to fit Keck Low-Resolution Imaging Spectrometer (LRIS) data ([Sexton et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)), but because BADASS is open-source and *not* written in an expensive proprietary language, one can easily contribute to or modify the code to fit data from other instruments.  Out of the box, BADASS fits SDSS spectra, MANGA IFU cube data, and examples are provided for fitting user-input spectra of any instrument.

Before getting started you should read the readme below.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  * [Fitting Options](#fitting-options)
  * [MCMC Options](#mcmc-options)
  * [Model Component Options](#model-component-options)
  * [Multiple Line Components](#multiple-line-components)
  * [Line Testing and Options](#line-testing-and-options)
  * [Configuration Testing](#configuration-testing)
  * [Line Component Options](#line-component-options)
  * [User Lines, Constraints, and Masks](#user-lines-constraints-and-masks)
  * [Combined Lines](#combined-lines)
  * [LOSVD Fitting Options (pPXF)](#losvd-fitting-options-ppxf)
  * [Host Model Options](#host-model-options)
  * [Power Law Options](#power-law-options)
  * [Polynomial Options](#polynomial-options)
  * [Optical FeII Options](#optical-feii-options)
  * [UV Iron Options](#uv-iron-options)
  * [Balmer Pseudo-Continuum Options](#balmer-psuedocontinuum-options)
  * [Plotting Options](#plotting-options)
  * [Output Options](#output-options)
  * [The Main Function and Calling Sequence](#the-main-function-and-calling-sequence)
- [Output](#output)
  * [Best-fit Model](#best-fit-model)
  * [Parameter Chains, Histograms, Best-fit Values and Uncertainties](#parameter-chains-histograms-best-fit-values-and-uncertainties)
  * [Log File](#log-file)
  * [Best-fit Model Components](#best-fit-model-components)
  * [Best-fit Parameters and Uncertainties](#best-fit-parameters-and-uncertainties)
  * [Autocorrelation Analysis](#autocorrelation-analysis)
- [Examples & Tutorials](#examples)
  * [Basic Example: A Single SDSS Spectrum](#single-sdss-spectrum)
  * [A Single Non-SDSS Spectrum](#single-non-sdss-spectrum)
  * [Multiple Spectra with Multiprocessing](#multiple-spectra-with-multiprocessing)
  * [MANGA IFU Cube Data](#manga-ifu-cube-data)
  * [Non-MANGA IFU Cube Data](#non-manga-ifu-cube-data)
- [How to...](#how-to)
  * [Prior Distributions](#prior-distributions)
  * [Line Lists](#line-lists)
  * [Hard Constraints](#hard-constraints)
  * [Soft Constraints](#soft-constraints)
- [Known Issues](#known-issues)
- [Credits](#credits)
- [License](#license)



# Installation

The easiest way to get started is to simply clone the repository. 

As of the most recent version, the following packages are required (Python 3.8.12):

- [`numpy 1.22.3`](https://numpy.org/doc/stable/index.html)
- [`pandas 1.4.2`](https://pandas.pydata.org/)
- [`scipy 1.7.3` ](https://scipy.org/)
- [`matplotlib 3.7.1`](https://matplotlib.org/)
- [`astropy 5.2.1`](https://www.astropy.org/)
- [`astroquery 0.4.6`](https://astroquery.readthedocs.io/en/latest/)
- [`emcee 3.1.4`](https://emcee.readthedocs.io/en/stable/)
- [`numexpr 2.8.4`](https://github.com/pydata/numexpr)
- [`natsort 8.3.1`](https://natsort.readthedocs.io/en/master/)
- [`psutil 5.9.4`](https://psutil.readthedocs.io/en/latest/)
- [`vorbin 3.1.5`](https://www-astro.physics.ox.ac.uk/~cappellari/software/#binning)
<!-- - [`astro-bifrost 2.0.3`](https://pypi.org/project/astro-bifrost/) -->
- [`spectres 2.2.0`](https://spectres.readthedocs.io/en/latest/)
- [`corner 2.2.1`](https://corner.readthedocs.io/en/latest/)
- [`prettytable 3.6.0`](https://pypi.org/project/prettytable/)
- **Note**: [`ppxf`](https://www-astro.physics.ox.ac.uk/~cappellari/software/#ppxf) was integrated within the BADASS source code early on, and does not require installation.
- **Optional**: [`plotly 5.13.1`](https://plotly.com/python/getting-started/) (for interactive HTML plots)

The code is run entirely through the Jupyter Notebook interface, and is set up to run on the included spectrum files in the ".../examples/" folder.  If one wants to fit multiple spectra consecutively, simply add folders for each spectrum to the folder.  This is the recommended directory structure:

![directory structure](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_example_spectra.png)

Simply create a folder containing the SDSS FITS format spectrum, and BADASS will generate a working folder for each fit titled "MCMC_output_#" inside each spectrum folder.  BADASS automatically generates a new output folder if the same object is fit again (it does not delete previous fits).

# Usage

## Fitting Options
 
```python 
################################## Fit Options #################################
# Fitting Parameters
fit_options={
fit_options={
"fit_reg"    : (4400,5500),# Fitting region; Note: Indo-US Library=(3460,9464)
"good_thresh": 0.0, # percentage of "good" pixels required in fig_reg for fit.
"mask_bad_pix": False, # mask pixels SDSS flagged as 'bad' (careful!)
"mask_emline" : False, # automatically mask lines for continuum fitting.
"mask_metal": False, # interpolate over metal absorption lines for high-z spectra
"fit_stat": "RCHI2", # fit statistic; ML = Max. Like. , OLS = Ordinary Least Squares, RCHI2 = reduced chi2
"n_basinhop": 15, # Number of consecutive basinhopping thresholds before solution achieved
"test_lines": False, # Perform line/configuration testing for multiple components
"max_like_niter": 10, # number of maximum likelihood iterations
"output_pars": False, # only output free parameters of fit and stop code (diagnostic)
"cosmology": {"H0":70.0, "Om0": 0.30}, # Flat Lam-CDM Cosmology
}
################################################################################
```

**`fit_reg`**: (*tuple/list of length (2,)*; *Default: (4400,5500)*)<br/>
the minimum and maximum desired fitting wavelength in angstroms, for example (4400,7000).  This is passed to the `determine_fit_reg()` function to check if this region is valid and and which emission lines to fit.

**`good_thresh`**: (*float [0.0,1.0]*; *Default: 0.0*)<br/>
the cutoff for minimum fraction of "good" pixels (determined by SDSS) within the fitting range to allow for fitting of a given spectrum.  If the spectrum has fewer good pixels than this value, BADASS skips over it and moves onto the next spectrum.

**`mask_bad_pix`**: (*bool*; *Default: False*)<br/>
Mask pixels which SDSS flagged as bad due to sky line subtraction or cosmic rays.  Warning: if large portions of the fitting region are marked as bad pixels, this can cause BADASS to crash.  One should only use this if only a few pixels are affected by contamination.  

**`mask_emline`**: (*bool*; *Default: False*)<br/>
Mask any significant absorption and emission features relative to the continuum.  This uses an automated iterative moving median filter of various sizes to detect significant flux differences between window sizes.  Good for continuum fitting but tends to over mask lots of features near the edges of the spectrum.

**`mask_metal`**: (*bool*; *Default: False*)<br/>
Performs the same moving median filter algorithm as `mask_emline` but only to absorption features.  Works well for metal absorption features seen typically in high-redshift spectra.

**`fit_stat`**: (*str*; *Default: : "ML"*)<br/>
The fit statistic used for the likelihood.  The default is "RCHI2" which converges on a reduced chi-squared of 1 by scaling the input noise by a `noise_scale` free parameter.  Other options include "ML" for standard maximum likelihood (pixels weighted by noise with no noise scaling), and "LS" for ordinary least-squares fitting (all pixels weighted by same amount).

**`n_basinhop`**: (*int*; *Default: 25*)<br/>
Number of successive `niter_success` times the basinhopping algorithm needs to achieve a solution.  The fit becomes much better with more success times, however this can increase the time to a solution significantly  Recommended 5-10. 

**`reweighting`**: (*bool*; *Default: True*)<br/>
If true, BADASS will reweight the noise vector to achieve a reduced chi-squared ~ 1.  This is done after the initial basinhopping fit, and applied to any bootstrapped uncertainties and MCMC fitting performed afterward.  This does not affect the chi-squared ratio metric used in line and configuration testing, but does effect the amplitude-over-noise and SNR calculations in BADASS.

**`test_lines`**: (`bool`:*bool*; *Default: False*)<br/>
Performs tests for lines with multiple components (parent-child pairs).  Options are specified in `test_options`.

**`max_like_niter`**: (*int*; *Default: 10*)<br/>
Number of bootstrapping iterations to perform after the initial basinhopping fit.  This is a means to obtain uncertainties on parameters without performing MCMC fitting, however, do not produce as robust uncertainties as MCMC.

**`output_pars`**: (*bool*; *Default: False*)  <br/>
Convenience feature that prints out all free parameters so the user can check and then terminates without fitting.

**`cosmology`**: (*Default*: `{"H0":70.0, "Om0": 0.30}`)<br/>
The flat Lambda-CDM cosmology assumed for calculating luminosities from fluxes. 


# MCMC Options

```python
########################### MCMC algorithm parameters ##########################
mcmc_options={
  "mcmc_fit"    : False, # Perform robust fitting using emcee
  "nwalkers"    : 100,  # Number of emcee walkers; min = 2 x N_parameters
  "auto_stop"   : False, # Automatic stop using autocorrelation analysis
  "conv_type"   : "all", # "median", "mean", "all", or (tuple) of parameters
  "min_samp"    : 1000,  # min number of iterations for sampling post-convergence
  "ncor_times"  : 1.0,  # number of autocorrelation times for convergence
  "autocorr_tol": 10.0,  # percent tolerance between checking autocorr. times
  "write_iter"  : 100,   # write/check autocorrelation times interval
  "write_thresh": 100,   # iteration to start writing/checking parameters
  "burn_in"     : 1500, # burn-in if max_iter is reached
  "min_iter"    : 1000, # min number of iterations before stopping
  "max_iter"    : 2500, # max number of MCMC iterations
}
################################################################################
```

**`mcmc_fit`**: (*bool*; *Default: False*) <br/>
Perform fit with MCMC using the initial maximum likelihood fit as initial parameters for the fit.  It is *highly recommended* that one use MCMC to perform the fit, although sampling will require a significant amount of time compared to a maximum likelihood fit using `scipy.optimize.minimize()`.

**`nwalkers`**: (*int*; *Default: 100*) <br/> 
number of "walkers" per parameter used by emcee to explore each parameter space.  The minimum number of walkers is 2 x ( # of free parameters), set by emcee.

**`auto_stop`**: (*bool*; *Default: True*) <br/> 
if set to *True*, autocorrelation is used to automatically stop the fitting process when a convergence criteria (`conv_type`) is achieved. 

**`conv_type`**: (*str*; *Default: "median"*; *options: "all", "median", "mean", list of parameters*)  
mode of convergence.  Convergence of 'all' ensures all fitting parameters have achieved the desired `ncor_times` and `autocorr_tol` criteria, while "median" and "mean" only ensure that `ncor_times` and `autocorr_tol` criteria have been met for the median or mean of all parameters, respectively.  A list of valid parameters is also acceptable to ensure specific parameters have achieved convergence even if others have not.  In general "median" requires the fewest number of iterations and is not sensitive to poorly-constrained parameters, and "all" and "mean" require the most number of iterations and are much more sensitive to fluctuations in calculated autocorrelation times and tolerances.  A list of parameters is suitable in cases where one is only interested in certain spectral features.

**`min_samp`**: (*int*; *Default: 1000*)  <br/>
if `auto_stop=True`, then the `burn_in` is the iteration at which convergence is achieved, and `min_samp` is the number of iterations *past convergence* used for posterior sampling (the samples used for histograms and estimating best-fit parameters and uncertainties.  If for some reason the parameters "jump out" of convergence, the `burn_in` will reset and BADASS will continue to sample until convergence is met again.  If emcee completes `min_samp` iterations after convergence is achieved without jumping out of convergence, this concludes the MCMC sampling.

**`ncor_times`**: (*int* or *float*; *Default=10*) <br/> 
The number of integrated autocorrelation times (iterations) needed for convergence.  We recommend a minimum of `ncor_times=2.0`.  In general, it will require more than 2.0 autocorrelation times to calculate the autocorrelation time for a parameter chain.  Increasing `ncor_times` ensures that the parameter chain has stopped exploring the parameter space and is ready to begin sampling for the posterior distribution. 

**`autocorr_tol`**: (*int* or *float*; *Default=10*) <br/>
The percent change in the current integrated autocorrelation time and the previously calculated integrated autocorrelation time.  The `write_iter` determines how often BADASS checks a parameter's integrated autocorrelation time.  In general, we find that `autocorr_tol=5` (a 5% change) is acceptable for a converged parameter chain.  A parameter chain that diverges more than 10% in 100 iterations could still be exploring the parameter space for a stable solution.  A `autocorr_tol=1` (a 1% change) typically requires many more iterations than necessary for convergence. 

**`write_iter`**: (*int*; *Default=100*)<br/>
The frequency at which BADASS writes the current parameter values (median walker positions).  If `auto_stop=True`, then BADASS checks for convergence every `write_iter` iteration for convergence.

**`write_thresh`**: (*int*; *Default=100*) <br/> 
The iteration at which writing (and checking for convergence if `auto_stop=True`) begins.  BADASS does not check for convergence before this value.

**`burn_in`**: (*int*; *Default=1500*)  <br/>
If `auto_stop=False` then this serves as the burn-in for a maximum number of iterations.  If `auto_stop=True`, this value is ignored.

**`min_iter`**: (*int*; *Default=100*)  <br/>
The minimum number of iterations BADASS performs before it is allowed to stop.  This is true regardless of the value of `auto_stop`.

**`max_iter`**: (*int*; *Default=2500*)  <br/>
The maximum number of iterations BADASS performs before stopping.  This value is adhered to regardless of the value of `auto_stop` to set a limit on the number of iterations before BADASS should "give up."


## Model Component Options

```python
############################ Fit component options #############################
comp_options={
"fit_opt_feii"     : True, # optical FeII
"fit_uv_iron"      : False, # UV Iron 
"fit_balmer"       : False, # Balmer continuum (<4000 A)
"fit_losvd"        : False, # stellar LOSVD
"fit_host"         : True, # host template
"fit_power"        : True, # AGN power-law
"fit_poly"         : True, # Add polynomial continuum component
"fit_narrow"       : True, # narrow lines
"fit_broad"        : True, # broad lines
"fit_absorp"       : False, # absorption lines
"tie_line_disp"    : False, # tie line widths (dispersions)
"tie_line_voff"    : False, # tie line velocity offsets
}
################################################################################
```

These options are more-or-less self explanatory.  One can fit components appropriate (or not appropriate) for the types of objects they are fitting.  We summarize each component below:

**`fit_feii`**: (*Default=True*)<br/>
Broad and narrow optical FeII templates are taken from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) with each line modeled using a Gaussian.  One can also optionally using the template from [Kovačević et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MSAIS..15..176K/abstract), however with limited coverage (4400 Å - 5500 Å). FeII emission can be very strong in some Type 1 (broad line) AGN, but is almost undetectable in Type 2 (narrow line) AGN.

**`fit_uv_iron`**: (*Default=False*)<br/>
Fits the empirical UV iron template from [Vestergaard and Wilkes (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJS..134....1V/abstract), for high-redshift spectra with coverage < 3500 Å.

**`fit_balmer`**: (*Default=False*)<br/>
Fits a series of higher-order Balmer lines and Balmer pseudo-continuum for high-redshift spectra with coverage < 3500 Å.

**`fit_losvd`**: (*Default=True*)<br/>
Stellar line-of-sight velocity distribution (LOSVD) using Penalized Pixel-Fitting ([pPXF](https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), [Cappellari et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract)) using templates from the [Indo-U.S. Library of Coudé Feed Stellar Spectra](https://www.noao.edu/cflib/) ([Valdes et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ApJS..152..251V/abstract)) in the optical region 3460 Å - 9464 Å.  This is used to obtain stellar kinematics in spectra with resolvable absorption features, such as stellar velocity and dispersion. 

**`fit_host`**: (*Default=False*)<br/>
Fits a host galaxy template using single-stellar population templates from the EMILES library.  Note that this method does not estimate stellar LOSVD, but can shift in velocity and convolve to match the data as best as it can.

**`fit_power`**: (*Default=True*)<br/>
this fits a power-law component to simulate the effect of the AGN "blue-bump" continuum. 

**`fit_poly`**: (*Default=False*)<br/>
Fit a polynomial continuum component of a specified order.  Polynomial options are specified by `poly_options` dictionary.  Options are additive Legendre polynomial or multiplicative Legendre polynomial.  The order must be within the range 0 <= order <= 99.  Note: caution should be used when using polynomial components, as these can be degenerate with other continuum components, and higher-order polynomials can lead to overfitting.

**`fit_narrow`**: (*Default=True*)<br/>
Fit lines of the `line_type`:`na` in the line list.  Narrow forbidden emission lines are seen in both Type 1 and Type 2 AGNs, as well as starforming galaxies. 

**`fit_broad`**: (*Default=True*)<br/>
Fit lines of the `line_type`:`br` in the line list.  Broad permitted emission lines are commonly seen in Type 1 AGN.  

**`fit_absorp`**: (*Default=False*)<br/>
Fit lines of the `line_type`:`abs` in the line list.  Occasionally one might need to fit a strong absorption feature that isn't described by stellar processes, such as a broad absorption line in a quasar.

**`tie_line_disp`**: (*Default=False*)<br/>
Ties the widths of all respective line types (all narrow lines are tied, all broad lines are tied, etc.).  This can be done to significantly reduce the number of free parameters in the fit if fitting many lines, however it is not recommended. 

**`tie_line_voff`**: (*Default=False*)<br/>
Ties the velocity offsets of all respective line types (all narrow lines are tied, all broad lines are tied, etc.).  This can be done to significantly reduce the number of free parameters in the fit if fitting many lines, however it is not recommended. 

Examples of the aforementioned spectral components can be seen in the example fit below:

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_model_options.png)

## Multiple Line Components

Line testing of multiple components is the newest feature of BADASS, and completely changes how multitple line components (i.e., outflows) are treated.  Outflows are no longer a `line_type` (now only `narrow`, `broad`, and `absorp` are valid `line_types`), however, it is still possible to fit "outflow" components by adding *child* lines to corresponding *parent* lines.  The `parent` parameter keyword for lines ties an additional component of a line to its corresponding parent line.  

Here is an example of defining 5 components for the [OIII]5007 emission line:

```python
"NA_OIII_5007"   :{"center":5008.240,
                   "amp":"free",
                   "disp":"free",
                   "voff":"free",
                   "line_type":"na",
                   "ncomp":1,
                  },

"NA_OIII_5007_2" :{"center":5008.240,
                   "amp":"free",
                   "disp":"free",
                   "voff":"free",
                   "line_type":"na",
                   "ncomp":2,
                   "parent":"NA_OIII_5007"
                  },

"NA_OIII_5007_3" :{"center":5008.240,
                   "amp":"free",
                   "disp":"free",
                   "voff":"free",
                   "line_type":"na",
                   "ncomp":3,
                   "parent":"NA_OIII_5007"
                  },

"NA_OIII_5007_4" :{"center":5008.240,
                   "amp":"free",
                   "disp":"free",
                   "voff":"free",
                   "line_type":"na",
                   "ncomp":4,
                   "parent":"NA_OIII_5007"
                  },

"NA_OIII_5007_5" :{"center":5008.240,
                   "amp":"free",
                   "disp":"free",
                   "voff":"free",
                   "line_type":"na",
                   "ncomp":5,
                   "parent":"NA_OIII_5007"
                  },

```
A few things to point out here:
*  `"ncomp":1` defines the *parent* line; it is the first component
* Additional components with `"ncomp"` > 1 must have a `parent` defined, which is the name of the *parent* line; these are the *child* components.
* Each line component must be given a unique name; we recommend the above naming convention. 
* Each additional component must have a different component number `ncomp`, with the *parent* defined as `"ncomp":1`, and each additional component with `"ncomp"`>1. 

While it may seem tedius, this is the only way to tie parent lines to their corresponding child components, and subsequently test for them. 

## Line Testing and Options

Line testing (and outflow testing) has also changed to accomodate the new emission line framework for multiple components (see above section).  Line testing is intended for testing for multiple components in *individual* lines, preferably ones that can be resolved from others.  Testing for lines that are partially resolve from each other, for example broad and narrow lines, is more difficult to do, but can still be done.  We can test for multiple lines in the same fit, choose different significance metrics to "choose" the right number of components, and choose specfic parts of the fitting region (defined by `fit_reg`) to test line on.  The `test_lines` option must be set to `True` in the `fit_options` passed to BADASS.

The way line testing works (and the only way to do it) is to perform a fit of a given set of lines, compute various statistics on the residuals of the fit, and compare each fit to each other to determine the appropriate number of components.  BADASS performs a fit of a "simple" model, followed by a "complex" model, and then computes statistics between them.  For example, a 1 component model (simple) and a 2 component model (complex), 2 and 3, 3 and 4, etc...  The calculated metric is how *confident* you are that adding an additional component *does not* significantly improve the fit.  That is, when the `metric` is *below* its corresponding `threshold`, you have added the appropriate number of components.

Here is a simple example of specifying testing options for testing for narrow components of H-beta/[OIII]:
```python
test_options = {
    "test_mode":"line",
    "lines": [["NA_OIII_5007","NA_OIII_4960","NA_H_BETA"]], 
    "metrics": ["BADASS", "ANOVA", "CHI2_RATIO", "AON"],
    "thresholds": [0.95, 0.95, 0.10, 3.0],
    "conv_mode": "any", 
    "auto_stop":True, 
    "plot_tests":True, 
    "force_best":True, 
    "continue_fit":True, 
}
```

**`test_mode`**: (str; *Default="line"*)<br/>
Future releases of BADASS will have different testing methods; the only option available now is `line`.

**`lines`**: (list; *Default=[]*)<br/>
The lines (or group of lines) to test.  BADASS will perform these tests in the order the are defined in the list.

**`metrics`**: (list; *Default=["BADASS", "CHI2_RATIO", "AON"]*)<br/>
The testing metrics used to determine when the appropriate number of components is reached. Options are "BADASS", "ANOVA", "CHI2_RATIO", "SSR_RATIO", "F_RATIO", "AON".  We describe these below:
* *BADASS*: this test generates a confidence between 0 and 1 of the Monte-Carlo resampled maximum likelihood values of the simple and complex models.  For example, if the confidence is 0.95, you can say that the difference between residuals is significant, and an additional component is justified with 95% confidence.
* *ANOVA*: this test is anlogous to an analysis-of-variance (ANOVA) test.  It haves similarly to the BADASS test, but tends to prefer a more complex model, while the BADASS metric tends to prefer a simpler one.
* *CHI2_RATIO*: the ratio of the reduced chi-squared values for each fitted model.  This tends to be a robust method with thresholds between 0.1 and 0.25.  Interpreted as the fraction of the difference in the residuals weighted by the noise.
* *SSR_RATIO*: the ratio of the sum-of-squares of the residuals (SSR).  Values close to 1 indicate residuals that are very similar.  
* *F_RATIO*: the ratio of variances betweent he two fitted models, behaving similarly to the SSR ratio.
* *AON*: amplitude-over-noise (AON) of the final line.  This is not a test between models, but a final check to determine if the lines that comprise the best model have an amplitude-over-noise (otherwise known as signal-to-noise) above a certain threshold.  This metric is good to have to ensure you aren't fitting noise.

**`thresholds`**: (list; *Default=[0.95, 0.10, 3.0]*)<br/>
Thresholds for the above chosen metrics.  When a calculated metric goes *below* a threshold for a given metric, "convergence" is achieved.

**`conv_mode`**: (str; *Default="any"*; options are "any" or "all")<br/>
Whether "all" or "any" of the metrics must achieve convergence to determine the appropriate number of line components.  If "all" is chosen, then all metrics must go below the given threshold for convergence.  If "any" is chosen, any single metric must achive convergence.  

**`auto_stop`**: (bool; *Default=True*)<br/>
Automatically stop testing for a line when convergence is met.  For example, if we define five components for a line, but convergence is reached at three components, BADASS will not test that line further.

**`full_verbose`**: (bool; *Default=False*)<br/>
Print out the results of fitting each test (basinhopping callbacks, etc.) to screen.  This is `False` by default because it can be excessive (no exagerration, it will print *everything*).  This is useful to monitor the fit of each test as it is performed, but not recommended for the uninitiated.

**`plot_tests`**: (bool; *Default=True*)<br/>
Plot the results of each test and save them for visual comparison.

**`force_best`**: (bool; *Default=True*)<br/>
Forces the complex model to achieve a root-mean-squared-error (RMSE) comparable to or less than the simpler model.  This is highly recommended because of the caveats we give below.

**`continue_fit`**: (bool; *Default=True*)<br/>
Continue the fit (to max likelihood and/or MCMC) after the test is completed.  If False, BADASS terminates when line testing is completed.

We also test for different lines in the same fit.  Here we specify we want to fit the three narrow lines from above, but now we also want to test the broad H-beta line (`BR_H_BETA`) and some unknown line at 5100 Å (`NA_UNK_1`) in a separate test region:

```python
test_options = {
"test_mode":"line",
"lines": [["NA_OIII_5007","NA_OIII_4960","NA_H_BETA"],"BR_H_BETA","NA_UNK_1"], # The lines to test
"metrics": ["BADASS", "ANOVA", "CHI2_RATIO","AON"],# Fitting metrics to use when determining the 
"thresholds": [0.95, 0.95, 0.10, 3.0],
"auto_stop":True, # automatically stop testing once threshold is reached; False test all no matter 
"plot_tests":True,
"force_best":True, # this forces the more-complex model to have a fit better than the previous.
"continue_fit":True, # continue the fit with the best chosen model
}
```

**A few caveats to line testing the user must be aware of!**

The process of determining the "correct" number of components for a line is a naturally degenerate problem, especially if the individual components are not resolved.  A number of underlying gaussian processes can define the observed emission line shape, and the "best fit" to those processes can be achived multiple ways depending on the allowed widths, velocities, and number of components you include in the model.  Because of this, a local minimization technique usually fails (such as linear least squares or Levenberg-Marquardt) unless the user supplies very accurate apriori guesses for each component.  Determining these apriori values defeats the purpose of automated fitting.  
Instead, BADASS implements a stochastic global minimizer (basinhopping) to ensure that the fit can achive a global minimum and not get stuck in local minima.  Basinhopping tends to be just as accurate as simulated or dual-annealing, but much faster than brute-force methods.  However, because it *is* stochastic, this can lead to inconsistent results between line components.  To this I say: it doesn't matter; if individual line components are not resolved, then only the total fit to the line profile shape matters, and is the greatest amount of information you can actually recover from your data.  
There are cases in which decomposing partially resolved narrow "core" components from another more-broad "outflow" component can be done (i.e., a two-component fit), and one can make reasonable assumptions about the physical nature of those components, but this typically ends at two components.  With more than 2 components, the fit can become highly degenerate *between* components, that is, you can achieve the same fit multiple ways depending on the allowed withs, velocities, and even initial guesses.  Discerning the physical nature of individual components for `ncomp`>2 for a given line should be treated with many ceveats unless those underlying physical processes are understood apriori. 

Now, if you're getting unexpected behavior from your multiple component tests, there are a number of things you can do:
* plot out each test to visually confirm that the fit is doing what you expected (set `plot_tests` to `True` in `test_options`.
* set higher `n_basinhop` threshold (15-25); this is the number of sucessive basinhopping thresholds before a solution is achieved. A low number means basinhopping gives up sooner (less time), and a higher threshold gives basinhopping a greater chance of finding the better fit (more time).  Often times, if `n_basinhop` is too low, the best fit is usually not achieved for a given test, which means that even if `force_best` is used, successive tests might not achieve their best fit either, because they just have to be better than the previous fit.
* check parameter limits; if a paramter such as line dispersion or velocity hits its limit, it might mean that a better fit could not be achieved because a line was not allowed to go outside of its defined parameter space to achieve a better fit.  This will happen if `voff` or `disp` parameter limits are too restrictive.  You can control global line parameter limits using `narrow_options`, `broad_options`, or `absorp_options`, or set them individually for each line using `disp_plim` (for dispersion) or `voff_plim` (for velocity offset). 
* check parameter constraints; if you have soft constraints (for example `["OIII_5007_2_DISP","OIII_5007_DISP"]`, which forces the second component dispersion to be greater than the first component dispersion), you may be over-constraining the model.  This can lead to `force_best` never reaching a good solution (or just taking a very long time).  The best fits are usually acheived when you don't use any soft constraints, especially when the number of components exceeds 2.
* simplify the continuum.  Lots of continuum flexibility can lead to strange behavior.  For this reason, we already restrict the polynomial continuum in the test regions to be 2. Removing some continuum components (e.g., FeII) might help. 
* check behavior of fit; if you really want to know what the testing is doing under the hood, set `"full_verbose":True` and BADASS will print every step of the testing process to screen.  Warning: its a lot of output, but its the only way to monitor how the fit is performing for each test.

As an example, lets see what happens if your parameter limits are too small.  Let's say we want to test up to 5 components for the [OIII]4960,5007 doublet, and we specified `"disp_plim":(0,300)"` for our `narrow_options`.  This means the none of the components for the lines we are testing may exceed 300 km/s, so let's see how well they do.  Here are the five tests in order:

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/test_bad_50.png)

This is a pretty complicated line, and we can see that it needs at least three components.  However, you can tell that by the time you get to three components, you still don't have a good fit.  More importantly, you can see that all the components (for each line) have the same width.  If you were to check the dispersion values for each component, they all maxed out at 300 km/s, which was our parameter limit on dispersion for narrow lines.

So now lets set `"disp_plim":(0,1000)"` for our `narrow_options`:

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/test_good_50.png)

Now we can see that by the time we test for three components the fit is pretty good, and doesn't get significantly better after that.  To check, let's look at the test results (printed to the log file and screen):

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/test_results.png)

We can see that covergence was reached at the test for 4 components (red box), which means that our confidence that 4 components significantly improves the fit is below our chosen thresholds, *thus* we choose the 3-component model.  Remember you can choose these thresholds to be as strict as you want. 


## Configuration Testing

Configuration testing is similar to line testing with the exception that the user explicitly provides the combinations of lines to be used in each test.  During line testing, BADASS controls the increasing complexity of the fit by adding line components, but in configuration testing BADASS has no such control.  Therefore, it is up to the user to explicitly provide the *order* of the configurations in increasing complexity if one desires to use the *force_best* option (each subsequent fit must be at least as good as the one before it).  To use configuration testing instead of line testing, set the `test_mode` to `config`
in the `test_options` dictionary passed to BADASS.  Be sure to pass the configurations you want to test to the `lines` in test options, like so:



```python

configs = [
    ["NA_H_BETA","NA_OIII_4960","NA_OIII_5007"], # Type 2 Case, single component
    ["NA_H_BETA","NA_OIII_4960","NA_OIII_5007","BR_H_BETA"], # Type 1 case, single component
    ["NA_H_BETA","NA_OIII_4960","NA_OIII_5007","NA_H_BETA_2","NA_OIII_4960_2","NA_OIII_5007_2","BR_H_BETA"], # Type 1 Case, double component,
    ["NA_H_BETA","NA_OIII_4960","NA_OIII_5007","NA_H_BETA_2","NA_OIII_4960_2","NA_OIII_5007_2","NA_H_BETA_3","NA_OIII_4960_3","NA_OIII_5007_3","BR_H_BETA"], # Type 1 Case, triple component,
    ["NA_H_BETA","NA_OIII_4960","NA_OIII_5007","NA_H_BETA_2","NA_OIII_4960_2","NA_OIII_5007_2","NA_H_BETA_3","NA_OIII_4960_3","NA_OIII_5007_3","BR_H_BETA","BR_H_BETA_2"], # Type 1 Case, triple component,
]

test_options = {
"test_mode":"config",
"lines": configs, # The lines to test
"metrics": ["BADASS", "ANOVA", "CHI2_RATIO","AON"],# Fitting metrics to use when determining the 
"thresholds": [0.95, 0.95, 0.10, 3.0],
"auto_stop":True, # automatically stop testing once threshold is reached; False test all no matter 
"plot_tests":True,
"force_best":True, # this forces the more-complex model to have a fit better than the previous.
"continue_fit":True, # continue the fit with the best chosen model
}
```

In the above `configs` list of line configurations, one can see that they are in *increasing* level of complexity.  This is required if you want the expected behavior from the `force_best` options.  If one mistakenly place a more-complex configuration before a simpler configuration (for instance, a configuration with fewer lines), BADASS will keep on attempting to achieve a better fit even though it can't, and BADASS will max-out the attemps at 1000 basinhopping iterations without finding a better fit.


## Line Component Options

These options are new for BADASS.  Previously, controlling the range of allowed velocity offset (`voff`) or line dispersion (`disp`) required the user to delve into the code to adjust these paramters.  Now the user can explicitly set these options before passing them to BADASS.  When not specified, BADASS will assume reasonable defaults.  Here are some examples of this usage for the available three line types (`narrow`, `broad`, or `absorp`):

```python
narrow_options = {
#     "amp_plim": (0,1), # line amplitude parameter limits; default (0,); changing this not recommended 
    "disp_plim": (0,500), # line dispersion parameter limits; default (0,)
    "voff_plim": (-500,500), # line velocity offset parameter limits; default (0,)
    "line_profile": "gaussian", # line profile shape*
    "n_moments": 4, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)
}
```

```python
broad_options ={
#     "amp_plim": (0,40), # line amplitude parameter limits; default (0,)
    "disp_plim": (500,3000), # line dispersion parameter limits; default (0,)
    "voff_plim": (-1000,1000), # line velocity offset parameter limits; default (0,)
    "line_profile": "gauss-hermite", # line profile shape*
    "n_moments": 4, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)
}
```

```python
absorp_options = {
#     "amp_plim": (-1,0), # line amplitude parameter limits; default (0,)
#     "disp_plim": (0,10), # line dispersion parameter limits; default (0,)
#     "voff_plim": (-2500,2500), # line velocity offset parameter limits; default (0,)
    "line_profile": "voigt", # line profile shape*
    "n_moments": 4, # number of higher order Gauss-Hermite moments (if line profile is gauss-hermite, laplace, or uniform)        
}
```

## User Lines, Constraints, and Masks

Additionally, the user can specify a non-default line list directly through the Notebook.  These override the default line list given in `line_list_default()`.

```python
user_lines = {
  "NA_UNKNOWN_1":{"center":6085., "line_type":"na", "line_profile":"gaussian"},
}
```

Similarly, one can provided additional soft constraints.  For example, if we wanted to constrain the narrow component of MgII to be narrower in width than the broad component, we specify that inequality in terms of the free parameters of that line:

```python
user_constraints = [
  ("BR_MGII_2799_DISP","NA_MGII_2799_DISP"),
]
```

Note that the format for soft constraints follows that of inequality constraints defined in `scipy.optimize.minimize()`, that is `(parameter_1 - parameter_2) >= 0` or `parameter_1 >= parameter_2`. 

Finally, we can also manually define masks in the fitting range using tuples specified as the lower and upper extent of the mask `(lower,upper)`:

```python
user_mask = [
     (4840,5015),
     (6552,6580),
     (6674,6685),
]
```
## Combined Lines 

One might be interested in the combined sum of two individual line components, and want to calculate the combined FWHM, dispersions, or velocity offsets.  One can define combinations of individual line components, the parameters of which will be computed at every iteration of the fit to include uncertainties on the combined components, which would otherwise be non-trivial in a post-analysis step.

Below we define a combined line for the narrow and broad components of H-beta:

```python
combined_lines = {
  "H_BETA_COMP"   :["NA_H_BETA","BR_H_BETA"],
}
```
*Note*: that combined lines are automatically generated for lines defined as multiple components (parent-child lines).  


## LOSVD Fitting Options (pPXF) 
```python
losvd_options = {
  "library"   : "IndoUS", # Options: IndoUS, Vazdekis2010
  "vel_const" :  {"bool":False, "val":0.0}, # Hold velocity constant?
  "disp_const":  {"bool":False, "val":250.0}, # Hold dispersion constant?
}
```

## Host Model Options

The host model is used as a simplified placeholder in the event that the stellar continuum isn't of any interest.  These are single-stellar population templates from the EMILES library, and do not have a low enough resolution for reliable stellar LOSVD fitting.

```python
host_options = {
  "age"       : [1.0,5.0,10.0], # Ages to include in Gyr; [0.09 Gyr - 14 Gyr] 
  "vel_const" : {"bool":False, "val":0.0}, # hold velocity constant?
  "disp_const": {"bool":False, "val":150.0} # hold dispersion constant?
}
```

## Power Law Options
```python
power_options = {
  "type" : "simple" # alternatively, "broken" for smoothly-broken power-law
}
```

## Polynomial Options
```python
poly_options = {
"apoly" : {"bool": True , "order": 3}, # Legendre additive polynomial 
"mpoly" : {"bool": False, "order": 3}, # Legendre multiplicative polynomial 
}
```

## Optical FeII Options

There are two FeII templates built into BADASS.  The default is the broad and narrow templates from [Véron-Cetty et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...417..515V/abstract) (`VC04`).  This model allows the user to have amplitude, dispersion, and velocity offset as free-parameters, with options to constrain them to constant values during the fit.  BADASS can also use the temperature-dependent template from [Kovačević et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010MSAIS..15..176K/abstract) (`K10`), which allows for the fitting of individual F, S, G, and I Zw 1 atomic transitions, as well as temperature.  The K10 template is best suited for modeling FeII in NLS1 objects with strong FeII emission.

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
# disp_const : constant disp (default True)
# voff_const : constant velocity offset (default True)
# temp_const : constant temp ('K10' only)

feii_options={
  'template'  :{'type':'VC04'}, 
  'amp_const' :{'bool':False,'br_feii_val':1.0,'na_feii_val':1.0},
  'disp_const':{'bool':True,'br_feii_val':3000.0,'na_feii_val':500.0},
  'voff_const':{'bool':True,'br_feii_val':0.0,'na_feii_val':0.0},
}
# or
# feii_options={
# 'template'  :{'type':'K10'},
# 'amp_const' :{'bool':False,'f_feii_val':1.0,'s_feii_val':1.0,'g_feii_val':1.0,'z_feii_val':1.0},
# 'disp_const':{'bool':False,'val':1500.0},
# 'voff_const':{'bool':False,'val':0.0},
# 'temp_const':{'bool':True,'val':10000.0} 
# }
################################################################################
```

## UV Iron Options
```python
uv_iron_options={
  "uv_amp_const"  :{"bool":False, "uv_iron_val":1.0}, # hold amplitude constant?
  "uv_disp_const" :{"bool":False, "uv_iron_val":3000.0},  # hold dispersion constant?
  "uv_voff_const" :{"bool":True,  "uv_iron_val":0.0}, # hold velocity constant?
}
```

## Balmer Pseudo-Continuum Options
```python
balmer_options = {
  "R_const" :{"bool":True,  "R_val":1.0}, # ratio between balmer continuum and higher-order lines
  "balmer_amp_const" :{"bool":False, "balmer_amp_val":1.0}, # hold amplitude constant?
  "balmer_disp_const" :{"bool":True,  "balmer_disp_val":5000.0}, # hold dispersion constant?
  "balmer_voff_const" :{"bool":True,  "balmer_voff_val":0.0}, # hold velocity constant?
  "Teff_const" :{"bool":True,  "Teff_val":15000.0}, # effective temperature
  "tau_const" :{"bool":True,  "tau_val":1.0}, # optical depth
}
```

## Plotting Options

```python
############################### Plotting options ###############################
plot_options={
  "plot_param_hist"    : True,# Plot MCMC histograms and chains for each parameter
  "plot_HTML"          : True,# make interactive plotly HTML best-fit plot
}
################################################################################
```

**`plot_param_hist`**: (*Default: True*)<br/>
For each free parameter fit by emcee, BADASS outputs a figure that contains a histogram of the MCMC samples, the best-fit values and uncertainties, and a plot of the MCMC chain, as shown below:

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_chain.png)

**`plot_HTML`**: (*Default: False*)<br/>
This will produce a best fit plot using [plotly](https://plotly.com/)  in an HTML format that can viewed interactively in a browser.  Note, plotly must be installed to use this function.

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
"write_options": False,  # output restart file
'verbose'  : True,  # prints steps of fitting process in Jupyter output
}
################################################################################
```

**`write_chain`**: (*Default: False*)<br/>
Write the full flattened MCMC chain (# parameters x # walkers x # iterations) to a FITS file.  We set this to *False*, because the file can get quite large, and takes up a lot of space if one is fitting many spectra.  One should only need this file if one wants to reconstruct chains and re-compute histograms. 

**`verbise`**: (*Default: True*)<br/>
Print output during the fit.  This is a good idea since it can tell you what BADASS is doing at that moment, and may help debug.  This should be disabled when fitting multiple spectra via multiprocessing.


## The Main Function and Calling Sequence

All of the above options are fed into the `run_BADASS()` function as such:

```python
# Call the main function in BADASS
badass.run_BADASS(pathlib.Path(file),
#                   options_file = "BADASS_options",
#                   restart_file         = restart_file,
                  fit_options          = fit_options,
                  mcmc_options         = mcmc_options,
                  comp_options         = comp_options,
                  # New line options
                  narrow_options       = narrow_options,
                  broad_options        = broad_options,
                  absorp_options       = absorp_options,
                  #
                  user_lines           = user_lines,       # User-lines
                  user_constraints     = user_constraints, # User-constraints
                  user_mask            = user_mask,        # User-mask
                  combined_lines       = combined_lines,
                  losvd_options        = losvd_options,
                  host_options         = host_options,
                  power_options        = power_options,
                  poly_options         = poly_options,
                  opt_feii_options     = opt_feii_options,
                  uv_iron_options      = uv_iron_options,
                  balmer_options       = balmer_options,
                  plot_options         = plot_options,
                  output_options       = output_options,
                 )
```

## Options/Configuration Files

A relatively new feature in BADASS are options files.  A method for implementing a configuration file for BADASS isn't easy to maintain the Python syntax as in the Notebook interface.  As a compromise, one can place all the above options in a `.py` script, and call `run_BADASS()` with the options file, which will overwrite any and all defaults:

```python
options_file = "BADASS_options"
# Call the main function in BADASS with an options file
badass.run_BADASS(pathlib.Path(file),
                  options_file = options_file,
                 )
```

**Warning**: Because this file overwrites all defaults, one should specify all lines and soft constraints in the options file as well.

# Output

BADASS produces a number of different outputs for the user at the end of the fitting process.  We summarize these below.

## Best-fit Model

This is simply a figure that shows the data, model, residuals, and best-fit components, to visually ascertain the quality of the fit. 

![_](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_bestfit.png)

## Parameter Chains, Histograms, Best-fit Values and Uncertainties

For every parameter that is fit, BADASS outputs a figure to visualize the full parameter chain and all walkers, the burn-in, and the final posterior histogram with the best-fit values and uncertainties.  The purpose of these is for visual inspection of the parameter chain to observe its behavior during the fitting process, to get a sense of how well initial parameters were fit, and how walkers behave as the fit progresses.

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_chain.png)

## Log File

The entire fitting process, including selected options, is recorded in a text file for reference purposes.  It provides:
- File location
- FITS Header information, such as (RA,DEC) coordinates, SDSS redshift, velocity scale (km/s/pixel), and *E(B-V)*, which is retrieved from NED online during the fitting process and used to correct for Galactic extinction.
- Results from outflow or line testing
- All best-fit values and/or uncertainties of all free and calculated model parameters.

![](https://github.com/remingtonsexton/BADASS3/blob/master/figures/BADASS_output_logfile.png)

The log file is meant as a summary of what occurred during the fit and the results, but it is not the easiest way to compile the results of fitting many objects...

## Best-fit Model Components

Best-fit model components are stored as arrays in `best_model_components.fits` files to reproduce the best-fit model figures as shown above.  This can be accessed using the [`astropy.io.fits`](https://docs.astropy.org/en/stable/io/fits/) module, for example:
```python
from astropy.io import fits

hdu = fits.open("best_model_components.fits")
tbdata = hdu[1].data     # FITS table data is stored on FITS extension 1
data   = tbdata['DATA']  # the SDSS spectrum 
wave   = tbdata['WAVE']  # the rest-frame wavelength vector
model  = tbdata['MODEL'] # the best-fit model
hdu.close()
```

Below we show an example data model for the FITS-HDU of `best_model_components.fits`.  You can print out the columns using 
```python
print(tbdata.columns)
```
which shows

```python
ColDefs(
    name = 'POWER'; format = 'E'
    name = 'NA_OPT_FEII_TEMPLATE'; format = 'E'
    name = 'BR_OPT_FEII_TEMPLATE'; format = 'E'
    name = 'NA_H_BETA'; format = 'E'
    name = 'NA_OIII_4960'; format = 'E'
    name = 'NA_OIII_5007'; format = 'E'
    name = 'BR_H_BETA'; format = 'E'
    name = 'OUT_H_BETA'; format = 'E'
    name = 'OUT_OIII_4960'; format = 'E'
    name = 'OUT_OIII_5007'; format = 'E'
    name = 'HOST_GALAXY'; format = 'E'
    name = 'DATA'; format = 'E'
    name = 'WAVE'; format = 'E'
    name = 'NOISE'; format = 'E'
    name = 'MODEL'; format = 'E'
    name = 'RESID'; format = 'E'
    name = 'MASK'; format = 'E'
)
```

## Best-fit Parameters and Uncertainties 

All best-fit parameter values and their upper and lower 1-sigma uncertainties are stored in `par_table.fits` files so they can be more quickly accessed than from a text file.  These are most easily accessed using the (`astropy.table`](https://docs.astropy.org/en/stable/table/pandas.html) module, which can convert a FITS table into a Pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html):

```
from astropy.table import Table  

table = Table.read('par_table.fits')
pandas_df = table.to_pandas()
print pandas_df
```
which shows

|    |             parameter |     best_fit  |   sigma_low  |   sigma_upp  |  | flag  |
| ---| --------------------- | -------------:| -----------: | -----------: |---:| ----: |
| 0  |      HOST_GALAXY_FLUX |  43179.789062 |  2146.635986 |  2523.700439 |...|  0.0 |
| 1  |       HOST_GALAXY_LUM |      2.993121 |     0.148800 |     0.174937 |...|  0.0 |
| 2  |        NA_HALPHA_AMP |    458.180511 |     2.664992 |     2.122619 |...|   0.0 |
| 3  |       NA_HALPHA_FLUX|   2304.035889 |    13.468710 |    19.964769 |...|   0.0 |
| .  |            .          |     .       |     .      |    .     |...|  .  |
| .  |            .          |     .       |     .      |    .     |...|  .  |
| .  |            .          |     .       |     .      |    .     |...|  .  |
| 57 |             POWER_AMP |     13.716220 |     0.670133 |     0.736425 |...|   0.0 |
| 58 |            POWER_FLUX |  47760.011719 |  2512.798340 |  2167.185303 |...|   0.0 |
| 59 |             POWER_LUM |      3.310611 |     0.174181 |     0.150224 |...|  0.0 |
| 60 |           POWER_SLOPE |     -0.705488 |     0.294422 |     0.352110 |...|   0.0 |
| 61 |             STEL_DISP |     99.216248 |     1.282358 |     0.785158 |...|   0.0 |
| 62 |              STEL_VEL |     97.768555 |     3.329233 |     2.168582 |...|   0.0 |

## Autocorrelation Analysis

BADASS will output the full history of parameter autocorrelation times and tolerances for every `write_iter` iterations.  This is done for post-fit analysis to assess how individual parameters behave as MCMC walkers converge on a solution.   Parameter autocorrelation times and tolerances are stored as arrays in a dictionary, which is saved as a numpy `.npy` file named `autocorr_dict.npy`, which can be accessed using the `numpy.load()` function: 

```python
autocorr_dict = np.load('autocorr_dict.npy')

# Display parameters in dictionary
for key in autocorr_dict.item():
        print key

# Print the autocorrelation times and tolerances for the 
# 'NA_OIII5007_VOFF' parameter and store them as 
# "tau" and "tol", respectively:
tau = autocorr_dict.item().get('NA_OIII5007_VOFF').get('tau')
tol = autocorr_dict.item().get('NA_OIII5007_VOFF').get('tol')
```
Note: `auto_stop` must be `True` in order to perform any autocorrelation analysis and output the autocorrelation files.

# Examples & Tutorials

## Single SDSS Spectrum

The [BADASS3_single_spectrum.ipynb](https://github.com/remingtonsexton/BADASS3/blob/master/example_notebooks/BADASS3_single_spectrum.ipynb) notebook shows the basics of setting up the fit of a single SDSS spectrum, from defining fit parameters to calling sequence.

![_](https://github.com/remingtonsexton/BADASS3/blob/master/figures/single_sdss_spectrum.png)

## Single Non-SDSS Spectrum

The [BADASS3_nonSDSS_single_spectrum.ipynb](https://github.com/remingtonsexton/BADASS3/blob/master/example_notebooks/BADASS3_nonSDSS_single_spectrum.ipynb) notebook illustrates the use of BADASS for a non-SDSS spectrum.  The user is expected to provide some basic information such as redshift, FWHM resolution, wavelength scale, and some form of a noise vector.  The FWHM resolution is necessary to accurately correct for instrumental dispersion and estimate the stellar LOSVD.  The noise vector need not be exact, since BADASS will scale the noise appropriately to achieve a reduced chi-squared of 1.

This example performs a fit on a Keck LRIS spectrum of a Seyfert 1 galaxy from [Sexton et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract):

![_](https://github.com/remingtonsexton/BADASS3/blob/master/figures/non_sdss_spectrum.png)

## Multiple Spectra with Multiprocessing

The [BADASS3_multi_spectra.ipynb](https://github.com/remingtonsexton/BADASS3/blob/master/example_notebooks/BADASS3_multi_spectra.ipynb) notebook shows how to set up BADASS to use Python's multiprocessing capabilities to fit any number of spectra simultaneously.  The number of spectra that can be *efficiently* fit simultaneously ultimately depends on the number of CPUs your machine has.  The number of simultaneous processes is the only parameter the user needs to specify, and BADASS assigns a process (a fit) to each core.

## Multiple Spectra with Multiprocessing

The [BADASS3_multi_spectra_options.ipynb](https://github.com/remingtonsexton/BADASS3/blob/master/example_notebooks/BADASS3_multi_spectra_options.ipynb) notebook is identical to the multi-spectra example above, except with fitting options passed via an options file.


## MANGA IFU Cube Data

Support for fitting IFU cubes is the newest feature of BADASS, spurred by the increasingly growing interest in studying outflows and AGN feedback using IFU data.  The [BADASS3_ifu_MANGA.ipynb](https://github.com/remingtonsexton/BADASS3/blob/master/example_notebooks/BADASS3_ifu_MANGA.ipynb) notebook shows how to fit the standardized cubes produced by MANGA.  BADASS can also utilize the voronoi binning [VorBin](https://www-astro.physics.ox.ac.uk/~cappellari/software/#binning) algorithm from [Cappellari & Copin (2003, MNRAS, 342, 345)](https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C/abstract), as well as multiprocessing to quickly (for Python at least) fit cubes.

![_](https://github.com/remingtonsexton/BADASS3/blob/master/figures/manga_cube_example.png)


## MUSE IFU Cube Data

By popular demand, we've included an additional example using a MUSE subcube in [BADASS3_ifu_MUSE.ipynb](https://github.com/remingtonsexton/BADASS3/blob/master/example_notebooks/BADASS3_ifu_MUSE.ipynb).  This is different from the Generic Cube example, which also fits a MUSE subcube, in that this example is an actual MUSE subcube, and not converted into NumPy arrays for the generic case.


## Generic IFU Cube Data

The [BADASS3_ifu_LVS_Rodeo_Cube.ipynb](https://github.com/remingtonsexton/BADASS3/blob/master/example_notebooks/BADASS3_ifu_LVS_Rodeo_Cube.ipynb) notebook, similarly illustrates how to fit generic cube data, similar to how non-SDSS spectra are fit in BADASS.  The user must provide some basic information about the data, but BADASS handles the data as standard NumPy arrays.

The LVS cube is a MUSE sub-cube of NGC 1386, which exhibits some complex line profiles, some of which are double-peaked.  In the provided example, we show how we can use BADASS' line testing functionality to *force* increasingly better fits with an increasing number of components to ensure that we fit the line in its entirety.  This is done by setting the largest number of lines for BADASS to include and setting all test metrics (except AON) to -1.  Since none of the test metrics can achieve values less than zero, BADASS will continue to add components until finished.  The result is a slow ramp-up in the number of components for a given fit to a line.  This stepwise fitting forces every additional fit with an extra component to be at least as good as the one before it, and *significantly increases the chances* (but never guaranteed) in obtaining the best possible fit.  This is particularly useful if the user doesn't care about how many components, and only if the best possible fit is needed.

Below are some results of the Rodeo Cube (MUSE subcube of NGC 1386) from the [Large-Volume Spectroscopic Analyses of AGN and Star Forming Galaxies in the Era of JWST](https://www.stsci.edu/contents/events/stsci/2022/march/large-volume-spectroscopic-analyses-of-agn-and-star-forming-galaxies-in-the-era-of-jwst) workshop, during which BADASS and its new features were showcased:
![_](https://github.com/remingtonsexton/BADASS3/blob/master/figures/LVS_double_peak.png)
![_](https://github.com/remingtonsexton/BADASS3/blob/master/figures/LVS_rodeo_example.png)




# How to

## Prior Distributions

The newest version of BADASS provides an easy way to implement some non-uniform priors, which can give the user more control over some parameters that may be poorly-constrained by a uniform prior or soft-constraints.

By default, all parameters are subject to an (improper, i.e. unnormalized) uniform prior with values [0,1], or [-inf,0] in log.  The value of this prior is zero if its respective parameter violates bounds and/or any constraints, and 1 if it does not.  This log-prior value is added to the log-likelihood.  

BADASS currently allows for three different non-uniform priors on any free parameter, all of which follow the `scipy.stats` standard:

**Normal (`gaussian`) Prior**: `stats.norm.logpdf(x,loc,scale)`; `loc`  and `scale` are the mean and standard deviation of the distribution, respectively.  If `loc` and/or `scale` are not provided, they will be calculated from the `init` (initial value) and `plim` (parameter limits) keywords that define each free parameter.  If not provided, the assumed `loc` is set equal to `init`, and `scale` is determined such that the difference between `init` and the absolute maximum `plim` is at 5 standard deviations, i.e., `scale = (|plim|-init)/5`. 

**Half-Norm (`halfnorm`) Prior**: `stats.halfnorm.logpdf(x,loc,scale)`; `loc`  and `scale` are the mean and standard deviation of the  positively bounded [0,+inf] distribution, respectively.  If `loc` and/or `scale` are not provided, they will be calculated from the `init` (initial value) and `plim` (parameter limits) keywords that define each free parameter.  If not provided, the assumed `loc` is set equal to `init`, and `scale` is determined such that the difference between `init` and the absolute maximum `plim` is at 5 standard deviations, i.e., `scale = (|plim|-init)/5`. 

**Log-Uniform (`jeffreys`) Prior**: `stats.loguniform.logpdf(x,a,b,loc)`; `loc` is the mean of the distribution, and `a` and `b` are shape parameters, where `0<a<b`.  The parameters `a` and `b` are determined from `plim` automatically, whereas `loc` can be specified by the user. 

By default, all model continuum priors (i.e., stellar kinematics, power law continuum, FeII emission) have uniform priors.  Changes to these priors can be made in the `initialize_pars()` function.  For line parameters, the only prior implemented across all lines is the velocity (`voff`) parameter, which is chosen to be `gaussian`, with a `loc=0` and `scale=100`.  This is done so that lines are biased to not stray too far way from where they are expected to be.   Gaussian priors are also put on any higher-order moments (`h3`, `h4`, ...) of the gauss-hermite, laplace, or uniform line profiles. 

To specify a prior on a parameter, one must provide the `kind`  (`gaussian`, `halfnorm`, or `jeffreys`).  Other parameters such as `loc` or `scale` are optional because they can be inferred from the `init` and `plim` keywords for each free parameter.

Here is an example of a 4-moment gauss-hermite line with a strict gaussian prior on `h3` and `h4`.  By doing this, it biases the moments to be very close to zero, and thus the gauss-hermite profile to be more gaussian, i.e., `h3~0` and `h4~0`.

```python
"BR_H_BETA"   :{"center":4862.691, 
                    "amp":"free", 
                    "disp":"free", 
                    "voff":"free",
                    "h3":"free",
                    "h3_prior":{"type":"gaussian","loc":0.0,"scale":0.01},
                    "h4":"free",
                    "h4_prior":{"type":"gaussian","loc":0.0,"scale":0.01},
                    "disp_plim":(500,5000),
                    "disp_init":1000.0,
                    "line_profile":"GH",
                    "line_type":"br"
                   },
```

## Line Lists

The default line list built into BADASS references most of the [standard SDSS lines](http://classic.sdss.org/dr6/algorithms/linestable.html).  The actual reference list is inside the `line_list_default()` function in the `badass.py` script.   BADASS expects a line entry to be in the following form

```python
"NA_OIII_5007" :{"center"   :5008.240, # rest-frame wavelength of line
         "amp"      :"free", # "free" parameter or tied to another valid parameter
         "amp_init" : float, # initial guess value 
         "amp_plim" : tuple, # tuple of (lower,upper) bounds of parameter
         
         "disp"     :"free", # "free" parameter or tied to another valid parameter
         "disp_init": float, # initial guess value 
         "disp_plim": tuple, # tuple of (lower,upper) bounds of parameter
         
         "voff"     :"free",  # "free" parameter or tied to another valid parameter
         "voff_init": float, # initial guess value 
         "voff_plim": tuple, # tuple of (lower,upper) bounds of parameter
         
         "line_type": "na", # line type ["na","br","out","abs", or "user]
         "line_profile": "gaussian" # gaussian, lorentzian, voigt, gauss-hermite, laplace, or uniform
         "label"    : string, # a name for the line for plotting purposes
         },
```

If `_init` or `_plim` keys are not explicitly assigned, BADASS will assume some reasonable values based on the `line_type`.  There are additional keys available when `line_profile` is Voigt (a `shape` key) or Gauss-Hermite (higher orders `h3`, `h4`, etc.).  Keep in mind that BADASS will enforce line profiles defined in the `fit_options` for `line_types` `na`, `br`, `out`, and `abs`; if one wants to define a custom line that isn't the same line profile shape as those defined in `fit_options`, one should use the `user` `line_type`.

For example:

```python
"NA_OIII_5007" :{"center"   : 5008.240, 
         "amp"      : "free",
         "disp"     : "free",
         "voff"     : "free",
         "line_type": "na",
         "label"    : r"[O III]"
         },
```

or in a more general case

```python
"RANDOM_USER_LINE" :{"center"      : 3094.394, # some random wavelength 
             "amp"         : "free",
             "disp"        : "free",
             "voff"        : "free",
             "h3"          : "free",
             "h4"          : "free",
             "line_type"   : "user",
             "line_profile": "gauss-hermite"
             "label"       : r"User Line"
         },
```

## Hard Constraints 

BADASS uses the `numexpr` module to allow for hard constraints on line parameters (i.e., "tying" one line's parameter's to another line's parameters).  To do this, the only requirement is that the constraint be a valid free parameter.  The most common case is tying the [OIII] doublet widths and velocity offsets:

```python
    "NA_OIII_4960" :{"center":4960.295,
             "amp":"(NA_OIII_5007_AMP/2.98)", 
             "disp":"NA_OIII_5007_DISP", 
             "voff":"NA_OIII_5007_VOFF", 
             "line_type":"na" ,
             "label":r"[O III]"
             },
             
    "NA_OIII_5007" :{"center":5008.240, 
             "amp":"free", 
             "disp":"free", 
             "voff":"free", 
             "line_type":"na" ,
             "label":r"[O III]"
             },

```
This works because when we define `NA_OIII_5007`, free parameters are created for the amplitude (`NA_OIII_5007_AMP`), dispersion (`NA_OIII_5007_DISP`) and velocity offset (`NA_OIII_5007_VOFF`), because we specified that they are *free* parameters.  These free parameters are the actual parameters that are solved for.  We can then reference those free valid parameters for `NA_OIII_4960`.  The power of the `numexpr` module is that we can also perform mathematical operations on those parameters *during* the fit, for example we can fix the amplitude of [OIII]4960 to be the [OIII]5007 amplitude divided by 2.93.  This makes implementing hard constraints very easy and is a very powerful feature.  With that said, you can do some pretty wild and unrealistic stuff, so use it responsibly.

## Soft Constraints

BADASS also uses the `numexpr` module to implement soft constrains on free parameters.  A soft constraint is defined here as a limit of a free parameter with respect to another free parameter, i.e., soft constraints are inequality constraints.  For example, if we want BADASS to enforce the requirement that broad H-beta has a greater dispersion than narrow [OIII]5007, we would say 

<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{110}\bg{black}(\rm{broad~H}\beta\rm{~DISP})&space;>=&space;(\rm{narrow~[OIII]5007~DISP})" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}(\rm{broad~H}\beta\rm{~DISP}) >= (\rm{narrow~[OIII]5007~DISP})" />

or in the way the `scipy.optimize()` module requires it

<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{110}\bg{black}(\rm{broad~H}\beta\rm{~DISP}&space;)-&space;&space;(\rm{narrow~[OIII]5007~DISP})&space;>=&space;0" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}(\rm{broad~H}\beta\rm{~DISP} )- (\rm{narrow~[OIII]5007~DISP}) >= 0" />

In BADASS, this soft constraint would be implemented as a tuple of length 2: 
```python
("BR_H_BETA_DISP","NA_OIII_5007_DISP")
```
By default BADASS includes the following list of soft constraints in the `initialize_pars()` function:
```python
soft_cons = [
      ("BR_H_BETA_DISP","NA_OIII_5007_DISP"), # broad H-beta width > narrow [OIII] width
      ("BR_H_BETA_DISP","OUT_OIII_5007_DISP"), # broad H-beta width > outflow width
      ("OUT_OIII_5007_DISP","NA_OIII_5007_DISP"), # outflow width > narrow [OIII] width
      ]
```


# Known Issues

We've done our best to find any bugs or issues.  BADASS is under constant development for this reason.  Please let us know if there are any features you'd like us to implement, or bugs that need to be fixed.

# Credits

- [Dr. Remington Oliver Sexton (USNO/GMU)](https://r-magnitude.com/) 
- [Sara M. Doan (GMU)](https://mason.gmu.edu/~sdoan2/)
- [Michael Reefe (GMU](https://github.com/Michael-Reefe)
- William Matzko (GMU)

# License

MIT License

Copyright (c) 2022 Remington Oliver Sexton

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

> Written with [StackEdit](https://stackedit.io/).
