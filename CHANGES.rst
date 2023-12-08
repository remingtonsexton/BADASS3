10.0.0 - 10.3.0
===============
- New generalized line component option for easily adding n number of line components; deprecates 'outflow' components.
- W80 now a standard line parameter.
- New testing suite that incorporates testing for multiple components, lines, metrics, etc. with the 
  ability to continue fitting with the best model.
- Fixed autocorrelation time calculation; now always produces a time.
- PPOLY polynomial no longer an option pending bug fixes.
- To avoid an excessive number of plots, we now limit plotting of histograms to free fitted parameters.
- Configuration testing.
- Removed astro-bifrost due to grpcio dependency problems; using old metal masking algorithm until fixed.
- BADASS now internally normalizes the spectrum for optimization reasons.
- Bug fixes and minor changes.
- Reweighting noise to achieve RCHI2=1 now done prior to bootstrapping and MCMC as needed (no longer a free parameter, which made it numerically unstable).


9.3.1
=====
- BADASS-IFU
	- S/R computed at 5100 angstroms (rest frame) by default for use when using Voronoi binning
- Bug fixes, and edits to default line list
- Add explicit flat prior
- Add flux normalization option (default is SDSS normalization of 1.E-17)
- Fixed output line SNR to be calculated even if NPIX <1
- Constraint and initial value checking before fit takes place to prevent crashing
- Implemented restart file; saves all fitting options to restart fit


9.3.0
=====
- NPIX and SNR (signal-to-noise ratio) is computed for all lines and now includes an uncertainty.
- Removed interpolation inside of fit_model() to reduce computational expense.
- General bug fixes and cleaning up.


9.2.0 - 9.2.2
=============
- Options for different priors on free parameters
- Normalization for log-likelihoods
- Outflow test region fix


9.1.7
=====
- Switched width parameter of all lines from FWHM to dispersion to accomodate more lines and avert problems with biased integrated velocities and dispersions. As a result, integrated dispersions and velocities are only calculated for combined lines, and FWHM are calculated for ALL lines.
- Added Laplace and Uniform line profiles from Sanders et al. (2020) (https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5806S/abstract; https://github.com/jls713/gh_alternative)
- Changed instrumental fwhm keyword to instrumental dispersion "disp_res".  Input resolution for user input spectra is still a "fwhm_res" but changes to disp_res internally.


9.1.6
=====
- Polynomial continuum components independent from LOSVD component
- Linearization of non-linearized non-SDSS spectra using spectres module


9.0.0 - 9.1.1
=============
- Options for likelihood function
- Consolidated outflow and line testing routines


8.0.14 - 8.0.15
===============
- Regular expressions now supported for soft constraints
- IFU support for MANGA and MUSE (General) datasets


8.0.0 - 8.0.13
==============
- Added smoothly broken power-law spectrum for high-z objects
- Optimized FeII template fitting by utilizing PPXF framework
- Added UV FeII+FeIII template from Vestergaard & Wilkes (2001)
- Added Balmer continuum component
- Added equivalent width calculations
- Added additional chisquared fit statistic for outflow test
- Voigt and Gauss-Hermite line profile options, with any number of higher order moments
- Emission line list options (default and user-specified)
- Control over soft- and hard constraints
- Option for non-SDSS spectrum input
- Interpolation over metal absorption lines
- Masking of bad pixels, strong emission+absorption lines (automated), and user-defined masks
- Various bug fixes, plotting improvements
- New hypothesis testing for lines and outflows (F-test remains unchanged)
- Continuum luminosities at 1350 Å, 3000 Å, and 5100 Å.
- pathlib support
- Corner plots (corner.py) no longer supported; user should make their own corner plots with fewer free parameters
- Removed BPT diagram function; user should make BPT diagrams post processing


7.7.2 - 7.7.6
=============
- Fixed problem with FeII emission lines at the edge of the fitting region. This is done by setting the variable edge_pad=0.
- Fixed F-test NaN confidence bug
- Updated initial fitting parameters in Jupyter notebook
- Bug fixes and fixes to plots


7.7.1
=====
- MNRAS Publication Version
- Added statistical F-test for ascertaining confidence between single-Gaussian and double-Gaussian models for the outflow test. Removed the ratio-of-variance test and replaced it with a sum-of-squares of residuals ratio.
- Added "n_basinhop" to fit_options, which allows user to choose how many initial basinhopping success iterations before a solution is achieved. This can drastically reduce the basinhopping fit time, at the expense of fit quality.
- Bug fixes


7.7.0
=====
- NLS1 support; more detailed option for FeII template fitting (fwhm and voff fitting options); Lorentzian emission line profile option.
- Kovacevic et al. 2010 FeII template added, which includes a paramter for temperature.
- Relaxed wavelength requirement for outflow tests for higher-redshift targets.


7.6.0 - 7.6.8
=============
- Writing no-outflow parameters from test_outflows run to log file.
- bug fixes


7.5.0 - 7.5.3
=============
- Test outflow residual statistic replaced with f-statistic (ratio-of-variances) to compare model residuals.
- Added interpolation of bad pixels based on SDSS flagged pixels.
- bug fixes


7.4.1 - 7.4.3
=============
- Writing outflow test metrics to log file for post-fit analysis.
- Improved outflow/max-likelihood fitting using scipy.optimize.basinhopping. While basinhopping algorithm requires more runtime, it produces a significantly better fit, namely for the power-law slope parameter which never varies with the SLSQP algorithm due to the fact that it is stuck in a local minima.
- Added F-statistic (ratio of variances between no outflow and outflow model).
- Changed default outflow statistic settings.
- Bug fixes; fixed problems with parameters in 'list' option conv_type getting removed.  Now if a user-defined parameter in conv_type is wrong or removed, it uses the remaining valid parameters for convergence, or defaults to 'median'.


7.4.0
=====
- Changes to how outflow tests are performed; different residual improvement metric.
- New default host galaxy template for non-LOSVD fitting; using MILES 10.0 Gyr SSP with a dispersion of 100 km/s that better matches absorption features.


7.3.1 - 7.3.3
=============
- bug fixes


7.3.0
=====
- Feature additions; Jupyter Notebook now supports multiprocessing in place of for loops which do not release memory.
- Outflow test options; outflow fitting no longer constrains velocity offset to be less than core (blueshifted), and now only tests for blueshifts if 
  option is selected. Only amplitude and FHWM are constrained.
- Better outflow testing; test now compare outflow to no-outflow models to check if there is significant improvement in residuals, as well as flags
  models in which the bounds are reached and good fits cannot be determined.


7.2.0
=====
- Feature additions; one can suppress print output completely for use when running multiprocessing pool.


7.1.0
=====
- Fixed a critical bug in resolution correction for emission lines.
- misc. bug fixes


7.0.0
=====
- Added minimum width for emission lines which improves outflow testing; this is based on the dispersion element of a single noise spike.
- Emission lines widths are now measured as Gaussian dispersion (disp) instead of Gaussian FWHM (fwhm).
- Added warning flags to best fit parameter files and logfile if parameters are consistent with lower or upper limits to within 1-sigma.
- While is is *not recommended*, one can now test for outflows in the H-alpha/[NII] region independently of the H-beta/[OIII] region, as well as
  fit for outflows in this region.  However, if the region includes H-beta/[OIII], then the default constraint is to still use [OIII]5007 to constrain 
  outflow amplitude, dispersion, and velocity offset.
- Plotting options, as well as corner plot added (defualt is *not* to output this file because there is lots of overhead).
- More stable outflow testing and maximum likelihood estimation.


6.0.0
=====
- Improved autocorrelation analysis and options.  One can now choose the number of autocorrelation times and tolerance for convergence.
  Posterior sampling now restarts if solution jumps prematurely out of convergence.
- Simplified the Jupyter Notebook control panel and layout.  Most of the BADASS machinery is now contained in the badass_v6_0.py file.
- Output of black hole mass based on availability of broad line (based on Woo et al. (2015) (https://ui.adsabs.harvard.edu/abs/2015ApJ...801...38W/abstract) H-alpha BH mass estimate, and Sexton et al. (2019) (https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract) H-beta BH mass estimate.
- Output of systemic stellar velocity (redshift) and it's uncertainty.
- Output of BPT diagnostic ratios and plot if both H$\alpha$ and H$\beta$ regions are fit simultaneously.
- Minor memory leak improvements by optimizing plotting functions and deleting large arrays from memory via garbage collection.
- Fixed issues with the outflow test function.
- Added minimum S/N option for fitting the LOSVD.
- MCMC fitting with emcee is now optional with `mcmc_fit`; one can fit using only Monte Carlo bootstrapping with any number of `max_like_niter` iterations
  to estimate uncertainties if one does not require a fit of the LOSVD.  If you need LOSVD measurements, you still must (and *should*) use emcee.
- One can now perform more than a single maximum likelihood fit for intial parameter values for emcee by changing `max_like_niter`, be advised this will 
  take longer for large regions of spectra, but generally produces better initial parameter values.
- BPT diagnostic classification includes the classic Kewley+01 & Kauffmann+03 diagram to separate starforming from AGN dominated objects, but also the [SII] diagnostic to distinguish Seyferts from LINERs.  The BPT classification is now written to the log file.
- Store autocorrelation times and tolerances for each parameter in a dictionary and save to a `.npy` file.
- Cleaned up Notebook.
- Major changes and improvements in how monte carlo bootstrapping is performed for maximum likelihood and outflow testing functions.


1.0.0 - 5.0.0
=============
- Very unstable, lots of bugs, kinda messy, not many options or features.  We've made a lot of front- and back-end changes and improvements.
- Versions 1-4 were not very flexible, and were originally optimized for Keck LRIS spectra (See 
  [Sexton et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...878..101S/abstract)) and then optimized for large samples of SDSS spectra.
- In Version 5 we performed a complete overhaul with more options, features.  The most improved-upon feature was the addition of autocorrelation
  analysis for parameter chain convergence, which now produces the most robust estimates.
