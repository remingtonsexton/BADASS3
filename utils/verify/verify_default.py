from utils.verify.verify import ValueVerifier, CrossOptionVerifier

class DefaultVerifySet:
    def __init__(self, opts):
        self.opts = opts

    def verify(self):
        verifiers = []

        # I/O Options
        verifiers.append(ValueVerifier(['io_options', 'outdir'],
                                        (type(None), str),
                                        [],
                                        None,
                                        'Output directory must be a string'
                        ))

        # Fit Options
        verifiers.append(ValueVerifier(['fit_options', 'fit_reg'],
                                        (tuple,list),
                                        [
                                           lambda x: len(x)==2,
                                           lambda x: isinstance(x[0],(int,float)),
                                           lambda x: isinstance(x[1],(int,float)),
                                           lambda x: x[1]>x[0],
                                        ],
                                        (4400.0,5500.0),
                                        'Fitting region (fit_reg) must be an ordered tuple or list of the lower and upper wavelength limits (in Å) of the spectral region to be fit.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'good_thresh'],
                                        (int,float),
                                        [
                                            lambda x: (x>=0 and x<=1),
                                        ],
                                        0.0,
                                        'Good pixel threshold (good_thresh) must be an int or float between 0 and 1.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'mask_bad_pix'],
                                        (bool),
                                        [],
                                        False,
                                        'Mask SDSS-flagged bad pixels (mask_bad_pix) must be either True or False.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'mask_emline'],
                                        (bool),
                                        [],
                                        False,
                                        'Masking of emission lines (mask_emline) must be either True or False.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'mask_metal'],
                                        (bool),
                                        [],
                                        False,
                                        'Interpolation over metal absorption lines (mask_metal) must be either True or False.'
                        ))

        # TODO: make fit_stat options constants/enum
        verifiers.append(ValueVerifier(['fit_options', 'fit_stat'],
                                        (str),
                                        [
                                            lambda x: x in ["ML","OLS","RCHI2"],
                                        ],
                                        'ML',
                                        'Fit statistic can be either ML (Maximum Likelihood) or LS (Least Squares) , RCHI2 (Reduced Chi-Squared).'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'n_basinhop'],
                                        (int),
                                        [
                                            lambda x: (x>=0),
                                        ],
                                        5,
                                        'Number of consecutive successful basin-hopping iterations before maximum-likelihood fit is acheived must be an integer.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'test_outflows'],
                                        (bool),
                                        [],
                                        False,
                                        'Testing for outflows (test_outflows) must be either True or False.'
                        ))

        # TODO: remove need for 'bool' dict key
        verifiers.append(ValueVerifier(['fit_options', 'test_line'],
                                        (dict),
                                        [],
                                        {'bool':False, 'line':'br_H_beta'},
                                        'Line test (test_line) must be a dict.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'test_line', 'bool'],
                                        (bool),
                                        [],
                                        False,
                                        'test_line bool must be True or False.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'test_line', 'line'],
                                        (str,tuple,list),
                                        [],
                                        'br_H_beta',
                                        'test_line line must be string and a valid line, or list of lines, from the line list.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'max_like_niter'],
                                        (int),
                                        [
                                            lambda x: (x>=0),
                                        ],
                                        10,
                                        'Number of bootstrap iterations used for initial maximum-likelihood fit must be an integer.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'output_pars'],
                                        (bool),
                                        [],
                                        False,
                                        'Only output parameters of fit (output_pars) must be True or False.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'cosmology'],
                                        (dict),
                                        [],
                                        {"H0":70.0, "Om0": 0.30},
                                        'Flat Lambda-CDM Cosmology must be a dict with H0 and Om0 specified.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'cosmology', 'H0'],
                                        (int,float),
                                        [],
                                        70.0,
                                        'Hubble parameter must be an integer or float.'
                        ))

        verifiers.append(ValueVerifier(['fit_options', 'cosmology', 'Om0'],
                                        (float),
                                        [],
                                        0.30,
                                        'Density parameter must be a float.'
                        ))


        # MCMC Options
        verifiers.append(ValueVerifier(['mcmc_options', 'mcmc_fit'],
                                        (bool),
                                        [],
                                        False,
                                        'Toggle MCMC fitting (mcmc_fit) must be a bool.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'nwalkers'],
                                        (int),
                                        [
                                            lambda x: x>0,
                                        ],
                                        100,
                                        'Number of MCMC walkers (nwalkers) must be an integer.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'auto_stop'],
                                        (bool),
                                        [],
                                        False,
                                        'Automatic stop using autocorrelation (auto_stop) must be a bool.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'conv_type'],
                                        None,
                                        [
                                            lambda x: x in ["mean","median","all"] or isinstance(x,tuple),
                                        ],
                                        'median',
                                        'Type of autocorrelation convergence (conv_type) must be a string.  Options are: \'mean\', \'median\', \'all\', or a tuple of valid free parameters.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'min_samp'],
                                        (int),
                                        [
                                            lambda x: x>0,
                                        ],
                                        100,
                                        'Minimum number of sampling iterations (min_samp) before autocorrelation convergence must be an integer.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'ncor_times'],
                                        (int,float),
                                        [
                                            lambda x: x>=0,
                                        ],
                                        5.0,
                                        'Number of autocorrelation times (ncor_times) required before convergence is reached must be an integer or float.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'autocorr_tol'],
                                        (int,float),
                                        [
                                            lambda x: x>=0,
                                        ],
                                        10.0,
                                        'Autocorrelation tolerance (autocorr_tol) required before convergence is reached must be an integer or float.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'write_iter'],
                                        (int),
                                        [
                                            lambda x: x>0,
                                        ],
                                        100,
                                        'Iteration frequency to write to chain (write_iter) must be an integer at least equal to max_iter.'
                        ))

        verifiers.append(CrossOptionVerifier([lambda x: x['mcmc_options']['write_iter'] <= x['mcmc_options']["max_iter"]],
                                                'Iteration frequency to write to chain (write_iter) must be an integer at least equal to max_iter.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'write_thresh'],
                                        (int),
                                        [
                                            lambda x: x>0,
                                        ],
                                        100,
                                        'Iteration at which to begin writing (write_thresh) to chain must be an integer.'
                        ))

        verifiers.append(CrossOptionVerifier([lambda x: x['mcmc_options']['write_thresh'] <= x['mcmc_options']["max_iter"]],
                                                'Iteration at which to begin writing (write_thresh) to chain must be an integer.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'burn_in'],
                                        (int),
                                        [
                                            lambda x: x>0,
                                        ],
                                        1000,
                                        'Burn-in iteration (burn_in) to use if maximum number of iterations (max_iter) is reached must be an integer.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'min_iter'],
                                        (int),
                                        [
                                            lambda x: x>0,
                                        ],
                                        100,
                                        'Minimum number of iterations (min_iter) to perform before autocorrelation checking is performed.'
                        ))

        verifiers.append(ValueVerifier(['mcmc_options', 'max_iter'],
                                        (int),
                                        [
                                            lambda x: x>0,
                                        ],
                                        1500,
                                        'Maximum number of iterations (max_iter) to perform before stopping.'
                        ))

        verifiers.append(CrossOptionVerifier([lambda x: x['mcmc_options']['max_iter'] >= x['mcmc_options']["min_iter"]],
                                                'Maximum number of iterations (max_iter) to perform before stopping.'
                        ))


        # Comp Options
        verifiers.append(ValueVerifier(['comp_options', 'fit_opt_feii'], (bool), [], True, 'fit_opt_feii must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_uv_iron'], (bool), [], False, 'fit_uv_iron must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_balmer'], (bool), [], False, 'fit_balmer must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_losvd'], (bool), [], False, 'fit_losvd must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_host'], (bool), [], True, 'fit_host must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_power'], (bool), [], True, 'fit_power must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_narrow'], (bool), [], True, 'fit_narrow must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_broad'], (bool), [], True, 'fit_broad must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_outflow'], (bool), [], True, 'fit_outflow must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'fit_absorp'], (bool), [], True, 'fit_absorp must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'tie_line_fwhm'], (bool), [], False, 'tie_line_fwhm must be a bool.'))
        verifiers.append(ValueVerifier(['comp_options', 'tie_line_voff'], (bool), [], False, 'tie_line_voff must be a bool.'))

        # TODO: make enum
        verifiers.append(ValueVerifier(['comp_options', 'na_line_profile'],
                                        (str),
                                        [
                                            lambda x: x in ['G', 'L', 'GH', 'V'],
                                        ],
                                        'G',
                                        'na_line_profile must be a string.'
                        ))

        verifiers.append(ValueVerifier(['comp_options', 'br_line_profile'],
                                        (str),
                                        [
                                            lambda x: x in ['G', 'L', 'GH', 'V'],
                                        ],
                                        'G',
                                        'br_line_profile must be a string.'
                        ))

        verifiers.append(ValueVerifier(['comp_options', 'out_line_profile'],
                                        (str),
                                        [
                                            lambda x: x in ['G', 'L', 'GH', 'V'],
                                        ],
                                        'G',
                                        'out_line_profile must be a string.'
                        ))

        verifiers.append(ValueVerifier(['comp_options', 'abs_line_profile'],
                                        (str),
                                        [
                                            lambda x: x in ['G', 'L', 'GH', 'V'],
                                        ],
                                        'G',
                                        'abs_line_profile must be a string.'
                        ))

        verifiers.append(ValueVerifier(['comp_options', 'n_moments'],
                                        (int),
                                        [
                                            lambda x: (x>=2) & (x<=10),
                                        ],
                                        4,
                                        'Higher-order moments for Gauss-Hermite line profiles must be >= 2 or <= 10.'
                        ))

        # TODO:
        #   if fit_losvd & fit_host: fit_host = False




        verifiers.append(ValueVerifier(['outflow_test_options'], (dict), [], {}, ''))





        [v.verify(self.opts) for v in verifiers]
