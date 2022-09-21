from cerberus import Validator
from cerberus import rules_set_registry

import utils.constants as consts

# short hands for some common type rules
rules_set_registry.add('bool_false', {'type':'boolean', 'default':False})
rules_set_registry.add('bool_true', {'type':'boolean', 'default':True})
rules_set_registry.add('intfl_min0_d1', {
                                            'type': ['integer', 'float'],
                                            'min': 0.0,
                                            'default': 1.0,
                                        })
rules_set_registry.add('intfl_d0', {
                                        'type': ['integer', 'float'],
                                        'default': 0.0,
                                   })
rules_set_registry.add('line_profile', {'type':'string', 'allowed':consts.LINE_PROFILES, 'default':'G'})


class DefaultValidator(Validator):
    def _validate_min_ex(self, test_val, field, value):
        if value <= test_val:
            self._error(field, '%s must be strictly greater than %s' % (field, str(test_val)))


    def _validate_max_ex(self, test_val, field, value):
        if value >= test_val:
            self._error(field, '%s must be strictly less than %s' % (field, str(test_val)))


    def _validate_lt_other(self, other, field, value):
        if other not in self.document:
            return False
        if value >= self.document[other]:
            self._error(field, '%s must be less than %s' % (field, other))


    def _validate_gt_other(self, other, field, value):
        if other not in self.document:
            return False
        if value <= self.document[other]:
            self._error(field, '%s must be greater than %s' % (field, other))


    def _validate_le_other(self, other, field, value):
        if other not in self.document:
            return False
        if value > self.document[other]:
            self._error(field, '%s must be less than or equal to %s' % (field, other))


    def _validate_ge_other(self, other, field, value):
        if other not in self.document:
            return False
        if value < self.document[other]:
            self._error(field, '%s must be greater than or equal to %s' % (field, other))


    def _validate_is_lohi(self, constraint, field, value):
        if not constraint:
            return

        if (isinstance(value, list)) or (len(value) != 2) or (value[1] < value[0]):
            self._error(field, '%s must be a list of length 2' % field)


    def _normalize_coerce_nonzero(self, value):
        if value <= 0:
            return 1.e-3
        return value



# NOTE: Any container objects (ie. dicts, lists, etc.) needs to have a default
#       empty object ({}, [], etc.) in order for its child objects/values to
#       have their defaults set

# io_options
DEFAULT_IO_OPTIONS = {
    'infmt': {
        'type': 'string',
        'default': 'sdss_spec',
    },
    'output_dir': {
        'type': 'string',
        'nullable': True,
    },
    'dust_cache': {
        'type': 'string',
        'nullable': True,
    },
    'log_level': {
        'type': 'string',
        'default': 'info',
    },
}


# fit_options
DEFAULT_FIT_OPTIONS = {
    # Fitting region; Note: Indo-US Library=(3460,9464)
    'fit_reg': {
        'type': 'list',
        'minlength': 2,
        'maxlength': 2,
        'schema': {'type': ['integer', 'float']},
        'default': (4400.0, 5500.0),
    },
    # percentage of "good" pixels required in fig_reg for fit.
    'good_thresh': {
        'type': ['integer', 'float'],
        'min': 0,
        'max': 1,
        'default': 0.0,
    },
    # mask pixels SDSS flagged as 'bad' (careful!)
    'mask_bad_pix': 'bool_false',
    # mask emission lines for continuum fitting.
    'mask_emline': 'bool_false',
    # interpolate over metal absorption lines for high-z spectra
    'mask_metal': 'bool_false',
    # fit statistic; ML = Max. Like. , LS = Least Squares
    'fit_stat': {
        'type': 'string',
        'allowed': consts.FIT_STATS,
        'default': 'ML',
    },
    # Number of consecutive basinhopping thresholds before solution achieved
    'n_basinhop': {
        'type': 'integer',
        'min': 0,
        'default': 5,
    },
    # only test for outflows; stops after test
    'test_outflows': 'bool_false',
    'test_line': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'line': {
                'type': ['string', 'list'],
                'default': 'br_H_beta',
            },
        },
    },
    # number of maximum likelihood iterations
    'max_like_niter': {
        'type': 'integer',
        'min': 0,
        'default': 10,
    },
    # only output free parameters of fit and stop code (diagnostic)
    'output_pars': 'bool_false',
    'cosmology': {
        'type': 'dict',
        'default': {},
        'schema': {
            'H0': {
                'type': ['integer', 'float'],
                'default': 70.0,
            },
            'Om0': {
                'type': 'float',
                'default': 0.30,
            },
        },
    },
}


# ifu_options
DEFAULT_IFU_OPTIONS = {
    'z': 'intfl_d0',
    'aperture': {
        'type': 'list',
        'minlength': 4,
        'maxlength': 4,
        'schema': {'type': 'integer',},
    },
    'voronoi_binning': 'bool_true',
    # Target S/N ratio to bin for
    'targetsn': {
        'type': ['integer', 'float'],
        'nullable': True,
        'default': None,
    },
    'cvt': 'bool_true',
    'voronoi_plot': 'bool_true',
    'quiet': 'bool_true',
    'wvt': 'bool_false',
    'maxbins': {
        'type': 'integer',
        'default': 800,
    },
    'snr_threshold': {
        'type': ['integer', 'float'],
        'min': 0,
        'default': 3.0,
    },
    'use_and_mask': 'bool_true',
}


# mcmc_options
DEFAULT_MCMC_OPTIONS = {
    # Perform robust fitting using emcee
    'mcmc_fit': 'bool_false',
    # Number of emcee walkers; min = 2 x N_parameters
    'nwalkers': {
        'type': 'integer',
        'min_ex': 0,
        'default': 100,
    },
    # Automatic stop using autocorrelation analysis
    'auto_stop': 'bool_false',
    'conv_type': {
        'type': ['string', 'list'],
        'allowed': ['mean','median','all'],
        'default': 'median',
    },
    # min number of iterations for sampling post-convergence
    'min_samp': {
        'type': 'integer',
        'min_ex': 0,
        'default': 100,
    },
    # number of autocorrelation times for convergence
    'ncor_times': {
        'type': ['integer', 'float'],
        'min': 0,
        'default': 5.0,
    },
    # percent tolerance between checking autocorr. times
    'autocorr_tol': {
        'type': ['integer', 'float'],
        'min': 0,
        'default': 10.0,
    },
    # write/check autocorrelation times interval
    'write_iter': {
        'type': 'integer',
        'min_ex': 0,
        'lt_other': 'max_iter',
        'default': 100,
    },
    # when to start writing/checking parameters
    'write_thresh': {
        'type': 'integer',
        'min_ex': 0,
        'lt_other': 'max_iter',
        'default': 100,
    },
    # burn-in if max_iter is reached
    'burn_in': {
        'type': 'integer',
        'min_ex': 0,
        'default': 1000,
    },
    # min number of iterations before stopping
    'min_iter': {
        'type': 'integer',
        'min_ex': 0,
        'default': 100,
    },
    # max number of MCMC iterations
    'max_iter': {
        'type': 'integer',
        'min_ex': 0,
        'ge_other': 'min_iter',
        'default': 1500,
    },
}


# comp_options
DEFAULT_COMP_OPTIONS = {
    'fit_opt_feii': 'bool_true', # optical FeII
    'fit_uv_iron': 'bool_false', # UV Iron 
    'fit_balmer': 'bool_false', # Balmer continuum (<4000 A)
    'fit_losvd': 'bool_false', # stellar LOSVD
    'fit_host': 'bool_true', # host template
    'fit_power': 'bool_true', # AGN power-law
    'fit_narrow': 'bool_true', # narrow lines
    'fit_broad': 'bool_true', # broad lines
    'fit_outflow': 'bool_true', # outflow lines
    'fit_absorp': 'bool_true', # absorption lines
    'tie_line_fwhm': 'bool_false', # tie line widths
    'tie_line_voff': 'bool_false', # tie line velocity offsets
    'na_line_profile': 'line_profile', # narrow line profile
    'br_line_profile': 'line_profile', # broad line profile
    'out_line_profile': 'line_profile', # outflow line profile
    'abs_line_profile': 'line_profile', # absorption line profile
    # number of higher-order moments for GH line profiles
    'n_moments': {
        'type': 'integer',
        'min': 2,
        'max': 10,
        'default': 4
    }
}


# user_constraints
DEFAULT_USER_CONSTRAINTS = {
    'type': 'list',
    'minlength': 2,
    'maxlength': 2,
    'schema': {
        'type': 'string',
    },
}


# user_mask
DEFAULT_USER_MASK = {
    'type': 'list',
    'minlength': 2,
    'maxlength': 2,
    'schema': {
        'type': ['integer', 'float'],
    },
    'is_lohi': True,
}


# power_options
DEFAULT_POWER_OPTIONS = {
    'type': {
        'type': 'string',
        'allowed': ['simple', 'broken'],
        'default': 'simple',
    },
}


# losvd_options
DEFAULT_LOSVD_OPTIONS = {
    'library': {
        'type': 'string',
        'allowed': list(consts.LOSVD_LIBRARIES.keys()),
        'default': 'IndoUS'
    },
    'vel_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'val': {
                'type': ['integer', 'float'],
                'min': 0,
                'default': 0.0
            },
        },
    },
    'disp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'val': {
                'type': ['integer', 'float'],
                'min': 1.e-3,
                'coerce': 'nonzero',
                'default': 100.0,
            },
        },
    },
    'losvd_apoly': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'order': {
                'type': 'integer',
                'min': 0,
                'default': 3,
            },
        },
    },
}


# host_options
DEFAULT_HOST_OPTIONS  = {
    'age' : {
        'type': 'list',
        'default': [0.1,1.0,10.0],
        'schema': {
            'type': ['integer', 'float'],
            'allowed': [0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0],
        }
    },
    'vel_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'val': {
                'type': ['integer', 'float'],
                'min': 0,
                'default': 0.0
            },
        },
    },
    'disp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'val': {
                'type': ['integer', 'float'],
                'min': 1.e-3,
                'coerce': 'nonzero',
                'default': 100.0,
            },
        },
    },
}


DEFAULT_OPT_FEII_VC04_OPTIONS = {
    'opt_template' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'type': {
                'type': 'string',
                'allowed': ['VC04'],
                'default': 'VC04',
            },
        },
    },
    'opt_amp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'br_opt_feii_val' : 'intfl_min0_d1',
            'na_opt_feii_val' : 'intfl_min0_d1',
        },
    },
    'opt_fwhm_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'br_opt_feii_val' : {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 3000.0,
            },
            'na_opt_feii_val' : {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 500.0,
            },
        },
    },
    'opt_voff_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'br_opt_feii_val' : 'intfl_d0',
            'na_opt_feii_val' : 'intfl_d0',
        },
    },
}


DEFAULT_OPT_FEII_K10_OPTIONS = {
    'opt_template' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'type': {
                'type': 'string',
                'allowed': ['K10'],
                'default': 'K10',
            },
        },
    },
    'opt_amp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'f_feii_val' : 'intfl_min0_d1',
            's_feii_val' : 'intfl_min0_d1',
            'g_feii_val' : 'intfl_min0_d1',
            'z_feii_val' : 'intfl_min0_d1',
        },
    },
    'opt_fwhm_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'opt_feii_val' : {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 1500.0,
            },
        },
    },
    'opt_voff_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'opt_feii_val' : 'intfl_d0',
        },
    },
    'opt_temp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'opt_feii_val' : {
                'type': ['integer', 'float'],
                'min': 0.0,
                'default': 10000.0,
            },
        },
    },
}


DEFAULT_UV_IRON_OPTIONS = {
    'uv_amp_const' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'uv_iron_val': 'intfl_min0_d1',
        },
    },
    'uv_fwhm_const' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'uv_iron_val': {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 3000.0,
            },
        },
    },
    'uv_voff_const' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'uv_iron_val': 'intfl_d0',
        },
    },
    'uv_legendre_p' : {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'uv_iron_val': {
                'type': 'integer',
                'default': 3,
            },
        },
    },
}


DEFAULT_BALMER_OPTIONS = {
    # ratio between balmer continuum and higher-order balmer lines
    'R_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'R_val': {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 0.5,
            },
        },
    },
    # amplitude of overall balmer model (continuum + higher-order lines)
    'balmer_amp_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_false',
            'balmer_amp_val': 'intfl_min0_d1',
        },
    },
    # broadening of higher-order Balmer lines
    'balmer_fwhm_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'balmer_fwhm_val': {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 5000.0,
            },
        },
    },
    # velocity offset of higher-order Balmer lines
    'balmer_voff_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'balmer_voff_val': 'intfl_d0',
        },
    },
    # effective temperature
    'Teff_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'Teff_val': {
                'type': ['integer', 'float'],
                'min_ex': 0.0,
                'default': 15000.0,
            },
        },
    },
    # optical depth
    'tau_const': {
        'type': 'dict',
        'default': {},
        'schema': {
            'bool': 'bool_true',
            'tau_val': {
                'type': ['integer', 'float'],
                'min': 0.0,
                'max': 1.0,
                'default': 1.0,
            },
        },
    },
}


DEFAULT_PLOT_OPTIONS = {
    # Plot MCMC histograms and chains for each parameter
    'plot_param_hist': 'bool_true',
    # Plot MCMC hist. and chains for component fluxes
    'plot_flux_hist': 'bool_false',
    # Plot MCMC hist. and chains for component luminosities
    'plot_lum_hist': 'bool_false',
    # Plot MCMC hist. and chains for equivalent widths 
    'plot_eqwidth_hist': 'bool_false',
    # make interactive plotly HTML best-fit plot
    'plot_HTML': 'bool_false',
}


DEFAULT_OUTPUT_OPTIONS = {
    # Write MCMC chains for all paramters, fluxes, and
    # luminosities to a FITS table We set this to false 
    # because MCMC_chains.FITS file can become very large, 
    # especially  if you are running multiple objects.  
    # You only need this if you want to reconstruct chains 
    # and histograms. 
    'write_chain': 'bool_false',
    'verbose': 'bool_true',
}


# Combine all options into one schema

dict_options = {
    'io_options': DEFAULT_IO_OPTIONS,
    'fit_options': DEFAULT_FIT_OPTIONS,
    'ifu_options': DEFAULT_IFU_OPTIONS,
    'mcmc_options': DEFAULT_MCMC_OPTIONS,
    'comp_options': DEFAULT_COMP_OPTIONS,
    'power_options': DEFAULT_POWER_OPTIONS,
    'losvd_options': DEFAULT_LOSVD_OPTIONS,
    'host_options': DEFAULT_HOST_OPTIONS,
    'uv_iron_options': DEFAULT_UV_IRON_OPTIONS,
    'balmer_options': DEFAULT_BALMER_OPTIONS,
    'plot_options': DEFAULT_PLOT_OPTIONS,
    'output_options': DEFAULT_OUTPUT_OPTIONS,
}

list_options = {
    'user_constraints': DEFAULT_USER_CONSTRAINTS,
    'user_mask': DEFAULT_USER_MASK,
}


DEFAULT_OPTIONS_SCHEMA = {}
for option_name, schema in dict_options.items():
    DEFAULT_OPTIONS_SCHEMA[option_name] = {
        'type': 'dict',
        'default': {},
        'schema': schema,
    }

for option_name, schema in list_options.items():
    DEFAULT_OPTIONS_SCHEMA[option_name] = {
        'type': 'list',
        'default': [],
        'schema': schema,
    }


# Some options need special treatment

# TODO: more validation on user line dict
DEFAULT_OPTIONS_SCHEMA['user_lines'] = {
    'type': 'dict',
    'default': {},
    'keysrules': {'type': 'string'},
    'valuesrules': {'type': 'dict',},
}

DEFAULT_OPTIONS_SCHEMA['combined_lines'] = {
    'type': 'dict',
    'default': {},
    'keysrules': {'type': 'string'},
    'valuesrules': {
        'type': 'list',
        'schema': {'type': 'string'},
    },
}

DEFAULT_OPTIONS_SCHEMA['opt_feii_options'] = {
    'oneof_schema': [DEFAULT_OPT_FEII_VC04_OPTIONS, DEFAULT_OPT_FEII_K10_OPTIONS],
    'default': DefaultValidator().normalized({}, DEFAULT_OPT_FEII_VC04_OPTIONS),
}

