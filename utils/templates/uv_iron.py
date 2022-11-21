import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import utils.constants as consts
from utils.templates.common import BadassTemplate, template_rfft, convolve_gauss_hermite
from utils.utils import log_rebin

UV_IRON_TEMP_WAVE_MIN = 1074.0
UV_IRON_TEMP_WAVE_MAX = 3100.0

UV_IRON_TEMPLATE_FILE = consts.BADASS_DATA_DIR.joinpath('feii_templates', 'vestergaard-wilkes_2001', 'VW01_UV_B.csv')

UV_IRON_DISP_MIN = 0.01

class UVIronTemplate(BadassTemplate):

    @classmethod
    def initialize_template(cls, ctx):
        if not ctx.options.comp_options.fit_uv_iron:
            return None

        if (ctx.wave[0] > UV_IRON_TEMP_WAVE_MAX) or (ctx.wave[-1] < UV_IRON_TEMP_WAVE_MIN):
            ctx.options.comp_options.fit_uv_iron = False
            ctx.log.warning('UV Iron template disabled because template is outside of fitting region.')
            ctx.log.update_uv_iron()
            return None

        if not UV_IRON_TEMPLATE_FILE.exists():
            ctx.log.error('Could not find UV Iron template file: %s' % str(UV_IRON_TEMPLATE_FILE))
            return None

        return cls(ctx)


    def __init__(self, ctx):
        self.ctx = ctx

        npad = 100 # anstroms
        df_uviron = pd.read_csv(UV_IRON_TEMPLATE_FILE) # UV B only

        # Generate a new grid with the original resolution, but the size of the fitting region
        dlam_uviron = df_uviron['angstrom'][1] - df_uviron['angstrom'][0]
        lam_uviron = np.arange(np.min(self.ctx.wave)-npad, np.max(self.ctx.wave)+npad, dlam_uviron) # angstroms

        # Interpolate the original template onto the new grid
        interp_ftn_uv = interp1d(df_uviron['angstrom'].to_numpy(),df_uviron['flux'].to_numpy(),kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
        spec_uviron = interp_ftn_uv(lam_uviron)

        # log-rebin the spectrum to same velocity scale as the input galaxy
        lamRange_uviron = [np.min(lam_uviron), np.max(lam_uviron)]
        spec_uviron_new, loglam_uviron, velscale_uviron = log_rebin(lamRange_uviron, spec_uviron, velscale=self.ctx.velscale)

        # Pre-compute FFT of templates, since they do not change (only the LOSVD and convolution changes)
        self.uv_iron_fft, self.npad = template_rfft(spec_uviron_new)

        # The FeII templates are offset from the input galaxy spectrum by 100 A, so we 
        # shift the spectrum to match that of the input galaxy.
        self.vsyst = np.log(lam_uviron[0]/self.ctx.wave[0])*consts.c


    def convolve(self, uv_iron_voff, uv_iron_disp):
        return convolve_gauss_hermite(self.uv_iron_fft, self.npad, self.ctx.velscale,
                                              [uv_iron_voff, uv_iron_disp], self.ctx.wave.shape[0], 
                                               velscale_ratio=1, sigma_diff=0, vsyst=self.vsyst)


    def initialize_parameters(self, params):
        # TODO: implement
        return params


    def add_components(self, params, comp_dict, host_model):

        uv_iron_options = self.ctx.options.uv_iron_options
        val = lambda ok, ov, pk : uv_iron_options[ok][ov] if uv_iron_options[ok].bool else params[pk]

        # TODO: would this option ever change? ie. if amp, etc. are const, just set in init
        uv_iron_amp = val('uv_amp_const', 'uv_iron_val', 'UV_IRON_AMP')
        uv_iron_disp = val('uv_disp_const', 'uv_iron_val', 'UV_IRON_DISP')
        uv_iron_voff = val('uv_voff_const', 'uv_iron_val', 'UV_IRON_VOFF')

        conv_temp = self.convolve(uv_iron_voff, uv_iron_disp)

        # Reshape
        conv_temp = conv_temp.reshape(-1)
        # Re-normalize to 1
        conv_temp = conv_temp/np.max(conv_temp)
        # Multiply by amplitude
        template = uv_iron_amp * conv_temp

        # Set fitting region outside of template to zero to prevent convolution loops
        template[(lam_gal < UV_IRON_TEMP_WAVE_MIN) & (lam_gal > UV_IRON_TEMP_WAVE_MAX)] = 0

        # If the summation results in 0.0, it means that features were too close 
        # to the edges of the fitting region (usually because the region is too 
        # small), then simply return an array of zeros.
        if (isinstance(template,(int,float))) or (np.isnan(np.sum(template))):
            template = np.zeros(len(self.ctx.wave))

        comp_dict['UV_IRON_TEMPLATE'] = template
        host_model -= template

        return comp_dict, host_model

