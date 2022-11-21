import astropy.units as u
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import utils.constants as consts
from utils.templates.common import BadassTemplate, gaussian_filter1d, template_rfft, convolve_gauss_hermite
from utils.utils import log_rebin, find_nearest

BALMER_TEMP_WAVE_MAX = 3500.0 # Angstroms
BALMER_EDGE_WAVE = 3646.0 # Angstroms

BALMER_TEMPLATE_FILE = consts.BADASS_DATA_DIR.joinpath('balmer_template', 'higher_order_balmer_n8_500.csv')

class BalmerTemplate(BadassTemplate):

    @classmethod
    def initialize_template(cls, ctx):
        if not ctx.options.comp_options.fit_balmer:
            return None

        if ctx.wave[0] >= BALMER_TEMP_WAVE_MAX:
            ctx.log.warn('Balmer continuum disabled because template is outside of fitting region.')
            ctx.log.update_balmer()
            return None

        if not BALMER_TEMPLATE_FILE.exists():
            ctx.log.error('Could not find Balmer template file: %s' % str(BALMER_TEMPLATE_FILE))
            return None

        return cls(ctx)


    def __init__(self, ctx):
        self.ctx = ctx

        npad = 100 # Angstroms
        # Import the template for the higher-order balmer lines (7 <= n <= 500)
        df = pd.read_csv(BALMER_TEMPLATE_FILE)
        # Generate a new grid with the original resolution, but the size of the fitting region
        dlam_balmer = df['angstrom'][1] - df['angstrom'][0]
        lam_balmer = np.arange(np.min(self.ctx.wave)-npad, np.max(self.ctx.wave)+npad, dlam_balmer) # angstroms

        # Interpolate the original template onto the new grid
        interp_ftn_balmer = interp1d(df["angstrom"].to_numpy(),df["flux"].to_numpy(),kind='linear',bounds_error=False,fill_value=(1.e-10,1.e-10))
        spec_high_balmer = interp_ftn_balmer(lam_balmer)

        # Calculate the difference in instrumental dispersion between SDSS and the template
        lamRange_balmer = [np.min(lam_balmer), np.max(lam_balmer)]
        fwhm_balmer = 1.0
        disp_balmer = fwhm_balmer/2.3548
        disp_res_interp = np.interp(lam_balmer, self.ctx.wave, self.ctx.disp_res)
        disp_diff = np.sqrt((disp_res_interp**2 - disp_balmer**2).clip(0))
        sigma = disp_diff/dlam_balmer # Sigma difference in pixels

        # Convolve the FeII templates to the SDSS resolution
        spec_high_balmer = gaussian_filter1d(spec_high_balmer, sigma)

        # Log-rebin to same velocity scale as galaxy
        self.spec_high_balmer, loglam_balmer, self.velscale_balmer = log_rebin(lamRange_balmer, spec_high_balmer, velscale=self.ctx.velscale)
        if (np.sum(self.spec_high_balmer)>0):
            # Normalize to 1
            self.spec_high_balmer = self.spec_high_balmer/np.max(self.spec_high_balmer)
        self.lam_balmer = np.exp(loglam_balmer)


    def initialize_parameters(self, params):
        # TODO: implement
        return params


    def add_components(self, params, comp_dict, host_model):

        balmer_options = self.ctx.options.balmer_options
        val = lambda ok, ov, pk : balmer_options[ok][ov] if balmer_options[ok].bool else params[pk]

        balmer_ratio = val('R_const', 'R_val', 'BALMER_RATIO')
        balmer_amp = val('balmer_amp_const', 'balmer_amp_val', 'BALMER_AMP')
        balmer_disp = val('balmer_disp_const', 'balmer_disp_val', 'BALMER_DISP')
        if balmer_disp <= 0.01: balmer_disp = 0.01
        balmer_voff = val('balmer_voff_const', 'balmer_voff_val', 'BALMER_VOFF')
        balmer_Teff = val('Teff_const', 'Teff_val', 'BALMER_TEFF')
        balmer_tau = val('tau_const', 'tau_val', 'BALMER_TAU')

        # We need to generate a new grid for the Balmer continuum that matches
        # that we made for the higher-order lines
        def blackbody(lam, balmer_Teff):
            c = (consts.c*(u.km/u.s)).to(u.AA/u.s).value
            h = (consts.h*(((u.m)**2) * u.kg / u.s)).to(u.g*((u.AA**2)/(u.s**2))*u.s).value
            k = (consts.k*((u.m**2)*u.kg/(u.s**2)/u.K)).to(u.g * (u.AA**2) / (u.s**2) / u.K).value
            return ((2.0*h*c**2.0)/lam**5.0)*(1.0/(np.exp((h*c)/(lam*k*balmer_Teff))-1.0))

        # Construct Balmer continuum from lam_balmer
        Blam = blackbody(self.lam_balmer, balmer_Teff) # blackbody function [erg/s]
        cont = Blam * (1.0-1.0/np.exp(balmer_tau*(self.lam_balmer/BALMER_EDGE_WAVE)**3.0))
        # Normalize at 3000 Å
        cont = cont / np.max(cont)
        # Set Balmer continuum to zero after Balmer edge
        cont[find_nearest(self.lam_balmer, BALMER_EDGE_WAVE)[1]:] = 0.0

        # TODO: this is also done in initialization, need to be done here?
        if (np.sum(self.spec_high_balmer)>0):
            self.spec_high_balmer = self.spec_high_balmer/np.max(self.spec_high_balmer) * balmer_ratio

        # Sum the two components
        full_balmer = self.spec_high_balmer + cont

        # Pre-compute the FFT and vsyst
        balmer_fft, balmer_npad = template_rfft(full_balmer)
        vsyst = np.log(self.lam_balmer[0]/self.ctx.wave[0])*consts.c

        # Broaden the higher-order Balmer lines
        conv_temp = convolve_gauss_hermite(balmer_fft, balmer_npad, float(self.ctx.velscale),\
                                           [balmer_voff, balmer_disp], self.ctx.wave.shape[0], 
                                           velscale_ratio=1, sigma_diff=0, vsyst=vsyst)

        conv_temp = conv_temp/conv_temp[find_nearest(self.ctx.wave,BALMER_EDGE_WAVE)[1]] * balmer_ratio
        conv_temp = conv_temp.reshape(-1)

        # Normalize the full continuum to 1
        balmer_cont = conv_temp/np.max(conv_temp) * balmer_amp

        comp_dict['BALMER_CONT'] = balmer_cont
        host_model -= balmer_cont

        return comp_dict, host_model

