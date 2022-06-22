from astropy.io import fits
import numpy as np
import pathlib

from utils.input.input import BadassInput, SDSS_FMT
from utils.utils import find_nearest

class SDSSSpec(BadassInput):
    @classmethod
    def read_sdss_spec(cls, input_data, options):

        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading SDSS spectra from data currently unsupported')

        spec = cls()
        spec.infile = input_data

        with fits.open(spec.infile) as hdu:
            spec.specobj = hdu[2].data
            spec.z = spec.specobj['z'][0]

            if 'RA' in hdu[0].header:
                spec.ra = hdu[0].header['RA']
                spec.dec = hdu[0].header['DEC']
            elif 'PLUG_RA' in hdu[0].header:
                spec.ra = spec.specobj['PLUG_RA'][0]
                spec.dec = spec.specobj['PLUG_DEC'][0]
            else:
                spec.ra = None
                spec.dec = None

            t = hdu[1].data

            # Unpack the spectra
            spec.flux = t['flux']
            spec.wave = np.power(10, t['loglam'])
            spec.lam = spec.wave / (1+spec.z)
            spec.ivar = t['ivar']
            spec.error = np.sqrt(1 / spec.ivar)

        return spec


    def validate_input(self):
        for attr in ['z', 'ra', 'dec', 'flux', 'wave', 'error']:
            if not hasattr(self, attr):
                raise Exception('SDSS Spec input missing expected value: {attr}'.format(attr=attr))
        return True


    # def generate_mask(self):


    # def prepare_input(self, ba_ctx):
    #     fit_min, fit_max = ba_ctx.options.fit_options.fit_reg
