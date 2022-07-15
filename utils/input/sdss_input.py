from astropy.io import fits
import numpy as np
import pathlib

from utils.input.input import BadassInput, SDSS_FMT
from utils.utils import find_nearest
from utils.constants import C

class SDSSSpec(BadassInput):

    @classmethod
    def read_sdss_spec(cls, input_data, options):
        return cls(input_data)


    def __init__(self, input_data):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading SDSS spectra from data currently unsupported')

        super().__init__()
        self.infile = input_data
        with fits.open(self.infile) as hdu:
            specobj = hdu[2].data
            self.z = specobj['z'][0]

            if 'RA' in hdu[0].header:
                self.ra = hdu[0].header['RA']
                self.dec = hdu[0].header['DEC']
            elif 'PLUG_RA' in hdu[0].header:
                self.ra = specobj['PLUG_RA'][0]
                self.dec = specobj['PLUG_DEC'][0]
            else:
                self.ra = None
                self.dec = None

            t = hdu[1].data

            # Unpack the spectra
            self.spec = t['flux']
            self.wave = np.power(10, t['loglam']) / (1+self.z)
            self.noise = np.sqrt(1 / t['ivar'])
            self.mask = t['and_mask'] # TODO: need?

            frac = self.wave[1]/self.wave[0] # Constant lambda fraction per pixel
            dlam_gal = (frac - 1)*self.wave # Size of every pixel in Angstrom
            wdisp = t['wdisp'] # Intrinsic dispersion of every pixel, in pixels units
            self.fwhm_res = 2.355*wdisp*dlam_gal # Resolution FWHM of every pixel, in angstroms
            self.velscale = np.log(frac) * C
