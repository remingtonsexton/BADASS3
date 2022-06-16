import astropy
import numpy as np
import pathlib

from utils.input import BadassInput, SDSS_FMT

class SDSSSpec(BadassInput):
    @classmethod
    def read_sdss_spec(cls, input_data):

        spec = SDSSSpec()

        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading SDSS spectra from data currently unsupported')

        spec.infile = input_data

        hdu = astropy.io.fits.open(spec.infile)
        specobj = hdu[2].data
        spec.z = specobj['z'][0]

        try:
            spec.ra = hdu[0].header['RA']
            spec.dec = hdu[0].header['DEC']
        except KeyError:
            spec.ra = specobj['PLUG_RA'][0]
            spec.dec = specobj['PLUG_DEC'][0]

        t = hdu[1].data

        # Unpack the spectra
        spec.flux = t['flux']
        spec.wave = np.power(10, t['loglam'])
        spec.error = np.sqrt(1 / t['ivar'])
        return spec


    def validate_input(self):
        for attr in ['z', 'ra', 'dec', 'flux', 'wave', 'error']:
            if not hasattr(self, attr):
                raise Exception('SDSS Spec input missing expected value: {attr}')
        return True
