import numpy as np
import pathlib
from astropy.io import fits

from utils.input.ifu_input import IFUInput
import utils.verify.verify

class MuseIFU(IFUInput):
    @classmethod
    def read_muse_ifu(cls, input_data, options):
        # Reference: https://www.eso.org/rm/api/v1/public/releaseDescriptions/78

        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading MUSE IFU from data currently unsupported')

        spec = cls()
        spec.infile = input_data

        with fits.open(spec.infile) as hdu:
            # First axis is wavelength, then 2nd and 3rd are image x/y
            try:
                spec.nx, spec.ny, spec.nz = hdu[1].header['NAXIS1'], hdu[1].header['NAXIS2'], hdu[1].header['NAXIS3']
                spec.ra = hdu[0].header['RA']
                spec.dec = hdu[0].header['DEC']
            except:
                spec.nx, spec.ny, spec.nz = hdu[0].header['NAXIS1'], hdu[0].header['NAXIS2'], hdu[0].header['NAXIS3']
                spec.ra = hdu[0].header['CRVAL1']
                spec.dec = hdu[0].header['CRVAL2']

            primary = hdu[0].header
            spec.objname = primary.get('OBJECT', None)

            spec.dataid = []
            i = 1
            while True:
                try:
                    spec.dataid.append(primary['OBID'+str(i)])
                    i += 1
                except:
                    break

            # Get unit of flux, assuming 10^-x erg/s/cm2/Angstrom/spaxel
            # unit = hdu[0].header['BUNIT']
            # power = int(re.search('10\*\*(\(?)(.+?)(\))?\s', unit).group(2))
            # scale = 10**(-17) / 10**power
            try:
                # 3d rectified cube in units of 10(-20) erg/s/cm2/Angstrom/spaxel [NX x NY x NWAVE], convert to 10(-17)
                spec.flux = hdu[1].data
                # Variance (sigma2) for the above [NX x NY x NWAVE], convert to 10(-17)
                var = hdu[2].data
                # Wavelength vector must be reconstructed, convert from nm to angstroms
                header = hdu[1].header
                spec.wave = np.array(header['CRVAL3'] + header['CD3_3']*np.arange(header['NAXIS3']))
                # Default behavior for MUSE data cubes using https://www.aanda.org/articles/aa/pdf/2017/12/aa30833-17.pdf equation 7
                dlambda = 5.835e-8 * spec.wave**2 - 9.080e-4 * spec.wave + 5.983
                spec.specres = spec.wave / dlambda
                # Scale by the measured spec_res at the central wavelength
                spec_cent = primary['SPEC_RES']
                cwave = np.nanmedian(spec.wave)
                c_dlambda = 5.835e-8 * cwave**2 - 9.080e-4 * cwave + 5.983
                scale = 1 + (spec_cent - cwave/c_dlambda) / spec_cent
                spec.specres *= scale
            except:
                spec.flux = hdu[0].data
                var = (0.1 * spec.flux)**2
                spec.wave = np.arange(primary['CRVAL3'], primary['CRVAL3']+primary['CDELT3']*(nz-1), primary['CDELT3'])
                # specres = wave / 2.6
                dlambda = 5.835e-8 * spec.wave**2 - 9.080e-4 * spec.wave + 5.983
                spec.specres = spec.wave / dlambda

        spec.ivar = 1/var
        spec.mask = np.zeros_like(spec.flux)

        spec.ifu_common(options)
        return spec
