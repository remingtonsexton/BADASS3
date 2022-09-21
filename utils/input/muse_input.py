import numpy as np
import pathlib
from astropy.io import fits

from utils.input.ifu_input import IFUInput

class MuseIFU(IFUInput):
    @classmethod
    def read_muse(cls, input_data, options):
        # Reference: https://www.eso.org/rm/api/v1/public/releaseDescriptions/78

        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading MUSE IFU from data currently unsupported')

        inobj = cls()
        inobj.infile = input_data
        inobj.options = options
        inobj.z = inobj.options.ifu_options.z # TODO: options validate: exists and is float or int

        with fits.open(inobj.infile) as hdu:
            # First axis is wavelength, then 2nd and 3rd are image x/y
            try:
                inobj.nx, inobj.ny, inobj.nz = hdu[1].header['NAXIS1'], hdu[1].header['NAXIS2'], hdu[1].header['NAXIS3']
                inobj.ra = hdu[0].header['RA']
                inobj.dec = hdu[0].header['DEC']
            except:
                inobj.nx, inobj.ny, inobj.nz = hdu[0].header['NAXIS1'], hdu[0].header['NAXIS2'], hdu[0].header['NAXIS3']
                inobj.ra = hdu[0].header['CRVAL1']
                inobj.dec = hdu[0].header['CRVAL2']

            primary = hdu[0].header
            inobj.objname = primary.get('OBJECT', None)

            inobj.dataid = []
            i = 1
            while True:
                try:
                    inobj.dataid.append(primary['OBID'+str(i)])
                    i += 1
                except:
                    break

            try:
                # 3d rectified cube in units of 10(-20) erg/s/cm2/Angstrom/spaxel [NX x NY x NWAVE], convert to 10(-17)
                spec = hdu[1].data
                # Variance (sigma2) for the above [NX x NY x NWAVE], convert to 10(-17)
                var = hdu[2].data
                # Wavelength vector must be reconstructed, convert from nm to angstroms
                header = hdu[1].header
                wave = np.array(header['CRVAL3'] + header['CD3_3']*np.arange(header['NAXIS3']))
                # Default behavior for MUSE data cubes using https://www.aanda.org/articles/aa/pdf/2017/12/aa30833-17.pdf equation 7
                dlambda = 5.835e-8 * wave**2 - 9.080e-4 * wave + 5.983
                specres = wave / dlambda
                # Scale by the measured spec_res at the central wavelength
                spec_cent = primary['SPEC_RES']
                cwave = np.nanmedian(wave)
                c_dlambda = 5.835e-8 * cwave**2 - 9.080e-4 * cwave + 5.983
                scale = 1 + (spec_cent - cwave/c_dlambda) / spec_cent
                specres *= scale
            except:
                spec = hdu[0].data
                var = (0.1 * spec)**2
                wave = np.arange(primary['CRVAL3'], primary['CRVAL3']+primary['CDELT3']*(nz-1), primary['CDELT3'])
                dlambda = 5.835e-8 * wave**2 - 9.080e-4 * wave + 5.983
                specres = wave / dlambda

        noise = np.sqrt(var)
        mask = np.zeros_like(spec)
        inobj.wave = wave / (1 + inobj.z) # Convert to restframe
        inobj.specres = specres

        return inobj.ifu_common(inobj, spec, noise, mask)
