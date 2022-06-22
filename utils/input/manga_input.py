from utils.input.ifu_input import IFUInput

class MangaIFU(IFUInput):
    @classmethod
    def read_manga_ifu(cls, input_data, options):
        # Reference: https://data.sdss.org/datamodel/files/MANGA_SPECTRO_REDUX/DRPVER/PLATE4/stack/manga-CUBE.html#hdu1

        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading MUSE IFU from data currently unsupported')

        spec = cls()
        spec.infile = input_data
        spec.objname = None

        with fits.open(fits_file) as hdu:
            # First axis is wavelength, then 2nd and 3rd are image x/y
            spec.nx, spec.ny, spec.nz = hdu[1].header['NAXIS1'], hdu[1].header['NAXIS2'], hdu[1].header['NAXIS3']
            try:
                spec.ra = hdu[0].header['OBJRA']
                spec.dec = hdu[0].header['OBJDEC']
            except:
                spec.ra = hdu[1].header['IFURA']
                spec.dec = hdu[1].header['IFUDEC']

            primary = hdu[0].header
            spec.dataid = primary['MANGAID']

            # 3d rectified cube in units of 10(-17) erg/s/cm2/Angstrom/spaxel [NX x NY x NWAVE]
            spec.flux = hdu[1].data
            # Inverse variance (1/sigma2) for the above [NX x NY x NWAVE]
            spec.ivar = hdu[2].data
            # Pixel mask [NX x NY x NWAVE]. Defined values are set in sdssMaskbits.par
            spec.mask = hdu[3].data
            # Wavelength vector [NWAVE]
            spec.wave = hdu[6].data
            # Median spectral resolution as a function of wavelength for the fibers in this IFU [NWAVE]
            spec.specres = hdu[7].data
            # ebv = hdu[0].header['EBVGAL']

        spec.ifu_common(options)
        return spec
