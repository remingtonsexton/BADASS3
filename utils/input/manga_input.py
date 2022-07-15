from utils.input.ifu_input import IFUInput

class MangaIFU(IFUInput):
    @classmethod
    def read_manga(cls, input_data, options):
        # Reference: https://data.sdss.org/datamodel/files/MANGA_SPECTRO_REDUX/DRPVER/PLATE4/stack/manga-CUBE.html#hdu1

        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading MUSE IFU from data currently unsupported')

        inobj = cls()
        inobj.infile = input_data
        inobj.options = options
        inobj.z = inobj.options.ifu_options.z

        with fits.open(fits_file) as hdu:
            # First axis is wavelength, then 2nd and 3rd are image x/y
            inobj.nx, inobj.ny, inobj.nz = hdu[1].header['NAXIS1'], hdu[1].header['NAXIS2'], hdu[1].header['NAXIS3']
            try:
                inobj.ra = hdu[0].header['OBJRA']
                inobj.dec = hdu[0].header['OBJDEC']
            except:
                inobj.ra = hdu[1].header['IFURA']
                inobj.dec = hdu[1].header['IFUDEC']

            primary = hdu[0].header
            inobj.dataid = primary['MANGAID']

            # 3d rectified cube in units of 10(-17) erg/s/cm2/Angstrom/spaxel [NX x NY x NWAVE]
            spec = hdu[1].data
            # Inverse variance (1/sigma2) for the above [NX x NY x NWAVE]
            ivar = hdu[2].data
            noise = np.sqrt(1 / ivar)
            # Pixel mask [NX x NY x NWAVE]. Defined values are set in sdssMaskbits.par
            mask = hdu[3].data
            # Wavelength vector [NWAVE]
            inobj.wave = hdu[6].data / (1 + inobj.z)
            # Median spectral resolution as a function of wavelength for the fibers in this IFU [NWAVE]
            inobj.specres = hdu[7].data
            # ebv = hdu[0].header['EBVGAL']

        return inobj.ifu_common(inobj, spec, noise, mask)
