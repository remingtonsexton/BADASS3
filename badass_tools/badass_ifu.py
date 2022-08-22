import numpy as np
from astropy.io import fits
import os
import re
import glob
import copy
from vorbin.voronoi_2d_binning import voronoi_2d_binning
import matplotlib.pyplot as plt
from scipy import interpolate, stats, optimize
import gc
from matplotlib import gridspec, animation
try:
    import tqdm
except:
    tqdm = None
from joblib import Parallel, delayed

plt.style.use('dark_background')


def read_muse_ifu(fits_file,z=0):
    """
    Read in a MUSE-formatted IFU cube

    :param fits_file: str
        File path to the FITS IFU cube
    :param z: float, optional
        The redshift of the spectrum, since MUSE cubes often do not provide this information

    :return nx: int
        x-dimension (horizontal axis) of the cube
    :return ny: int
        y-dimension (vertical axis) of the cube
    :return nz: int
        z-dimension (wavelength axis) of the cube
    :return ra: float
        Right ascension
    :return dec: float
        Declination
    :return museid: str
        The MUSE ID of the observation
    :return wave: array
        1-D Wavelength array with dimension (nz,)
    :return flux: array
        3-D flux array with dimensions (nz, ny, nx)
    :return ivar: array
        3-D inverse variance array with dimensions (nz, ny, nx)
    :return specres: array
        1-D spectral resolution ("R") array with dimension (nz,)
    :return mask: array
        3-D mask array with dimensions (nz, ny, nx)
    :return object_name: str
        The name of the object, if provided in the FITS header
    """

    # Load the file
    # https://www.eso.org/rm/api/v1/public/releaseDescriptions/78
    with fits.open(fits_file) as hdu:
        # First axis is wavelength, then 2nd and 3rd are image x/y
        try:
            nx, ny, nz = hdu[1].header['NAXIS1'], hdu[1].header['NAXIS2'], hdu[1].header['NAXIS3']
            ra = hdu[0].header['RA']
            dec = hdu[0].header['DEC']
        except:
            # ra = hdu[0].header['ESO ADA GUID RA']
            # dec = hdu[0].header['ESO ADA GUID DEC']
            nx, ny, nz = hdu[0].header['NAXIS1'], hdu[0].header['NAXIS2'], hdu[0].header['NAXIS3']
            ra = hdu[0].header['CRVAL1']
            dec = hdu[0].header['CRVAL2']

        primary = hdu[0].header
        try:
            object_name = primary['OBJECT']
        except:
            object_name = None
        i = 1
        museid = []
        while True:
            try:
                museid.append(primary['OBID'+str(i)])
                i += 1
            except:
                break

        # Get unit of flux, assuming 10^-x erg/s/cm2/Angstrom/spaxel
        # unit = hdu[0].header['BUNIT']
        # power = int(re.search('10\*\*(\(?)(.+?)(\))?\s', unit).group(2))
        # scale = 10**(-17) / 10**power
        try:
            # 3d rectified cube in units of 10(-20) erg/s/cm2/Angstrom/spaxel [NX x NY x NWAVE], convert to 10(-17)
            flux = hdu[1].data
            # Variance (sigma2) for the above [NX x NY x NWAVE], convert to 10(-17)
            var = hdu[2].data
            # Wavelength vector must be reconstructed, convert from nm to angstroms
            header = hdu[1].header
            wave = np.array(header['CRVAL3'] + header['CD3_3']*np.arange(header['NAXIS3']))
            # wave = np.linspace(primary['WAVELMIN'], primary['WAVELMAX'], nz) * 10
            # Median spectral resolution at (wavelmin + wavelmax)/2
            # dlambda = cwave / primary['SPEC_RES']
            # specres = wave / dlambda
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
            flux = hdu[0].data
            var = (0.1 * flux)**2
            wave = np.arange(primary['CRVAL3'], primary['CRVAL3']+primary['CDELT3']*(nz-1), primary['CDELT3'])
            # specres = wave / 2.6
            dlambda = 5.835e-8 * wave**2 - 9.080e-4 * wave + 5.983
            specres = wave / dlambda

    ivar = 1/var
    mask = np.zeros_like(flux)

    return nx,ny,nz,ra,dec,museid,wave,flux,ivar,specres,mask,object_name


def read_manga_ifu(fits_file,z=0):
    """
    Read in a MANGA-formatted IFU cube

    :param fits_file: str
        File path to the FITS IFU cube
    :param z: float, optional
        The redshift of the spectrum, this is unused.

    :return nx: int
        x-dimension (horizontal axis) of the cube
    :return ny: int
        y-dimension (vertical axis) of the cube
    :return nz: int
        z-dimension (wavelength axis) of the cube
    :return ra: float
        Right ascension
    :return dec: float
        Declination
    :return mangaid: str
        The MANGA ID of the observation
    :return wave: array
        1-D Wavelength array with dimension (nz,)
    :return flux: array
        3-D flux array with dimensions (nz, ny, nx)
    :return ivar: array
        3-D inverse variance array with dimensions (nz, ny, nx)
    :return specres: array
        1-D spectral resolution ("R") array with dimension (nz,)
    :return mask: array
        3-D mask array with dimensions (nz, ny, nx)
    :return None:
        To mirror the output length of read_muse_ifu
    """

    # Load the file
    # https://data.sdss.org/datamodel/files/MANGA_SPECTRO_REDUX/DRPVER/PLATE4/stack/manga-CUBE.html#hdu1
    with fits.open(fits_file) as hdu:
        # First axis is wavelength, then 2nd and 3rd are image x/y
        nx, ny, nz = hdu[1].header['NAXIS1'], hdu[1].header['NAXIS2'], hdu[1].header['NAXIS3']
        try:
            ra = hdu[0].header['OBJRA']
            dec = hdu[0].header['OBJDEC']
        except:
            ra = hdu[1].header['IFURA']
            dec = hdu[1].header['IFUDEC']

        primary = hdu[0].header
        ebv = primary['EBVGAL']
        mangaid = primary['MANGAID']

        # 3d rectified cube in units of 10(-17) erg/s/cm2/Angstrom/spaxel [NX x NY x NWAVE]
        flux = hdu[1].data
        # Inverse variance (1/sigma2) for the above [NX x NY x NWAVE]
        ivar = hdu[2].data
        # Pixel mask [NX x NY x NWAVE]. Defined values are set in sdssMaskbits.par
        mask = hdu[3].data
        # Wavelength vector [NWAVE]
        wave = hdu[6].data
        # Median spectral resolution as a function of wavelength for the fibers in this IFU [NWAVE]
        specres = hdu[7].data
        # ebv = hdu[0].header['EBVGAL']

    return nx,ny,nz,ra,dec,mangaid,wave,flux,ivar,specres,mask,None



def prepare_ifu(fits_file,z,format,aperture=None,voronoi_binning=True,fixed_binning=False,targetsn=None,cvt=True,voronoi_plot=True,quiet=True,wvt=False,
                maxbins=800,snr_threshold=0.5,fixed_bin_size=10,use_and_mask=True,nx=None,ny=None,nz=None,ra=None,dec=None,dataid=None,wave=None,flux=None,ivar=None,
                specres=None,mask=None,objname=None):
    """
    Deconstruct an IFU cube into individual spaxel files for fitting with BADASS

    :param fits_file: str
        The file path to the IFU FITS file; if format == 'user', this field may be left as None, '', or any other filler value
    :param z: float
        The redshift of the spectrum
    :param aperture: array, optional
        The lower-left and upper-right corners of a square aperture, formatted as [y0, y1, x0, x1]
    :param voronoi_binning: bool
        Whether or not to bin spaxels using the voronoi method (grouping to read a certain SNR threshold). Default True.
        Mutually exclusive with fixed_binning.
    :param fixed_binning: bool
        Whether or not to bin spaxels using a fixed size. Default False.
        Mutually exclusive with voronoi_binning.
    :param targetsn: float, optional
        The target SNR to bin by, if using voronoi binning.
    :param cvt: bool
        Vorbin CVT option (see the vorbin package docs). Default True.
    :param voronoi_plot: bool
        Whether or not to plot the voronoi bin structure. Default True.
    :param quiet: bool
        Vorbin quiet option (see the vorbin package docs). Default True.
    :param wvt: bool
        Vorbin wvt option (see the vorbin package docs). Default False.
    :param maxbins: int
        If no target SNR is provided for voronoi binning, maxbins may be specified, which will automatically calculate
        the target SNR required to reach the number of bins desired. Default 800.
    :param snr_threshold: float
        Minimum SNR threshold, below which spaxel data will be removed and not fit.
    :param fixed_bin_size: int
        If using fixed binning, this is the side length of the square bins, in units of spaxels.
    :param use_and_mask: bool
        Whether or not to save the and_mask data.
    :param nx: int, optional
        x-dimension of the cube, only required if format == 'user'
    :param ny: int, optional
        y-dimension of the cube, only required if format == 'user'
    :param nz: int, optional
        z-dimension of the cube, only required if format == 'user'
    :param ra: float, optional
        Right ascension of the cube, only required if format == 'user'
    :param dec: float, optional
        Declination of the cube, only required if format == 'user'
    :param dataid: str, optional
        ID of the cube, only required if format == 'user'
    :param wave: array, optional
        1-D wavelength array with shape (nz,), only required if format == 'user'
    :param flux: array, optional
        3-D flux array with shape (nz, ny, nx), only required if format == 'user'
    :param ivar: array, optional
        3-D inverse variance array with shape (nz, ny, nx), only required if format == 'user'
    :param specres: array, optional
        1-D spectral resolution ("R") array with shape (nz,), only required if format == 'user'
    :param mask: array, optional
        3-D mask array with shape (nz, ny, nx), only required if format == 'user'
    :param objname: str, optional
        The name of the object, only required if format == 'user'
    
    :return wave: array
        1-D wavelength array with shape (nz,)
    :return flux: array
        3-D masked flux array with shape (nz, ny, nx)
    :return ivar: array
        3-D masked inverse variance array with shape (nz, ny, nx)
    :return mask: array
        3-D mask array with shape (nz, ny, nx)
    :return fwhm_res: array
        1-D FWHM resolution array with shape (nz,)
    :return binnum: array
        Bin number array that specifies which spaxels are in each bin (see the vorbin docs)
    :return npixels: array
        Number of spaxels in each bin (see the vorbin docs)
    :return xpixbin: array
        The x positions of spaxels in each bin
    :return ypixbin: array
        The y positions of spaxels in each bin
    :return z: float
        The redshift
    :return dataid: str
        The data ID
    :return objname: str
        The object name
    """

    assert format in ('manga', 'muse', 'user'), "format must be either 'manga' or 'muse'; no others currently supported!"
    # Read the FITS file using the appropriate parsing function
    # no more eval 🥲
    if format == 'manga':
        nx,ny,nz,ra,dec,dataid,wave,flux,ivar,specres,mask,objname = read_manga_ifu(fits_file,z)
    elif format == 'muse':
        nx,ny,nz,ra,dec,dataid,wave,flux,ivar,specres,mask,objname = read_muse_ifu(fits_file,z)
    else:
        # wave array shape = (nz,)
        # flux, ivar array shape = (nz, ny, nx)
        # specres can be a single value or an array of shape (nz,)

        # VALIDATE THAT USER INPUTS ARE IN THE CORRECT FORMAT
        for value in (nx, ny, nz, ra, dec, wave, flux, specres):
            assert value is not None, "For user spec, all of (nx, ny, nz, ra, dec, wave, flux, specres) must be specified!"
        if ivar is None:
            print("WARNING: No ivar was input.  Defaulting to sqrt(flux).")
            ivar = np.sqrt(flux)
        if mask is None:
            mask = np.zeros(flux.shape, dtype=int)
        assert wave.shape == (nz,), "Wave array shape should be (nz,)"
        assert flux.shape == (nz, ny, nx), "Flux array shape should be (nz, ny, nx)"
        assert ivar.shape == (nz, ny, nx), "Ivar array shape should be (nz, ny, nx)"
        assert mask.shape == (nz, ny, nx), "Mask array shape should be (nz, ny, nx)"
        assert (type(specres) in (int, float, np.int_, np.float_)) or (specres.shape == (nz,)), "Specres should be a float or an array of shape (nz,)"

    loglam = np.log10(wave)
    # FWHM Resolution in angstroms:
    fwhm_res = wave / specres  # dlambda = lambda / R; R = lambda / dlambda
    if not use_and_mask:
        mask = np.zeros(flux.shape, dtype=int)

    # Converting to wdisp -- so that 2.355*wdisp*dlam_gal = fwhm_res
    # if format == 'manga':
    # c = 299792.458  # speed of light in km/s
    # frac = wave[1]/wave[0]    # Constant lambda fraction per pixel
    # dlam_gal = (frac-1)*wave  # Size of every pixel in Angstrom
    # vdisp = c / (2.355*specres)       # delta v = c / R in km/s
    # velscale = np.log(frac) * c  # Constant velocity scale in km/s per pixel
    # wdisp = vdisp / velscale     # Intrinsic dispersion of every pixel, in pixels units

    minx, maxx = 0, nx
    miny, maxy = 0, ny
    if aperture:
        miny, maxy, minx, maxx = aperture
        maxy += 1
        maxx += 1

    x = np.arange(minx, maxx, 1)
    y = np.arange(miny, maxy, 1)
    # Create x/y grid for the voronoi binning
    X, Y = np.meshgrid(x, y)
    _x, _y = X.ravel(), Y.ravel()

    if voronoi_binning:

        # Average along the wavelength axis so each spaxel has one s/n value
        # Note to self: Y AXIS IS ALWAYS FIRST ON NUMPY ARRAYS
        signal = np.nanmean(flux[:, miny:maxy, minx:maxx], axis=0)
        noise = np.sqrt(np.nanmean(1/ivar[:, miny:maxy, minx:maxx], axis=0))

        sr = signal.ravel()
        nr = noise.ravel()
        good = np.where(np.isfinite(sr) & np.isfinite(nr) & (sr > 0) & (nr > 0))[0]
            
        # Target S/N ratio to bin for. If none, defaults to value such that the highest pixel isnt binned
        # In general this isn't a great choice.  Should want to maximize resolution without sacrificing too much
        # computation time.
        if not targetsn:
            # binnum = np.array([maxbins+1])
            targetsn0 = np.max([np.sort((sr / nr)[good], kind='quicksort')[-1] / 16, 10])

            def objective(targetsn, return_data=False):
                vplot = voronoi_plot if return_data else False
                qt = quiet if return_data else True
                try:
                    binnum, xbin, ybin, xbar, ybar, sn, npixels, scale = voronoi_2d_binning(_x[good], _y[good], sr[good], nr[good],
                                                                                        targetsn, cvt=cvt, pixelsize=1, plot=vplot,
                                                                                        quiet=qt, wvt=wvt)
                except ValueError:
                    return np.inf

                if return_data:
                    return binnum, xbin, ybin, xbar, ybar, sn, npixels, scale
                return (np.max(binnum)+1 - maxbins)**2

            print(f'Performing S/N optimization to reach {maxbins} bins.  This may take a while...')
            soln = optimize.minimize(objective, [targetsn0], method='Nelder-Mead', bounds=[(1, X.size)])
            targetsn = soln.x[0]

            binnum, xbin, ybin, xbar, ybar, SNR, npixels, scale = objective(targetsn, return_data=True)

        else:
            binnum, xbin, ybin, xbar, ybar, SNR, npixels, scale = voronoi_2d_binning(_x[good], _y[good], sr[good], nr[good],
                                                                                    targetsn, cvt=cvt, pixelsize=1, plot=voronoi_plot,
                                                                                    quiet=quiet, wvt=wvt)
        print(f'Voronoi binning successful with target S/N = {targetsn}! Created {np.max(binnum)+1} bins.')

        if voronoi_plot:
            # For some reason voronoi makes the plot but doesnt save it or anything
            filename = os.path.join(os.path.dirname(fits_file), 'voronoi_binning.pdf')
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
        _x = _x[good]
        _y = _y[good]
        # Create output arrays for flux, ivar, mask
        out_flux = np.zeros((flux.shape[0], np.nanmax(binnum)+1))
        out_ivar = np.zeros((ivar.shape[0], np.nanmax(binnum)+1))
        out_mask = np.zeros((mask.shape[0], np.nanmax(binnum)+1))
        xpixbin = np.full(np.nanmax(binnum)+1, fill_value=np.nan, dtype=object)
        ypixbin = np.full(np.nanmax(binnum)+1, fill_value=np.nan, dtype=object)
        for j in range(xpixbin.size):
            xpixbin[j] = []
            ypixbin[j] = []
        # Sum flux/ivar in each bin
        for i, bin in enumerate(binnum):
            # there is probably a better way to do this, but I'm lazy
            xi, yi = _x[i], _y[i]
            out_flux[:, bin] += flux[:, yi, xi]
            out_ivar[:, bin] += 1/ivar[:, yi, xi]
            out_mask[:, bin] += mask[:, yi, xi]
            xpixbin[bin].append(xi)
            ypixbin[bin].append(yi)
        # out_flux /= npixels
        # out_ivar /= npixels
        out_ivar = 1/out_ivar
        irange = np.nanmax(binnum)+1

        for bin in binnum:
            if SNR[bin] < snr_threshold:
                flux[:, np.asarray(ypixbin[bin]), np.asarray(xpixbin[bin])] = np.nan
                ivar[:, np.asarray(ypixbin[bin]), np.asarray(xpixbin[bin])] = np.nan
                mask[:, np.asarray(ypixbin[bin]), np.asarray(xpixbin[bin])] = 1
    
    elif fixed_binning:
        print(f'Performing binning with fixed bin size of {fixed_bin_size}')
        # Create square bins of a fixed size
        binnum = np.zeros((maxy-miny, maxx-minx), dtype=int)
        wy = int(np.ceil((maxy-miny)/fixed_bin_size))
        wx = int(np.ceil((maxx-minx)/fixed_bin_size))
        indx = 0
        nbins = wy*wx
        out_flux = np.zeros((flux.shape[0], nbins))
        out_ivar = np.zeros((ivar.shape[0], nbins))
        out_mask = np.zeros((mask.shape[0], nbins))
        xpixbin = np.full(nbins, fill_value=np.nan, dtype=object)
        ypixbin = np.full(nbins, fill_value=np.nan, dtype=object)
        npixels = np.zeros((nbins,), dtype=int)
        SNR = np.zeros((nbins,))
        for iy in range(wy):
            for ix in range(wx):
                # Relative axes indices
                ylo = iy*fixed_bin_size
                yhi = np.min([(iy+1)*fixed_bin_size, binnum.shape[0]])
                xlo = ix*fixed_bin_size
                xhi = np.min([(ix+1)*fixed_bin_size, binnum.shape[1]])
                binnum[ylo:yhi, xlo:xhi] = indx

                # Shift axes limits by the aperture
                ylo += miny
                yhi += miny
                xlo += minx
                xhi += minx
                ybin, xbin = np.meshgrid(np.arange(ylo, yhi, 1), np.arange(xlo, xhi, 1))
                ypixbin[indx] = ybin.flatten().tolist()
                xpixbin[indx] = xbin.flatten().tolist()
                out_flux[:, indx] = np.apply_over_axes(np.nansum, flux[:, ylo:yhi, xlo:xhi], (1,2)).flatten()
                out_ivar[:, indx] = 1/np.apply_over_axes(np.nansum, 1/ivar[:, ylo:yhi, xlo:xhi], (1,2)).flatten()
                out_mask[:, indx] = np.apply_over_axes(np.nansum, mask[:, ylo:yhi, xlo:xhi], (1,2)).flatten()
                npixels[indx] = len(ybin)
                
                signal = np.nanmean(flux[:, ylo:yhi, xlo:xhi], axis=0)
                noise = np.sqrt(np.nanmean(1/ivar[:, ylo:yhi, xlo:xhi], axis=0))
                SNR[indx] = np.nansum(signal) / np.sqrt(np.nansum(noise**2))
                if SNR[indx] < snr_threshold:
                    flux[:, ylo:yhi, xlo:xhi] = np.nan
                    ivar[:, ylo:yhi, xlo:xhi] = np.nan
                    mask[:, ylo:yhi, xlo:xhi] = 1

                indx += 1

        binnum = binnum.flatten()
        irange = nbins
        print(f'Fixed binning successful, created {nbins} bins')

    else:
        xpixbin = None
        ypixbin = None
        out_flux = flux[:, miny:maxy, minx:maxx].reshape(nz, (maxx-minx)*(maxy-miny))
        out_ivar = ivar[:, miny:maxy, minx:maxx].reshape(nz, (maxx-minx)*(maxy-miny))
        out_mask = mask[:, miny:maxy, minx:maxx].reshape(nz, (maxx-minx)*(maxy-miny))
        binnum = np.zeros((maxx-minx)*(maxy-miny))
        npixels = np.ones((maxx-minx)*(maxy-miny)) * (maxx-minx)*(maxy-miny)
        irange = (maxx-minx)*(maxy-miny)

        signal = np.nanmean(flux, axis=0)
        noise = np.sqrt(np.nanmean(1/ivar, axis=0))
        SNR = signal / noise

        flux[:, SNR < snr_threshold] = np.nan
        ivar[:, SNR < snr_threshold] = np.nan
        mask[:, SNR < snr_threshold] = 1

    for i in range(irange):
        # Unpack the spaxel
        galaxy_spaxel = out_flux[:,i]  # observed flux
        ivar_spaxel = out_ivar[:,i]    # 1-sigma spectral noise
        mask_spaxel = out_mask[:,i]    # bad pixels
        if voronoi_binning or fixed_binning:
            xi = xpixbin[i]            # x and y pixel position
            yi = ypixbin[i]
            snr_thresh = SNR[i] >= snr_threshold  # make sure bin has an overall SNR greater than the threshold
        else:
            xi = [_x[i]]
            yi = [_y[i]]
            snr_thresh = SNR[_y[i], _x[i]] >= snr_threshold  # make sure spaxel has an SNR greater than the threshold
        binnum_i = 0 if (not voronoi_binning) and (not fixed_binning) else i   # Voronoi bin index that this pixel belongs to

        # Package into a FITS file -- but only if the SNR is high enough, otherwise throw out the data
        if snr_thresh:
            primaryhdu = fits.PrimaryHDU()
            primaryhdu.header.append(("FORMAT", format.upper(), "Data format"), end=True)
            if type(dataid) is list:
                for j, did in enumerate(dataid):
                    primaryhdu.header.append((f'{format.upper()}ID{j}', did, f'{"MANGA" if format == "manga" else "MUSE"} ID number'), end=True)
            else:
                primaryhdu.header.append((f'{format.upper()}ID', dataid, f'{"MANGA" if format == "manga" else "MUSE"} ID number'), end=True)
            primaryhdu.header.append(('OBJNAME', objname, 'Object Name'), end=True)
            primaryhdu.header.append(('RA', ra, 'Right ascension'), end=True)
            primaryhdu.header.append(('DEC', dec, 'Declination'), end=True)
            primaryhdu.header.append(('BINNUM', binnum_i, 'bin index of the spaxel (Voronoi)'), end=True)
            primaryhdu.header.append(('NX', nx, 'x dimension of the full MANGA cube'), end=True)
            primaryhdu.header.append(('NY', ny, 'y dimension of the full MANGA cube'), end=True)
            coadd = fits.BinTableHDU.from_columns(fits.ColDefs([
                fits.Column(name='flux', array=galaxy_spaxel, format='D'),
                fits.Column(name='loglam', array=loglam, format='D'),
                fits.Column(name='ivar', array=ivar_spaxel, format='D'),
                fits.Column(name='and_mask', array=mask_spaxel, format='D'),
                fits.Column(name='fwhm_res', array=fwhm_res, format='D')
            ]))
            specobj = fits.BinTableHDU.from_columns(fits.ColDefs([
                fits.Column(name='z', array=np.array([z]), format='D'),
                # fits.Column(name='ebv', array=np.array([ebv]), format='E')
            ]))
            specobj.header.append(('PLUG_RA', ra, 'Right ascension'), end=True)
            specobj.header.append(('PLUG_DEC', dec, 'Declination'), end=True)
            binobj = fits.BinTableHDU.from_columns(fits.ColDefs([
                fits.Column(name='spaxelx', array=np.array(xi), format='E'),
                fits.Column(name='spaxely', array=np.array(yi), format='E')
            ]))

            out_hdu = fits.HDUList([primaryhdu, coadd, specobj, binobj])

            # Save output to sub-folder
            if voronoi_binning or fixed_binning:
                tag = '_'.join(['spaxel', 'bin', str(binnum_i)])
            else:
                tag = '_'.join(['spaxel', str(xi[0]), str(yi[0])])
            outdir = os.path.join(os.path.dirname(fits_file), fits_file.split(os.sep)[-1].replace('.fits',''), tag)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = os.path.join(outdir, tag+'.fits')
            out_hdu.writeto(outfile, overwrite=True)

        # else:
        #     for xx, yy in zip(xi, yi):
        #         flux[:, yy, xx] = np.nan
        #         ivar[:, yy, xx] = np.nan
        #         mask[:, yy, xx] = 1

    return wave,flux,ivar,mask,fwhm_res,binnum,npixels,xpixbin,ypixbin,z,dataid,objname


def plot_ifu(fits_file,wave,flux,ivar,mask,binnum,npixels,xpixbin,ypixbin,z,dataid,aperture=None,object_name=None):
    """
    Plot a binned IFU cube and aperture.

    :param fits_file: str
        The file path to the FITS IFU file.
    :param wave: array
        1-D wavelength array with shape (nz,)
    :param flux: array
        3-D masked flux array with shape (nz, ny, nx)
    :param ivar: array
        3-D masked inverse variance array with shape (nz, ny, nx)
    :param mask: array
        3-D mask array with shape (nz, ny, nx)
    :param binnum: array
        Bin number array that specifies which spaxels are in each bin (see the vorbin docs)
    :param npixels: array
        Number of spaxels in each bin (see the vorbin docs)
    :param xpixbin: array
        The x positions of spaxels in each bin
    :param ypixbin: array
        The y positions of spaxels in each bin
    :param z: float
        The redshift
    :param dataid: str
        The data ID
    :param aperture: array
        The lower-left and upper-right corners of a square aperture, formatted as [y0, y1, x0, x1]
    :param objname: str
        The object name
    
    :return None:
    """

    # fig = plt.figure(figsize=(14,4))
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(nrows=8, ncols=8)
    ax1 = fig.add_subplot(gs[0:5, 0:4])
    ax2 = fig.add_subplot(gs[0:5, 4:8])
    ax3 = fig.add_subplot(gs[5:8, 0:8])
    fig.subplots_adjust(wspace=0.1, hspace=0.5)

    ny, nx = flux.shape[1:]
    # breakpoint()
    center = (nx / 2, ny / 2)
    minx, maxx = 0, nx
    miny, maxy = 0, ny
    if aperture:
        miny, maxy, minx, maxx = aperture
        maxy += 1
        maxx += 1

    flux_sum = np.nansum(flux, axis=0)
    # flux_sum[flux_sum==0] = np.nan
    flux_avg = flux_sum / flux.shape[0]
    noise_sum = np.nanmedian(np.sqrt(1/ivar), axis=0)
    flux_max_unbinned = np.nanmax(flux, axis=0)
    noise_max_unbinned = np.nanmax(np.sqrt(1/ivar), axis=0)


    if np.any(binnum):
        flux_bin = np.zeros(np.nanmax(binnum)+1)
        noise_bin = np.zeros(np.nanmax(binnum)+1)
        # flux_max = np.zeros(np.nanmax(binnum)+1)
        # noise_max = np.zeros(np.nanmax(binnum)+1)

        for bin in range(np.nanmax(binnum)+1):
            _x = xpixbin[bin]
            _y = ypixbin[bin]
            for i in range(len(_x)):
                flux_bin[bin] += flux_avg[_y[i], _x[i]]
                noise_bin[bin] += noise_sum[_y[i], _x[i]]
                # flux_max[bin] = np.nanmax([flux_max[bin], np.nanmax(flux[:, _y[i], _x[i]])])
                # noise_max[bin] = np.nanmax([noise_max[bin], np.nanmax(np.sqrt(1/ivar)[:, _y[i], _x[i]])])
        flux_bin /= npixels
        noise_bin /= npixels

        for bin in range(np.nanmax(binnum)+1):
            _x = xpixbin[bin]
            _y = ypixbin[bin]
            for i in range(len(_x)):
                flux_avg[_y[i], _x[i]] = flux_bin[bin]
                noise_sum[_y[i], _x[i]] = noise_bin[bin]
                # flux_max_unbinned[_y[i], _x[i]] = flux_max[bin]
                # noise_max_unbinned[_y[i], _x[i]] = noise_max[bin]

    # This is rapidly making me lose the will to live
    base = 10
    cbar_data = ax1.imshow(np.log(flux_avg*base+1)/np.log(base), origin='lower', cmap='cubehelix')
    cbar_noise = ax2.imshow(np.log(noise_sum*base+1)/np.log(base), origin='lower', cmap='cubehelix')
    cbar = plt.colorbar(cbar_data, ax=ax1, label=r'$\log_{10}{(f_{\lambda,max})}$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ spaxel$^{-1}$)')
    cbar2 = plt.colorbar(cbar_noise, ax=ax2, label=r'$\log_{10}{(\Sigma\sigma)}$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ spaxel$^{-1}$)')

    if aperture:
        aper = plt.Rectangle((aperture[2]-.5, aperture[0]-.5), aperture[3]-aperture[2]+1, aperture[1]-aperture[0]+1, color='red',
                             fill=False, linewidth=2)
        ax1.add_patch(aper)
        aper = plt.Rectangle((aperture[2]-.5, aperture[0]-.5), aperture[3]-aperture[2]+1, aperture[1]-aperture[0]+1, color='red',
                             fill=False, linewidth=2)
        ax2.add_patch(aper)

    # Oh you're a python coder? Name every numpy function.
    coadd = np.nansum(np.nansum(flux, axis=2), axis=1) / (flux.shape[1]*flux.shape[2])
    coadd_noise = np.nansum(np.nansum(np.sqrt(1/ivar), axis=2), axis=1) / (ivar.shape[1]*ivar.shape[2])

    fontsize = 14
    ax3.plot(wave, coadd, linewidth=0.5, color='xkcd:bright aqua', label='Coadded Flux')
    ax3.plot(wave, coadd_noise, linewidth=0.5, color='xkcd:bright orange', label='$1\sigma$ uncertainty')
    ax3.axhline(0.0, color='white', linewidth=0.5, linestyle='--')
    ax3.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)', fontsize=fontsize)
    # ax4.plot(wave, fwhm)
    # ax4.set_ylabel(r'$\Delta\lambda = \lambda/R (\AA)$', fontsize=fontsize)
    ax3.set_xlabel(r'$\lambda_{\rm{obs}}$ ($\mathrm{\AA}$)', fontsize=fontsize)
    ax3.legend(loc='best')

    fig.suptitle(f'OBJECT ID: {dataid}, z={z}' if object_name is None else
                 f'{object_name}, z={z}', fontsize=fontsize)
    plt.tight_layout()
    filepath = os.path.join(os.path.dirname(fits_file), 'fitting_aperture.pdf')
    plt.savefig(filepath)

    ax1.clear()
    ax2.clear()
    ax3.clear()
    fig.clear()
    plt.close(fig)


def reconstruct_ifu(fits_file,mcmc_label=None):
    """
    Reconstruct an IFU cube using the fit MCMC data from BADASS

    :param fits_file: str
        The file path to the original IFU FITS file
    :param mcmc_label: int, optional
        The index of the MCMC_output_* files that should be used in the reconstruction. Defaults to the largest one found.
    
    :return par_out: FITS HDUList
        FITS-formatted HDUList mirroring the par_table format from BADASS, but arranged in the cube shape
        Each HDU in the list corresponds to one parameter, mapped with a shape (ny, nx)
    :return bmc_out: FITS HDUList
        FITS-formatted HDUList mirroring the best_model_components format from BADASS, but arranged in the cube shape
        Each HDU in the list corresponds to a model component, mapped with a shape (nz, ny, nx)
    :return last_mcmc+1: int
        The index of the output MCMC_output_* file for the overall cube (independent of the individual MCMC_output_* folders for each spaxel)
    """

    # Make sure outputs exist
    path = fits_file.replace('.fits', '') + os.sep
    if not os.path.exists(path):
        raise NotADirectoryError(f"The unpacked folders for {fits_file} do not exist! Fit before calling reconstruct")
    subdirs = glob.glob(path + 'spaxel_*_*')
    voronoi = subdirs[0].split('_')[1] == 'bin'
    subdirs.sort()
    if len(subdirs) == 0:
        raise NotADirectoryError(f"The unpacked folders for {fits_file} do not exist! Fit before calling reconstruct")

    # Get number of bins
    if voronoi:
        nbins = max([int(subdir.split('_')[-1]) for subdir in subdirs]) + 1
    else:
        nbins = len(subdirs)
    xpixbin = np.full(nbins, fill_value=np.nan, dtype=object)
    ypixbin = np.full(nbins, fill_value=np.nan, dtype=object)

    i = 0
    subdir = subdirs[0]

    # Find each MCMC output
    if mcmc_label is None:
        most_recent_mcmc = glob.glob(subdir + os.sep + 'MCMC_output_*')
        if len(most_recent_mcmc) == 0:
            raise NotADirectoryError(f"The unpacked folders for {fits_file} do not exist! Fit before calling reconstruct")
        most_recent_mcmc = sorted(most_recent_mcmc)[-1]
    else:
        most_recent_mcmc = glob.glob(subdir + os.sep + f"MCMC_output_{mcmc_label}")
        if len(most_recent_mcmc) == 0:
            raise NotADirectoryError(f"The unpacked folders for {fits_file}, MCMC_output{mcmc_label} do not exist! Fit before calling reconstruct")
        most_recent_mcmc = most_recent_mcmc[0]
    par_table = sorted(glob.glob(os.path.join(most_recent_mcmc, 'log', '*par_table.fits')))
    best_model_components = sorted(glob.glob(os.path.join(most_recent_mcmc, 'log', '*best_model_components.fits')))
    test_stats = sorted(glob.glob(os.path.join(most_recent_mcmc, 'log', 'test_stats.fits')))
    if len(par_table) < 1 or len(best_model_components) < 1:
        raise FileNotFoundError(
            f"The FITS files for {most_recent_mcmc} do not exist! Fit before calling reconstruct")
    par_table = par_table[0]
    best_model_components = best_model_components[0]

    # Load in the FITS files
    with fits.open(par_table) as parhdu, fits.open(best_model_components) as bmchdu:
        # Get the bin number and x/y coord(s)
        hdr = parhdu[0].header
        data1 = parhdu[1].data
        data2 = parhdu[2].data
        bdata = bmchdu[1].data

    if len(test_stats) > 0:
        test_stats = test_stats[0]
        with fits.open(test_stats) as tshdu:
            tdata = tshdu[1].data
    else:
        tdata = None

    binnum = copy.deepcopy(hdr['binnum']) if voronoi else i
    xpixbin[binnum] = copy.deepcopy(data2['spaxelx'])
    ypixbin[binnum] = copy.deepcopy(data2['spaxely'])

    # if it's the first iteration, create the arrays based on the proper shape
    parameters = data1['parameter']
    if tdata is not None:
        parameters = np.concatenate((parameters, tdata['parameter']))
    parvals = np.full(shape=(nbins,), fill_value=np.nan, dtype=[
        (param, float) for param in np.unique(parameters)
    ])
    parvals_low = copy.deepcopy(parvals)
    parvals_upp = copy.deepcopy(parvals)
    bmcparams = np.array(bdata.columns.names, dtype=str)
    bmcvals = np.full(shape=(bdata.size, nbins), fill_value=np.nan, dtype=[
        (param, float) for param in np.unique(bmcparams)
    ])

    # Set the par table parameters
    mcmc = 'ci_68_low' in data1.names and 'ci_68_upp' in data1.names
    for param in parameters:
        w = np.where(data1['parameter'] == param)[0]
        if w.size > 0:
            w = w[0]
            parvals[param][binnum] = copy.deepcopy(data1['best_fit'][w])
            if mcmc:
                parvals_low[param][binnum] = copy.deepcopy(data1['ci_68_low'][w])
                parvals_upp[param][binnum] = copy.deepcopy(data1['ci_68_upp'][w])
        elif tdata is not None:
            w2 = np.where(tdata['parameter'] == param)[0]
            if w2.size > 0:
                parvals[param][binnum] = copy.deepcopy(tdata['best_fit'][w2])
                if mcmc:
                    parvals_low[param][binnum] = copy.deepcopy(tdata['ci_68_low'][w2])
                    parvals_upp[param][binnum] = copy.deepcopy(tdata['ci_68_upp'][w2])


    # Set the best model components
    for param in bmcparams:
        bmcvals[param][:, binnum] = copy.deepcopy(bdata[param])

    parsize = data1.size
    if tdata is not None:
        parsize += tdata.size
    bmcsize = bdata.size

    def append_spaxel(i, subdir):
        nonlocal parvals, parvals_low, parvals_upp, bmcvals, parameters, xpixbin, ypixbin, voronoi

        # Find each MCMC output
        if mcmc_label is None:
            most_recent_mcmc = glob.glob(subdir + os.sep + 'MCMC_output_*')
            if len(most_recent_mcmc) == 0:
                # raise NotADirectoryError(
                # f"The unpacked folders for {fits_file} do not exist! Fit before calling reconstruct")
                print(f"WARNING: MCMC folder for {subdir} not found!")
                return
            most_recent_mcmc = sorted(most_recent_mcmc)[-1]
        else:
            most_recent_mcmc = glob.glob(subdir + os.sep + f"MCMC_output_{mcmc_label}")
            if len(most_recent_mcmc) == 0:
                print(f"WARNING: MCMC folder for {subdir} not found!")
                return
            most_recent_mcmc = most_recent_mcmc[0]
        par_table = sorted(glob.glob(os.path.join(most_recent_mcmc, 'log', '*par_table.fits')))
        best_model_components = sorted(glob.glob(os.path.join(most_recent_mcmc, 'log', '*best_model_components.fits')))
        test_stats = sorted(glob.glob(os.path.join(most_recent_mcmc, 'log', 'test_stats.fits')))
        if len(par_table) < 1 or len(best_model_components) < 1:
            # raise FileNotFoundError(
            # f"The FITS files for {most_recent_mcmc} do not exist! Fit before calling reconstruct")
            print(f"WARNING: FITS files for {most_recent_mcmc} not found!")
            return
        par_table = par_table[0]
        best_model_components = best_model_components[0]

        # Load in the FITS files
        with fits.open(par_table) as parhdu, fits.open(best_model_components) as bmchdu:
            # Get the bin number and x/y coord(s)
            hdr = parhdu[0].header
            data1 = parhdu[1].data
            data2 = parhdu[2].data
            bdata = bmchdu[1].data

        if len(test_stats) > 0:
            test_stats = test_stats[0]
            with fits.open(test_stats) as tshdu:
                tdata = tshdu[1].data
        else:
            tdata = None

        binnum = copy.deepcopy(hdr['binnum']) if voronoi else i
        xpixbin[binnum] = copy.deepcopy(data2['spaxelx'])
        ypixbin[binnum] = copy.deepcopy(data2['spaxely'])

        # Set the par table parameters
        mcmc = 'ci_68_low' in data1.names and 'ci_68_upp' in data1.names
        for param in parameters:
            w = np.where(data1['parameter'] == param)[0]
            if w.size > 0:
                w = w[0]
                parvals[param][binnum] = copy.deepcopy(data1['best_fit'][w])
                if mcmc:
                    parvals_low[param][binnum] = copy.deepcopy(data1['ci_68_low'][w])
                    parvals_upp[param][binnum] = copy.deepcopy(data1['ci_68_upp'][w])
            elif tdata is not None:
                w2 = np.where(tdata['parameter'] == param)[0]
                if w2.size > 0:
                    parvals[param][binnum] = copy.deepcopy(tdata['best_fit'][w2])
                    if mcmc:
                        parvals_low[param][binnum] = copy.deepcopy(tdata['ci_68_low'][w2])
                        parvals_upp[param][binnum] = copy.deepcopy(tdata['ci_68_upp'][w2])

        # Set the best model components
        for param in bmcparams:
            bmcvals[param][:, binnum] = copy.deepcopy(bdata[param])

    iterable = enumerate(subdirs) if tqdm is None else tqdm.tqdm(enumerate(subdirs), total=len(subdirs))
    Parallel(n_jobs=-1, require='sharedmem')(delayed(append_spaxel)(i, subdir) for i, subdir in iterable)
    for i in range(len(xpixbin)):
        if type(xpixbin[i]) in (float, np.float_) and np.isnan(xpixbin[i]):
            xpixbin[i] = []
        if type(ypixbin[i]) in (float, np.float_) and np.isnan(ypixbin[i]):
            ypixbin[i] = []

    maxx = -np.inf
    maxy = -np.inf
    minx = np.inf
    miny = np.inf
    for j in range(nbins):
        maxx = np.nanmax([maxx, np.nanmax(xpixbin[j]) if len(xpixbin[j]) > 0 else np.nan])
        maxy = np.nanmax([maxy, np.nanmax(ypixbin[j]) if len(ypixbin[j]) > 0 else np.nan])
        minx = np.nanmin([minx, np.nanmin(xpixbin[j]) if len(xpixbin[j]) > 0 else np.nan])
        miny = np.nanmin([miny, np.nanmin(ypixbin[j]) if len(ypixbin[j]) > 0 else np.nan])

    # Reconstruct original shape
    nx = int(maxx - minx + 1)
    ny = int(maxy - miny + 1)
    bmcvals_out = np.full(shape=(bmcparams.size, bmcsize, ny, nx), fill_value=np.nan, dtype=float)
    parvals_out = np.full(shape=(parsize, ny, nx), fill_value=np.nan, dtype=float)
    parvals_out_low = copy.deepcopy(parvals_out)
    parvals_out_upp = copy.deepcopy(parvals_out)
    binpix = np.zeros((nx*ny, 3), dtype=int)
    ii = 0
    for n in range(nbins):
        for xi, yi in zip(xpixbin[n], ypixbin[n]):
            for j, param in enumerate(parameters):
                parvals_out[j, int(yi-miny), int(xi-minx)] = parvals[param][n]
                if mcmc:
                    parvals_out_low[j, int(yi-miny), int(xi-minx)] = parvals_low[param][n]
                    parvals_out_upp[j, int(yi-miny), int(xi-minx)] = parvals_upp[param][n]
            binpix[ii, :] = (int(xi-minx), int(yi-miny), n)
            ii += 1
        for j, param in enumerate(bmcparams):
            for xi, yi in zip(xpixbin[n], ypixbin[n]):
                bmcvals_out[j, :, int(yi-miny), int(xi-minx)] = bmcvals[param][:, n]

    # Construct FITS outputs
    bmc_out = fits.HDUList()
    primary = fits.PrimaryHDU()
    primary.header.append(('ORIGINX', minx, 'x-coordinate of position (0,0) in full cube'), end=True)
    primary.header.append(('ORIGINY', miny, 'y-coordinate of position (0,0) in full cube'), end=True)
    primary.header.append(('NBINS', nbins, 'number of Voronoi bins'), end=True)
    primary2 = copy.deepcopy(primary)

    bininfo = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name='x', array=binpix[:, 0], format='I'),
        fits.Column(name='y', array=binpix[:, 1], format='I'),
        fits.Column(name='bin', array=binpix[:, 2], format='I')
    ]))
    bininfo2 = copy.deepcopy(bininfo)

    bmc_out.append(primary)
    for k, name in enumerate(bmcparams):
        if name.upper() == 'WAVE':
            # good = np.where(np.isfinite(bmcvals_out[k, ...]))
            bmc_out.append(
                fits.BinTableHDU.from_columns(fits.ColDefs([
                    fits.Column(name='wave', array=bmcvals_out[k, :, ny//2, nx//2], format='E'),
            ]), name=name))
        else:
            bmc_out.append(
                fits.ImageHDU(bmcvals_out[k, ...], name=name)
            )
    bmc_out.append(bininfo)
    par_out = fits.HDUList()
    par_out.append(primary2)
    for k, name in enumerate(parameters):
        par_out.append(
            fits.ImageHDU(parvals_out[k, ...], name=name)
        )
        if mcmc:
            par_out.append(
                fits.ImageHDU(parvals_out_low[k, ...], name=name + '_SIGMA_LOW')
            )
            par_out.append(
                fits.ImageHDU(parvals_out_upp[k, ...], name=name + '_SIGMA_UPP')
            )
    par_out.append(bininfo2)

    # Write outputs
    folders = os.listdir(os.path.dirname(fits_file))
    mcmc_outputs = [int(fold.split('_')[-1]) for fold in folders if 'MCMC_output' in fold]
    if len(mcmc_outputs) >= 1:
        last_mcmc = max(mcmc_outputs)
    else:
        last_mcmc = 0
    logdir = os.path.join(os.path.dirname(fits_file), 'MCMC_output_'+str(last_mcmc+1), 'log')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    bmc_out.writeto(logdir + os.sep + 'cube_best_model_components.fits', overwrite=True)
    par_out.writeto(logdir + os.sep + 'cube_par_table.fits', overwrite=True)

    return par_out, bmc_out, last_mcmc+1


def plot_reconstructed_cube(mcmc_output_dir, partable_to_plot=None, bmc_to_plot=None, animated=False):
    """
    Make 2D maps and/or videos of the reconstructed par_table and best_model_components parameters

    :param mcmc_output_dir: str
        The folder to the overall MCMC_output_* folder for the whole cube (not individual spaxels)
    :param partable_to_plot: list, optional
        List of the par_table parameter names to plot. If None, plots them all.
    :param bmc_to_plot: list, optional
        List of best_model_components parameter names to plot. If None, plots them all.
    :param animated: bool   
        Whether or not to make the best_model_components plots into videos. Required an installation of FFMpeg.
    
    :return None:
    """
    # Get directories
    partable = os.path.join(mcmc_output_dir, 'log', 'cube_par_table.fits')
    bmc = os.path.join(mcmc_output_dir, 'log', 'cube_best_model_components.fits')
    if not os.path.isfile(partable) or not os.path.isfile(bmc):
        raise FileNotFoundError(f"Could not find cube_par_table.fits or cube_best_model_components.fits in"
                                f"{mcmc_output_dir}/log/")

    # Load in data
    parhdu = fits.open(partable)
    bmchdu = fits.open(bmc)
    ox, oy = parhdu[0].header['ORIGINX'], parhdu[0].header['ORIGINY']

    # First make 2D image maps for each parameter in par table
    if not os.path.exists(os.path.join(mcmc_output_dir, 'partable_plots')):
        os.mkdir(os.path.join(mcmc_output_dir, 'partable_plots'))
    if not os.path.exists(os.path.join(mcmc_output_dir, 'best_model_components_plots')):
        os.mkdir(os.path.join(mcmc_output_dir, 'best_model_components_plots'))

    if partable_to_plot is None:
        partable_to_plot = [p.name for p in parhdu[1:-1]]
    if bmc_to_plot is None:
        bmc_to_plot = [b.name for b in bmchdu[1:-1]]

    for imagehdu in parhdu[1:-1]:
        if imagehdu.name not in partable_to_plot:
            continue
        fig, ax = plt.subplots()
        data = imagehdu.data
        std = np.nanstd(data)
        mad = stats.median_absolute_deviation(data[np.isfinite(data)])
        # data[np.abs(data - np.nanmedian(data)) > 10*std] = np.nan
        if "FLUX" in imagehdu.name and "SIGMA" not in imagehdu.name:
            mask = data >= 0
            data[mask] = np.nan

        map_ = ax.imshow(data, origin='lower', cmap='cubehelix',
                  vmin=np.nanpercentile(data, 1),
                  vmax=np.nanpercentile(data, 99),
                  extent=[ox-.5, ox+imagehdu.data.shape[1]-.5, oy-.5, oy+imagehdu.data.shape[0]-.5])
        plt.colorbar(map_, ax=ax, label=imagehdu.name)
        ax.set_title(mcmc_output_dir.split(os.sep)[-1])
        plt.savefig(os.path.join(mcmc_output_dir, 'partable_plots', f'{imagehdu.name}.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

    # Now loop over and plot the model components, averaging/summing over wavelength
    if not animated:
        for imagehdu in bmchdu[1:-1]:
            if imagehdu.name.upper() == 'WAVE' or imagehdu.name not in bmc_to_plot:
                continue
            # Sum over the 1st axis, aka the wavelength axis
            datasum = np.nansum(imagehdu.data, axis=0)
            datasum[datasum == 0] = np.nan
            # datasum[np.abs(datasum) > 1e5] = np.nan
            dataavg = datasum / imagehdu.data.shape[0]
            std = np.nanstd(dataavg)
            # mad = stats.median_absolute_deviation(dataavg.flatten()[np.isfinite(dataavg.flatten())])
            # dataavg[np.abs(dataavg - np.nanmedian(dataavg)) > 10*std] = np.nan

            fig, ax = plt.subplots()
            map_ = ax.imshow(dataavg, origin='lower', cmap='cubehelix',
                             vmin=np.nanpercentile(dataavg, 1),
                             vmax=np.nanpercentile(dataavg, 99),
                             extent=[ox-.5, ox+imagehdu.data.shape[2]-.5, oy-.5, oy+imagehdu.data.shape[1]-.5])
            plt.colorbar(map_, ax=ax, label=imagehdu.name)
            ax.set_title(mcmc_output_dir.split(os.sep)[-1])
            plt.savefig(os.path.join(mcmc_output_dir, 'best_model_components_plots', f'{imagehdu.name}.pdf'), bbox_inches='tight', dpi=300)
            plt.close()
    else:
        for imagehdu in bmchdu[1:-1]:
            if imagehdu.name.upper() == 'WAVE' or imagehdu.name not in bmc_to_plot:
                continue
            FFMpegWriter = animation.writers['ffmpeg']
            # ensure no matter how many frames there are, the video lasts 30 seconds
            if bmchdu['WAVE'].data['wave'].size > (5*30):
                fps = bmchdu['WAVE'].data['wave'].size / 30
            else:
                fps = 5
            metadata = {'title': imagehdu.name, 'artist': 'BADASS', 'fps': fps}
            writer = FFMpegWriter(fps=fps, metadata=metadata)

            fig = plt.figure()
            gs = gridspec.GridSpec(ncols=10, nrows=10)
            ax1 = fig.add_subplot(gs[0:8, 0:8])
            ax2 = fig.add_subplot(gs[9:10, :])
            ax3 = fig.add_subplot(gs[0:8, 8:9])
            # fig.subplots_adjust(wspace=.5, hspace=.5)
            a = imagehdu.data[0, ...]

            datasum = np.nansum(imagehdu.data, axis=0)
            datasum[datasum == 0] = np.nan
            # datasum[np.abs(datasum) > 1e5] = np.nan
            dataavg = datasum / imagehdu.data.shape[0]
            # mad = stats.median_absolute_deviation(a[np.isfinite(a)])
            # a[np.abs(a - np.nanmedian(a)) > 10*np.nanstd(a)] = np.nan

            im = ax1.imshow(a, origin='lower', cmap='cubehelix',
                            vmin=np.nanpercentile(dataavg, 1),
                            vmax=np.nanpercentile(dataavg, 99),
                            extent=[ox-.5, ox+imagehdu.data.shape[2]-.5, oy-.5, oy+imagehdu.data.shape[1]-.5])
            plt.colorbar(im, cax=ax3, label=imagehdu.name)
            ax2.hlines(6,bmchdu['WAVE'].data['wave'][0],bmchdu['WAVE'].data['wave'][-1])
            ln, = ax2.plot(bmchdu['WAVE'].data['wave'][0], 24, '|', ms=20, color='y')
            ax2.axis('off')
            ax2.set_ylim(-10, 24)
            ax2.text(bmchdu['WAVE'].data['wave'][bmchdu['WAVE'].data['wave'].size//2], -8, r'$\lambda$ [$\AA$]', horizontalalignment='center', verticalalignment='center')
            time_text = ax2.text(bmchdu['WAVE'].data['wave'][bmchdu['WAVE'].data['wave'].size//2], 16, f"{bmchdu['WAVE'].data['wave'][0]:.1f}",
                                 horizontalalignment='center', verticalalignment='center')
            ax2.text(bmchdu['WAVE'].data['wave'][0], -8, str(bmchdu['WAVE'].data['wave'][0]), horizontalalignment='center', verticalalignment='center')
            ax2.text(bmchdu['WAVE'].data['wave'][-1], -8, str(bmchdu['WAVE'].data['wave'][-1]), horizontalalignment='center', verticalalignment='center')

            with writer.saving(fig, os.path.join(mcmc_output_dir, 'best_model_components_plots', f'{imagehdu.name}.mp4'), 100):
                for i in range(imagehdu.data.shape[0]):
                    ai = imagehdu.data[i, ...]
                    im.set_array(ai)
                    ln.set_data(bmchdu['WAVE'].data['wave'][i], 24)
                    time_text.set_text(f"{bmchdu['WAVE'].data['wave'][i]:.1f}")
                    writer.grab_frame()
            plt.close()

    parhdu.close()
    bmchdu.close()
