import copy
import numpy as np

from astropy.io import fits
from vorbin.voronoi_2d_binning import voronoi_2d_binning

from utils.input.input import BadassInput


class IFUInput(BadassInput):

    def __init__(self):
        super().__init__()

        self.nx, self.ny, self.nz = [None]*3
        self.dataid = -1
        self.objname = None


    @classmethod
    def read_ifu(cls, input_data, options):
        if not isinstance(input_data, dict):
            raise Exception('Default IFU input data must be dict')

        for attr in ['infile', 'nx', 'ny', 'nz', 'spec', 'wave', 'noise', 'specres', 'mask']:
            if attr not in input_data:
                raise Exception('Default IFU input reader expects {attr} key'.format(attr=attr))

        inobj = cls()
        inobj.infile = input_data['infile']
        inobj.options = options

        inobj.nx = input_data['nx']
        inobj.ny = input_data['ny']
        inobj.nz = input_data['nz']

        return inobj.ifu_common(inobj, input_data['spec'], input_data['wave'], input_data['ivar'], input_data['specres'], input_data['mask'])


    # TODO: better way to call this from child classes automatically?
    # TODO: need ivar? just pass noise?
    @classmethod
    def ifu_common(cls, base, spec, noise, mask):
        # TODO: validation before?
        # if not base.validate_custom():
        #     raise Exception('IFU validation failed')

        options = base.options

        # FWHM Resolution in angstroms:
        base.fwhm_res = base.wave / base.specres  # dlambda = lambda / R; R = lambda / dlambda
        base.velscale = np.log(base.wave[1] / base.wave[0]) * C

        if not options.ifu_options.use_and_mask:
            mask = np.zeros(spec.shape, dtype=int)

        snr_threshold = options.ifu_options.snr_threshold

        minx, maxx = 0, base.nx
        miny, maxy = 0, base.ny
        if options.ifu_options.aperture:
            # TODO: options validation that aperture is a 4 element list
            miny, maxy, minx, maxx = options.ifu_options.aperture
            maxy += 1
            maxx += 1

        x = np.arange(minx, maxx, 1)
        y = np.arange(miny, maxy, 1)
        # Create x/y grid for the voronoi binning
        X, Y = np.meshgrid(x, y)
        _x, _y = X.ravel(), Y.ravel()

        if options.ifu_options.voronoi_binning:

            # Average along the wavelength axis so each spaxel has one s/n value
            # Note to self: Y AXIS IS ALWAYS FIRST ON NUMPY ARRAYS
            signal = np.nanmean(spec[:, miny:maxy, minx:maxx], axis=0)
            ap_noise = np.nanmean(noise[:, miny:maxy, minx:maxx])

            sr = signal.ravel()
            nr = ap_noise.ravel()
            good = np.where(np.isfinite(sr) & np.isfinite(nr) & (sr > 0) & (nr > 0))[0]

            # Target S/N ratio to bin for. If none, defaults to value such that the highest pixel isnt binned
            # In general this isn't a great choice.  Should want to maximize resolution without sacrificing too much
            # computation time.
            if not options.ifu_options.targetsn:
                targetsn0 = np.max([np.sort((sr / nr)[good], kind='quicksort')[-1] / 16, 10])

                def objective(targetsn, return_data=False):
                    vplot = options.ifu_options.voronoi_plot if return_data else False
                    qt = options.ifu_options.quiet if return_data else True
                    try:
                        binnum, xbin, ybin, xbar, ybar, sn, npixels, scale = voronoi_2d_binning(_x[good], _y[good], sr[good], nr[good],
                                                                                            options.ifu_options.targetsn, cvt=options.ifu_options.cvt, pixelsize=1, plot=vplot,
                                                                                            quiet=qt, wvt=options.ifu_options.wvt)
                    except ValueError:
                        return np.inf

                    if return_data:
                        return binnum, xbin, ybin, xbar, ybar, sn, npixels, scale
                    return (np.max(binnum)+1 - maxbins)**2

                print(f'Performing S/N optimization to reach {maxbins} bins.  This may take a while...')
                soln = optimize.minimize(objective, [targetsn0], method='Nelder-Mead', bounds=[(1, X.size)])
                options.ifu_options.targetsn = soln.x[0]

                binnum, xbin, ybin, xbar, ybar, SNR, npixels, scale = objective(options.ifu_options.targetsn, return_data=True)

            else:
                binnum, xbin, ybin, xbar, ybar, SNR, npixels, scale = voronoi_2d_binning(_x[good], _y[good], sr[good], nr[good],
                                                                                        options.ifu_options.targetsn, cvt=options.ifu_options.cvt, pixelsize=1, plot=options.ifu_options.voronoi_plot,
                                                                                        quiet=options.ifu_options.quiet, wvt=options.ifu_options.wvt)
                print('Voronoi binning successful with target S/N = {t}! Created {bins} bins.'.format(t=options.ifu_options.targetsn, bins=np.max(binnum)+1))

            if options.ifu_options.voronoi_plot:
                # For some reason voronoi makes the plot but doesnt save it or anything
                filename = base.infile.joinpath('voronoi_binning.pdf') # TODO: option for other out directory
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close()

            _x = _x[good]
            _y = _y[good]
            # Create output arrays for spec, noise, mask
            out_spec = np.zeros((spec.shape[0], np.nanmax(binnum)+1))
            out_noise = np.zeros((noise.shape[0], np.nanmax(binnum)+1))
            out_mask = np.zeros((mask.shape[0], np.nanmax(binnum)+1))
            xpixbin = np.full(np.nanmax(binnum)+1, fill_value=np.nan, dtype=object)
            ypixbin = np.full(np.nanmax(binnum)+1, fill_value=np.nan, dtype=object)
            for j in range(xpixbin.size):
                xpixbin[j] = []
                ypixbin[j] = []
            # Average spec and noise in each bin
            for i, bin in enumerate(binnum):
                # there is probably a better way to do this, but I'm lazy
                xi, yi = _x[i], _y[i]
                out_spec[:, bin] += spec[:, yi, xi]
                out_noise[:, bin] += noise[:, yi, xi]
                out_mask[:, bin] += mask[:, yi, xi]
                xpixbin[bin].append(xi)
                ypixbin[bin].append(yi)
            out_spec /= npixels
            out_noise /= npixels
            irange = np.nanmax(binnum)+1

            for bin in binnum:
                if SNR[bin] < snr_threshold:
                    spec[:, np.asarray(ypixbin[bin]), np.asarray(xpixbin[bin])] = np.nan
                    noise[:, np.asarray(ypixbin[bin]), np.asarray(xpixbin[bin])] = np.nan
                    mask[:, np.asarray(ypixbin[bin]), np.asarray(xpixbin[bin])] = 1

        else:
            xpixbin = None
            ypixbin = None
            out_spec = spec[:, miny:maxy, minx:maxx].reshape(base.nz, (maxx-minx)*(maxy-miny))
            out_noise = noise[:, miny:maxy, minx:maxx].reshape(base.nz, (maxx-minx)*(maxy-miny))
            out_mask = mask[:, miny:maxy, minx:maxx].reshape(base.nz, (maxx-minx)*(maxy-miny))
            binnum = np.zeros((maxx-minx)*(maxy-miny))
            npixels = np.ones((maxx-minx)*(maxy-miny)) * (maxx-minx)*(maxy-miny)
            irange = (maxx-minx)*(maxy-miny)

            signal = np.nanmean(spec, axis=0)
            SNR = signal / np.nanmean(noise)

            spec[:, SNR < snr_threshold] = np.nan
            noise[:, SNR < snr_threshold] = np.nan
            mask[:, SNR < snr_threshold] = 1

        inputs = []
        for i in range(irange):

            # Unpack the spaxel
            if options.ifu_options.voronoi_binning:
                xi = spaxel.xpixbin[i]            # x and y pixel position
                yi = spaxel.ypixbin[i]
                snr_thresh = SNR[i] >= snr_threshold  # make sure bin has an overall SNR greater than the threshold
            else:
                xi = [_x[i]]
                yi = [_y[i]]
                snr_thresh = SNR[_y[i], _x[i]] >= snr_threshold  # make sure spaxel has an SNR greater than the threshold

            # Package into a FITS file -- but only if the SNR is high enough, otherwise throw out the data
            if not snr_thresh:
                continue

            # Create a BadassInput instance for each spaxel
            spaxel = copy.deepcopy(base)

            spaxel.spec = out_spec[:,i]  # observed spectrum flux
            spaxel.noise = out_noise[:,i]    # 1-sigma spectral noise
            spaxel.mask = out_mask[:,i]    # bad pixels
            spaxel.xi, spaxel.yi = xi, yi
            spaxel.binnum_i = i if options.ifu_options.voronoi_binning else 0   # Voronoi bin index that this pixel belongs to

            spaxel.write_fits()
            inputs.append(spaxel)

        return inputs


    def write_fits(self):

        primaryhdu = fits.PrimaryHDU()
        format = self.options.io_options.infmt
        primaryhdu.header.append(("FORMAT", format.upper(), "Data format"), end=True)
        if type(self.dataid) is list:
            for j, did in enumerate(self.dataid):
                primaryhdu.header.append((f'{format.upper()}ID{j}', did, f'{"MANGA" if format == "manga" else "MUSE"} ID number'), end=True)
        else:
            primaryhdu.header.append((f'{format.upper()}ID', self.dataid, f'{"MANGA" if format == "manga" else "MUSE"} ID number'), end=True)
        primaryhdu.header.append(('OBJNAME', self.objname, 'Object Name'), end=True)
        primaryhdu.header.append(('RA', self.ra, 'Right ascension'), end=True)
        primaryhdu.header.append(('DEC', self.dec, 'Declination'), end=True)
        primaryhdu.header.append(('BINNUM', self.binnum_i, 'bin index of the spaxel (Voronoi)'), end=True)
        primaryhdu.header.append(('NX', self.nx, 'x dimension of the full MANGA cube'), end=True)
        primaryhdu.header.append(('NY', self.ny, 'y dimension of the full MANGA cube'), end=True)
        coadd = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name='spec', array=self.spec, format='E'),
            fits.Column(name='wave', array=self.wave, format='E'),
            fits.Column(name='noise', array=self.noise, format='E'),
            fits.Column(name='mask', array=self.mask, format='E'),
            fits.Column(name='fwhm_res', array=self.fwhm_res, format='E')
        ]))
        specobj = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name='z', array=np.array([self.z]), format='E'),
        ]))
        specobj.header.append(('PLUG_RA', self.ra, 'Right ascension'), end=True)
        specobj.header.append(('PLUG_DEC', self.dec, 'Declination'), end=True)
        binobj = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name='spaxelx', array=np.array(self.xi), format='E'),
            fits.Column(name='spaxely', array=np.array(self.yi), format='E')
        ]))

        out_hdu = fits.HDUList([primaryhdu, coadd, specobj, binobj])

        # Save output to sub-folder
        if self.options.ifu_options.voronoi_binning:
            tag = '_'.join(['spaxel', 'bin', str(binnum_i)])
        else:
            tag = '_'.join(['spaxel', str(self.xi[0]), str(self.yi[0])])

        outdir = self.infile.parent.joinpath(self.infile.stem, tag)
        outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir.joinpath(tag + '.fits')
        out_hdu.writeto(outfile, overwrite=True)
        self.infile = outfile


    # Validate all custom IFU readers included the expected values
    def validate_custom(self):
        for attr in ['nx', 'ny', 'nz']:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                print('Custom IFU reader did not provide {attr}'.format(attr))
                return False

        if (not hasattr(self, 'noise')) or (self.noise is None):
            print("WARNING: No ivar was input.  Defaulting to sqrt(flux).")
            self.noise = np.sqrt(1 / np.sqrt(self.flux))

        if (not hasattr(self, 'mask')) or (self.mask is None):
            self.mask = np.zeros(self.spec.shape, dtype=int)

        # TODO: some of these are set in ifu_common, need pre and post validation
        tests = [
                    (self.wave.shape == (self.nz,), 'Wave array shape should be (nz,)'),
                    (self.spec.shape == (self.nz, self.ny, self.nx), 'Flux array shape should be (nz, ny, nx)'),
                    (self.noise.shape == (self.nz, self.ny, self.nx), 'Ivar array shape should be (nz, ny, nx)'),
                    (self.mask.shape == (self.nz, self.ny, self.nx), 'Mask array shape should be (nz, ny, nx)'),
                    ((type(self.specres) in (int, float, np.int_, np.float_)) or (self.specres.shape == (self.nz,)), 'Specres should be a float or an array of shape (nz,)'),
                ]

        for (test, err_msg) in tests:
            if not test:
                print(err_msg)
                return False

        return True


    def validate_input(self):
        # TODO: any other validation post-common
        return True
