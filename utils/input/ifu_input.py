import numpy as np

from astropy.io import fits
from vorbin.voronoi_2d_binning import voronoi_2d_binning

from utils.input.input import BadassInput


class IFUInput(BadassInput):

    @classmethod
    def read_ifu(cls, input_data, options):
        if not isinstance(input_data, pathlib.Path):
            raise Exception('Reading MUSE IFU from data currently unsupported')

        spec = cls()
        spec.infile = input_data
        spec.dataid = -1
        spec.objname = None
        spec.nx = options.ifu_options.nx
        spec.ny = options.ifu_options.ny
        spec.nz = options.ifu_options.nz
        spec.ra = options.ifu_options.ra
        spec.dec = options.ifu_options.dec
        spec.wave = options.ifu_options.wave
        spec.flux = options.ifu_options.flux
        spec.specres = options.ifu_options.specres
        spec.ifu_common(options)
        return spec


    # TODO: better way to call this from child classes automatically?
    def ifu_common(self, options):
        if not self.validate_custom():
            raise Exception('IFU validation failed')

        self.z = options.ifu_options.z # TODO: options validate: exists and is float or int
        self.loglam = np.log10(self.wave)
        # FWHM Resolution in angstroms:
        self.fwhm_res = self.wave / self.specres  # dlambda = lambda / R; R = lambda / dlambda
        if not options.ifu_options.use_and_mask:
            self.mask = np.zeros(self.flux.shape, dtype=int)

        snr_threshold = options.ifu_options.snr_threshold
        format = options.io_options.infmt

        minx, maxx = 0, self.nx
        miny, maxy = 0, self.ny
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
            signal = np.nanmean(self.flux[:, miny:maxy, minx:maxx], axis=0)
            noise = np.sqrt(1 / np.nanmean(self.ivar[:, miny:maxy, minx:maxx], axis=0))

            sr = signal.ravel()
            nr = noise.ravel()
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

                self.binnum, xbin, ybin, xbar, ybar, SNR, self.npixels, scale = objective(options.ifu_options.targetsn, return_data=True)

            else:
                self.binnum, xbin, ybin, xbar, ybar, SNR, self.npixels, scale = voronoi_2d_binning(_x[good], _y[good], sr[good], nr[good],
                                                                                        options.ifu_options.targetsn, cvt=options.ifu_options.cvt, pixelsize=1, plot=options.ifu_options.voronoi_plot,
                                                                                        quiet=options.ifu_options.quiet, wvt=options.ifu_options.wvt)
                print('Voronoi binning successful with target S/N = {t}! Created {bins} bins.'.format(t=options.ifu_options.targetsn, bins=np.max(self.binnum)+1))

            if options.ifu_options.voronoi_plot:
                # For some reason voronoi makes the plot but doesnt save it or anything
                filename = self.infile.joinpath('voronoi_binning.pdf') # TODO: option for other out directory
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close()

            _x = _x[good]
            _y = _y[good]
            # Create output arrays for flux, ivar, mask
            out_flux = np.zeros((self.flux.shape[0], np.nanmax(self.binnum)+1))
            out_ivar = np.zeros((self.ivar.shape[0], np.nanmax(self.binnum)+1))
            out_mask = np.zeros((self.mask.shape[0], np.nanmax(self.binnum)+1))
            self.xpixbin = np.full(np.nanmax(self.binnum)+1, fill_value=np.nan, dtype=object)
            self.ypixbin = np.full(np.nanmax(self.binnum)+1, fill_value=np.nan, dtype=object)
            for j in range(self.xpixbin.size):
                self.xpixbin[j] = []
                self.ypixbin[j] = []
            # Average flux/ivar in each bin
            for i, bin in enumerate(self.binnum):
                # there is probably a better way to do this, but I'm lazy
                xi, yi = _x[i], _y[i]
                out_flux[:, bin] += self.flux[:, yi, xi]
                out_ivar[:, bin] += self.ivar[:, yi, xi]
                out_mask[:, bin] += self.mask[:, yi, xi]
                self.xpixbin[bin].append(xi)
                self.ypixbin[bin].append(yi)
            out_flux /= self.npixels
            out_ivar /= self.npixels
            irange = np.nanmax(self.binnum)+1

            for bin in self.binnum:
                if SNR[bin] < snr_threshold:
                    self.flux[:, np.asarray(self.ypixbin[bin]), np.asarray(self.xpixbin[bin])] = np.nan
                    self.ivar[:, np.asarray(self.ypixbin[bin]), np.asarray(self.xpixbin[bin])] = np.nan
                    self.mask[:, np.asarray(self.ypixbin[bin]), np.asarray(self.xpixbin[bin])] = 1

        else:
            self.xpixbin = None
            self.ypixbin = None
            out_flux = self.flux[:, miny:maxy, minx:maxx].reshape(self.nz, (maxx-minx)*(maxy-miny))
            out_ivar = self.ivar[:, miny:maxy, minx:maxx].reshape(self.nz, (maxx-minx)*(maxy-miny))
            out_mask = self.mask[:, miny:maxy, minx:maxx].reshape(self.nz, (maxx-minx)*(maxy-miny))
            self.binnum = np.zeros((maxx-minx)*(maxy-miny))
            self.npixels = np.ones((maxx-minx)*(maxy-miny)) * (maxx-minx)*(maxy-miny)
            irange = (maxx-minx)*(maxy-miny)

            signal = np.nanmean(self.flux, axis=0)
            noise = np.sqrt(1 / np.nanmean(self.ivar, axis=0))
            SNR = signal / noise

            self.flux[:, SNR < snr_threshold] = np.nan
            self.ivar[:, SNR < snr_threshold] = np.nan
            self.mask[:, SNR < snr_threshold] = 1

        for i in range(irange):
            # Unpack the spaxel
            galaxy_spaxel = out_flux[:,i]  # observed flux
            ivar_spaxel = out_ivar[:,i]    # 1-sigma spectral noise
            mask_spaxel = out_mask[:,i]    # bad pixels
            if options.ifu_options.voronoi_binning:
                xi = self.xpixbin[i]            # x and y pixel position
                yi = self.ypixbin[i]
                snr_thresh = SNR[i] >= snr_threshold  # make sure bin has an overall SNR greater than the threshold
            else:
                xi = [_x[i]]
                yi = [_y[i]]
                snr_thresh = SNR[_y[i], _x[i]] >= snr_threshold  # make sure spaxel has an SNR greater than the threshold
            binnum_i = 0 if not options.ifu_options.voronoi_binning else i   # Voronoi bin index that this pixel belongs to

            # Package into a FITS file -- but only if the SNR is high enough, otherwise throw out the data
            if not snr_thresh:
                continue

            primaryhdu = fits.PrimaryHDU()
            primaryhdu.header.append(("FORMAT", format.upper(), "Data format"), end=True)
            if type(self.dataid) is list:
                for j, did in enumerate(self.dataid):
                    primaryhdu.header.append((f'{format.upper()}ID{j}', did, f'{"MANGA" if format == "manga" else "MUSE"} ID number'), end=True)
            else:
                primaryhdu.header.append((f'{format.upper()}ID', self.dataid, f'{"MANGA" if format == "manga" else "MUSE"} ID number'), end=True)
            primaryhdu.header.append(('OBJNAME', self.objname, 'Object Name'), end=True)
            primaryhdu.header.append(('RA', self.ra, 'Right ascension'), end=True)
            primaryhdu.header.append(('DEC', self.dec, 'Declination'), end=True)
            primaryhdu.header.append(('BINNUM', binnum_i, 'bin index of the spaxel (Voronoi)'), end=True)
            primaryhdu.header.append(('NX', self.nx, 'x dimension of the full MANGA cube'), end=True)
            primaryhdu.header.append(('NY', self.ny, 'y dimension of the full MANGA cube'), end=True)
            coadd = fits.BinTableHDU.from_columns(fits.ColDefs([
                fits.Column(name='flux', array=galaxy_spaxel, format='E'),
                fits.Column(name='loglam', array=self.loglam, format='E'),
                fits.Column(name='ivar', array=ivar_spaxel, format='E'),
                fits.Column(name='and_mask', array=mask_spaxel, format='E'),
                fits.Column(name='fwhm_res', array=self.fwhm_res, format='E')
            ]))
            specobj = fits.BinTableHDU.from_columns(fits.ColDefs([
                fits.Column(name='z', array=np.array([self.z]), format='E'),
                # fits.Column(name='ebv', array=np.array([ebv]), format='E')
            ]))
            specobj.header.append(('PLUG_RA', self.ra, 'Right ascension'), end=True)
            specobj.header.append(('PLUG_DEC', self.dec, 'Declination'), end=True)
            binobj = fits.BinTableHDU.from_columns(fits.ColDefs([
                fits.Column(name='spaxelx', array=np.array(xi), format='E'),
                fits.Column(name='spaxely', array=np.array(yi), format='E')
            ]))

            out_hdu = fits.HDUList([primaryhdu, coadd, specobj, binobj])

            # Save output to sub-folder
            if options.ifu_options.voronoi_binning:
                tag = '_'.join(['spaxel', 'bin', str(binnum_i)])
            else:
                tag = '_'.join(['spaxel', str(xi[0]), str(yi[0])])

            outdir = self.infile.parent.joinpath(self.infile.stem, tag)
            outdir.mkdir(exist_ok=True, parents=True)
            outfile = outdir.joinpath(tag + '.fits')
            out_hdu.writeto(outfile, overwrite=True)


    # Validate all custom IFU readers included the expected values
    def validate_custom(self):
        for attr in ['infile', 'dataid', 'nx', 'ny', 'nz', 'ra', 'dec', 'wave', 'flux', 'specres']:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                print('Custom IFU reader did not provide {attr}'.format(attr))
                return False

        if (not hasattr(self, 'ivar')) or (self.ivar is None):
            print("WARNING: No ivar was input.  Defaulting to sqrt(flux).")
            self.ivar = np.sqrt(self.flux)

        if (not hasattr(self, 'mask')) or (self.mask is None):
            self.mask = np.zeros(self.flux.shape, dtype=int)

        tests = [
                    (self.wave.shape == (self.nz,), 'Wave array shape should be (nz,)'),
                    (self.flux.shape == (self.nz, self.ny, self.nx), 'Flux array shape should be (nz, ny, nx)'),
                    (self.ivar.shape == (self.nz, self.ny, self.nx), 'Ivar array shape should be (nz, ny, nx)'),
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


    def prepare_input(self, ba_ctx):
        return

