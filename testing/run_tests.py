import pathlib
import shutil
import unittest

import badass
import badass_tools.badass_ifu as ifu

TESTING_DIR = pathlib.Path(__file__).resolve().parent
BADASS_DIR = TESTING_DIR.parent
EXAMPLES_DIR =  BADASS_DIR.joinpath('examples')
OPTIONS_DIR = TESTING_DIR.joinpath('options')


class TestBADASS(unittest.TestCase):

    def test_single_examples(self):

        options_file = OPTIONS_DIR.joinpath('single_tests.py')

        single_tests = [
            EXAMPLES_DIR.joinpath('0-test', 'spec-1087-52930-0084.fits'),
            EXAMPLES_DIR.joinpath('1-test', 'spec-7748-58396-0782.fits'),
            EXAMPLES_DIR.joinpath('2-test', 'spec-2756-54508-0579.fits'),
            EXAMPLES_DIR.joinpath('3-test', 'spec-0266-51602-0151.fits'),
        ]

        for fits_file in single_tests:
            self.assertTrue(fits_file.exists())
            test_dir = fits_file.parent

            # Clean up old test output directories
            for d in test_dir.glob('MCMC_output_*'):
                shutil.rmtree(d)

        # Run tests all at once so BADASS will use multiprocessing
        ret = badass.BadassRunner(single_tests, options_file, infmt='sdss_spec').run()
        self.assertIsNone(ret) # BadassRunner.run() should return an error string on failure, None on success

        for fits_file in single_tests:
            test_dir = fits_file.parent
            output_dir = test_dir.joinpath('MCMC_output_1')
            self.assertTrue(output_dir.exists())

            for file in ['best_fit_model.pdf', 'fitting_region.pdf', 'max_likelihood_fit.pdf']:
                self.assertTrue(output_dir.joinpath(file).exists())

            log_dir = output_dir.joinpath('log')
            for file in ['log_file.txt', 'MCMC_chain.csv', 'best_model_components.fits', 'par_table.fits']:
                self.assertTrue(log_dir.joinpath(file).exists())


    def test_muse(self):

        self.skipTest('ignore')

        options_file = str(OPTIONS_DIR.joinpath('muse_test.py').relative_to(BADASS_DIR)).replace('/', '.')[:-3]

        z = 0.00379
        ap = [8, 9, 3, 4]

        cube = EXAMPLES_DIR.joinpath('MUSE', 'NGC1068_subcube.fits')
        self.assertTrue(cube.exists())

        cube_subdir = cube.with_suffix('')
        if cube_subdir.exists():
            shutil.rmtree(str(cube_subdir))

        wave, flux, ivar, mask, fwhm_res, binnum, npixels, xpixbin, ypixbin, z, dataid, objname = ifu.prepare_ifu(str(cube), z, format='muse', 
                                                                                                    aperture=ap, 
                                                                                                    targetsn=25.0 ,
                                                                                                    snr_threshold=0.,
                                                                                                    voronoi_binning=False,
                                                                                                    use_and_mask=False,
                                                                                                    )

        self.assertTrue(cube_subdir.exists())
        spaxel_dirs = list(cube_subdir.glob('spaxel*'))
        self.assertGreater(len(spaxel_dirs), 0)

        badass.run_BADASS(cube_subdir, nprocesses=4, options_file=options_file, sdss_spec=False, ifu_spec=True)

        for spaxel_dir in spaxel_dirs:
            output_dir = spaxel_dir.joinpath('MCMC_output_1')
            self.assertTrue(output_dir.exists())

            for file in ['fitting_region.pdf', 'max_likelihood_fit.pdf']:
                self.assertTrue(output_dir.joinpath(file).exists())

            log_dir = output_dir.joinpath('log')
            for file in ['best_model_components.fits', 'log_file.txt', 'par_table.fits']:
                self.assertTrue(log_dir.joinpath(file).exists())


if __name__ == '__main__':
    unittest.main()
