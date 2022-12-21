import pathlib
import shutil
import sys
import unittest

TESTING_DIR = pathlib.Path(__file__).resolve().parent
BADASS_DIR = TESTING_DIR.parent
EXAMPLES_DIR =  BADASS_DIR.joinpath('examples')
OPTIONS_DIR = TESTING_DIR.joinpath('options')

sys.path.insert(0, str(BADASS_DIR))

import badass


def get_log_line_values(log_file):
    lines = [line.strip() for line in open(log_file, 'r').readlines()]

    START = 0
    LINE_LIST = 1
    LINE = 2
    state = START

    target_props = ['center_pix','disp_res_ang','disp_res_kms',]
    line_dict = {} 

    cur_line = ''
    for line in lines:

        if 'Line List' in line:
            state = LINE_LIST
            continue

        if state == LINE_LIST:
            if '------' in line:
                continue
            cur_line = line
            line_dict[cur_line] = {}
            state = LINE
            continue

        if state == LINE:
            if '------' in line:
                break

            if len(line.split()) == 1:
                cur_line = line
                line_dict[cur_line] = {}
                continue

            key, value = line.split()[:2]
            if not key in target_props:
                continue

            line_dict[cur_line][key] = float(value)

    return line_dict



class TestBADASS(unittest.TestCase):

    def test_single_examples(self):

        # self.skipTest('ignore')
        options_file = OPTIONS_DIR.joinpath('single_tests2.py')

        single_tests = [
            EXAMPLES_DIR.joinpath('0-test', 'spec-1087-52930-0084.fits'),
            EXAMPLES_DIR.joinpath('1-test', 'spec-7748-58396-0782.fits'),
            EXAMPLES_DIR.joinpath('2-test', 'spec-2756-54508-0579.fits'),
            EXAMPLES_DIR.joinpath('3-test', 'spec-0266-51602-0151.fits'),
            EXAMPLES_DIR.joinpath('4-test', 'spec-7721-57360-0412.fits'),
            EXAMPLES_DIR.joinpath('5-test', 'spec-0276-51909-0251.fits'),
        ]

        for fits_file in single_tests:
            self.assertTrue(fits_file.exists(), msg='%s does not exist' % str(fits_file))
            test_dir = fits_file.parent

            # Clean up old test output directories
            for d in test_dir.glob('MCMC_output*'):
                shutil.rmtree(d)

        # Run tests all at once so BADASS will use multiprocessing
        ret = badass.BadassRunner(single_tests, options_file).run()
        self.assertIsNone(ret, msg='BadassRunner.run failed') # BadassRunner.run() should return an error string on failure, None on success

        for fits_file in single_tests:
            test_dir = fits_file.parent
            output_dir = test_dir.joinpath('MCMC_output')
            log_dir = output_dir.joinpath('log')
            self.assertTrue(output_dir.exists(), msg='%s does not exist' % str(output_dir))

            # Exception: bad wavelength range
            if test_dir.name == '4-test':
                self.assertTrue(log_dir.joinpath('log_file.txt').exists(), msg='%s does not exist' % str(log_dir.joinpath('log_file.txt')))
                continue

            # TODO: add fitting_region.pdf back in after implementing plotting
            # for file in ['best_fit_model.pdf', 'fitting_region.pdf', 'max_likelihood_fit.pdf']:
            for file in ['max_likelihood_fit.pdf']:
                self.assertTrue(output_dir.joinpath(file).exists(), msg='%s does not exist' % str(output_dir.joinpath(file)))

            # for file in ['log_file.txt', 'MCMC_chain.csv', 'best_model_components.fits', 'par_table.fits']:
            for file in ['log_file.txt', 'best_model_components.fits', 'par_table.fits']:
                self.assertTrue(log_dir.joinpath(file).exists(), msg='%s does not exist' % str(log_dir.joinpath(file)))

            bm_log_file = fits_file.parent.joinpath('benchmark', 'log', 'log_file.txt')
            # self.assertTrue(bm_log_file.exists())
            if not bm_log_file.exists():
                print('Benchmark does not exist for %s' % str(fits_file))
                continue
            bm_line_values = get_log_line_values(bm_log_file)

            log_file = log_dir.joinpath('log_file.txt')
            self.assertTrue(log_file.exists(), msg='%s does not exist' % str(log_file))
            line_values = get_log_line_values(log_file)

            for line, line_dict in bm_line_values.items():
                for attr, val in line_dict.items():
                    self.assertEqual(int(val), int(line_values[line][attr]), msg='%s != %s for %s' % (int(val), int(line_values[line][attr]), attr))



    def test_muse(self):

        self.skipTest('ignore')
        options_file = OPTIONS_DIR.joinpath('muse_test.py')

        cube = EXAMPLES_DIR.joinpath('MUSE', 'NGC1068_subcube.fits')
        self.assertTrue(cube.exists())

        cube_subdir = cube.with_suffix('')
        if cube_subdir.exists():
            shutil.rmtree(str(cube_subdir))

        # Clean up old test output directories
        for d in cube.parent.glob('MCMC_output*'):
            shutil.rmtree(d)

        ret = badass.BadassRunner(cube, options_file).run()

        spaxel_dirs = list(cube_subdir.glob('spaxel*'))
        self.assertGreater(len(spaxel_dirs), 0)

        for spaxel_dir in spaxel_dirs:
            output_dir = spaxel_dir.joinpath('MCMC_output')
            self.assertTrue(output_dir.exists())

            # TODO: add fitting_region.pdf back in after implementing plotting
            # for file in ['fitting_region.pdf', 'max_likelihood_fit.pdf']:
            for file in ['max_likelihood_fit.pdf']:
                self.assertTrue(output_dir.joinpath(file).exists())

            log_dir = output_dir.joinpath('log')
            for file in ['best_model_components.fits', 'log_file.txt', 'par_table.fits']:
                self.assertTrue(log_dir.joinpath(file).exists())


    # TODO: Add manga test

    # TODO: Add miri test

    # TODO: Add nirspec test

    # TODO: Add user spec test


    # TODO: Option tests:
    #   fit_reg: 'auto' and 'full'




if __name__ == '__main__':
    unittest.main()
    # TestBADASS().test_single_examples()
