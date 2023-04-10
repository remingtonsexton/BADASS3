
# BADASS Options File
# Use the file as the input for the options_file in BADASS to re-run the fit with the same options.
# --------------------------------------------------------------------------------------------------
#
fit_options = {'fit_reg': (4400, 5500), 'good_thresh': 0.0, 'mask_bad_pix': False, 'mask_emline': False, 'mask_metal': False, 'fit_stat': 'RCHI2', 'n_basinhop': 15, 'test_lines': False, 'max_like_niter': 25, 'output_pars': False, 'cosmology': {'H0': 70.0, 'Om0': 0.3}}
#
mcmc_options = {'mcmc_fit': True, 'nwalkers': 100, 'auto_stop': False, 'conv_type': 'all', 'min_samp': 1000, 'ncor_times': 10.0, 'autocorr_tol': 10.0, 'write_iter': 100, 'write_thresh': 100, 'burn_in': 1500, 'min_iter': 1000, 'max_iter': 2500}
#
comp_options = {'fit_opt_feii': True, 'fit_uv_iron': False, 'fit_balmer': False, 'fit_losvd': False, 'fit_host': True, 'fit_power': True, 'fit_poly': True, 'fit_narrow': True, 'fit_broad': True, 'fit_absorp': False, 'tie_line_disp': False, 'tie_line_voff': False}
#
losvd_options = {'library': 'IndoUS', 'vel_const': {'bool': False, 'val': 0.0}, 'disp_const': {'bool': False, 'val': 250.0}}
#
host_options = {'age': [1.0, 5.0, 10.0], 'vel_const': {'bool': False, 'val': 0.0}, 'disp_const': {'bool': False, 'val': 150.0}}
#
power_options = {'type': 'simple'}
#
poly_options = {'apoly': {'bool': True, 'order': 3}, 'mpoly': {'bool': False, 'order': 3}}
#
opt_feii_options = {'opt_template': {'type': 'VC04'}, 'opt_amp_const': {'bool': False, 'br_opt_feii_val': 1.0, 'na_opt_feii_val': 1.0}, 'opt_disp_const': {'bool': False, 'br_opt_feii_val': 3000.0, 'na_opt_feii_val': 500.0}, 'opt_voff_const': {'bool': False, 'br_opt_feii_val': 0.0, 'na_opt_feii_val': 0.0}}
#
uv_iron_options = {'uv_amp_const': {'bool': False, 'uv_iron_val': 1.0}, 'uv_disp_const': {'bool': False, 'uv_iron_val': 3000.0}, 'uv_voff_const': {'bool': True, 'uv_iron_val': 0.0}}
#
balmer_options = {'R_const': {'bool': True, 'R_val': 1.0}, 'balmer_amp_const': {'bool': False, 'balmer_amp_val': 1.0}, 'balmer_disp_const': {'bool': True, 'balmer_disp_val': 5000.0}, 'balmer_voff_const': {'bool': True, 'balmer_voff_val': 0.0}, 'Teff_const': {'bool': True, 'Teff_val': 15000.0}, 'tau_const': {'bool': True, 'tau_val': 1.0}}
#
plot_options = {'plot_param_hist': True, 'plot_HTML': True, 'plot_pca': False, 'plot_corner': False, 'corner_options': {'pars': [], 'labels': []}}
#
output_options = {'write_chain': False, 'write_options': True, 'verbose': True}
#
pca_options = {'do_pca': False, 'n_components': 20, 'pca_masks': []}
#
user_mask = []
#
user_constraints = [('NA_OIII_5007_AMP', 'NA_OIII_5007_2_AMP'), ('NA_OIII_5007_2_DISP', 'NA_OIII_5007_DISP')]
#
user_lines = {'NA_H_BETA': {'center': 4862.691, 'amp': 'free', 'disp': 'NA_OIII_5007_DISP', 'voff': 'free', 'line_type': 'na', 'label': 'H$\\beta$', 'ncomp': 1, 'line_profile': 'gaussian', 'center_pix': 435.21099039301293, 'disp_res_ang': 0.7266926172967304, 'disp_res_kms': 44.80172932021387}, 'NA_H_BETA_2': {'center': 4862.691, 'amp': 'NA_H_BETA_AMP*NA_OIII_5007_2_AMP/NA_OIII_5007_AMP', 'disp': 'NA_OIII_5007_2_DISP', 'voff': 'NA_OIII_5007_2_VOFF', 'line_type': 'na', 'ncomp': 2, 'parent': 'NA_H_BETA', 'line_profile': 'gaussian', 'center_pix': 435.21099039301293, 'disp_res_ang': 0.7266926172967304, 'disp_res_kms': 44.80172932021387}, 'NA_OIII_4960': {'center': 4960.295, 'amp': '(NA_OIII_5007_AMP/2.98)', 'disp': 'NA_OIII_5007_DISP', 'voff': 'NA_OIII_5007_VOFF', 'line_type': 'na', 'label': '[O III]', 'ncomp': 1, 'line_profile': 'gaussian', 'center_pix': 521.5197602739727, 'disp_res_ang': 0.7349264587941482, 'disp_res_kms': 44.41780368529158}, 'NA_OIII_4960_2': {'center': 4960.295, 'amp': '(NA_OIII_5007_2_AMP/2.98)', 'disp': 'NA_OIII_5007_2_DISP', 'voff': 'NA_OIII_5007_2_VOFF', 'line_type': 'na', 'ncomp': 2, 'parent': 'NA_OIII_4960', 'line_profile': 'gaussian', 'center_pix': 521.5197602739727, 'disp_res_ang': 0.7349264587941482, 'disp_res_kms': 44.41780368529158}, 'NA_OIII_5007': {'center': 5008.24, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'label': '[O III]', 'ncomp': 1, 'line_profile': 'gaussian', 'center_pix': 563.2959830508472, 'disp_res_ang': 0.7407543175269816, 'disp_res_kms': 44.341436837197556}, 'NA_OIII_5007_2': {'center': 5008.24, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'na', 'ncomp': 2, 'parent': 'NA_OIII_5007', 'line_profile': 'gaussian', 'center_pix': 563.2959830508472, 'disp_res_ang': 0.7407543175269816, 'disp_res_kms': 44.341436837197556}, 'BR_H_BETA': {'center': 4862.691, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'ncomp': 1, 'line_profile': 'gauss-hermite', 'h3': 'free', 'h4': 'free', 'center_pix': 435.21099039301293, 'disp_res_ang': 0.7266926172967304, 'disp_res_kms': 44.80172932021387}}
#
combined_lines = {'H_BETA_COMP': ['NA_H_BETA', 'BR_H_BETA']}
#
# --------------------------------------------------------------------------------------------------
# End of BADASS Options File