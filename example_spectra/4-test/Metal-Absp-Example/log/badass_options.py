
# BADASS Options File
# Use the file as the input for the options_file in BADASS to re-run the fit with the same options.
# --------------------------------------------------------------------------------------------------
#
fit_options = {'fit_reg': (1000, 3500), 'good_thresh': 0.0, 'mask_bad_pix': False, 'mask_emline': False, 'mask_metal': True, 'fit_stat': 'RCHI2', 'n_basinhop': 25, 'test_lines': False, 'max_like_niter': 25, 'output_pars': False, 'cosmology': {'H0': 70.0, 'Om0': 0.3}}
#
mcmc_options = {'mcmc_fit': False, 'nwalkers': 100, 'auto_stop': False, 'conv_type': 'all', 'min_samp': 1000, 'ncor_times': 1.0, 'autocorr_tol': 10.0, 'write_iter': 100, 'write_thresh': 100, 'burn_in': 1500, 'min_iter': 2500, 'max_iter': 2500}
#
comp_options = {'fit_opt_feii': True, 'fit_uv_iron': True, 'fit_balmer': True, 'fit_losvd': False, 'fit_host': True, 'fit_power': True, 'fit_poly': True, 'fit_narrow': False, 'fit_broad': True, 'fit_absorp': False, 'tie_line_disp': False, 'tie_line_voff': False}
#
losvd_options = {'library': 'IndoUS', 'vel_const': {'bool': False, 'val': 0.0}, 'disp_const': {'bool': False, 'val': 250.0}}
#
host_options = {'age': [1.0, 5.0, 10.0], 'vel_const': {'bool': False, 'val': 0.0}, 'disp_const': {'bool': False, 'val': 150.0}}
#
power_options = {'type': 'simple'}
#
poly_options = {'apoly': {'bool': True, 'order': 13}, 'mpoly': {'bool': False, 'order': 3}}
#
opt_feii_options = {'opt_template': {'type': 'VC04'}, 'opt_amp_const': {'bool': False, 'br_opt_feii_val': 1.0, 'na_opt_feii_val': 1.0}, 'opt_disp_const': {'bool': False, 'br_opt_feii_val': 3000.0, 'na_opt_feii_val': 500.0}, 'opt_voff_const': {'bool': False, 'br_opt_feii_val': 0.0, 'na_opt_feii_val': 0.0}}
#
uv_iron_options = {'uv_amp_const': {'bool': False, 'uv_iron_val': 1.0}, 'uv_disp_const': {'bool': False, 'uv_iron_val': 3000.0}, 'uv_voff_const': {'bool': True, 'uv_iron_val': 0.0}}
#
balmer_options = {'R_const': {'bool': False, 'R_val': 1.0}, 'balmer_amp_const': {'bool': False, 'balmer_amp_val': 1.0}, 'balmer_disp_const': {'bool': True, 'balmer_disp_val': 5000.0}, 'balmer_voff_const': {'bool': True, 'balmer_voff_val': 0.0}, 'Teff_const': {'bool': True, 'Teff_val': 15000.0}, 'tau_const': {'bool': True, 'tau_val': 1.0}}
#
plot_options = {'plot_param_hist': False, 'plot_HTML': True, 'plot_pca': False, 'plot_corner': False, 'corner_options': {'pars': [], 'labels': []}}
#
output_options = {'write_chain': False, 'write_options': True, 'verbose': True}
#
pca_options = {'do_pca': False, 'n_components': 20, 'pca_masks': []}
#
user_mask = []
#
user_constraints = []
#
user_lines = {'BR_OI_1305': {'center': 1305.53, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'O I', 'line_profile': 'gaussian', 'ncomp': 1, 'center_pix': 166.96169105691047, 'disp_res_ang': 0.35018203670709797, 'disp_res_kms': 80.41326781603419}, 'BR_CII_1335': {'center': 1335.31, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'C II', 'line_profile': 'gaussian', 'ncomp': 1, 'center_pix': 264.91475357710635, 'disp_res_ang': 0.35292060891632915, 'disp_res_kms': 79.23473712162945}, 'BR_SIIV_1398': {'center': 1397.61, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'Si IV + O IV', 'line_profile': 'gaussian', 'ncomp': 1, 'center_pix': 462.9519211822657, 'disp_res_ang': 0.3589892412577814, 'disp_res_kms': 77.00450557181568}, 'BR_SIIV+OIV': {'center': 1399.8, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'Si IV + O IV', 'line_profile': 'gaussian', 'ncomp': 1, 'center_pix': 469.7521180030256, 'disp_res_ang': 0.3592105858056593, 'disp_res_kms': 76.9314362468199}, 'BR_CIV_1549': {'center': 1549.48, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'C IV', 'line_profile': 'gaussian', 'ncomp': 1, 'center_pix': 910.9534635149024, 'disp_res_ang': 0.37508937962843575, 'disp_res_kms': 72.5720674603763}, 'BR_HEII_1640': {'center': 1640.4, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'He II', 'line_profile': 'gaussian', 'ncomp': 1, 'center_pix': 1158.5903591070853, 'disp_res_ang': 0.3850604050586092, 'disp_res_kms': 70.37198568092909}, 'BR_CIII_1908': {'center': 1908.734, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'C III]', 'line_profile': 'gaussian', 'ncomp': 1, 'center_pix': 1816.5472547274749, 'disp_res_ang': 0.41147766271000386, 'disp_res_kms': 64.62812519498631}, 'BR_CII_2326': {'center': 2326.0, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_profile': 'gaussian', 'line_type': 'br', 'label': 'C II]', 'ncomp': 1, 'center_pix': 2675.1893248175184, 'disp_res_ang': 0.508842541244781, 'disp_res_kms': 65.58347213015446}, 'BR_FEIII_UV47': {'center': 2418.0, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_profile': 'gaussian', 'line_type': 'br', 'label': 'Fe III', 'ncomp': 1, 'center_pix': 2843.656277436348, 'disp_res_ang': 0.511914229555023, 'disp_res_kms': 63.468993037004374}, 'BR_MGII_2799': {'center': 2799.117, 'amp': 'free', 'disp': 'free', 'voff': 'free', 'line_type': 'br', 'label': 'Mg II', 'line_profile': 'gaussian', 'ncomp': 1, 'center_pix': 3479.302704588548, 'disp_res_ang': 0.516744822408248, 'disp_res_kms': 55.34466778935719}}
#
combined_lines = {}
#
# --------------------------------------------------------------------------------------------------
# End of BADASS Options File