import logging
import sys

# TODO: create error file with warning +
# 			check err_level option

class BadassLogger:
	def __init__(self, ba_ctx):
		self.ctx = ba_ctx # BadassContext

		self.log_dir = ba_ctx.outdir.joinpath('log')
		self.log_dir.mkdir(parents=True, exist_ok=True)

		# File for useful BADASS output
		self.log_file_path = self.log_dir.joinpath('log_file.txt')
		# File for all BADASS logging
		self.log_out_path = self.log_dir.joinpath('out_log.txt')

		log_lvl = logging.getLevelName(self.ctx.options.io_options.log_level.upper())
		log_lvl = log_lvl if isinstance(log_lvl, int) else logging.INFO

		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

		self.logger = logging.getLogger('BADASS_log')
		self.logger.setLevel(log_lvl) # TODO: have a separate log level for default to INFO
		fh = logging.FileHandler(self.log_file_path)
		self.logger.addHandler(fh)

		self.logout = logging.getLogger('BADASS_out')
		self.logout.setLevel(log_lvl)
		fh = logging.FileHandler(self.log_out_path)
		fh.setFormatter(formatter)
		self.logout.addHandler(fh)
		sh = logging.StreamHandler(sys.stdout)
		sh.setFormatter(formatter)
		self.logout.addHandler(sh)

		self.log_title()


	def debug(self, msg):
		self.logout.debug(msg)

	def info(self, msg):
		self.logout.info(msg)

	def warning(self, msg):
		self.logout.warning(msg)

	def error(self, msg):
		self.logout.error(msg)

	def critical(self, msg):
		self.logout.critical(msg)


	def log_title(self):
		# TODO: get version from central source
		self.logger.info('############################### BADASS v9.1.1 LOGFILE ####################################')


	def log_target_info(self):
		self.logger.info('\n-----------------------------------------------------------------------------------------------------------------\n')
		self.logger.info('{0:<30}{1:<30}'.format('file:', self.ctx.target.infile.name))
		self.logger.info('{0:<30}{1:<30}'.format('(RA, DEC):', '(%0.6f,%0.6f)' % (self.ctx.ra, self.ctx.dec)))
		self.logger.info('{0:<30}{1:<30}'.format('SDSS redshift:' , '%0.5f' % self.ctx.z))
		self.logger.info('{0:<30}{1:<30}'.format('fitting region:', '(%d,%d) [A]' % (self.ctx.fit_reg.min, self.ctx.fit_reg.max)))
		self.logger.info('{0:<30}{1:<30}'.format('velocity scale:', '%0.2f [km/s/pixel]' % self.ctx.velscale))
		self.logger.info('{0:<30}{1:<30}'.format('Galactic E(B-V):', '%0.3f' % self.ctx.ebv))
		self.logger.info('\n')
		self.logger.info('{0:<30}'.format('Units:'))
		self.logger.info('{0:<30}'.format('	- Note: SDSS Spectra are in units of [1.e-17 erg/s/cm2/Å]'))
		self.logger.info('{0:<30}'.format('	- Velocity, dispersion, and FWHM have units of [km/s]'))
		self.logger.info('{0:<30}'.format('	- Fluxes and Luminosities are in log-10'))
		self.logger.info('')
		self.logger.info('{0:<30}'.format('Cosmology:'))
		self.logger.info('{0:<30}'.format('	H0 = %0.1f' % self.ctx.options.fit_options.cosmology['H0']))
		self.logger.info('{0:<30}'.format('	Om0 = %0.2f' % self.ctx.options.fit_options.cosmology['Om0']))
		self.logger.info('\n')
		self.logger.info('-----------------------------------------------------------------------------------------------------------------')


	def log_fit_information(self):
		# TODO: does it make more sense to just pretty print the entire options dict to a file?
		# TODO: use options.<sub_option>.items() to just print all items?
		self.logger.info('\n### User-Input Fitting Paramters & Options ###')
		self.logger.info('-----------------------------------------------------------------------------------------------------------------')

		# General fit options
		fit_options = self.ctx.options.fit_options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('fit_options:','',''))
		for key in ['fit_reg', 'good_thresh', 'mask_bad_pix', 'n_basinhop', 'test_outflows', 'test_line', 'max_like_niter', 'output_pars']:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(fit_options[key])))
		self.logger.info('\n')

		# MCMC options
		mcmc_options = self.ctx.options.mcmc_options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('mcmc_options:','',''))
		if mcmc_options.mcmc_fit:
			for key in ['mcmc_fit', 'nwalkers', 'auto_stop', 'conv_type', 'min_samp', 'ncor_times', 'autocorr_tol', 'write_iter', 'write_thresh', 'burn_in', 'min_iter', 'max_iter']:
				self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(mcmc_options[key])))
		else:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','MCMC fitting is turned off.' ))
		self.logger.info('\n')

		# Fit Component options
		comp_options = self.ctx.options.comp_options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('comp_options:','',''))
		for key in ['fit_opt_feii', 'fit_uv_iron', 'fit_balmer', 'fit_losvd', 'fit_host', 'fit_power', 'fit_narrow', 'fit_broad', 'fit_outflow', 'fit_absorp', 'tie_line_fwhm', 'tie_line_voff', 'na_line_profile', 'br_line_profile', 'out_line_profile', 'abs_line_profile']:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(comp_options[key])))
		self.logger.info('\n')

		# LOSVD options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('losvd_options:','',''))
		if comp_options.fit_losvd:
			losvd_options = self.ctx.options.losvd_options
			for key in ['library', 'vel_const', 'disp_const']:
				self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(losvd_options[key])))
		else:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','LOSVD fitting is turned off.'))
		self.logger.info('\n')

		# Host Options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('host_options:','',''))
		if comp_options.fit_host:
			host_options = self.ctx.options.host_options
			for key in ['age', 'vel_const', 'disp_const']:
				self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':', str(host_options[key])))
		else:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','Host-galaxy template fitting is turned off.'))
		self.logger.info('\n')

		# Power-law continuum options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('power_options:','',''))
		if comp_options.fit_power:
			power_options = self.ctx.options.power_options
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format('type',':', str(power_options['type'])))
		else:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','Power Law fitting is turned off.'))
		self.logger.info('\n')

		# Optical FeII fitting options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('opt_feii_options:','',''))
		if comp_options.fit_opt_feii:
			opt_feii_options = self.ctx.options.opt_feii_options
			self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_template',':','type: %s' % str(opt_feii_options['opt_template']['type']) ))
			if opt_feii_options.opt_template.type == 'VC04':
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_amp_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_amp_const']['bool']),str(opt_feii_options['opt_amp_const']['br_opt_feii_val']),str(opt_feii_options['opt_amp_const']['na_opt_feii_val']))))
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_fwhm_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_fwhm_const']['bool']),str(opt_feii_options['opt_fwhm_const']['br_opt_feii_val']),str(opt_feii_options['opt_fwhm_const']['na_opt_feii_val']))))
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_voff_const',':','bool: %s, br_opt_feii_val: %s, na_opt_feii_val: %s' % (str(opt_feii_options['opt_voff_const']['bool']),str(opt_feii_options['opt_voff_const']['br_opt_feii_val']),str(opt_feii_options['opt_voff_const']['na_opt_feii_val']))))
			elif opt_feii_options.opt_template.type =='K10':
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_amp_const',':','bool: %s, f_feii_val: %s, s_feii_val: %s, g_feii_val: %s, z_feii_val: %s' % (str(opt_feii_options['opt_amp_const']['bool']),str(opt_feii_options['opt_amp_const']['f_feii_val']),str(opt_feii_options['opt_amp_const']['s_feii_val']),str(opt_feii_options['opt_amp_const']['g_feii_val']),str(opt_feii_options['opt_amp_const']['z_feii_val']))))
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_fwhm_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_fwhm_const']['bool']),str(opt_feii_options['opt_fwhm_const']['opt_feii_val']),)))
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_voff_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_voff_const']['bool']),str(opt_feii_options['opt_voff_const']['opt_feii_val']),)))
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format('opt_temp_const',':','bool: %s, opt_feii_val: %s' % (str(opt_feii_options['opt_temp_const']['bool']),str(opt_feii_options['opt_temp_const']['opt_feii_val']),)))
		else:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','Optical FeII fitting is turned off.'))
		self.logger.info('\n')

		# UV Iron options'
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('uv_iron_options:','',''))
		if comp_options.fit_uv_iron:
			uv_iron_options = self.ctx.options.uv_iron_options
			for key in ['uv_amp_const', 'uv_fwhm_const', 'uv_voff_const']:
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format(key,':','bool: %s, uv_iron_val: %s' % (str(uv_iron_options[key]['bool']), str(uv_iron_options[key]['uv_iron_val']))))
		else:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','UV Iron fitting is turned off.'))
		self.logger.info('\n')

		# Balmer options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('balmer_options:','',''))
		if comp_options.fit_balmer:
			for key in ['R', 'balmer_amp', 'balmer_fwhm', 'balmer_voff', 'Teff', 'tau']:
				self.logger.info('{0:>30}{1:<2}{2:<100}'.format('%s_const'%key,':','bool: %s, %s_val: %s' % (str(balmer_options['%s_const'%key]['bool']), key, str(balmer_options['%s_const'%key]['%s_val'%key]))))
		else:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format('','','Balmer pseudo-continuum fitting is turned off.' )) 
		self.logger.info('\n')

		 # Plotting options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('plot_options:','',''))
		plot_options = self.ctx.options.plot_options
		for key in ['plot_param_hist', 'plot_flux_hist', 'plot_lum_hist', 'plot_eqwidth_hist']:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':',str(plot_options[key])))
		self.logger.info('\n')

		# Output options
		self.logger.info('\t{0:<30}{1:<30}{2:<30}'.format('output_options:','',''))
		output_options = self.ctx.options.output_options
		for key in ['write_chain', 'verbose']:
			self.logger.info('{0:>30}{1:<2}{2:<30}'.format(key,':',str(output_options[key]) )) 

		# TODO: other options?

		self.logger.info('\n')
		self.logger.info('-----------------------------------------------------------------------------------------------------------------')


	def update_opt_feii(self):
		self.logger.info('\t* optical FeII templates outside of fitting region and disabled.')




