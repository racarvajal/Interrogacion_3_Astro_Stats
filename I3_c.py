#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Interrogacion 3
# parte c

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.mlab import PCA
import batman
import emcee
import corner
import george
from george import kernels

# Plot style
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('bmh')

# Function definitions for MCMC

# Log of Likelihood with white noise
def lnlike(theta, x, y):
	c, rp, a, inc, sigma = theta[0:5]
	alpha = theta[5:]
	params.rp = rp
	params.a = a
	params.inc = inc
	m = batman.TransitModel(params, x / 24.)
	pca_signals = np.sum([alpha[i] * results.Y[:, i] for i in range(np.shape(alpha)[0])], axis=0)
	model = c + np.log(m.light_curve(params) + 1e-8) + pca_signals
	inv_sigma2 = 1 / (sigma**2)
	return -0.5 * (np.sum((y - model)**2) / (sigma**2) - (n_data * 0.5) * np.log(2 * np.pi * sigma**2))

# Log of Likelihood with Gaussian Process
def lnlikeGP(theta, x, y):
	am, tau = np.exp(theta[:2])
	c, rp, a, inc, sigma = theta[2:7]
	alpha = theta[7:]
	gp = george.GP(am * kernels.Matern32Kernel(tau))
	gp.compute(x, sigma)
	params.rp = rp
	params.a = a
	params.inc = inc
	m = batman.TransitModel(params, x / 24.)
	pca_signals = np.sum([alpha[i] * results.Y[:, i] for i in range(np.shape(alpha)[0])], axis=0)
	model = c + np.log(m.light_curve(params) + 1e-8) + pca_signals
	inv_sigma2 = 1 / (sigma**2)
	# return -0.5 * (np.sum((y - model)**2) / (sigma**2) - (n_data * 0.5) * np.log(2 * np.pi * sigma**2))
	return gp.lnlikelihood(y - model_outGP(theta, x))

# Log of prior with white noise
def lnprior(theta):
	c, rp, a, inc, sigma = theta[0:5]
	alpha = theta[5:]
	alpha_cond = [0 < alpha[i] <= 1.5 * results.s[i] for i in range(np.shape(alpha)[0])]
	if -3. < c < 0. and 0. < rp < 2. and 0. < a < 100. and 0. < inc < 90. and 1e-5 < sigma < 1e-3 and all(alpha_cond):
		return 0.0
	return -np.inf

# Log of prior with Gaussian Process	
def lnpriorGP(theta):
	lnam, lntau = theta[:2]
	c, rp, a, inc, sigma = theta[2:7]
	alpha = theta[7:]
	alpha_cond = [0 < alpha[i] <= 1.5 * results.s[i] for i in range(np.shape(alpha)[0])]
	if -5. < lnam < 5. and -5. < lntau < 5. and -3. < c < 0. and 0. < rp < 2. and 0. < a < 100. and 0. < inc < 90. and 1e-5 < sigma < 1e-3 and all(alpha_cond):
		return 0.0
	return -np.inf

# Log of Posterior with white noise
def lnprob(theta, x, y):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y)

# Log of Posterior with Gaussian Process
def lnprobGP(theta, x, y):
	lp = lnpriorGP(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlikeGP(theta, x, y)

# Model evaluation. White noise
# logF(t) = c + log(T + 1e-8) + sum(alpha[i].dot(results.Wt)) + epsilon(t)
# 1e-8 to avoid negative values in logarithm
def model_out(theta, x_data):
	c, rp, a, inc, sigma = theta[0:5]
	alpha = theta[5:]
	params.rp = rp
	params.a = a
	params.inc = inc
	m = batman.TransitModel(params, x_data / 24.)
	pca_signals = np.sum([alpha[i] * results.Y[:, i] for i in range(np.shape(alpha)[0])], axis=0)
	model = c + np.log(m.light_curve(params) + 1e-8) + pca_signals
	return model

# Model evaluation. Gaussian Process (GP)
# logF(t) = c + log(T + 1e-8) + sum(alpha[i].dot(results.Wt)) + GP
# 1e-8 to avoid negative values in logarithm
def model_outGP(theta, x_data):
	_, _, c, rp, a, inc, sigma = theta[0:7]
	alpha = theta[7:]
	params.rp = rp
	params.a = a
	params.inc = inc
	m = batman.TransitModel(params, x_data / 24.)
	pca_signals = np.sum([alpha[i] * results.Y[:, i] for i in range(np.shape(alpha)[0])], axis=0)
	model = c + np.log(m.light_curve(params) + 1e-8) + pca_signals
	return model

# Import data from file. R Carvajal: dataset_1.dat
datos_file = 'datasets/dataset_1.dat'
datos = np.loadtxt(datos_file)

# Separate data in arrays
time_data   = datos[:, 0]
flux_target = datos[:, 1]
'''
flux_comp_1 = datos[:, 2]
flux_comp_2 = datos[:, 3]
flux_comp_3 = datos[:, 4]
flux_comp_4 = datos[:, 5]
flux_comp_5 = datos[:, 6]
flux_comp_6 = datos[:, 7]
flux_comp_7 = datos[:, 8]
flux_comp_8 = datos[:, 9]
flux_comp_9 = datos[:, 10]
'''
# One matrix with log of data without transit
reduced_data = np.log(datos[:, 2:])

# Log of flux with transit
flux_target_red = np.log(flux_target)

# Calculate PCA with mlab module
results = PCA(reduced_data)

# Initialize transit model and parameters
params           = batman.TransitParams()
params.t0        = 0.                #time of inferior conjunction
params.per       = 0.78884           #orbital period
# params.rp      = 0.1               #planet radius (in units of stellar radii). Free
# params.a       = 15.               #semi-major axis (in units of stellar radii). Free
# params.inc     = 87.               #orbital inclination (in degrees). Free
params.ecc       = 0.                #eccentricity
params.w         = 90.               #longitude of periastron (in degrees)
params.u         = [0.1, 0.3]        #limb darkening coefficients
params.limb_dark = "quadratic"       #limb darkening model

n_data = 100  # Number of points
labels = ['$\mathrm{amp}$', '$\\tau$', '$\mathrm{c}$', '$\mathrm{rp}$', '$\mathrm{a}$', '$\mathrm{inc}$', '$\sigma$', '$\\alpha_{0}$', '$\\alpha_{1}$', '$\\alpha_{2}$', '$\\alpha_{3}$', '$\\alpha_{4}$', '$\\alpha_{5}$', '$\\alpha_{6}$', '$\\alpha_{7}$', '$\\alpha_{8}$']

# for ndim == 8, 6 for normal model + 2 for GP:
for ndim in [(6 + 2)]:

	nwalkers = 200

	# Initial parameters for mcmc
	am_0      = 0.
	tau_0     = 0.
	c_0       = -2.
	rp_0      = 1.
	a_0       = 15.
	inc_0     = 87.
	sigma_0   = 1e-4
	alpha_0_0 = results.s[0]
	# alpha_1_0 = results.s[1]
	# alpha_2_0 = results.s[2]

	# Create array with initial values for parameters
	alpha_x_0 = np.array([results.s[i] for i in range(ndim - 5)])
	theta_0 = np.concatenate((np.array([am_0, tau_0, c_0, rp_0, a_0, inc_0, sigma_0]), [alpha_0_0]))
	
	# Create walkers around initial values
	pos = [theta_0 + 1e-4 * np.random.rand(ndim) for i in range(nwalkers)]

	# Initialize mcmc sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobGP, args=(time_data, flux_target_red))

	# Burn-in phase
	burnin_it = 1000  # Number of iterations
	pos, prob, state = sampler.run_mcmc(pos, burnin_it)
	sampler.reset()

	iterations = 3500
	sampler.run_mcmc(pos, iterations)

	# Obtain individual values and percentiles: mean +- 1 sigma
	samples = sampler.flatchain
	quantiles_plot = [0.16, 0.5, 0.84]
	
	# Generate corner plot with marginalized distributions
	'''
	fig = corner.corner(samples[:, 3], plot_contours=False, labels='rp', no_fill_contours=False, quantiles=quantiles_plot, show_titles=True)
	fig.savefig('I3_c_corner_rp_%i_pca_gp.pdf' % (ndim - 7))
	plt.show()
	'''
	percentiles = np.asarray([np.percentile(samples[:, ax], [16, 50, 84]) for ax in range(ndim)])
	
	# Plot values (with walkers) for every parameter
	'''
	for i in range(ndim):
		for walker in sampler.chain[:, :, i]:  # sampler.chain dims: (nwalkers, niter, ndim)
			plt.plot(walker, drawstyle='steps', marker=None, alpha=0.3)
			plt.xlabel('$\mathrm{Step\,number}$')
			plt.ylabel(labels[i])
			plt.axhline(percentiles[i, 0])  # -1 sigma
			plt.axhline(percentiles[i, 1])  # mean value
			plt.axhline(percentiles[i, 2])  # +1 sigma
		plt.show()
	'''

	# Plot original data and random sample from mcmc values
	for theta in samples[np.random.randint(len(samples), size=120)]:
		am, tau = np.exp(theta[:2])
		gp = george.GP(am * kernels.Matern32Kernel(tau))
		gp.compute(time_data, theta[6])
		mode = gp.sample_conditional(flux_target_red - model_outGP(theta, time_data), time_data) + model_out(theta[2:], time_data)
		# plt.plot(time_data, model_out(theta, time_data), color="k", alpha=0.1)
		plt.plot(time_data, mode, color='k', alpha=0.8)
	plt.plot(time_data, flux_target_red, color="r", lw=2, alpha=0.8, label='$\mathrm{Original}$')
	plt.xlabel('$\mathrm{Time\,from\,mid-transit\,[hour]}$')
	plt.ylabel('$\log{\mathrm{Relative\,flux}}$')
	plt.legend(loc='best')
	# plt.savefig('I3_c_samples_GP_%i_pca.pdf' % (ndim - 7))
	plt.show()

	# Plot original data and mean value from mcmc samples
	samples[:, 2] = np.exp(samples[:, 2])
	
	# Create array with percentiles for parameters
	theta_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	am_mcmc, tau_mcmc, c_mcmc, rp_mcmc, a_mcmc, inc_mcmc, sigma_mcmc = theta_mcmc[0:7][:]
	alpha_mcmc = theta_mcmc[7:][:]
	alpha_mcmc_mean = np.array([alpha_mcmc[i][0] for i in range(ndim - 7)])
	theta_mcmc_mean = np.array([am_mcmc[0], tau_mcmc[0], c_mcmc[0], rp_mcmc[0], a_mcmc[0], inc_mcmc[0], sigma_mcmc[0]])
	theta_mcmc_mean = np.append(theta_mcmc_mean, alpha_mcmc_mean)
	theta_mcmc_mean[0] = np.exp(theta_mcmc_mean[0])  # am
	theta_mcmc_mean[1] = np.exp(theta_mcmc_mean[1])  # tau
	gp = george.GP(theta_mcmc_mean[0] * kernels.Matern32Kernel(theta_mcmc_mean[1]))
	gp.compute(time_data, theta_mcmc_mean[6])
	mode = gp.sample_conditional(flux_target_red - model_outGP(theta_mcmc_mean, time_data), time_data) + model_out(theta_mcmc_mean[2:], time_data)
	plt.plot(time_data, flux_target_red, color="r", lw=2, alpha=0.8, label='$\mathrm{Original}$')
	# plt.plot(time_data, model_out(theta_mcmc_mean, time_data), color='g', lw=2, alpha=0.8)
	plt.plot(time_data, mode, color='g', lw=2, alpha=0.8, label='$\mathrm{mcmc\,mean\,GP}$')
	# plt.fill_between(time_data, model_out(theta_mcmc_mean, time_data) - sigma_mcmc[0], model_out(theta_mcmc_mean, time_data) + sigma_mcmc[0], alpha=0.6)
	plt.xlabel('$\mathrm{Time\,from\,mid-transit\,[hour]}$')
	plt.ylabel('$\log{\mathrm{Relative\,flux}}$')
	plt.legend(loc='best')
	# plt.savefig('I3_c_samples_mean_GP_%i_pca.pdf' % (ndim - 7))
	plt.show()

	# AIC_temp = -2 * lnlike(theta_mcmc_mean, time_data, model_out(theta_mcmc_mean, time_data)) + 2 * ndim + (2 * ndim * (ndim + 1)) / (n_data - ndim - 1)
	# AIC = np.append(AIC, AIC_temp)
	plt.close('all')


# Iterate over the number of PCA components included in model. White Noise
for ndim in [(6)]:
	nwalkers = 200  # Walkers for MCMC

	# Initial parameters for mcmc
	c_0       = -2.
	rp_0      = 1.
	a_0       = 15.
	inc_0     = 87.
	sigma_0   = 1e-4
	# alpha_0_0 = results.s[0]
	# alpha_1_0 = results.s[1]
	# alpha_2_0 = results.s[2]

	# Create array with initial values for parameters
	alpha_x_0 = np.array([results.s[i] for i in range(ndim - 5)])
	theta_0 = np.concatenate((np.array([c_0, rp_0, a_0, inc_0, sigma_0]), alpha_x_0))
	
	# Create walkers around initial values
	pos = [theta_0 + 1e-4 * np.random.rand(ndim) for i in range(nwalkers)]

	# Initialize mcmc sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time_data, flux_target_red))

	# Burn-in phase
	burnin_it = 1000  # Number of iterations
	pos, prob, state = sampler.run_mcmc(pos, burnin_it)
	sampler.reset()

	iterations = 3500
	sampler.run_mcmc(pos, iterations)

	# Obtain individual values and percentiles: mean +- 1 sigma
	samples = sampler.flatchain
	quantiles_plot = [0.16, 0.5, 0.84]
	
	# Generate corner plot with marginalized distributions
	'''
	fig = corner.corner(samples[:, 1], plot_contours=False, labels='rp', no_fill_contours=False, quantiles=quantiles_plot, show_titles=True)
	fig.savefig('I3_c_corner_rp_%i_pca_white.pdf' % (ndim - 5))
	plt.show()
	'''

	percentiles = np.asarray([np.percentile(samples[:, ax], [16, 50, 84]) for ax in range(ndim)])
	
	# Plot values (with walkers) for every parameter
	'''
	for i in range(ndim):
		for walker in sampler.chain[:, :, i]:  # sampler.chain dims: (nwalkers, niter, ndim)
			plt.plot(walker, drawstyle='steps', marker=None, alpha=0.3)
			plt.xlabel('$\mathrm{Step\,number}$')
			plt.ylabel(labels[i])
			plt.axhline(percentiles[i, 0])  # -1 sigma
			plt.axhline(percentiles[i, 1])  # mean value
			plt.axhline(percentiles[i, 2])  # +1 sigma
		plt.show()
	'''

	# Plot original data and random sample from mcmc values
	for theta in samples[np.random.randint(len(samples), size=120)]:
		plt.plot(time_data, model_out(theta, time_data), color="k", alpha=0.1)
	plt.plot(time_data, flux_target_red, color="r", lw=2, alpha=0.8, label='$\mathrm{Original}$')
	plt.xlabel('$\mathrm{Time\,from\,mid-transit\,[hour]}$')
	plt.ylabel('$\log{\mathrm{Relative\,flux}}$')
	plt.legend(loc='best')
	# plt.savefig('I3_c_samples_%i_pca.pdf' % (ndim - 5))
	plt.show()
	
	# Plot original data and mean value from mcmc samples
	samples[:, 2] = np.exp(samples[:, 2])
	
	# Create array with percentiles for parameters
	theta_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	c_mcmc, rp_mcmc, a_mcmc, inc_mcmc, sigma_mcmc = theta_mcmc[0:5][:]
	alpha_mcmc = theta_mcmc[5:][:]
	alpha_mcmc_mean = np.array([alpha_mcmc[i][0] for i in range(ndim - 5)])
	theta_mcmc_mean = np.array([c_mcmc[0], rp_mcmc[0], a_mcmc[0], inc_mcmc[0], sigma_mcmc[0]])
	theta_mcmc_mean = np.append(theta_mcmc_mean, alpha_mcmc_mean)
	plt.plot(time_data, flux_target_red, color="r", lw=2, alpha=0.8, label='$\mathrm{Original}$')
	plt.plot(time_data, model_out(theta_mcmc_mean, time_data), color='g', lw=2, alpha=0.8, label='$\mathrm{mcmc\,mean}$')
	plt.fill_between(time_data, model_out(theta_mcmc_mean, time_data) - sigma_mcmc[0], model_out(theta_mcmc_mean, time_data) + sigma_mcmc[0], alpha=0.6)
	plt.xlabel('$\mathrm{Time\,from\,mid-transit\,[hour]}$')
	plt.ylabel('$\log{\mathrm{Relative\,flux}}$')
	plt.legend(loc='best')
	# plt.savefig('I3_c_samples_mean_%i_pca.pdf' % (ndim - 5))
	plt.show()
	
	
	plt.close('all')


# plt.plot(range(6, 15), AIC)
# plt.show()

