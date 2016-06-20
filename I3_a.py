#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Interrogacion 3
# part a

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.mlab import PCA

# Plot style
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('bmh')

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

# Plot log original data
plt.plot(time_data, np.log(flux_target), label='$\mathrm{Flux\,Target}$')
plt.plot(time_data, reduced_data[:,0], label='$\mathrm{Flux\,Comp}1$')
plt.plot(time_data, reduced_data[:,1], label='$\mathrm{Flux\,Comp}2$')
plt.plot(time_data, reduced_data[:,2], label='$\mathrm{Flux\,Comp}3$')
plt.plot(time_data, reduced_data[:,3], label='$\mathrm{Flux\,Comp}4$')
plt.plot(time_data, reduced_data[:,4], label='$\mathrm{Flux\,Comp}5$')
plt.plot(time_data, reduced_data[:,5], label='$\mathrm{Flux\,Comp}6$')
plt.plot(time_data, reduced_data[:,6], label='$\mathrm{Flux\,Comp}7$')
plt.plot(time_data, reduced_data[:,7], label='$\mathrm{Flux\,Comp}8$')
plt.plot(time_data, reduced_data[:,8], label='$\mathrm{Flux\,Comp}9$')
plt.xlabel('$\mathrm{Time\,from\,mid-transit\,[hour]}$')
plt.ylabel('$\log{\mathrm{Relative\,flux}}$')
plt.legend(loc='best', title='$\mathrm{Reduced\,data}$')
# plt.savefig('I3_pre.pdf')
plt.show()

# Calculate PCA with mlab module
results = PCA(reduced_data)

# All principal components plot
plt.plot(time_data, results.Y[:, 0] + 0.0)
plt.plot(time_data, results.Y[:, 1] + 0.1)
plt.plot(time_data, results.Y[:, 2] + 0.2)
plt.plot(time_data, results.Y[:, 3] + 0.3)
plt.plot(time_data, results.Y[:, 4] + 0.4)
plt.plot(time_data, results.Y[:, 5] + 0.5)
plt.plot(time_data, results.Y[:, 6] + 0.6)
plt.plot(time_data, results.Y[:, 7] + 0.7)
plt.plot(time_data, results.Y[:, 8] + 0.8)
plt.xlabel('$\mathrm{Time\,from\,mid-transit\,[hour]}$')
plt.ylabel('$\log{\mathrm{Relative\,flux}}$')
# plt.savefig('I3_a_all_pca.pdf')
plt.show()

# All pca but the first
# plt.plot(time_data, results.Y[:, 0] + 0.0)
plt.plot(time_data, results.Y[:, 1] + 0.1)
plt.plot(time_data, results.Y[:, 2] + 0.2)
plt.plot(time_data, results.Y[:, 3] + 0.3)
plt.plot(time_data, results.Y[:, 4] + 0.4)
plt.plot(time_data, results.Y[:, 5] + 0.5)
plt.plot(time_data, results.Y[:, 6] + 0.6)
plt.plot(time_data, results.Y[:, 7] + 0.7)
plt.plot(time_data, results.Y[:, 8] + 0.8)
plt.xlabel('$\mathrm{Time\,from\,mid-transit\,[hour]}$')
plt.ylabel('$\log{\mathrm{Relative\,flux}}$')
# plt.savefig('I3_a_pca_no1.pdf')
plt.show()

# Plot weights (fraction of total) of components
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(len(results.s)), results.fracs)
ax.set_yscale('log')
ax.set_xlabel('$\mathrm{PC}$')
ax.set_ylabel('$\mathrm{eigenvalue}$')
# plt.savefig('I3_a_pca_fraction.pdf')
plt.show()

# Plot cumulative component fractions
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(len(results.s)), np.cumsum(results.fracs))
# ax.set_yscale('log')
ax.set_xlabel('$\mathrm{PC}$')
ax.set_ylabel('$\mathrm{cumulative\,eigenvalue}$')
# plt.savefig('I3_a_pca_cumu.pdf')
plt.show()
