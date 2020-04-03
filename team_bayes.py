# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import sys
import numpy as np
import scipy.stats as sps
from scipy.ndimage import gaussian_filter
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary, set_trainable
from netCDF4 import Dataset

sys.path.append(r'C:\Users\chral\Nextcloud\code\HIDA2020\Climate_Model_Data')
os.chdir(r'C:\Users\chral\Dropbox\ipp\HIDA2020')
from common import *

# %% [markdown]
# First we load the time-series data of the forcing data to see how they look like.

# %%
_, AOD = get_volcanic_data()
_, TSI = get_solar_data()

t = np.arange(len(AOD))

plt.figure()
plt.plot(t, AOD)
plt.xlabel('Time / years')
plt.title('Volcanic forcing')
plt.figure()
plt.plot(t, TSI)
plt.xlabel('Time / years')
plt.title('Solar forcing')

# %% [markdown]
# We notice a very different kind of time-series. While the volcanic forcing is mostly zero with distinct peaks, the solar data is strongly periodic on top of a slow trend. Let's have a look at the frequecy domain:

# %%
AOD_f = np.fft.rfft(normalize(AOD))
TSI_f = np.fft.rfft(normalize(TSI))
f = np.fft.fftfreq(len(TSI))[0:len(TSI_f)]

plt.figure()
plt.semilogy(f, np.abs(AOD_f))
plt.xlabel('f / years^-1')
plt.title('Volcanic forcing in frequency space')

plt.figure()
plt.semilogy(f, np.abs(TSI_f))
plt.xlabel('f / years^-1')
plt.title('Solar forcing in frequency space')


# %%



# %%
# Load temperature data
# The mean over time differs by less than 0.05% locally, so this is a good basis

_, lon, lat, T1 = get_geodata(1)
_, _, _, T2 = get_geodata(2)
T1_mean = np.mean(T1, 0)
T2_mean = np.mean(T2, 0)
T_mean = (T1_mean + T2_mean)/2.0

dT1 = T1-T1_mean
dT2 = T2-T2_mean

plt.figure()
plt.imshow(T1_mean, cmap='jet')
plt.colorbar()
plt.figure()
plt.imshow(T2_mean, cmap='jet')
plt.colorbar()
plt.figure()
plt.imshow(T1_mean-T_mean, cmap='jet')
plt.colorbar()
plt.figure()
plt.imshow(T2_mean-T_mean, cmap='jet')
plt.colorbar()


# %%



# %%
dT1glob = np.mean(np.mean(dT1, 2), 1)
dT2glob = np.mean(np.mean(dT2, 2), 1)
dTglob = (dT1glob + dT2glob)/2.0
var_dTglob = (dT1glob - dTglob)**2 + (dT2glob - dTglob)**2  # sample variance with n-1 = 1
err_dTglob = 1.96*np.sqrt(var_dTglob)

plt.figure()
plt.plot(t, dTglob)
plt.fill_between(t, dTglob - err_dTglob, dTglob + err_dTglob)
plt.plot(t, 4*AOD)
plt.xlabel('Time / years')
plt.title('Global temperature deviation vs volcanic forcing')
plt.legend([r'$\Delta T$', 'forcing'])

plt.figure()
plt.plot(t, dTglob)
plt.fill_between(t, dTglob - err_dTglob, dTglob + err_dTglob, alpha=0.5)
plt.plot(t, 4*AOD)
plt.xlim([500, 600])
plt.xlabel('Time / years')
plt.title('Global temperature deviation vs volcanic forcing')
plt.legend([r'$\Delta T$', 'forcing'])


# %%



# %%
AODmin = 1e-2*np.max(AOD)

con = filter_ts(AODmin)
x = np.sqrt(AOD[con])

plt.figure(figsize=(9, 6))
for lag in range(-2, 4):
    #yA = np.roll(dT1glob[AOD>AODmin], -lag)
    #yB = np.roll(dT2glob[AOD>AODmin], -lag)
    y = np.roll(dTglob[con], -lag)
    err_y = np.roll(err_dTglob[con], -lag)
    plt.subplot(2, 3, lag+3)
    plt.errorbar(x, y, yerr=err_y, fmt='.')
    plt.ylabel('Global temperature deviation')
    plt.xlabel('SQRT volcanic forcing')
    plt.title('{} years'.format(lag))
plt.tight_layout()

# %% [markdown]
# So the first question we are going to ask is:
# 
# # At year $t$, was there volcanic forcing in the previous year $t-1$?

# %%
x = np.sqrt(np.concatenate([AOD[AOD>AODmin],AOD[AOD>AODmin]]))
#X = sm.add_constant(x)
X = x
y = np.concatenate([np.roll(dT1glob[AOD>AODmin], -1), np.roll(dT2glob[AOD>AODmin], -1)])
#lam = -1 # Box Cox
#c = -1

#y := ((y-1)**lam-c)/lam

model = sm.OLS(y, X)
fit = model.fit()
print(fit.summary())

xpl = np.sqrt(AOD[AOD>AODmin])
Xpl = xpl
#Xpl = sm.add_constant(xpl)
ypred = fit.predict(Xpl)

plt.figure(figsize=(3,3))
plt.plot(x, y, '.')
plt.plot(xpl, ypred, '-')
plt.ylabel(r'$\Delta T$')
plt.xlabel(r'$\sqrt{\mathrm{ADO}}$')
plt.title('1 year')

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(t[AOD>AODmin], dTglob[AOD>AODmin], '.')
ax.plot(t[AOD>AODmin], ypred, 'o')
#plt.fill_between(t, dTglob - err_dTglob, dTglob + err_dTglob)

print(fit.cov_params())


# %%
y = np.sqrt(np.concatenate([AOD[AOD>AODmin],AOD[AOD>AODmin]]))
x = np.concatenate([np.roll(dT1glob[AOD>AODmin], -1), np.roll(dT2glob[AOD>AODmin], -1)])
X = x
#X = sm.add_constant(x)

model = sm.OLS(y, X)
fit = model.fit()
print(fit.summary())

xpl = np.roll(dTglob, -1)
Xpl = xpl
#Xpl = sm.add_constant(xpl)
ypred = fit.predict(Xpl)

pred = fit.get_prediction(Xpl)
ci = pred.conf_int()

plt.figure()
plt.plot(x, y, '.')
plt.plot(xpl, ypred, 'o')

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(t, AOD, '-')
ax.plot(t, ypred**2, '--', color='tab:red')
ax.fill_between(t, ci[:,0]**2, ci[:,1]**2, color='tab:red', alpha=0.8)
plt.legend(['data', 'prediction'])
plt.title(r'Linear predictor of volcanic forcing from $\Delta T$')
plt.ylabel('AOD')
plt.xlabel('Time / years')
#ax.plot(t,)
#plt.fill_between(t, dTglob - err_dTglob, dTglob + err_dTglob)





# %%
TSI_sm = gaussian_filter(TSI, 6)

plt.figure()
plt.plot(TSI)
plt.plot(TSI_sm)


# %%



# %%
dTmin = -0.4

y = TSI_sm
dT1glob_sm = gaussian_filter(dT1glob, 6)
dT2glob_sm = gaussian_filter(dT2glob, 6)
xA = dT1glob_sm
xB = dT2glob_sm

plt.figure()
plt.plot(xA, y, '.')
plt.plot(xB, y, '.')
plt.xlabel('Mean temperature')
plt.ylabel('Solar forcing')

x = np.concatenate([xA[dT1glob_sm > dTmin], xB[dT1glob_sm > dTmin]])
y = np.concatenate([y[dT1glob_sm > dTmin], y[dT1glob_sm > dTmin]])
X = sm.add_constant(x)

model = sm.OLS(y, X)
fit = model.fit()
print(fit.summary())

xpl = dT1glob_sm
Xpl = sm.add_constant(xpl)
ypred = fit.predict(Xpl)

plt.figure()
plt.plot(x, y, '.')
plt.plot(xpl, ypred, 'o')

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(t, TSI_sm, '-')
ax.plot(t, ypred, '--')


# %%



# %%

# datavar = np.mean(var_dTglob)

xtrain = t.reshape([-1,1]).astype(np.float64)
ytrain = dT1glob.reshape([-1,1]).astype(np.float64)

k = gpflow.kernels.SquaredExponential(variance=1e2*np.var(ytrain), lengthscales=50)
#meanf = gpflow.mean_functions.Constant()
m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, noise_variance=np.var(ytrain))#, mean_function=meanf)
opt = gpflow.optimizers.Scipy()

set_trainable(m.kernel.variance, False)
#set_trainable(m.likelihood.variance, False)

print_summary(m)

def objective_closure():
    return - m.log_marginal_likelihood()

opt_logs = opt.minimize(objective_closure,
                        m.trainable_variables,
                        options=dict(maxiter=100))

print_summary(m)

xpl = np.linspace(-20, 1020, 1041).reshape([-1, 1]).astype('float64')
mean, var = m.predict_f(xpl, full_cov=False)
mean0, var0 = m.predict_f(xtrain, full_cov=False)

plt.figure(figsize=(12, 6))
plt.plot(xtrain, ytrain, 'x')
plt.plot(xpl, mean, 'C0', lw=2)
plt.fill_between(xpl[:, 0],
                 mean[:, 0] - 1.96 * np.sqrt(var[:,0]),
                 mean[:, 0] + 1.96 * np.sqrt(var[:,0]),
                 color='C0', alpha=0.2)

plt.figure()
plt.plot(xtrain, ytrain-mean0)


# %%
# datavar = np.mean(var_dTglob)
t, lon, lat, T1 = get_geodata(1)
t = np.arange(len(AOD))
# t, AOD = get_volcanic_data()
xtrain = t.reshape([-1,1]).astype(np.float64)
ytrain = dT1glob.reshape([-1,1]).astype(np.float64)

k1 = gpflow.kernels.SquaredExponential(variance=1e2*np.var(ytrain), lengthscales=50)
k2 = gpflow.kernels.SquaredExponential(variance=1e2*np.var(ytrain), lengthscales=1)
k = k1 + k2
#meanf = gpflow.mean_functions.Constant()
m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, noise_variance=np.var(ytrain))#, mean_function=meanf)
opt = tf.optimizers.Adam(1.0)
set_trainable(m.likelihood.variance, False)
set_trainable(m.kernel.kernels[0].variance, False)
set_trainable(m.kernel.kernels[1].variance, False)
#set_trainable(m.likelihood.variance, False)
#print_summary(m)
def objective_closure():
    return - m.log_marginal_likelihood()
opt_logs = opt.minimize(objective_closure, m.trainable_variables)#, options=dict(maxiter=100))
iterations = 64
for i in range(iterations):
    opt.minimize(objective_closure, var_list= m.trainable_variables)
    likelihood = m.log_marginal_likelihood()
    tf.print(f"GPR with Adam: iteration {i + 1} likelihood {likelihood:.04f}")
print_summary(m)
xpl = np.linspace(-5, 1005, 1011).reshape([-1, 1]).astype('float64')
mean, var = m.predict_f(xpl, full_cov=False)
mean0, var0 = m.predict_f(xtrain, full_cov=False)
plt.figure(figsize=(12, 6))
plt.plot(xtrain, ytrain, 'x')
plt.plot(xpl, mean, 'C0', lw=2)
plt.plot(xtrain, mean0, 'C0', lw=2)
plt.fill_between(xpl[:, 0], mean[:, 0] - 1.96 * np.sqrt(var[:,0]), mean[:, 0] + 1.96 * np.sqrt(var[:,0]), color='C0', alpha=0.2)


# %%
# datavar = np.mean(var_dTglob)
t, lon, lat, T1 = get_geodata(1)
t = np.arange(len(AOD))
# t, AOD = get_volcanic_data()
xtrain = t.reshape([-1,1]).astype(np.float64)
ytrain = dT1glob.reshape([-1,1]).astype(np.float64)

k = gpflow.kernels.SquaredExponential(variance=1e2*np.var(ytrain), lengthscales=10 )
m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, noise_variance=np.var(ytrain))

opt = gpflow.optimizers.Scipy()
set_trainable(m.kernel.lengthscales, True)
set_trainable(m.kernel.variance, False)
set_trainable(m.likelihood.variance, True)

def objective_closure():
    return - m.log_marginal_likelihood()
opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))

print_summary(m)
pl = np.linspace(-20, 1020, 1041).reshape([-1, 1]).astype('float64')
mean, var = m.predict_f(xpl, full_cov=False)
mean0, var0 = m.predict_f(xtrain, full_cov=False)
plt.figure(figsize=(12, 6))
plt.plot(xtrain, ytrain, 'x')
plt.plot(xpl, mean, 'C0', lw=2)
plt.plot(xtrain, mean0, 'C0', lw=2)
plt.fill_between(xpl[:, 0], mean[:, 0] - 1.96 * np.sqrt(var[:,0]), mean[:, 0] + 1.96 * np.sqrt(var[:,0]), color='C0', alpha=0.2)

print(m.log_marginal_likelihood())


# %%

def volc_influence(volc_strength):
    #a0 = 0.102
    #a1 = -1.65
    a0 = 0.0
    a1 = -1.3294
    return a0 + a1*volc_strength

# %%
t = np.arange(len(AOD))
xtrain = t.reshape([-1,1]).astype(np.float64)
ytrain = (dT1glob - volc_influence(np.sqrt(AOD))).reshape([-1,1]).astype(np.float64)

k = gpflow.kernels.SquaredExponential(variance=1e2*np.var(ytrain), lengthscales=10 )
m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, noise_variance=np.var(ytrain))

opt = gpflow.optimizers.Scipy()
set_trainable(m.kernel.lengthscales, True)
set_trainable(m.kernel.variance, False)
set_trainable(m.likelihood.variance, True)

def objective_closure():
    return - m.log_marginal_likelihood()
opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))

print_summary(m)
pl = np.linspace(-20, 1020, 1041).reshape([-1, 1]).astype('float64')
mean, var = m.predict_f(xpl, full_cov=False)
mean0, var0 = m.predict_f(xtrain, full_cov=False)
plt.figure(figsize=(12, 6))
plt.plot(xtrain, ytrain, 'x')
plt.plot(xpl, mean, 'C0', lw=2)
plt.plot(xtrain, mean0, 'C0', lw=2)
plt.fill_between(xpl[:, 0], mean[:, 0] - 1.96 * np.sqrt(var[:,0]), mean[:, 0] + 1.96 * np.sqrt(var[:,0]), color='C0', alpha=0.2)

print(m.log_marginal_likelihood())

# %%
conla = filter_lat(35, 55)
conlo = filter_lon(150, 175)

LON, LAT = np.meshgrid(lon[conlo], lat[conla])

plt.figure()
plt.contourf(LON, LAT, dT1[0,conla,:][:,conlo], levels=50)

xtrain = np.vstack([LON.flatten(), LAT.flatten()]).astype(np.float64).T

print(xtrain.shape)

#%%

th = []
sig2 = []
for kt in range(0, 999, 5):
    ytrain = dT1[kt,conla,:][:,conlo].flatten().reshape([-1,1]).astype(np.float64)
    k = gpflow.kernels.SquaredExponential(variance=1e2*np.var(ytrain), lengthscales=5 )
    m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, noise_variance=np.var(ytrain))


    opt = gpflow.optimizers.Scipy()
    set_trainable(m.kernel.lengthscales, True)
    set_trainable(m.kernel.variance, False)
    set_trainable(m.likelihood.variance, True)

    def objective_closure():
        return - m.log_marginal_likelihood()
    opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))

    th.append(m.kernel.lengthscales.value())
    sig2.append(m.likelihood.variance.value())
    print(kt, th[-1])
# %%
plt.figure()
plt.plot(th)
plt.plot(AOD[::5]*40)

plt.figure()
plt.plot(sig2)
plt.plot(AOD[::5]/5)

#%%


# %%
nth = 10
nsig2 = 10

thmin = 3
thmax = 10
sig2min = 0.001
sig2max = 0.01

kt = 0

var0 = np.var(dT1[kt,conla,:][:,conlo].flatten())

k = gpflow.kernels.SquaredExponential(variance=1e2*var0, lengthscales=5 )

th = []
sig2 = []
for kt in range(0, 999, 2):
    ytrain = dT1[kt,conla,:][:,conlo].flatten().reshape([-1,1]).astype(np.float64)
    m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, noise_variance=var0)

    thmean = 0.0
    sig2mean = 0.0
    normmean = 0.0

    for thk in np.linspace(thmin, thmax, nth):
        for sig2k in np.linspace(sig2min, sig2max, nsig2):
            m.kernel.lengthscales.assign(thk)
            m.likelihood.variance.assign(sig2k)
            p = np.exp(m.log_marginal_likelihood())
            thmean += thk*p
            sig2mean += sig2k*p
            normmean += p

    thmean = thmean/normmean
    sig2mean = sig2mean/normmean

    th.append(thmean)
    sig2.append(sig2mean)
    print(kt, th[-1])



# %%
plt.figure()
plt.plot(th)

plt.figure()
plt.plot(sig2)


# %%
# %%
conla = filter_lat(35, 70)
klon = 80

plt.figure()
plt.plot(lat[conla], T1[0,conla,80])

xtrain = lat[conla].astype(np.float64).reshape([-1,1])

print(xtrain.shape)

#%%

th = []
sig2 = []
for kt in range(0, 999, 2):
    ytrain = dT1[kt,conla,klon].reshape([-1,1]).astype(np.float64)
    k = gpflow.kernels.SquaredExponential(variance=1e2*np.var(ytrain), lengthscales=10 )
    m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, noise_variance=np.var(ytrain))

    opt = gpflow.optimizers.Scipy()
    set_trainable(m.kernel.lengthscales, False)
    set_trainable(m.kernel.variance, False)
    set_trainable(m.likelihood.variance, True)

    def objective_closure():
        return -m.log_marginal_likelihood()
    opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))

    th.append(m.kernel.lengthscales.value())
    sig2.append(m.likelihood.variance.value())
    print(kt, th[-1])
# %%
# plt.figure()
# plt.plot(th)
# plt.plot(AOD[::2]*100)
# plt.ylim([0,60])

plt.figure()
plt.plot(sig2)
plt.plot(AOD[::2])

#%%


# %%
nth = 5
nsig2 = 5

thmin = 3
thmax = 10
sig2min = 0.001
sig2max = 0.01

kt = 0

var0 = np.var(dT1[kt,conla,:][:,conlo].flatten())

k = gpflow.kernels.SquaredExponential(variance=1e2*var0, lengthscales=5 )

th = []
sig2 = []
for kt in range(0, 999, 2):
    ytrain = dT1[kt,conla,klon].reshape([-1,1]).astype(np.float64)
    m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, noise_variance=var0)

    thmean = 0.0
    sig2mean = 0.0
    normmean = 0.0

    for thk in np.linspace(thmin, thmax, nth):
        for sig2k in np.linspace(sig2min, sig2max, nsig2):
            m.kernel.lengthscales.assign(thk)
            m.likelihood.variance.assign(sig2k)
            p = np.exp(m.log_marginal_likelihood())
            thmean += thk*p
            sig2mean += sig2k*p
            normmean += p

    thmean = thmean/normmean
    sig2mean = sig2mean/normmean

    th.append(thmean)
    sig2.append(sig2mean)
    print(kt, th[-1])



# %%
plt.figure()
plt.plot(th)

plt.figure()
plt.plot(sig2)


# %%
conla2 = filter_lat(30, 70)
conlo2 = filter_lon(140, 185)

dTvar = [np.var(dT1[kt,conla2,:][:,conlo2]) for kt in t]

# %%

plt.figure()
plt.plot(dTvar)
plt.plot(5*AOD)

# %%
