#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary

def f(x):
    return np.sin(3*x) + x

xtrain = np.linspace(0, 3, 50).reshape([-1, 1])
ytrain = f(xtrain) + 0.5*(np.random.randn(len(xtrain)).reshape([-1, 1]) - 0.5)

k = gpflow.kernels.SquaredExponential()
meanf = gpflow.mean_functions.Linear()
m = gpflow.models.GPR(data=(xtrain, ytrain), kernel=k, mean_function=meanf)
opt = gpflow.optimizers.Scipy()

def objective_closure():
    return - m.log_marginal_likelihood()

opt_logs = opt.minimize(objective_closure,
                        m.trainable_variables,
                        options=dict(maxiter=100))

print_summary(m)


xpl = np.linspace(-5, 10, 100).reshape(100, 1)
mean, var = m.predict_f(xpl)

plt.figure(figsize=(12, 6))
plt.plot(xtrain, ytrain, 'x')
plt.plot(xpl, mean, 'C0', lw=2)
plt.fill_between(xpl[:, 0],
                 mean[:, 0] - 1.96 * np.sqrt(var[:,0]),
                 mean[:, 0] + 1.96 * np.sqrt(var[:,0]),
                 color='C0', alpha=0.2)
