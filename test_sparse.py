import numpy as np
import matplotlib.pyplot as plt
import gpflow
import gpflow.optimizers
from gpflow.utilities import print_summary
from glob import glob

files = glob('*.prn')

data = list()
for k in range(len(files)):
    f = files[k]
    data.append(np.genfromtxt(f))


kern = gpflow.kernels.SquaredExponential(1)

k = 3
#x0 = 1.0*np.arange(12,12+len(data[k][12::24,-1])).reshape(-1, 1)
#y0 = np.log(data[k][12::24,-1]).reshape(-1, 1)


x0 = 1.0*np.arange(len(data[k])).reshape(-1, 1)
y0 = np.log(data[k][:,-1]).reshape(-1, 1)

x = x0[~np.isnan(y0)].reshape(-1, 1)
y = y0[~np.isnan(y0)].reshape(-1, 1)
z = x[12::24,:]

plt.plot(x0,y0)

m = gpflow.models.SGPR((x, y), kern, inducing_variable=z)
m.kernel.lengthscale.assign(100.0)
m.kernel.variance.assign(10.0)
gpflow.utilities.set_trainable(m.inducing_variable, False)
print_summary(m)
opt = gpflow.optimizers.Scipy()

def objective_closure():
    return - m.log_marginal_likelihood()

opt_logs = opt.minimize(objective_closure,
                        m.trainable_variables,
                        options=dict(maxiter=100))
print_summary(m)

xh = np.linspace(x0[0], x0[-1], len(data[k])).reshape(-1,1)
[ytest, vartest] = m.predict_f(xh)
plt.plot(xh, ytest, 'k')
plt.fill_between(xh.flatten(), 
    np.array(ytest - 2*np.sqrt(vartest)).flatten(),
    np.array(ytest + 2*np.sqrt(vartest)).flatten(),
    alpha=0.5, color='k')
plt.ylim([0.0, np.max(y)*1.2])

plt.figure()
plt.plot(np.log(data[k][:,-1]))

plt.figure()
plt.plot(np.log(data[k][:,-1]) - ytest.numpy().flatten())

print(np.sqrt(m.likelihood.variance.numpy()))

plt.figure()
plt.hist(np.log(data[k][:,-1]) - ytest.numpy().flatten(), bins=40)
