import numpy as np
import matplotlib.pyplot as plt
import gpflow
import gpflow.optimizers
from gpflow.utilities import print_summary
from glob import glob
import tensorflow as tf
import itertools
from gpflow.ci_utils import ci_niter
import random

files = glob('*.prn')

data = list()
for k in range(len(files)):
    f = files[k]
    data.append(np.genfromtxt(f))


kern = gpflow.kernels.SquaredExponential(1)

k = 1

x0 = 1.0*np.linspace(-1.0, 1.0, len(data[k])).reshape(-1, 1)
y0 = np.log(data[k][:,-1]).reshape(-1, 1)


x = x0[~np.isnan(y0)].reshape(-1, 1)
y = y0[~np.isnan(y0)].reshape(-1, 1)

N = len(x)
#reord = np.arange(N)
#random.shuffle(reord)
#x = x[reord].reshape(-1, 1)
#y = y[reord].reshape(-1,1)

plt.plot(x0,y0)

M = 50  # Number of inducing locations
Z = x[::M, :].copy()  # Initialise inducing locations to the first M inputs in the dataset

m = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), Z, num_data = N,)
m.kernel.lengthscale.assign(0.1)
m.kernel.variance.assign(10.0)
m.likelihood.variance.assign(0.05)
print_summary(m)
opt = gpflow.optimizers.Scipy()

minibatch_size = 100

train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) \
    .repeat() \
    .shuffle(N)

train_it = iter(train_dataset.batch(minibatch_size))

ground_truth = m.log_likelihood(x, y).numpy()
log_likelihood = tf.function(autograph=False)(m.log_likelihood)
evals = [log_likelihood(*minibatch).numpy()
for minibatch in itertools.islice(train_it, 100)]

def plot(title=''):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(-1, 1, 100)[:, None]  # Test locations
    pY, pYv = m.predict_y(pX)  # Predict Y values at test locations
    plt.plot(x, y, 'x', label='Training points', alpha=0.2)
    line, = plt.plot(pX, pY, lw=1.5, label='Mean of predictive posterior')
    col = line.get_color()
    plt.fill_between(pX[:, 0], (pY-2*pYv**0.5)[:, 0], (pY+2*pYv**0.5)[:, 0],
                     color=col, alpha=0.6, lw=1.5)
    Z = m.inducing_variable.Z.numpy()
    plt.plot(Z, np.zeros_like(Z), 'k|', mew=2, label='Inducing locations')
    plt.legend(loc='lower right')

plot(title="Predictions before training")


minibatch_size = 100

# We turn of training for inducing point locations
gpflow.utilities.set_trainable(m.inducing_variable, False)

@tf.function(autograph=False)
def optimization_step(optimizer, model: gpflow.models.SVGP, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = - model.elbo(*batch)
        grads = tape.gradient(objective, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return objective

def run_adam(model, iterations):
    """
    Utility function running the Adam optimiser

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimiser action
    logf = []
    train_it = iter(train_dataset.batch(minibatch_size))
    adam = tf.optimizers.Adam()
    for step in range(iterations):
        elbo = - optimization_step(adam, model, next(train_it))
        if step % 10 == 0:
            logf.append(elbo.numpy())
    return logf

maxiter = ci_niter(10000)

logf = run_adam(m, maxiter)
plt.figure()
plt.plot(np.arange(maxiter)[::10], logf)
plt.xlabel('iteration')
plt.ylabel('ELBO')

plot("Predictions after training")

print_summary(m)