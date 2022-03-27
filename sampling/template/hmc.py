"""demonstrating some utilties in the starter code"""
import argparse
import os
from re import S

import jax.numpy as np
from jax import random

import jax
import matplotlib.image
import matplotlib.pyplot
import numpy
from tqdm import tqdm
import os

import NPEET.npeet.entropy_estimators
from utils.metrics import get_discretized_tv_for_image_density
from utils.density import continuous_energy_from_image, prepare_image, sample_from_image_density

def hamiltonian_leapfrog(x, v, f, epsilon):
    """Start with position, x, velocity, v, energy, f, step size, epsilon
    And integrate for one step with leapfrog."""
    g = jax.grad(f)
    vp = v - 0.5 * epsilon * g(x)  # half step in v
    xp = x + epsilon * vp  # full step in x
    vpp = vp - 0.5 * epsilon * g(xp)  # half step in v
    return xp, vpp

def hamiltonian_monte_carlo(x0, f, g, k, epsilon):
    """Run HMC for k steps, with step size epsilon"""
    v = numpy.random.randn(*x0.shape)  # Not the correct way to get randoms in JAX
    x = x0  # save the original state, in case we reject the update
    for i in range(k):
        v = v - 0.5 * epsilon * g(x)  # half step in v
        x = x + epsilon * v  # full step in x
        v = v - 0.5 * epsilon * g(x)  # half step in v  
        # more efficient to combine half-steps
    reject=0
    if numpy.random.random() > np.exp(f(x0) - f(x)):
        #print("Metropolis- Hastings REJECT", f(x0), f(x))
        reject=1
        x = x0
    return x,reject

if __name__ == "__main__":
    
    img_name = 'spiral' # {'labrador','spiral'}
    d = 2 
    num_samples = 10000
    k = 10
    epsilon = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="results")

    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    os.makedirs(f"{args.result_folder}", exist_ok=True)

    # load some image
    img = matplotlib.image.imread(f'./data/{img_name}.jpg')

    # plot and visualize
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    matplotlib.pyplot.show()

    # convert to energy function
    # first we get discrete energy and density values
    crop_ranges = (10, 710, 240, 940) if img_name=='labrador' else None
    density, energy = prepare_image(
        img, crop=crop_ranges, white_cutoff=225, gauss_sigma=3, background=0.01
    )

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(density)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/{img_name}_density.png")

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(energy)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/{img_name}_energy.png")

    # create energy fn and its grad
    x_max, y_max = density.shape
    xp = jax.numpy.arange(x_max)
    yp = jax.numpy.arange(y_max)
    zp = jax.numpy.array(density)

    # You may use fill value to enforce some boundary conditions or some other way to enforce boundary conditions
    energy_fn = lambda coord: continuous_energy_from_image(coord, xp, yp, zp, fill_value=0)
    energy_fn_grad = jax.grad(energy_fn)

    
    x = sample_from_image_density(1, density, key)[0]  # x, position
    trajectories = numpy.zeros((num_samples, d), dtype=x.dtype)
    num_reject = 0; real_i = 0
    with tqdm(range(num_samples),unit='iterations') as p_bar:
        for i in p_bar:
            reject = True
            while reject:
                real_i+=1
                x, reject = hamiltonian_monte_carlo(x0=x,f=energy_fn,g=energy_fn_grad,k=k,epsilon=epsilon)
                num_reject+=reject; trajectories[i]=x
                reject_rate = num_reject/(real_i+1)
                p_bar.set_postfix({'reject_rate':reject_rate})
    with open(f'samples/samples_{img_name}_niter_{num_samples}_k_{k}_eps_{epsilon}.npy','wb') as f:
        numpy.save(f,trajectories)
        
    # generate samples from true distribution
    key, subkey = jax.random.split(key)
    samples = sample_from_image_density(num_samples, density, subkey)
    key, subkey = jax.random.split(key)
    samples2 = sample_from_image_density(num_samples, density, subkey)

    '''
    # (scatter) plot the samples with image in background
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(numpy.array(samples)[:, 1], numpy.array(samples)[:, 0], s=0.5, alpha=0.5)
    ax.imshow(density, alpha=0.3)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/labrador_sampled.png")

    # generate another set of samples from true distribution, to demonstrate comparisons
    key, subkey = jax.random.split(key)
    second_samples = sample_from_image_density(num_samples, density, subkey)
    '''

    # We have samples from two distributions. We use NPEET package to compute kldiv directly from samples.
    # NPEET needs nxd tensors
    kldiv = NPEET.npeet.entropy_estimators.kldiv(samples, trajectories)
    kldiv2 = NPEET.npeet.entropy_estimators.kldiv(samples, samples2)
    print(f"KL divergence b/w true samples and generated is {kldiv}")
    print(f"KL divergence is b/w two true samples: {kldiv2}")

    # TV distance between discretized density
    # The discrete density bin from the image give us a natural scale for discretization.
    # We compute discrete density from sample at this scale and compute the TV distance between the two densities
    tv_dist = get_discretized_tv_for_image_density(
        numpy.asarray(density), numpy.asarray(samples), bin_size=[7, 7]
    )
    print(f"True Samples: TV distance is {tv_dist}")
    
    tv_dist = get_discretized_tv_for_image_density(
        numpy.asarray(density), numpy.asarray(trajectories), bin_size=[7, 7]
    )
    print(f"HMC Samples: TV distance is {tv_dist}")


