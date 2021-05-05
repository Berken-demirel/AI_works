#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HW2 Question 1

UC Irvine CS274B
"""
from factor_utils import *
from ldpc_utils import *
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

max_iters = 50
conv_tol = 1e-6

# QUESTION 1b #######################
print('Question 1b')

ldpc = sio.loadmat('ldpc36-128.mat')
G = ldpc['G']
H = ldpc['H']
num_bits = np.size(G,0)
noise = 0.04
codeWord = np.zeros([num_bits, 1])

# generate data
noisyCodeWord = channel_noise(codeWord, noise)
F = init_ldpc_graph( H, noisyCodeWord, 1-noise )
actualError = np.sum(noisyCodeWord != codeWord) / len(noisyCodeWord)

# Run Parallel Loopy BP

nodeMarg_par = run_loopy_bp_parallel(F, max_iters, conv_tol)
iters = len(nodeMarg_par)
if iters < max_iters:
    print('Parallel LBP converged in', iters, 'iterations.')
else:
    print('Parallel LBP did not converge. Terminated at', max_iters, 'iterations')

p = [nodeMarg_par[-1][key][1] for key in nodeMarg_par[-1].keys()] # probabilities of xi = 1

plt.figure(1)
plt.plot(range(num_bits), p)
plt.title('Bitwise Marginal Probability p(x_i=1)')
plt.xlabel('Bit #')
plt.ylabel('p(x_i=1)')
plt.show()


## QUESTION 1c #######################
print('Question 1c')
num_runs = 10
noise = 0.06
codeWord = np.zeros([num_bits, 1])

# err_mat[i,j]: stores the hamming distance for run i at iteration j
err_mat = np.zeros([num_runs, max_iters])

longest = 0
for m in range(num_runs):
    print('## run ', m + 1, '##')
    # generate data
    noisyCodeWord = channel_noise(codeWord, noise)
    F = init_ldpc_graph( H, noisyCodeWord, 1-noise )
    actualError = np.sum(noisyCodeWord != codeWord) / len(noisyCodeWord)
    print('actual error: ' , actualError)

    # Run Parallel Loopy BP
    nodeMarg_par = run_loopy_bp_parallel(F, max_iters, conv_tol)
    iters = len(nodeMarg_par)
    if iters < max_iters:
        print('Parallel LBP converged in', iters, 'iterations.')
    else:
        print('Parallel LBP did not converge. Terminated at', max_iters, 'iterations')

    # determine code
    decoded = estimate_code( nodeMarg_par[-1] )
    ham_dist = np.sum( decoded != codeWord )
    print('Hamming Distance: ', ham_dist)
    print('Error after correction ' , ham_dist/len(noisyCodeWord))

    # store error at each iteration of this run
    for it in range(iters):
        decoded = estimate_code(nodeMarg_par[it])
        err_mat[m,it] = np.sum(decoded != codeWord)
    if iters < max_iters:
        err_mat[m, iters:] = err_mat[m,iters-1]

    # longest run (for plotting)
    if longest < iters:
        longest = iters
    # plot
    plt.figure(2)
    for m in range(num_runs):
        plt.plot(range(max_iters), err_mat[m])
    plt.xlabel('Iteration #')
    plt.ylabel('Hamming Distance')
    plt.title('Channel Noise '+str(noise))

## QUESTION 1d #######################
print('Question 1d')
num_runs = 10
noise = 0.08
codeWord = np.zeros([num_bits, 1])

# err_mat[i,j]: stores the hamming distance for run i at iteration j
err_mat = np.zeros([num_runs, max_iters])

longest = 0
for m in range(num_runs):
    print('## run ', m + 1, '##')
    # generate data
    noisyCodeWord = channel_noise(codeWord, noise)
    F = init_ldpc_graph(H, noisyCodeWord, 1 - noise)
    actualError = np.sum(noisyCodeWord != codeWord) / len(noisyCodeWord)
    print('actual error: ', actualError)

    # Run Parallel Loopy BP
    nodeMarg_par = run_loopy_bp_parallel(F, max_iters, conv_tol)
    iters = len(nodeMarg_par)
    if iters < max_iters:
        print('Parallel LBP converged in', iters, 'iterations.')
    else:
        print('Parallel LBP did not converge. Terminated at', max_iters, 'iterations')

    # determine code
    decoded = estimate_code(nodeMarg_par[-1])
    ham_dist = np.sum(decoded != codeWord)
    print('Hamming Distance: ', ham_dist)
    print('Error after correction ', ham_dist / len(noisyCodeWord))

    # store error at each iteration of this run
    for it in range(iters):
        decoded = estimate_code(nodeMarg_par[it])
        err_mat[m, it] = np.sum(decoded != codeWord)
    if iters < max_iters:
        err_mat[m, iters:] = err_mat[m, iters - 1]

    # longest run (for plotting)
    if longest < iters:
        longest = iters
    # plot
    plt.figure(2)
    for m in range(num_runs):
        plt.plot(range(max_iters), err_mat[m])
    plt.xlabel('Iteration #')
    plt.ylabel('Hamming Distance')
    plt.title('Channel Noise ' + str(noise))

print('Question 1d-2')
num_runs = 10
noise = 0.1
codeWord = np.zeros([num_bits, 1])

# err_mat[i,j]: stores the hamming distance for run i at iteration j
err_mat = np.zeros([num_runs, max_iters])

longest = 0
for m in range(num_runs):
    print('## run ', m + 1, '##')
    # generate data
    noisyCodeWord = channel_noise(codeWord, noise)
    F = init_ldpc_graph(H, noisyCodeWord, 1 - noise)
    actualError = np.sum(noisyCodeWord != codeWord) / len(noisyCodeWord)
    print('actual error: ', actualError)

    # Run Parallel Loopy BP
    nodeMarg_par = run_loopy_bp_parallel(F, max_iters, conv_tol)
    iters = len(nodeMarg_par)
    if iters < max_iters:
        print('Parallel LBP converged in', iters, 'iterations.')
    else:
        print('Parallel LBP did not converge. Terminated at', max_iters, 'iterations')

    # determine code
    decoded = estimate_code(nodeMarg_par[-1])
    ham_dist = np.sum(decoded != codeWord)
    print('Hamming Distance: ', ham_dist)
    print('Error after correction ', ham_dist / len(noisyCodeWord))

    # store error at each iteration of this run
    for it in range(iters):
        decoded = estimate_code(nodeMarg_par[it])
        err_mat[m, it] = np.sum(decoded != codeWord)
    if iters < max_iters:
        err_mat[m, iters:] = err_mat[m, iters - 1]

    # longest run (for plotting)
    if longest < iters:
        longest = iters
    # plot
    plt.figure(2)
    for m in range(num_runs):
        plt.plot(range(max_iters), err_mat[m])
    plt.xlabel('Iteration #')
    plt.ylabel('Hamming Distance')
    plt.title('Channel Noise ' + str(noise))

# TODO: use modified code from 1c

## QUESTION 1e #######################

print('Question 1e')
ldpc = sio.loadmat('ldpc36-1600.mat')
G = ldpc['G']
H = ldpc['H']
logo = ldpc['logo']
num_bits = np.size(G,0)
max_iters = 30

# generate data
noise = 0.06
w = np.reshape(logo, [np.size(logo),1])
codeWord = np.dot(G.astype(float), w) % 2
noisyCodeWord = channel_noise(codeWord, noise)
F = init_ldpc_graph( H, noisyCodeWord, 1-noise )
actualError = np.sum(noisyCodeWord != codeWord) / len(noisyCodeWord)
print('actual error: ' , actualError)
nodeMarg_par = run_loopy_bp_parallel(F, max_iters, conv_tol)
iters = len(nodeMarg_par)
if iters < max_iters:
    print('Parallel LBP converged in', iters, 'iterations.')
else:
    print('Parallel LBP did not converge. Terminated at', max_iters, 'iterations')

plotAtIters = [ 0, 1, 2, 3, 5, 10, 20, 30 ]
plt.figure(5)
fig,axs = plt.subplots(2,4,constrained_layout=True,subplot_kw={'xticks': [], 'yticks': []})
for index in range(len(plotAtIters)):
    i = plotAtIters[index]

    if i == 0:
        thisX = noisyCodeWord
    elif i >= len(nodeMarg_par):
        thisX = estimate_code(nodeMarg_par[-1])
    else:
        thisX = estimate_code(nodeMarg_par[i])

    msg = np.reshape(thisX[:1600], [40,40])
    par = np.reshape(thisX[1600:], [40,40])
    im = np.vstack([msg,par])
    axs.flat[index].imshow(im, cmap = 'gray')
    axs.flat[index].set_xlabel('Iter '+str(i))


## QUESTION 1f #######################
print('Question 1f')
ldpc = sio.loadmat('ldpc36-1600.mat')
G = ldpc['G']
H = ldpc['H']
logo = ldpc['logo']
num_bits = np.size(G, 0)
max_iters = 30

# generate data
noise = 0.1
w = np.reshape(logo, [np.size(logo), 1])
codeWord = np.dot(G.astype(float), w) % 2
noisyCodeWord = channel_noise(codeWord, noise)
F = init_ldpc_graph(H, noisyCodeWord, 1 - noise)
actualError = np.sum(noisyCodeWord != codeWord) / len(noisyCodeWord)
print('actual error: ', actualError)
nodeMarg_par = run_loopy_bp_parallel(F, max_iters, conv_tol)
iters = len(nodeMarg_par)
if iters < max_iters:
    print('Parallel LBP converged in', iters, 'iterations.')
else:
    print('Parallel LBP did not converge. Terminated at', max_iters, 'iterations')

plotAtIters = [0, 1, 2, 3, 5, 10, 20, 30]
plt.figure(5)
fig, axs = plt.subplots(2, 4, constrained_layout=True, subplot_kw={'xticks': [], 'yticks': []})
for index in range(len(plotAtIters)):
    i = plotAtIters[index]

    if i == 0:
        thisX = noisyCodeWord
    elif i >= len(nodeMarg_par):
        thisX = estimate_code(nodeMarg_par[-1])
    else:
        thisX = estimate_code(nodeMarg_par[i])

    msg = np.reshape(thisX[:1600], [40, 40])
    par = np.reshape(thisX[1600:], [40, 40])
    im = np.vstack([msg, par])
    axs.flat[index].imshow(im, cmap='gray')
    axs.flat[index].set_xlabel('Iter ' + str(i))


print('a')




