#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UC Irvine CS274B

Handout version:  Some unimplemented functionality indicated by TODO
"""

from factor_utils import SimpleFactorGraph, FastDiscreteFactor
import numpy as np
import copy


def init_ldpc_graph(H, y, theta):
    '''
    INIT_LDPC_GRAPH - Creates a factor graph from a codebook H, noisy data y,
    and binomial noise parameter 1-theta.
    '''
    G = SimpleFactorGraph()
    # z = channel_noise(y, 1-theta)
    # comparison = z == y
    # value = np.zeros((len(y),1))
    # counter = 0
    # for k in comparison:
    #     if k:
    #         value[counter,0] = theta
    #     else:
    #         value[counter,0] = 1-theta
    #     counter = counter + 1
    # Create variables for each column of H matrix
    list_2_add_nodes_from = []
    str1 = 'X'
    for i in range(len(H[0])):
        str_2_add = str1 + str(i)
        list_2_add_nodes_from.append(str_2_add)
    G.add_nodes_from(list_2_add_nodes_from)
    # Create factors for each row of H matrix
    factor_list = []
    for i in range(len(H)):
        list_2_factors = []
        a = np.where(H[i, :] == 1)
        for k in range(len(a[0])):
            str_2_add = str1 + str(a[0][k])
            list_2_factors.append(str_2_add)
        probs = give_the_matrix(len(list_2_factors))
        factor_list.append(FastDiscreteFactor(list_2_factors, [2] * len(list_2_factors),probs))
        G.add_factors(factor_list[i])

    G.add_nodes_from(factor_list)
    for i in range(len(factor_list)):
        a = np.where(H[i, :] == 1)
        for k in range(len(a[0])):
            str_2_add = str1 + str(a[0][k])
            G.add_edges_from([(str_2_add, factor_list[i])])

    factor_list_unary = []
    # Create unary factors
    for i in range(len(list_2_add_nodes_from)):
        if y[i] == 1:
            factor_list_unary.append(FastDiscreteFactor([list_2_add_nodes_from[i]], [2], np.array([1-theta, theta])))
            G.add_factors(FastDiscreteFactor([list_2_add_nodes_from[i]], [2], np.array([1-theta, theta])))
        else:
            factor_list_unary.append(FastDiscreteFactor([list_2_add_nodes_from[i]], [2], np.array([theta, 1-theta])))
            G.add_factors(FastDiscreteFactor([list_2_add_nodes_from[i]], [2], np.array([theta, 1-theta])))
        G.add_edges_from([(list_2_add_nodes_from[i], factor_list_unary[i])])

    G.add_nodes_from(factor_list_unary)

    # TODO: FILL IN THE REST OF THE CODE!
    # - For efficiency, define factors use the FastDiscreteFactor() class
    #   (It has the same interface as DiscreteFactor(), but allows faster indexing)
    # - Add factor/variable nodes using G.add_node (or G.add_nodes_from) and G.add_factors
    # - Don't forget to add edges between variables and factors by calling G.add_edges_from 
    # - Suggested variable names:  'X0', 'X1', 'X2' ...
    
    return G

def give_the_matrix(length_of_variables):
    limit = 2**length_of_variables
    output = np.zeros((limit,1))
    for k in range(limit):
        a = f'{k:010b}'
        if a.count('1') % 2 == 0:
            output[k] = 1
        else:
            output[k] = 0
    return output

def estimate_code(marg):
    '''
    ESTIMATE_CODE - Estimates a codeword based on its marginal distributions.
    This returns the most likely value of each bit under the marginals.

    Parameters
    ----------
    marg : n-entry dictionary containing the marginal distribution of each variable

    Returns
    -------
    decoded : n*1 numpy array containing the estimated codeword

    '''
    n = len(marg)
    decoded = np.zeros([n,1])
    # TODO: FILL IN THE REST OF THE CODE
    for i in marg:
        a = marg[i]
        if a[0] > a[1]:
            decoded[int(i[1:]),0] = 0
        else:
            decoded[int(i[1:]),0] = 1
    return decoded
        
def channel_noise(x, noise):
    '''
    CHANNEL_NOISE - Simulates a noisy channel with specified noise level.

    Parameters
    ----------
    x : n*1 numpy array of original data
    noise : noise level

    Returns
    -------
    y : n*1 numpy array of noisy data

    '''
    num_bits = len(x)
    I_flip = np.where( np.random.rand(num_bits, 1) < noise)[0]
    y = copy.deepcopy(x)
    y[I_flip] = 1 - y[I_flip]
    
    return y
