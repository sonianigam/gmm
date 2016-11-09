#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats


def main():
    """
    This function runs your code for problem 2.

    You can also use this to test your code for problem 1,
    but make sure that you do not leave anything in here that will interfere
    with problem 2. Especially make sure that gmm_est does not output anything
    extraneous, as problem 2 has a very specific expected output.
    """
    file_path = sys.argv[1]
    file = open(file_path)

    #ignore title
    first_line = file.readline()

    #read the remaining data within the file
    body = file.readlines()

    X1 = []
    X2 = []

    for line in body:
        values = line.strip().split(',')
        if values[1] == '1':
            X1.append(float(values[0]))
        else:
            X2.append(float(values[0]))


    # #plots histogram of raw data to determine initial params
    # bins = 50 
    # plt.subplot(2,1,1)
    # plt.title('Class 1 Data Distribution')
    # plt.ylabel('Frequency')
    # plt.hist(X1, bins) 
    # plt.subplot(2,1,2) 
    # plt.title('Class 2 Data Distribution')
    # plt.ylabel('Frequency')
    # plt.xlabel('Value')
    # plt.hist(X2, bins) 
    # plt.savefig('likelihood_class1.png')

    # YOUR CODE FOR PROBLEM 2 GOES HERE

    mu_results1, sigma2_results1, w_results1, L1 = gmm_est(X1, [10, 30], [7, 5], [.6, .4], 20)

    # mu_results1, sigma2_results1, w_results1 are all numpy arrays
    # with learned parameters from Class 1
    print 'Class 1'
    print 'mu =', mu_results1, '\nsigma^2 =', sigma2_results1, '\nw =', w_results1

    mu_results2, sigma2_results2, w_results2, L2 = gmm_est(X2, [-30, -10, 50], [10, 13, 15], [.2, .5, .3], 20)

    # mu_results2, sigma2_results2, w_results2 are all numpy arrays
    # with learned parameters from Class 2
    print 'Class 2'
    print 'mu =', mu_results2, '\nsigma^2 =', sigma2_results2, '\nw =', w_results2


    #plot log by iteration number
    X_plot = []

    for i in xrange(20):
        X_plot.append(i)

    plt.plot(X_plot, L1, "ro")
    plt.ylabel('Log-Likelihood Values')
    plt.xlabel('Iteration Number')
    plt.axis([0, 20, min(L1)-1, max(L1)+1])
    plt.title('X1 Results')
    plt.savefig('Log-LikelihoodX1.png')

    plt.clf()

    plt.plot(X_plot, L2, "ro")
    plt.ylabel('Log-Likelihood Values')
    plt.xlabel('Iteration Number')
    plt.axis([0, 20, min(L2)-1, max(L2)+1])
    plt.title('X2Results')
    plt.savefig('Log-LikelihoodX2.png')

def gmm_est(X, mu_init, sigmasq_init, wt_init, its):
    """
    Input Parameters:
      - X             : N 1-dimensional data points (a 1-by-N numpy array)
      - mu_init       : initial means of K Gaussian components (a 1-by-K numpy array)
      - sigmasq_init  : initial  variances of K Gaussian components (a 1-by-K numpy array)
      - wt_init       : initial weights of k Gaussian components (a 1-by-K numpy array that sums to 1)
      - its           : number of iterations for the EM algorithm

    Returns:
      - mu            : means of Gaussian components (a 1-by-K numpy array)
      - sigmasq       : variances of Gaussian components (a 1-by-K numpy array)
      - wt            : weights of Gaussian components (a 1-by-K numpy array, sums to 1)
      - L             : log likelihood
    """

    # YOUR CODE FOR PROBLEM 1 HERE
    N = len(X)
    k = len(mu_init)
    L = []

    #initialize
    mu = mu_init
    wt = wt_init
    sigmasq = sigmasq_init

    #number of iterations
    for x in xrange(its):
        #number of Gaussian components
        for y in xrange(k):
            #Big gamma
            total = 0
            #list of little gammas acorss all data points (j)
            data_points = []

            for data in X:
                #normal distribution
                probability = scipy.stats.norm(mu[y], np.sqrt(sigmasq[y])).pdf(data)
                denom = 0

                #calc demonimator component of little gamma
                for i in xrange(k):
                    #normal distribution
                    denom += wt[i] * scipy.stats.norm(mu[i], np.sqrt(sigmasq[i])).pdf(data)

                #calc little gamma
                if denom != 0:
                    value = wt[y] * probability / denom
                    data_points.append(value)
                #append to list of little gammas
                else:
                    data_points.append(0.0)

            #aggregate all little gammas across data points to get big gamma value
            total += np.sum(data_points)

            #update weight and mean 
            wt[y] = total / N
            mu[y] = np.dot(data_points, X) /total  

            #variance list
            sigma = []

            #across data points, calculate a component of new variance: (data-mean)^2
            for data2 in X:
                value = (data2 - mu[y])**2
                sigma.append(value)

            #update variance: sum little gamma * above component --> total
            sigmasq[y] = np.dot(data_points, sigma) / total


        L_temp = 0

        #across all data points
        for data3 in X:
            temp_total = 0
            #across all gaussians: k
            for i in xrange(k):
                #normal distribution
                temp_total += wt[i] * scipy.stats.norm(mu[i], np.sqrt(sigmasq[i])).pdf(data3)

            #aggregate log liklihood
            L_temp += np.log(temp_total)

        #component's log liklihood
        L.append(L_temp)
        print L
    
    return mu, sigmasq, wt, L


def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2

if __name__ == '__main__':
    main()
